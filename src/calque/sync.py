"""Drive a synchronisation run: reconcile events between calendars and mirror them between primary and auxiliaries.

The reconciliation core (:func:`to_mirrors`, :func:`designate`, :func:`adoptions`, :func:`relocate`, :func:`settle`,
:func:`reconcile`) stays pure; the impure edges — reading the clock and opening the calendar store — live in
:func:`synchronise`, the process entry.
"""

import logging
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from datetime import UTC, datetime, timedelta

from calque.config import Config
from calque.errors import DeleteLimitError
from calque.exclusions import included, intrinsics, rules
from calque.model import Adoption, Availability, Event, Mirror, Participation, Plan, Status, Tag, Window
from calque.store import Calendar, CalendarStore

# Ranks an event whose creation time the store does not report after every event that has one.
LATEST = datetime.max.replace(tzinfo=UTC)


def to_mirrors(events: Iterable[Event], title: str, origin: str) -> dict[str, Mirror]:
    """Transform events into mirror blocks from ``origin``, titled via ``title``, keyed by source id."""
    return {
        event.identifier: Mirror(
            source=event.identifier,
            window=event.window,
            title=title.format(**{field.name: getattr(event, field.name) for field in fields(event)}),
            location=event.location,
            url=event.url,
            notes=event.notes,
            time_zone=event.time_zone,
            availability=event.availability,
            all_day=event.all_day,
            original=event.title,
            origin=origin,
            dates=event.dates,
        )
        for event in events
    }


@dataclass(frozen=True, slots=True)
class Carry:
    """What one pass is told about the whole run.

    :param copies: Identifiers of the untagged events, in any calendar, that duplicate another calendar's event.
    :param pending: Events an earlier pass of a dry run would have left in this pass's source, had it written.
    """

    copies: frozenset[str] = frozenset()
    pending: tuple[Event, ...] = ()


def designate(calendars: Sequence[Iterable[Event]]) -> frozenset[str]:
    """Identify the untagged events that duplicate an event in another calendar, by identifier.

    Events sharing a title, start and end across two or more calendars are one event and its duplicates: copies
    another mirroring tool left behind, or one invitation received on several accounts. Exactly one untagged
    event of each such group is kept as the original — one a mirror block already points at, else an invitation,
    else the earliest created, else the one in the earliest-listed calendar — unless the group already holds a
    mirror block of ours, which stands for an original elsewhere. The rest are duplicates: they are not mirrored
    again, and :func:`adoptions` decides which of them become mirror blocks.

    An event some mirror block was copied from is never a duplicate, so it is never adopted: that would let a
    later run overwrite or delete an event that is not ours.
    """
    members = [(order, event) for order, events in enumerate(calendars) for event in events]
    sourced = {event.tag.identifier for _, event in members if event.tag}
    groups: dict[tuple[str, Window], list[tuple[int, Event]]] = defaultdict(list)
    for order, event in members:
        groups[event.title, event.window].append((order, event))

    def rank(member: tuple[int, Event]) -> tuple[bool, bool, datetime, int]:
        """Order a group's untagged events from the likeliest original to the least."""
        order, event = member
        return (event.identifier not in sourced, not event.invited, event.created or LATEST, order)

    copies: set[str] = set()
    for group in groups.values():
        if len({order for order, _ in group}) > 1:
            untagged = sorted((member for member in group if member[1].tag is None), key=rank)
            anchored = len(untagged) < len(group)
            copies.update(
                event.identifier
                for _, event in (untagged if anchored else untagged[1:])
                if event.identifier not in sourced
            )
    return frozenset(copies)


def adoptions(
    desired: Mapping[str, Mirror], existing: Collection[str], events: Iterable[Event], copies: Collection[str]
) -> tuple[tuple[Adoption, ...], frozenset[str]]:
    """Match each wanted block not yet in the target to an untagged event there of the same title, start and end.

    Such a block is never created, since the target already shows the event. Where the matching event is one of
    the ``copies`` and not an invitation it is adopted: tagged as the block, and from then on kept in step with
    the source. An invitation, or an event that is an original in its own right, is left untouched and the
    block is simply not written. Returns the adoptions, and the source ids of the blocks left unwritten.

    Each event is matched at most once, so two wanted blocks that look alike never claim the same event.
    """
    untagged: dict[tuple[str, Window], Event] = {}
    for event in events:
        if event.tag is None:
            untagged.setdefault((event.title, event.window), event)
    matched = {
        source: untagged.pop(key)
        for source, mirror in desired.items()
        if source not in existing and (key := (mirror.title, mirror.window)) in untagged
    }
    adoptable = {source: event for source, event in matched.items() if event.identifier in copies and not event.invited}
    adopt = tuple(Adoption(desired[source], event) for source, event in adoptable.items())
    return adopt, frozenset(matched.keys() - adoptable.keys())


def with_link(url: str, notes: str | None) -> str:
    """The notes carrying ``url`` on their first line."""
    return f"{url}\n\n{notes}" if notes else url


def relocate(mirror: Mirror) -> Mirror:
    """Move a block's URL to the first line of its notes, unless the notes or location already contain it.

    A block without a URL is returned as it is.
    """
    if not mirror.url:
        return mirror
    carried = mirror.url in (mirror.notes or "") or mirror.url in (mirror.location or "")
    return replace(mirror, url=None, notes=mirror.notes if carried else with_link(mirror.url, mirror.notes))


def settle(mirror: Mirror, existing: Mirror | None, availabilities: Collection[Availability]) -> Mirror:
    """Adapt a wanted block to what its target has shown it cannot store, so it is not rewritten on every run.

    - A target block that came back without the URL it was written with (some backends drop the field) carries
      the link on the first line of its notes instead, unless the notes or location already contain it.
    - A location the target moved into a trailer below our marker (Google does this with a meeting link) is
      left where the target put it.
    - An availability the target calendar cannot record is left as whatever the target block holds.
    """
    if existing is None:
        return mirror
    if existing.url is None:
        mirror = relocate(mirror)
    if mirror.location and existing.location is None and mirror.location in (existing.trailer or ""):
        mirror = replace(mirror, location=None)
    if mirror.availability not in availabilities:
        mirror = replace(mirror, availability=existing.availability)
    return mirror


def reconcile(desired: Mapping[str, Mirror], existing: Mapping[str, Mirror]) -> Plan:
    """Diff the blocks we want against the ones already present to produce a change set."""
    return Plan(
        create=tuple(mirror for source, mirror in desired.items() if source not in existing),
        update=tuple(mirror for source, mirror in desired.items() if source in existing and mirror != existing[source]),
        delete=tuple(mirror for source, mirror in existing.items() if source not in desired),
    )


def exceeds(plan: Plan, limit: int | None) -> bool:
    """Whether ``plan`` deletes more blocks than ``limit`` allows; a ``None`` limit never trips."""
    return limit is not None and len(plan.delete) > limit


def staged(plan: Plan, account: str) -> tuple[Event, ...]:
    """The events a plan's creations and adoptions would leave in its target, as a later pass would read them.

    A dry run writes nothing, so the primary is read back unchanged when it is fanned out. Overlaying these
    events lets the dry run report the fan-out as a real run would perform it. Knock-on effects of the plan's
    updates and deletions are not modelled.
    """

    def event(mirror: Mirror, identifier: str) -> Event:
        """The target event that writing ``mirror`` would produce."""
        return Event(
            identifier=identifier,
            title=mirror.title,
            account=account,
            window=mirror.window,
            all_day=mirror.all_day,
            participation=Participation.ACCEPTED,
            status=Status.NONE,
            notes=mirror.notes,
            location=mirror.location,
            url=mirror.url,
            time_zone=mirror.time_zone,
            availability=mirror.availability,
            tag=Tag(mirror.origin, mirror.source),
            dates=mirror.dates,
        )

    return (
        *(event(mirror, f"pending:{mirror.source}") for mirror in plan.create),
        *(event(adoption.mirror, adoption.target.identifier) for adoption in plan.adopt),
    )


def overlay(events: Iterable[Event], pending: Iterable[Event]) -> Iterable[Event]:
    """The events with each pending event added, or put in place of the event sharing its identifier."""
    return ({event.identifier: event for event in events} | {event.identifier: event for event in pending}).values()


def mirror(
    source: Calendar,
    target: Calendar,
    config: Config,
    window: Window,
    carry: Carry | None = None,
) -> Plan:
    """Run one mirror pass: read the source, reconcile against the target, and apply the plan, which is returned.

    The ``carry`` names the duplicate events, which are not mirrored from the source, and any pending events,
    which are overlaid on the source's. A block the target already shows as an untagged event is adopted or
    left unwritten rather than created (see :func:`adoptions`).

    A plan deleting more blocks than ``config.max_deletes`` allows is not applied at all, and nor is any plan
    on a dry run.
    """
    carry = carry or Carry()
    exclusions = rules(config, target.busy(window), source.qualified, target.qualified, carry.copies)
    title = config.title_for_calendar(source, target)
    events = overlay(source.events(window), carry.pending)
    desired = to_mirrors(included(events, exclusions), title, source.qualified)
    if not target.keeps_urls():
        desired = {identifier: relocate(block) for identifier, block in desired.items()}
    existing = target.mirrors(window, source.qualified)
    adopt, present = adoptions(desired, existing, target.events(window), carry.copies) if config.adopt else ((), ())
    handled = {*present, *(adoption.mirror.source for adoption in adopt)}
    availabilities = target.availabilities()
    plan = replace(
        reconcile(
            {
                identifier: settle(block, existing.get(identifier), availabilities)
                for identifier, block in desired.items()
                if identifier not in handled
            },
            existing,
        ),
        adopt=adopt,
    )
    if present:
        logging.debug(
            "%s -> %s: %d already in the target, left alone", source.qualified, target.qualified, len(present)
        )
    level = logging.DEBUG if plan.empty and not config.dry_run else logging.INFO
    logging.log(level, "%s -> %s: %s", source.qualified, target.qualified, plan)

    skipped = exceeds(plan, config.max_deletes)
    if skipped:
        logging.warning(
            "%s -> %s: skipped, %d deletes exceed the limit of %s",
            source.qualified,
            target.qualified,
            len(plan.delete),
            config.max_deletes,
        )

    if config.dry_run:
        plan.log()
        logging.info("dry run: no changes written")
    elif not skipped:
        target.apply(plan, window)
    return plan


def synchronise(config: Config, calendars: list[str]) -> None:
    """Collect every calendar's events into the primary, then fan the primary's combined events back out to the rest.

    The first calendar listed is the primary: each of the others is mirrored into it, so it holds every calendar's
    events, and it is then mirrored back out to each of the others. So a consultant's time with one client also
    shows as busy to the rest. A calendar named in ``config.muted`` is still read as a source but never written
    to, keeping its viewers from seeing the other commitments.

    A pass skipped for exceeding ``config.max_deletes`` is logged and does not stop the others; once all have run,
    :class:`~calque.errors.DeleteLimitError` is raised, unless this is a dry run.
    """
    now = datetime.now(UTC)
    window = Window(now - timedelta(days=config.lookback), now + timedelta(days=config.lookahead))
    store = CalendarStore()
    resolved = {title: store.calendar(title) for title in calendars}
    copies: frozenset[str] = frozenset()
    if config.adopt:
        genuine = tuple(intrinsics(config))
        copies = designate([tuple(included(resolved[title].events(window), genuine)) for title in calendars])
    primary, *auxiliaries = calendars
    plans: list[Plan] = []
    pending: list[Event] = []
    if primary not in config.muted:
        for auxiliary in auxiliaries:
            plans.append(mirror(resolved[auxiliary], resolved[primary], config, window, Carry(copies)))
            if config.dry_run:
                pending.extend(staged(plans[-1], resolved[primary].source().title()))
    plans.extend(
        mirror(resolved[primary], resolved[auxiliary], config, window, Carry(copies, tuple(pending)))
        for auxiliary in auxiliaries
        if auxiliary not in config.muted
    )
    if not config.dry_run and any(exceeds(plan, config.max_deletes) for plan in plans):
        raise DeleteLimitError(config.max_deletes)
