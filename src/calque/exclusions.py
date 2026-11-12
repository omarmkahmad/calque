"""Composable exclusion rules that keep selected source events out of the mirror.

An :data:`Exclusion` is a predicate over an event, true when that event should be dropped.
Builders turn configuration and target-calendar context into rules, :func:`rules` assembles
the active ones, and :func:`excluded` runs them.
"""

import logging
import re
from collections.abc import Callable, Collection, Container, Iterable, Iterator
from datetime import UTC, date, datetime, time, timedelta

from calque.config import Config
from calque.model import Availability, Event, Participation, Status

Exclusion = Callable[[Event], bool]


def dates_spanned(start: datetime, end: datetime) -> Iterator[date]:
    """Yield each local calendar date an event touches, from its start through its end."""
    day = start.date()
    while day <= end.date():
        yield day
        day += timedelta(days=1)


def by_title(patterns: tuple[re.Pattern[str], ...]) -> Exclusion:
    """Build a rule excluding events whose title matches any of the given patterns."""

    def rule(event: Event) -> bool:
        """Test the event's title against every configured pattern."""
        for pattern in patterns:
            if pattern.search(event.title):
                logging.debug("excluding %r by title (%r)", event.title, pattern.pattern)
                return True
        return False

    return rule


def unlisted(patterns: tuple[re.Pattern[str], ...]) -> Exclusion:
    """Build a rule excluding events whose title matches none of the given whitelist patterns.

    The inverse of :func:`by_title`: where a source calendar opts in with include patterns, only events
    carrying one of the markers are mirrored and every other event is dropped.
    """

    def rule(event: Event) -> bool:
        """True when the event's title matches none of the whitelist patterns."""
        result = not any(pattern.search(event.title) for pattern in patterns)
        if result:
            logging.debug("excluding %r as unlisted (no include pattern matched)", event.title)
        return result

    return rule


def by_clash(events: Iterable[Event]) -> Exclusion:
    """Build a rule excluding events that overlap any interval already busy in the target."""
    busy = tuple(event.window for event in events)

    def rule(event: Event) -> bool:
        """Half-open overlap test of the event against each busy interval."""
        for interval in busy:
            if event.start < interval.end and interval.start < event.end:
                logging.debug("excluding %r due to clash with %s", event.title, interval)
                return True
        return False

    return rule


def by_passed() -> Exclusion:
    """Build a rule excluding events whose end time has already passed.

    Enabled by cleanup: the cut-off is the moment the rule is built — cheap to read and captured by
    the returned closure — so a finished event drops out of the desired set and reconciliation
    deletes its now-stale mirror block instead of keeping it through the lookback window.
    """
    now = datetime.now(UTC)

    def rule(event: Event) -> bool:
        """True when the event has already ended."""
        result = event.end < now
        if result:
            logging.debug("excluding %r as already ended", event.title)
        return result

    return rule


def by_origin(target: str) -> Exclusion:
    """Build a rule excluding our own mirror blocks that we previously copied from ``target``.

    A mirror is dropped only when it would return to the calendar it came from — which would become a
    mirror of a mirror (this would otherwise be caused by a deleted event).
    A mirror in the primary, sourced from one calendar, still propagates onward to the others.
    """

    def rule(event: Event) -> bool:
        """True when the event is our mirror block originating from the target calendar."""
        result = event.tag is not None and event.tag.origin == target
        if result:
            logging.debug("excluding %r as our own mirror returning to %s", event.title, target)
        return result

    return rule


def by_copy(copies: Collection[str]) -> Exclusion:
    """Build a rule excluding untagged events identified as duplicates of an event in another calendar.

    Such a duplicate (a copy left by another mirroring tool, or the same invitation received on a second
    account) already stands for an event that is mirrored from elsewhere; mirroring it onward as well would
    duplicate that event everywhere. A copy that is adopted, and so tagged, is no longer excluded.
    """

    def rule(event: Event) -> bool:
        """True when the event is an untagged copy."""
        result = event.tag is None and event.identifier in copies
        if result:
            logging.debug("excluding %r as a duplicate of another calendar's event", event.title)
        return result

    return rule


def is_free(event: Event) -> bool:
    """Whether the event is marked as free, so it does not block any time."""
    result = event.availability is Availability.FREE
    if result:
        logging.debug("excluding %r as free", event.title)
    return result


def is_all_day(event: Event) -> bool:
    """Whether the event is an all-day event, which would otherwise block the whole day."""
    result = event.all_day
    if result:
        logging.debug("excluding %r by all-day", event.title)
    return result


def is_cancelled(event: Event) -> bool:
    """Whether the event has been cancelled according to its status."""
    result = event.status is Status.CANCELLED
    if result:
        logging.debug("excluding %r as cancelled", event.title)
    return result


def by_participation(statuses: frozenset[Participation]) -> Exclusion:
    """Build a rule excluding events whose participation is not among the mirrored statuses."""

    def rule(event: Event) -> bool:
        """True when the event's participation is not one we mirror."""
        result = event.participation not in statuses
        if result:
            logging.debug("excluding %r by participation (%s)", event.title, event.participation.value)
        return result

    return rule


def by_hours(days: Container[int], opening: time, closing: time) -> Exclusion:
    """Build a rule excluding events that fall entirely outside the working-hours window.

    Hours are compared in local time. An event is kept if it overlaps the window on any
    working day it touches, so an event straddling the edge (e.g. 17:00-19:00) still mirrors.
    """

    def rule(event: Event) -> bool:
        """True when no working-hours interval on any day the event spans overlaps it."""
        start = event.start.astimezone()
        end = event.end.astimezone()
        result = not any(
            start < datetime.combine(day, closing, start.tzinfo) and datetime.combine(day, opening, start.tzinfo) < end
            for day in dates_spanned(start, end)
            if day.weekday() in days
        )
        if result:
            logging.debug("excluding %r by hours (%s-%s)", event.title, opening, closing)
        return result

    return rule


def intrinsics(config: Config) -> Iterator[Exclusion]:
    """Yield the rules that judge an event on its own properties, whichever calendars it moves between."""
    yield by_participation(config.statuses)
    yield is_cancelled
    if config.exclude_patterns:
        yield by_title(config.exclude_patterns)
    if config.exclude_all_day:
        yield is_all_day
    if config.exclude_free:
        yield is_free
    if config.exclude_out_of_hours:
        yield by_hours(config.work_days, config.work_start, config.work_end)
    if config.cleanup:
        yield by_passed()


def rules(
    config: Config, events: Iterable[Event], source: str, target: str, copies: Collection[str] = ()
) -> tuple[Exclusion, ...]:
    """Return the active exclusion rule chain for mirroring from ``source`` into ``target`` given its busy ``events``.

    The full exclusion chain:
    - intrinsic rules:
      - participation status
      - cancelled events
      - title patterns
      - all-day
      - free
      - out-of-hours
      - finished events, when cleanup is enabled
    - our own mirror blocks returning to ``target``
    - untagged ``copies``: duplicates of another calendar's event, which are not mirrored a second time
    - events the ``source`` calendar's include patterns don't whitelist, when it has any
    - clash against target busy periods - which itself filters using the intrinsic rules

    A mirror block is excluded only when its origin is ``target``, so a block written by a previous sync is never
    returned to its source (which would chain into a mirror of a mirror) yet still propagates to the other
    calendars. Genuinely busy periods are the target events the intrinsic rules don't themselves exclude, so a
    focus block, all-day, out-of-hours or unaccepted event never blocks a source event. The include-pattern
    whitelist is keyed on ``source`` and applied only to source events, never to the target's busy periods.
    """
    intrinsic = tuple(intrinsics(config))
    exclusions = (*intrinsic, by_origin(target), by_copy(copies))
    if whitelist := config.calendar_include_patterns.get(source):
        exclusions = (*exclusions, unlisted(whitelist))
    if not config.exclude_clashes:
        return exclusions
    # The clash rule takes the target events, which we filter to the genuinely busy periods using the intrinsic rules.
    # Then, when it is used, it excludes any source event that overlaps the busy periods.
    return (*exclusions, by_clash(included(events, intrinsic)))


def excluded(event: Event, exclusions: Iterable[Exclusion]) -> bool:
    """Predicate: whether any rule in the chain excludes the event."""
    return any(rule(event) for rule in exclusions)


def included(events: Iterable[Event], exclusions: Iterable[Exclusion]) -> Iterator[Event]:
    """Filter: yield only the events that no exclusion rule rejects."""
    for event in events:
        if not excluded(event, exclusions):
            yield event
