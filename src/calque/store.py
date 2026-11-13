"""EventKit boundary: read a calendar's events and write their mirrored copies, all locally.

Every impure interaction with macOS lives here so the reconciliation core stays pure. PyObjC
returns untyped objects, so this module is the one place the type checker is relaxed.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import EventKit
import Foundation

from calque.errors import AccessError, CalendarError, WriteError
from calque.model import (
    Availability,
    Event,
    Mirror,
    Participation,
    Plan,
    Source,
    Status,
    Tag,
    Window,
    clean,
    parts,
    strip_marker,
    tag,
    untag,
)

AVAILABILITIES = {
    EventKit.EKEventAvailabilityBusy: Availability.BUSY,
    EventKit.EKEventAvailabilityFree: Availability.FREE,
    EventKit.EKEventAvailabilityTentative: Availability.TENTATIVE,
    EventKit.EKEventAvailabilityUnavailable: Availability.UNAVAILABLE,
}

# The ``EKCalendarEventAvailabilityMask`` bit a calendar sets for each availability it can store.
AVAILABILITY_MASKS = {
    Availability.BUSY: EventKit.EKCalendarEventAvailabilityBusy,
    Availability.FREE: EventKit.EKCalendarEventAvailabilityFree,
    Availability.TENTATIVE: EventKit.EKCalendarEventAvailabilityTentative,
    Availability.UNAVAILABLE: EventKit.EKCalendarEventAvailabilityUnavailable,
}


def to_nsdate(value: datetime) -> Any:
    """Convert a timezone-aware datetime to an ``NSDate``."""
    return Foundation.NSDate.dateWithTimeIntervalSince1970_(value.timestamp())


def to_datetime(value: Any) -> datetime:
    """Convert an ``NSDate`` to a UTC datetime."""
    return datetime.fromtimestamp(value.timeIntervalSince1970(), tz=UTC)


def to_window(event: Any) -> Window:
    """Convert an EventKit event to a Window."""
    return Window(to_datetime(event.startDate()), to_datetime(event.endDate()))


def identify(item: Any) -> str:
    """The identifier of one occurrence of an EventKit event.

    Every occurrence of a recurring event shares one ``eventIdentifier``, so the occurrence
    start is folded into the identifier to keep each instance a distinct mirror block.
    """
    return f"{item.eventIdentifier()}@{int(item.startDate().timeIntervalSince1970())}"


def to_availability(availability: int) -> Availability:
    """Map an ``EKEventAvailability`` code to an availability value."""
    return AVAILABILITIES.get(availability, Availability.UNSUPPORTED)


def to_url(item: Any) -> str | None:
    """The event's URL field as text, or ``None`` when it has none."""
    url = item.URL()
    return None if url is None else clean(url.absoluteString())


def to_time_zone(item: Any) -> str | None:
    """The name of the event's time zone, or ``None`` for a floating (and so every all-day) event."""
    time_zone = item.timeZone()
    return None if time_zone is None else str(time_zone.name())


def to_dates(item: Any) -> tuple[object, object] | None:
    """An all-day event's own native start and end dates, to be reused untouched when it is copied."""
    return (item.startDate(), item.endDate()) if item.isAllDay() else None


def to_event(item: Any, account: str) -> Event:
    """Convert an EventKit event to an Event for the given account, separating our marker from its notes."""
    created = item.creationDate()
    return Event(
        identifier=identify(item),
        title=item.title() or "",
        account=account,
        window=to_window(item),
        all_day=bool(item.isAllDay()),
        participation=response(item),
        status=to_status(item.status()),
        notes=strip_marker(item.notes()),
        location=clean(item.location()),
        url=to_url(item),
        time_zone=to_time_zone(item),
        availability=to_availability(item.availability()),
        tag=untag(item.notes()),
        invited=bool(item.hasAttendees()),
        created=None if created is None else to_datetime(created),
        dates=to_dates(item),
    )


def to_mirror(item: Any, source: str) -> Mirror:
    """Read back a mirror block we wrote for the source event ``source``, in the form it is compared in."""
    body, _, trailer = parts(item.notes())
    return Mirror(
        source=source,
        window=to_window(item),
        title=item.title() or "",
        location=clean(item.location()),
        url=to_url(item),
        notes=strip_marker(body),
        trailer=trailer,
        time_zone=to_time_zone(item),
        availability=to_availability(item.availability()),
        all_day=bool(item.isAllDay()),
    )


def to_participation(status: int) -> Participation:
    """Map an ``EKParticipantStatus`` code to a participation value."""
    match status:
        case EventKit.EKParticipantStatusAccepted:
            return Participation.ACCEPTED
        case EventKit.EKParticipantStatusTentative:
            return Participation.TENTATIVE
        case EventKit.EKParticipantStatusDeclined:
            return Participation.DECLINED
        case EventKit.EKParticipantStatusPending:
            return Participation.PENDING
        case _:
            return Participation.UNKNOWN


def to_status(status: int) -> Status:
    """Map an ``EKEventStatus`` code to a status value."""
    match status:
        case EventKit.EKEventStatusConfirmed:
            return Status.CONFIRMED
        case EventKit.EKEventStatusTentative:
            return Status.TENTATIVE
        case EventKit.EKEventStatusCanceled:
            return Status.CANCELLED
        case _:
            return Status.NONE


def response(item: Any) -> Participation:
    """Determine the current user's response to an event from their own attendee record."""
    attendee = item.selfAttendee()
    if attendee is not None:
        participation = to_participation(attendee.participantStatus())
        return Participation.PENDING if participation is Participation.UNKNOWN else participation
    return Participation.UNKNOWN if item.hasAttendees() else Participation.ACCEPTED


def qualify(calendar: Any) -> str:
    """The account-qualified name of a native calendar, e.g. ``Work.Calendar``."""
    return f"{calendar.source().title()}.{calendar.title()}"


def request_access(store: Any, *, timeout: float = 30.0) -> None:
    """Block until the user grants full calendar access, raising if denied or timed out."""
    status = EventKit.EKEventStore.authorizationStatusForEntityType_(EventKit.EKEntityTypeEvent)
    if status == EventKit.EKAuthorizationStatusFullAccess:
        return
    outcome: dict[str, object] = {}
    finished = threading.Event()

    def handler(granted: bool, error: Any) -> None:  # noqa: FBT001 - PyObjC callback signature
        """Record the access decision and release the wait."""
        outcome["granted"] = bool(granted)
        outcome["error"] = error
        finished.set()

    store.requestFullAccessToEventsWithCompletion_(handler)
    if not finished.wait(timeout) or not outcome.get("granted"):
        raise AccessError(outcome.get("error") or "timed out")


class Calendar:
    """A calendar in the local store, bound to the store that created it.

    Wraps the untyped PyObjC ``EKCalendar`` (``native``) and exposes the calendar-scoped
    operations the mirror needs, delegating the low-level EventKit work back to the store.
    """

    __slots__ = ("native", "qualified", "store")

    def __init__(self, store: CalendarStore, native: Any, qualified: str) -> None:
        """Bind a native ``EKCalendar`` to the store that produced it."""
        self.store = store
        self.native = native
        self.qualified = qualified

    def title(self) -> str:
        """The calendar's own display title."""
        return self.native.title()

    def source(self) -> Source:
        """The account this calendar belongs to."""
        return self.native.source()

    def availabilities(self) -> frozenset[Availability]:
        """The availabilities this calendar can store on an event; empty where it records none."""
        supported = self.native.supportedEventAvailabilities()
        return frozenset(availability for availability, mask in AVAILABILITY_MASKS.items() if supported & mask)

    def keeps_urls(self) -> bool:
        """Whether an event's URL field survives on this calendar's server.

        Exchange has no such field: the URL stays in the local store, so it reads back intact here, but it never
        reaches the server or any other device.
        """
        return bool(self.native.source().sourceType() != EventKit.EKSourceTypeExchange)

    def events(self, window: Window) -> Iterator[Event]:
        """Yield this calendar's events in the window, reduced to the fields the mirror needs."""
        account = self.source().title()
        return (to_event(item, account) for item in self.store.fetch(self, window))

    def busy(self, window: Window) -> Iterator[Event]:
        """Yield the genuine events here (excluding our own mirror blocks) as candidate busy periods."""
        return (event for event in self.events(window) if event.tag is None)

    def tagged(self, window: Window) -> Iterator[tuple[Tag, Any]]:
        """Yield our own mirror blocks in the window, paired with their decoded tag."""
        for item in self.store.fetch(self, window):
            marker = untag(item.notes())
            if marker is not None:
                yield marker, item

    def mirrors(self, window: Window, origin: str) -> dict[str, Mirror]:
        """Read back the mirror blocks we wrote here from ``origin``, keyed by source identifier.

        Scoping to one origin keeps reconciliation from treating another source's blocks in this
        calendar as orphans: each source only reconciles against its own blocks.
        """
        return {
            marker.identifier: to_mirror(item, marker.identifier)
            for marker, item in self.tagged(window)
            if marker.origin == origin
        }

    def apply(self, plan: Plan, window: Window) -> None:
        """Carry out a reconciliation plan against this calendar."""
        handles = {marker.identifier: item for marker, item in self.tagged(window)}
        untagged = {identify(item): item for item in self.store.fetch(self, window) if untag(item.notes()) is None}
        for adoption in plan.adopt:
            self.store.save(self.store.apply_fields(untagged[adoption.target.identifier], adoption.mirror))
        for mirror in plan.create:
            self.store.save(self.store.build_event(mirror, self))
        for mirror in plan.update:
            self.store.save(self.store.apply_fields(handles[mirror.source], mirror))
        for mirror in plan.delete:
            self.store.remove(handles[mirror.source])


class CalendarStore:
    """A thin wrapper over an authorised ``EKEventStore`` — the program's one external resource."""

    def __init__(self) -> None:
        """Create the underlying event store and block until the user grants full calendar access."""
        self.store = EventKit.EKEventStore.alloc().init()
        request_access(self.store)

    def qualified_names(self) -> list[str]:
        """Return the account-qualified names of all calendars in the local store, sorted alphabetically."""
        return sorted(qualify(calendar) for calendar in self.store.calendarsForEntityType_(EventKit.EKEntityTypeEvent))

    def calendar(self, name: str) -> Calendar:
        """Return the calendar whose account-qualified name matches exactly, or raise if none does."""
        for calendar in self.store.calendarsForEntityType_(EventKit.EKEntityTypeEvent):
            qualified = qualify(calendar)
            if qualified == name:
                return Calendar(store=self, native=calendar, qualified=qualified)
        raise CalendarError(name)

    def fetch(self, calendar: Calendar, window: Window) -> Iterator[Any]:
        """Yield the raw EventKit events in the given calendar and window."""
        predicate = self.store.predicateForEventsWithStartDate_endDate_calendars_(
            to_nsdate(window.start),
            to_nsdate(window.end),
            [calendar.native],
        )
        yield from self.store.eventsMatchingPredicate_(predicate)

    def build_event(self, mirror: Mirror, calendar: Calendar) -> Any:
        """Create a new event in the calendar that copies the mirror, tagged with its origin and source identifier."""
        event = EventKit.EKEvent.eventWithEventStore_(self.store)
        event.setCalendar_(calendar.native)
        return self.apply_fields(event, mirror)

    def apply_fields(self, event: Any, mirror: Mirror) -> Any:
        """Make an event, new or existing, match the mirror in every copied field, returning the event.

        The marker is rewritten as the last line of the notes, followed by any trailer the event had gained
        below its old marker, which is not ours to discard. Alarms and attendees are deliberately never
        copied: an alarm would fire twice, and an attendee could be sent an invitation.
        """
        event.setTitle_(mirror.title)
        event.setLocation_(mirror.location)
        event.setURL_(Foundation.NSURL.URLWithString_(mirror.url) if mirror.url else None)
        trailer = parts(event.notes())[2]
        notes = tag(mirror.origin, mirror.source, mirror.notes)
        event.setNotes_(f"{notes}\n\n{trailer}" if trailer else notes)
        # The all-day flag goes on before the dates, which for an all-day event are the source's own, so the
        # day cannot shift. An all-day event floats, so only a timed one is given a time zone.
        event.setAllDay_(mirror.all_day)
        if mirror.all_day and mirror.dates:
            start, end = mirror.dates
        else:
            start, end = to_nsdate(mirror.start), to_nsdate(mirror.end)
        if not mirror.all_day:
            zone = Foundation.NSTimeZone.timeZoneWithName_(mirror.time_zone) if mirror.time_zone else None
            event.setTimeZone_(zone)
        event.setStartDate_(start)
        event.setEndDate_(end)
        mask = AVAILABILITY_MASKS.get(mirror.availability, 0)
        if event.calendar().supportedEventAvailabilities() & mask:
            event.setAvailability_(next(code for code, value in AVAILABILITIES.items() if value is mirror.availability))
        return event

    def save(self, event: Any) -> None:
        """Persist an event to the store, raising on failure."""
        succeeded, error = self.store.saveEvent_span_error_(event, EventKit.EKSpanThisEvent, None)
        if not succeeded:
            raise WriteError(error)

    def remove(self, event: Any) -> None:
        """Delete an event from the store, raising on failure."""
        succeeded, error = self.store.removeEvent_span_error_(event, EventKit.EKSpanThisEvent, None)
        if not succeeded:
            raise WriteError(error)
