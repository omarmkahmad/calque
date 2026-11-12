"""Domain model for the mirror: events, their mirrored copies, and the reconciliation plan."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol

MARKER = "⟦calque⟧"


@dataclass(frozen=True, slots=True)
class Window:
    """A half-open time range in UTC — a query range or an interval occupied by an event."""

    start: datetime
    end: datetime

    def __str__(self) -> str:
        """A readable local-time rendering of the range, dropping seconds and the timezone."""
        start = self.start.astimezone()
        end = self.end.astimezone()
        until = f"{end:%H:%M}" if start.date() == end.date() else f"{end:%a %Y-%m-%d %H:%M}"
        return f"{start:%a %Y-%m-%d %H:%M} to {until}"


class Participation(Enum):
    """The current user's response to an event invitation."""

    ACCEPTED = "accepted"
    TENTATIVE = "tentative"
    DECLINED = "declined"
    PENDING = "pending"
    UNKNOWN = "unknown"


class Status(Enum):
    """The lifecycle status of an event itself (its iCalendar ``STATUS``), independent of participation."""

    NONE = "none"
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class Availability(Enum):
    """How an event counts towards free/busy time; ``UNSUPPORTED`` where the calendar does not record it."""

    BUSY = "busy"
    FREE = "free"
    TENTATIVE = "tentative"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"


class Source(Protocol):
    """The slice of an EventKit ``EKSource`` we read: the account a calendar belongs to.

    A narrowing of PyObjC's untyped objects; ``store.Calendar.source`` returns one.
    """

    def title(self) -> str:
        """The account name (e.g. the Google or Exchange account behind the calendar)."""
        ...


@dataclass(frozen=True, slots=True)
class Tag:
    """The provenance encoded on a mirror block: the calendar it was copied from and that source's event id."""

    origin: str
    identifier: str


def clean(text: str | None) -> str | None:
    """Normalise free text for storage and comparison: unified newlines, no outer whitespace, ``None`` when blank."""
    return (text or "").replace("\r\n", "\n").strip() or None


def split(notes: str | None) -> tuple[str | None, str | None]:
    """Split a notes field into the user-visible body and our marker line, either of which may be absent.

    The marker is the last line of the notes. Stock calque wrote it as the whole field (which is also a last
    line) and so, for a block whose notes gained text afterwards, it is still recognised on the first line.
    """
    lines = (clean(notes) or "").split("\n")
    if lines[-1].startswith(MARKER):
        return clean("\n".join(lines[:-1])), lines[-1]
    if lines[0].startswith(MARKER):
        return clean("\n".join(lines[1:])), lines[0]
    return clean(notes), None


def strip_marker(notes: str | None) -> str | None:
    """The notes without any marker of ours, so a copy of a copy does not carry the old marker along."""
    body, marker = split(notes)
    return strip_marker(body) if marker else body


def tag(origin: str, identifier: str, notes: str | None = None) -> str:
    """Encode a mirror block's origin calendar and source event id as the last line of its notes.

    The fields are space-separated: the identifier never contains a space, so it is the first token,
    and the origin (which may) is the remainder. A space survives any normalisation that some
    calendar backends apply, where a tab or newline does not. Any marker already in ``notes`` is replaced.
    """
    body = strip_marker(notes)
    marker = f"{MARKER} {identifier} {origin}"
    return f"{body}\n\n{marker}" if body else marker


def untag(notes: str | None) -> Tag | None:
    """Recover the origin and source id from a mirror block's notes, or ``None`` if not one of ours."""
    _, marker = split(notes)
    if marker is None:
        return None
    identifier, _, origin = marker.removeprefix(MARKER).strip().partition(" ")
    if not identifier:
        return None
    return Tag(origin=origin, identifier=identifier)


@dataclass(frozen=True, slots=True)
class Event:
    """A source-calendar event reduced to the fields the mirror cares about.

    ``notes`` never includes our marker: a mirror block's provenance is carried by ``tag`` instead.
    """

    identifier: str
    title: str
    account: str
    window: Window
    all_day: bool
    participation: Participation
    status: Status
    notes: str | None = None
    location: str | None = None
    url: str | None = None
    time_zone: str | None = None
    availability: Availability = Availability.BUSY
    tag: Tag | None = None
    # Whether the event has attendees: an invitation is never mistaken for a copy of another event.
    invited: bool = False
    created: datetime | None = None
    # An all-day event's own start and end as the store's native date objects, reused verbatim when it is copied
    # so the day can never shift through a conversion.
    dates: tuple[object, object] | None = field(default=None, compare=False, repr=False)

    @property
    def start(self) -> datetime:
        """The event's start, in UTC."""
        return self.window.start

    @property
    def end(self) -> datetime:
        """The event's end, in UTC."""
        return self.window.end

    def __str__(self) -> str:
        """A human-readable representation of the event for logging and diagnostics."""
        return f"title={self.title!r} window={self.window} account={self.account!r} status={self.status.value} participation={self.participation.value} identifier={self.identifier!r} notes={self.notes!r}"


@dataclass(frozen=True, slots=True)
class Mirror:
    """An anonymised busy block in the target calendar, linked to its source by identifier."""

    source: str
    window: Window
    title: str
    location: str | None = None
    url: str | None = None
    notes: str | None = None
    time_zone: str | None = None
    availability: Availability = Availability.BUSY
    all_day: bool = False
    # Excluded from equality:
    original: str | None = field(default=None, compare=False)
    # The source's native all-day dates (see ``Event.dates``), absent on blocks read back for diffing.
    dates: tuple[object, object] | None = field(default=None, compare=False, repr=False)
    # The calendar this block was copied from, stored in the tag so a mirror is never returned to its
    # source. Excluded from equality: it is fixed per direction and absent on blocks read back for diffing.
    origin: str = field(default="", compare=False)

    @property
    def start(self) -> datetime:
        """The block's start, in UTC."""
        return self.window.start

    @property
    def end(self) -> datetime:
        """The block's end, in UTC."""
        return self.window.end

    def __str__(self) -> str:
        """A human-readable representation of the mirror, in local time, for diagnostics."""
        return (
            f"title={self.title!r} original={self.original!r} window={self.window} all_day={self.all_day} "
            f"location={self.location!r} url={self.url!r} notes={self.notes!r} time_zone={self.time_zone!r} "
            f"availability={self.availability.value} source={self.source!r}"
        )


@dataclass(frozen=True, slots=True)
class Adoption:
    """An untagged event already in the target that matches a wanted block, to be tagged instead of duplicated."""

    mirror: Mirror
    target: Event

    def __str__(self) -> str:
        """A human-readable representation of the adoption for diagnostics."""
        return f"{self.mirror} adopting={self.target.identifier!r}"


@dataclass(frozen=True, slots=True)
class Plan:
    """The reconciliation outcome: blocks to create, update, and remove in the target, and events to adopt."""

    create: tuple[Mirror, ...]
    update: tuple[Mirror, ...]
    delete: tuple[Mirror, ...]
    adopt: tuple[Adoption, ...] = ()

    @property
    def empty(self) -> bool:
        """Whether the plan would change nothing."""
        return not (self.create or self.update or self.delete or self.adopt)

    def __str__(self) -> str:
        """A human-readable summary of the plan's contents for logging and diagnostics."""
        return f"create={len(self.create)} update={len(self.update)} delete={len(self.delete)} adopt={len(self.adopt)}"

    def log(self) -> None:
        """Log the plan's contents in a human-readable form."""
        logging.info("create:")
        for event in self.create:
            logging.info("  %s", event)
        logging.info("update:")
        for event in self.update:
            logging.info("  %s", event)
        logging.info("delete:")
        for event in self.delete:
            logging.info("  %s", event)
        logging.info("adopt:")
        for adoption in self.adopt:
            logging.info("  %s", adoption)
