"""Tests for the EventKit boundary: conversions, access, and calendar read/write operations.

PyObjC objects are stand-in mocks; the few operations that reach for an EventKit class method or
constructor get a minimal namespace swapped in for that call, leaving the real participation
constants intact for the pure conversion tests.
"""

import types
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, call

import EventKit
import Foundation
import pytest

from calque import store
from calque.errors import AccessError, CalendarError, WriteError
from calque.model import Adoption, Availability, Mirror, Participation, Plan, Status, Tag, Window, tag


@pytest.fixture
def start() -> datetime:
    return datetime(2026, 6, 5, 9, 0, tzinfo=UTC)


def window(start: datetime, hours: float = 1.0) -> Window:
    return Window(start, start + timedelta(hours=hours))


def stub_event(window: Window, **fields: object) -> Mock:
    """A mock EventKit event whose accessors return the given fields with sensible defaults."""
    item = Mock()
    item.startDate.return_value = store.to_nsdate(window.start)
    item.endDate.return_value = store.to_nsdate(window.end)
    item.eventIdentifier.return_value = fields.get("identifier", "event-1")
    item.title.return_value = fields.get("title", "Meeting")
    item.selfAttendee.return_value = fields.get("self_attendee")
    item.hasAttendees.return_value = fields.get("has_attendees", False)
    item.isAllDay.return_value = fields.get("all_day", False)
    item.status.return_value = fields.get("status", EventKit.EKEventStatusConfirmed)
    item.notes.return_value = fields.get("notes")
    item.location.return_value = fields.get("location")
    url = fields.get("url")
    item.URL.return_value = None if url is None else Foundation.NSURL.URLWithString_(url)
    zone = fields.get("time_zone")
    item.timeZone.return_value = None if zone is None else Foundation.NSTimeZone.timeZoneWithName_(zone)
    item.availability.return_value = fields.get("availability", EventKit.EKEventAvailabilityBusy)
    created = fields.get("created")
    item.creationDate.return_value = store.to_nsdate(created) if isinstance(created, datetime) else None
    item.calendar.return_value.supportedEventAvailabilities.return_value = fields.get("supported", ALL_AVAILABILITIES)
    return item


ALL_AVAILABILITIES = (
    EventKit.EKCalendarEventAvailabilityBusy
    | EventKit.EKCalendarEventAvailabilityFree
    | EventKit.EKCalendarEventAvailabilityTentative
    | EventKit.EKCalendarEventAvailabilityUnavailable
)


def self_attendee(status: int = EventKit.EKParticipantStatusAccepted) -> Mock:
    """A mock self-attendee record exposing the current user's participant status."""
    return Mock(**{"participantStatus.return_value": status})


def detached_store(native: object) -> store.CalendarStore:
    """A CalendarStore bound to a mock native store, bypassing the access-gated constructor."""
    instance = store.CalendarStore.__new__(store.CalendarStore)
    instance.store = native
    return instance


def test_nsdate_roundtrips_through_a_utc_datetime(start: datetime) -> None:
    assert store.to_datetime(store.to_nsdate(start)) == start


def test_to_window_spans_the_events_start_and_end(start: datetime) -> None:
    assert store.to_window(stub_event(window(start))) == window(start)


def test_to_event_folds_the_occurrence_start_into_the_identifier(start: datetime) -> None:
    event = store.to_event(stub_event(window(start), identifier="abc", title="Standup"), "Client")
    assert event.identifier.startswith("abc@")
    assert event.title == "Standup"
    assert event.account == "Client"
    assert event.window == window(start)


def test_to_event_substitutes_an_empty_title_when_absent(start: datetime) -> None:
    assert store.to_event(stub_event(window(start), title=None), "Client").title == ""


def test_to_event_carries_the_cancellation_status(start: datetime) -> None:
    item = stub_event(window(start), status=EventKit.EKEventStatusCanceled)
    assert store.to_event(item, "Client").status == Status.CANCELLED


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (EventKit.EKEventStatusNone, Status.NONE),
        (EventKit.EKEventStatusConfirmed, Status.CONFIRMED),
        (EventKit.EKEventStatusTentative, Status.TENTATIVE),
        (EventKit.EKEventStatusCanceled, Status.CANCELLED),
    ],
)
def test_to_status_maps_each_status(status: int, expected: Status) -> None:
    assert store.to_status(status) == expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (EventKit.EKParticipantStatusAccepted, Participation.ACCEPTED),
        (EventKit.EKParticipantStatusTentative, Participation.TENTATIVE),
        (EventKit.EKParticipantStatusDeclined, Participation.DECLINED),
        (EventKit.EKParticipantStatusPending, Participation.PENDING),
        (EventKit.EKParticipantStatusUnknown, Participation.UNKNOWN),
    ],
)
def test_to_participation_maps_each_status(status: int, expected: Participation) -> None:
    assert store.to_participation(status) == expected


def test_response_treats_a_self_authored_event_as_accepted(start: datetime) -> None:
    # No self-attendee record and no attendees at all: a block we authored ourselves.
    assert store.response(stub_event(window(start), has_attendees=False)) == Participation.ACCEPTED


def test_response_reads_the_current_users_status_from_the_self_attendee(start: datetime) -> None:
    item = stub_event(window(start), has_attendees=True, self_attendee=self_attendee())
    assert store.response(item) == Participation.ACCEPTED


def test_response_maps_an_unanswered_invitation_to_pending(start: datetime) -> None:
    # Exchange reports a not-yet-answered invitation as the Unknown participant status on our own record.
    item = stub_event(
        window(start), has_attendees=True, self_attendee=self_attendee(EventKit.EKParticipantStatusUnknown)
    )
    assert store.response(item) == Participation.PENDING


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (EventKit.EKParticipantStatusTentative, Participation.TENTATIVE),
        (EventKit.EKParticipantStatusDeclined, Participation.DECLINED),
    ],
)
def test_response_carries_a_definite_self_attendee_status(
    start: datetime, status: int, expected: Participation
) -> None:
    item = stub_event(window(start), has_attendees=True, self_attendee=self_attendee(status))
    assert store.response(item) == expected


def test_response_is_unknown_when_invited_without_a_self_record(start: datetime) -> None:
    # Attendees exist but EventKit exposes no record for us: genuinely indeterminate, not implicitly accepted.
    assert store.response(stub_event(window(start), has_attendees=True)) == Participation.UNKNOWN


def fake_eventkit(monkeypatch: pytest.MonkeyPatch, **attributes: object) -> None:
    """Swap a minimal EventKit namespace into the store module for one test."""
    monkeypatch.setattr(store, "EventKit", types.SimpleNamespace(**attributes))


def test_request_access_returns_when_full_access_is_already_granted(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_eventkit(
        monkeypatch,
        EKEntityTypeEvent=0,
        EKAuthorizationStatusFullAccess=3,
        EKEventStore=types.SimpleNamespace(authorizationStatusForEntityType_=lambda _entity: 3),
    )
    native = Mock()
    store.request_access(native)
    native.requestFullAccessToEventsWithCompletion_.assert_not_called()


def test_request_access_returns_when_the_user_grants_access(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_eventkit(
        monkeypatch,
        EKEntityTypeEvent=0,
        EKAuthorizationStatusFullAccess=3,
        EKEventStore=types.SimpleNamespace(authorizationStatusForEntityType_=lambda _entity: 0),
    )
    native = Mock(requestFullAccessToEventsWithCompletion_=lambda handler: handler(True, None))  # noqa: FBT003
    store.request_access(native)


def test_request_access_raises_when_the_user_denies(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_eventkit(
        monkeypatch,
        EKEntityTypeEvent=0,
        EKAuthorizationStatusFullAccess=3,
        EKEventStore=types.SimpleNamespace(authorizationStatusForEntityType_=lambda _entity: 0),
    )
    native = Mock(requestFullAccessToEventsWithCompletion_=lambda handler: handler(False, "refused"))  # noqa: FBT003
    with pytest.raises(AccessError):
        store.request_access(native)


def test_request_access_raises_when_the_decision_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_eventkit(
        monkeypatch,
        EKEntityTypeEvent=0,
        EKAuthorizationStatusFullAccess=3,
        EKEventStore=types.SimpleNamespace(authorizationStatusForEntityType_=lambda _entity: 0),
    )
    native = Mock(requestFullAccessToEventsWithCompletion_=lambda _handler: None)
    with pytest.raises(AccessError):
        store.request_access(native, timeout=0.01)


def test_calendar_store_creates_the_event_store_and_requests_access(monkeypatch: pytest.MonkeyPatch) -> None:
    native = Mock()
    fake_eventkit(
        monkeypatch,
        EKEventStore=types.SimpleNamespace(alloc=lambda: types.SimpleNamespace(init=lambda: native)),
    )
    granted = Mock()
    monkeypatch.setattr(store, "request_access", granted)
    instance = store.CalendarStore()
    assert instance.store is native
    granted.assert_called_once_with(native)


def calendar_named(account: str, title: str) -> Mock:
    """A mock native EKCalendar reporting the given account and own title."""
    return Mock(**{"source.return_value.title.return_value": account, "title.return_value": title})


def test_qualified_names_are_sorted() -> None:
    native = Mock()
    native.calendarsForEntityType_.return_value = [calendar_named("Work", "Office"), calendar_named("Home", "Personal")]
    assert detached_store(native).qualified_names() == ["Home.Personal", "Work.Office"]


def test_calendar_matches_a_qualified_title() -> None:
    native = Mock()
    native.calendarsForEntityType_.return_value = [calendar_named("Work", "Office")]
    found = detached_store(native).calendar("Work.Office")
    assert found.qualified == "Work.Office"


def test_calendar_raises_when_no_title_matches() -> None:
    native = Mock()
    native.calendarsForEntityType_.return_value = [calendar_named("Work", "Office")]
    store = detached_store(native)
    with pytest.raises(CalendarError):
        store.calendar("Missing")


def test_fetch_yields_events_matching_the_window_predicate(start: datetime) -> None:
    native = Mock()
    matched = [stub_event(window(start))]
    native.eventsMatchingPredicate_.return_value = matched
    calendar = store.Calendar(detached_store(native), Mock(), "Work.Office")
    assert list(calendar.store.fetch(calendar, window(start))) == matched


def test_build_event_tags_a_new_block(monkeypatch: pytest.MonkeyPatch, start: datetime) -> None:
    created = stub_event(window(start))
    fake_eventkit(
        monkeypatch,
        EKEvent=types.SimpleNamespace(eventWithEventStore_=lambda _store: created),
    )
    calendar = store.Calendar(detached_store(Mock()), Mock(), "Work.Office")
    mirror = Mirror("source-1", window(start), "Busy", origin="Personal.Home")
    built = calendar.store.build_event(mirror, calendar)
    assert built is created
    created.setTitle_.assert_called_once_with("Busy")
    created.setNotes_.assert_called_once_with(tag("Personal.Home", "source-1"))


def test_apply_fields_writes_the_mirror_bounds(start: datetime) -> None:
    event = stub_event(window(start))
    mirror = Mirror("source-1", window(start), "Busy")
    assert detached_store(Mock()).apply_fields(event, mirror) is event
    assert store.to_datetime(event.setStartDate_.call_args.args[0]) == start
    assert store.to_datetime(event.setEndDate_.call_args.args[0]) == start + timedelta(hours=1)


def test_apply_fields_writes_a_changed_title_on_an_existing_event(start: datetime) -> None:
    # Stock calque only reset the times on update, so a renamed source event kept its old title in the copy.
    event = stub_event(window(start), title="Old name")
    detached_store(Mock()).apply_fields(event, Mirror("source-1", window(start), "New name"))
    event.setTitle_.assert_called_once_with("New name")


def test_apply_fields_copies_location_url_notes_and_time_zone(start: datetime) -> None:
    event = stub_event(window(start))
    mirror = Mirror(
        "source-1",
        window(start),
        "Clinic",
        location="Room 4",
        url="https://zoom.us/j/123",
        notes="Bring the slides",
        time_zone="America/Chicago",
        origin="Work.Office",
    )
    detached_store(Mock()).apply_fields(event, mirror)
    event.setLocation_.assert_called_once_with("Room 4")
    assert event.setURL_.call_args.args[0].absoluteString() == "https://zoom.us/j/123"
    event.setNotes_.assert_called_once_with(tag("Work.Office", "source-1", "Bring the slides"))
    assert event.setNotes_.call_args.args[0].startswith("Bring the slides\n\n")
    assert event.setTimeZone_.call_args.args[0].name() == "America/Chicago"
    event.setAllDay_.assert_called_once_with(False)  # noqa: FBT003 - asserting the positional PyObjC argument


def test_apply_fields_clears_fields_the_source_no_longer_has(start: datetime) -> None:
    event = stub_event(window(start), location="Room 4", url="https://zoom.us/j/123", time_zone="America/Chicago")
    detached_store(Mock()).apply_fields(event, Mirror("source-1", window(start), "Clinic", origin="Work.Office"))
    event.setLocation_.assert_called_once_with(None)
    event.setURL_.assert_called_once_with(None)
    event.setTimeZone_.assert_called_once_with(None)
    event.setNotes_.assert_called_once_with(tag("Work.Office", "source-1"))


def test_apply_fields_never_touches_alarms_or_attendees(start: datetime) -> None:
    event = stub_event(window(start))
    detached_store(Mock()).apply_fields(event, Mirror("source-1", window(start), "Clinic"))
    touched = {name for name, _, _ in event.method_calls}
    assert not {name for name in touched if "larm" in name or "ttendee" in name}


def test_apply_fields_sets_all_day_before_reusing_the_sources_own_dates(start: datetime) -> None:
    first, last = object(), object()  # the source event's native dates: they must be passed through untouched
    event = stub_event(window(start))
    mirror = Mirror("source-1", window(start, hours=24), "Holiday", all_day=True, dates=(first, last))
    detached_store(Mock()).apply_fields(event, mirror)
    names = [name for name, _, _ in event.method_calls]
    assert names.index("setAllDay_") < names.index("setStartDate_") < names.index("setEndDate_")
    assert event.method_calls[names.index("setAllDay_")] == call.setAllDay_(True)
    assert event.setStartDate_.call_args.args[0] is first
    assert event.setEndDate_.call_args.args[0] is last
    # An all-day event floats: it is given no time zone, which could shift its day.
    event.setTimeZone_.assert_not_called()


def test_apply_fields_falls_back_to_the_window_for_an_all_day_block_without_native_dates(start: datetime) -> None:
    event = stub_event(window(start))
    detached_store(Mock()).apply_fields(event, Mirror("source-1", window(start, hours=24), "Holiday", all_day=True))
    assert store.to_datetime(event.setStartDate_.call_args.args[0]) == start
    event.setTimeZone_.assert_not_called()


def test_apply_fields_sets_an_availability_the_calendar_supports(start: datetime) -> None:
    event = stub_event(window(start))
    mirror = Mirror("source-1", window(start), "Hold", availability=Availability.FREE)
    detached_store(Mock()).apply_fields(event, mirror)
    event.setAvailability_.assert_called_once_with(EventKit.EKEventAvailabilityFree)


@pytest.mark.parametrize(
    ("availability", "supported"),
    [
        pytest.param(Availability.TENTATIVE, EventKit.EKCalendarEventAvailabilityBusy, id="calendar cannot store it"),
        pytest.param(Availability.BUSY, EventKit.EKCalendarEventAvailabilityNone, id="calendar stores none"),
        pytest.param(Availability.UNSUPPORTED, ALL_AVAILABILITIES, id="source did not record one"),
    ],
)
def test_apply_fields_leaves_an_unsupported_availability_alone(
    start: datetime, availability: Availability, supported: int
) -> None:
    event = stub_event(window(start), supported=supported)
    detached_store(Mock()).apply_fields(event, Mirror("source-1", window(start), "Hold", availability=availability))
    event.setAvailability_.assert_not_called()


def test_to_event_reads_every_copied_field(start: datetime) -> None:
    item = stub_event(
        window(start),
        location=" Room 4 ",
        url="https://zoom.us/j/123",
        notes="Bring the slides",
        time_zone="America/Chicago",
        availability=EventKit.EKEventAvailabilityFree,
        has_attendees=True,
        created=start - timedelta(days=3),
    )
    event = store.to_event(item, "Client")
    assert event.location == "Room 4"
    assert event.url == "https://zoom.us/j/123"
    assert event.notes == "Bring the slides"
    assert event.time_zone == "America/Chicago"
    assert event.availability is Availability.FREE
    assert event.invited
    assert event.created == start - timedelta(days=3)
    assert event.tag is None
    assert event.dates is None


def test_to_event_defaults_the_fields_an_event_lacks(start: datetime) -> None:
    event = store.to_event(stub_event(window(start)), "Client")
    assert (event.location, event.url, event.notes, event.time_zone, event.created) == (None, None, None, None, None)
    assert event.availability is Availability.BUSY
    assert not event.invited


def test_to_event_separates_our_marker_from_the_notes(start: datetime) -> None:
    item = stub_event(window(start), notes=tag("Work.Office", "source-1", "Bring the slides"))
    event = store.to_event(item, "Client")
    assert event.notes == "Bring the slides"
    assert event.tag == Tag(origin="Work.Office", identifier="source-1")


def test_to_event_recognises_a_block_tagged_by_stock_calque(start: datetime) -> None:
    event = store.to_event(stub_event(window(start), notes=tag("Work.Office", "source-1")), "Client")
    assert event.notes is None
    assert event.tag == Tag(origin="Work.Office", identifier="source-1")


def test_to_event_keeps_an_all_day_events_own_dates(start: datetime) -> None:
    item = stub_event(window(start, hours=24), all_day=True)
    event = store.to_event(item, "Client")
    assert event.all_day
    assert event.dates == (item.startDate.return_value, item.endDate.return_value)
    assert event.dates[0] is item.startDate.return_value


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (EventKit.EKEventAvailabilityBusy, Availability.BUSY),
        (EventKit.EKEventAvailabilityFree, Availability.FREE),
        (EventKit.EKEventAvailabilityTentative, Availability.TENTATIVE),
        (EventKit.EKEventAvailabilityUnavailable, Availability.UNAVAILABLE),
        (EventKit.EKEventAvailabilityNotSupported, Availability.UNSUPPORTED),
    ],
)
def test_to_availability_maps_each_code(code: int, expected: Availability) -> None:
    assert store.to_availability(code) is expected


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        pytest.param(EventKit.EKSourceTypeExchange, False, id="exchange drops it"),
        pytest.param(EventKit.EKSourceTypeCalDAV, True, id="caldav keeps it"),
    ],
)
def test_keeps_urls_is_false_only_for_exchange(kind: int, *, expected: bool) -> None:
    native = Mock(**{"source.return_value.sourceType.return_value": kind})
    assert store.Calendar(detached_store(Mock()), native, "Work.Office").keeps_urls() is expected


def test_availabilities_lists_what_the_calendar_can_store() -> None:
    native = Mock(**{"supportedEventAvailabilities.return_value": EventKit.EKCalendarEventAvailabilityBusy | 2})
    calendar = store.Calendar(detached_store(Mock()), native, "Work.Office")
    assert calendar.availabilities() == frozenset({Availability.BUSY, Availability.FREE})
    native.supportedEventAvailabilities.return_value = EventKit.EKCalendarEventAvailabilityNone
    assert calendar.availabilities() == frozenset()


def test_save_persists_a_successful_write() -> None:
    native = Mock()
    native.saveEvent_span_error_.return_value = (True, None)
    detached_store(native).save(Mock())
    native.saveEvent_span_error_.assert_called_once()


def test_save_raises_when_the_write_fails() -> None:
    native = Mock()
    event = Mock()
    native.saveEvent_span_error_.return_value = (False, "rejected")
    store = detached_store(native)
    with pytest.raises(WriteError):
        store.save(event)


def test_remove_deletes_a_successful_block() -> None:
    native = Mock()
    native.removeEvent_span_error_.return_value = (True, None)
    detached_store(native).remove(Mock())
    native.removeEvent_span_error_.assert_called_once()


def test_remove_raises_when_the_delete_fails() -> None:
    native = Mock()
    event = Mock()
    native.removeEvent_span_error_.return_value = (False, "rejected")
    store = detached_store(native)
    with pytest.raises(WriteError):
        store.remove(event)


def calendar_with(native_store: Mock, account: str = "Client") -> store.Calendar:
    """A Calendar over a mock native store whose source reports the given account."""
    native = Mock(**{"source.return_value.title.return_value": account})
    return store.Calendar(detached_store(native_store), native, "Work.Office")


def test_calendar_title_and_source_delegate_to_the_native_object() -> None:
    native = Mock(**{"title.return_value": "Office"})
    calendar = store.Calendar(detached_store(Mock()), native, "Work.Office")
    assert calendar.title() == "Office"
    assert calendar.source() is native.source.return_value


def test_events_converts_every_fetched_item(start: datetime) -> None:
    native_store = Mock()
    native_store.eventsMatchingPredicate_.return_value = [stub_event(window(start), identifier="a")]
    calendar = calendar_with(native_store)
    events = list(calendar.events(window(start)))
    assert [event.account for event in events] == ["Client"]
    assert events[0].identifier.startswith("a@")


def test_busy_excludes_our_own_mirror_blocks(start: datetime) -> None:
    native_store = Mock()
    native_store.eventsMatchingPredicate_.return_value = [
        stub_event(window(start), identifier="genuine", notes="lunch with team"),
        stub_event(window(start), identifier="ours", notes=tag("Personal.Home", "source-1")),
    ]
    calendar = calendar_with(native_store)
    assert [event.identifier.split("@")[0] for event in calendar.busy(window(start))] == ["genuine"]


def test_tagged_pairs_our_blocks_with_their_tag(start: datetime) -> None:
    ours = stub_event(window(start), notes=tag("Personal.Home", "source-1"))
    native_store = Mock()
    native_store.eventsMatchingPredicate_.return_value = [stub_event(window(start), notes="foreign"), ours]
    calendar = calendar_with(native_store)
    assert list(calendar.tagged(window(start))) == [(Tag(origin="Personal.Home", identifier="source-1"), ours)]


def test_mirrors_reads_back_blocks_keyed_by_source(start: datetime) -> None:
    native_store = Mock()
    native_store.eventsMatchingPredicate_.return_value = [
        stub_event(window(start), title="Busy", notes=tag("Personal.Home", "source-1")),
    ]
    mirrors = calendar_with(native_store).mirrors(window(start), "Personal.Home")
    assert mirrors == {"source-1": Mirror("source-1", window(start), "Busy")}


def test_mirrors_reads_back_every_copied_field_without_the_marker(start: datetime) -> None:
    native_store = Mock()
    native_store.eventsMatchingPredicate_.return_value = [
        stub_event(
            window(start, hours=24),
            title=None,
            notes=tag("Personal.Home", "source-1", "Bring the slides"),
            location="Room 4",
            url="https://zoom.us/j/123",
            availability=EventKit.EKEventAvailabilityFree,
            all_day=True,
        ),
    ]
    expected = Mirror(
        "source-1",
        window(start, hours=24),
        "",
        location="Room 4",
        url="https://zoom.us/j/123",
        notes="Bring the slides",
        availability=Availability.FREE,
        all_day=True,
    )
    assert calendar_with(native_store).mirrors(window(start), "Personal.Home") == {"source-1": expected}


def test_mirrors_excludes_blocks_from_a_different_origin(start: datetime) -> None:
    # A block another source wrote into this calendar must not appear when reconciling our own origin,
    # or it would be treated as an orphan and deleted.
    native_store = Mock()
    native_store.eventsMatchingPredicate_.return_value = [
        stub_event(window(start), title="Busy", notes=tag("Other.Calendar", "theirs")),
    ]
    assert calendar_with(native_store).mirrors(window(start), "Personal.Home") == {}


def test_apply_creates_updates_and_deletes_against_the_store(monkeypatch: pytest.MonkeyPatch, start: datetime) -> None:
    fake_eventkit(
        monkeypatch,
        EKEvent=types.SimpleNamespace(eventWithEventStore_=lambda _store: stub_event(window(start))),
        EKSpanThisEvent=EventKit.EKSpanThisEvent,
    )
    existing = stub_event(window(start), notes=tag("Personal.Home", "stays"))
    going = stub_event(window(start), notes=tag("Personal.Home", "gone"))
    native_store = Mock()
    native_store.saveEvent_span_error_.return_value = (True, None)
    native_store.removeEvent_span_error_.return_value = (True, None)
    native_store.eventsMatchingPredicate_.return_value = [existing, going]
    calendar = calendar_with(native_store)
    plan = Plan(
        create=(Mirror("new", window(start), "Busy"),),
        update=(Mirror("stays", window(start, hours=2), "Busy"),),
        delete=(Mirror("gone", window(start), "Busy"),),
    )
    calendar.apply(plan, window(start))
    assert native_store.saveEvent_span_error_.call_count == 2  # one create, one update
    native_store.removeEvent_span_error_.assert_called_once_with(going, EventKit.EKSpanThisEvent, None)


def test_apply_adopts_an_untagged_event_by_tagging_it_in_place(start: datetime) -> None:
    stray = stub_event(window(start), identifier="stray", title="Clinic", notes="left by another tool")
    bystander = stub_event(window(start), identifier="bystander", title="Clinic")
    native_store = Mock()
    native_store.saveEvent_span_error_.return_value = (True, None)
    native_store.eventsMatchingPredicate_.return_value = [bystander, stray]
    calendar = calendar_with(native_store)
    target = store.to_event(stray, "Client")
    mirror = Mirror("source-1", window(start), "Clinic", notes="Bring the slides", origin="Personal.Home")
    calendar.apply(Plan(create=(), update=(), delete=(), adopt=(Adoption(mirror, target),)), window(start))
    native_store.saveEvent_span_error_.assert_called_once_with(stray, EventKit.EKSpanThisEvent, None)
    stray.setNotes_.assert_called_once_with(tag("Personal.Home", "source-1", "Bring the slides"))
    bystander.setNotes_.assert_not_called()
    native_store.removeEvent_span_error_.assert_not_called()
