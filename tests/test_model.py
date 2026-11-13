"""Tests for the domain model: marker encoding, time rendering, and plan reporting."""

import logging
import time as clock
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from calque.model import (
    MARKER,
    Adoption,
    Availability,
    Event,
    Mirror,
    Participation,
    Plan,
    Status,
    Tag,
    Window,
    clean,
    parts,
    split,
    strip_marker,
    tag,
    untag,
)


@pytest.fixture(autouse=True)
def local_utc(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the local timezone to UTC so local-time rendering is deterministic."""
    monkeypatch.setenv("TZ", "UTC")
    clock.tzset()
    yield
    clock.tzset()


@pytest.fixture
def start() -> datetime:
    return datetime(2026, 6, 5, 9, 0, tzinfo=UTC)


def window(start: datetime, hours: float = 1.0) -> Window:
    return Window(start, start + timedelta(hours=hours))


def test_tag_then_untag_roundtrips() -> None:
    assert untag(tag("Work.Office", "ABC-123")) == Tag(origin="Work.Office", identifier="ABC-123")


def test_tag_roundtrips_an_origin_containing_spaces() -> None:
    # The identifier has no spaces, so splitting on the first space keeps a spaced account name intact.
    assert untag(tag("Example Team.Calendar", "ABC-123")) == Tag(origin="Example Team.Calendar", identifier="ABC-123")


def test_untag_ignores_foreign_notes() -> None:
    assert untag("a normal meeting note") is None


def test_untag_handles_missing_notes() -> None:
    assert untag(None) is None


def test_untag_rejects_marker_with_no_identifier() -> None:
    assert untag(MARKER) is None


def test_window_renders_a_same_day_range_without_repeating_the_date(start: datetime) -> None:
    assert str(window(start)) == "Fri 2026-06-05 09:00 to 10:00"


def test_window_renders_a_cross_day_range_with_the_end_date(start: datetime) -> None:
    assert str(Window(start, start + timedelta(days=2))) == "Fri 2026-06-05 09:00 to Sun 2026-06-07 09:00"


def test_event_exposes_window_bounds_and_renders(start: datetime) -> None:
    event = Event(
        identifier="id",
        title="Standup",
        account="Client",
        window=window(start),
        all_day=False,
        participation=Participation.ACCEPTED,
        status=Status.CONFIRMED,
    )
    assert event.start == start
    assert event.end == start + timedelta(hours=1)
    assert str(event) == (
        "title='Standup' window=Fri 2026-06-05 09:00 to 10:00 account='Client' "
        "status=confirmed participation=accepted identifier='id' notes=None"
    )


def test_mirror_exposes_window_bounds_and_renders(start: datetime) -> None:
    mirror = Mirror("id", window(start), "Busy", original="Standup")
    assert mirror.start == start
    assert mirror.end == start + timedelta(hours=1)
    assert str(mirror) == (
        "title='Busy' original='Standup' window=Fri 2026-06-05 09:00 to 10:00 all_day=False location=None "
        "url=None notes=None time_zone=None availability=busy source='id'"
    )


def test_plan_summary_counts_each_bucket(start: datetime) -> None:
    mirror = Mirror("id", window(start), "Busy")
    plan = Plan(create=(mirror,), update=(), delete=(mirror, mirror))
    assert not plan.empty
    assert str(plan) == "create=1 update=0 delete=2 adopt=0"


def test_plan_log_emits_every_block(start: datetime, caplog: pytest.LogCaptureFixture) -> None:
    plan = Plan(
        create=(Mirror("c", window(start), "Created"),),
        update=(Mirror("u", window(start), "Updated"),),
        delete=(Mirror("d", window(start), "Deleted"),),
    )
    with caplog.at_level(logging.INFO):
        plan.log()
    logged = caplog.text
    assert "Created" in logged
    assert "Updated" in logged
    assert "Deleted" in logged


OLD_FORMAT = f"{MARKER} ABC-123 Work.Office"  # stock calque: the marker is the whole notes field


def test_tag_on_empty_notes_is_just_the_marker() -> None:
    assert tag("Work.Office", "ABC-123") == OLD_FORMAT
    assert tag("Work.Office", "ABC-123", "") == OLD_FORMAT
    assert tag("Work.Office", "ABC-123", "  \n ") == OLD_FORMAT


def test_tag_puts_the_marker_on_the_last_line_after_the_notes() -> None:
    assert tag("Work.Office", "ABC-123", "Bring the slides") == f"Bring the slides\n\n{OLD_FORMAT}"


@pytest.mark.parametrize(
    "notes",
    [
        pytest.param(None, id="no notes"),
        pytest.param("", id="empty notes"),
        pytest.param("Bring the slides", id="one line"),
        pytest.param("Agenda:\n- budget\n\n- hiring\nhttps://zoom.us/j/123", id="several lines"),
        pytest.param("Windows\r\nline endings", id="carriage returns"),
    ],
)
def test_tag_untag_and_strip_roundtrip(notes: str | None) -> None:
    tagged = tag("Example Team.Calendar", "ABC-123", notes)
    assert untag(tagged) == Tag(origin="Example Team.Calendar", identifier="ABC-123")
    assert strip_marker(tagged) == clean(notes)
    assert tagged.split("\n")[-1] == f"{MARKER} ABC-123 Example Team.Calendar"


def test_tag_replaces_a_marker_already_in_the_notes() -> None:
    # A copy of a copy: the hub's block is mirrored onward and must not carry the hub's marker along.
    inherited = tag("Work.Office", "first", "Bring the slides")
    onward = tag("Home.Personal", "second", inherited)
    assert onward == f"Bring the slides\n\n{MARKER} second Home.Personal"
    assert onward.count(MARKER) == 1
    assert untag(onward) == Tag(origin="Home.Personal", identifier="second")


def test_strip_marker_removes_markers_stacked_by_older_copies() -> None:
    stacked = f"Bring the slides\n\n{MARKER} first Work.Office\n\n{MARKER} second Home.Personal"
    assert strip_marker(stacked) == "Bring the slides"


def test_strip_marker_leaves_ordinary_notes_alone() -> None:
    assert strip_marker("Bring the slides") == "Bring the slides"
    assert strip_marker(None) is None
    assert strip_marker(OLD_FORMAT) is None


def test_untag_recognises_the_old_whole_field_marker() -> None:
    assert untag(OLD_FORMAT) == Tag(origin="Work.Office", identifier="ABC-123")


def test_untag_recognises_an_old_marker_that_gained_text_after_it() -> None:
    notes = f"{OLD_FORMAT}\nadded later"
    assert untag(notes) == Tag(origin="Work.Office", identifier="ABC-123")
    assert strip_marker(notes) == "added later"


def test_untag_survives_trailing_whitespace_a_backend_appends() -> None:
    assert untag(f"Bring the slides\r\n\r\n{OLD_FORMAT}\r\n\r\n") == Tag("Work.Office", "ABC-123")


def test_untag_ignores_a_marker_that_does_not_start_a_line() -> None:
    assert untag(f"see {MARKER} ABC-123 Work.Office") is None


def test_untag_recognises_a_marker_with_text_on_both_sides() -> None:
    # Google appends a meeting-link block to the description, below a marker that has notes above it.
    assert untag(f"Agenda\n\n{OLD_FORMAT}\n\nJoin Zoom Meeting") == Tag("Work.Office", "ABC-123")


def test_parts_separates_the_body_the_marker_and_the_trailer() -> None:
    assert parts(f"Agenda\n\n{OLD_FORMAT}\n\nJoin Zoom Meeting\nhttps://zoom.us/j/1") == (
        "Agenda",
        OLD_FORMAT,
        "Join Zoom Meeting\nhttps://zoom.us/j/1",
    )
    assert parts(f"Agenda\n\n{OLD_FORMAT}") == ("Agenda", OLD_FORMAT, None)
    assert parts("Agenda") == ("Agenda", None, None)
    assert parts(None) == (None, None, None)


def test_split_separates_the_body_from_the_marker_line() -> None:
    assert split(f"Bring the slides\n\n{OLD_FORMAT}") == ("Bring the slides", OLD_FORMAT)
    assert split(f"Agenda\n\n{OLD_FORMAT}\n\nadded later") == ("Agenda\n\nadded later", OLD_FORMAT)
    assert split("Bring the slides") == ("Bring the slides", None)
    assert split(None) == (None, None)


def test_clean_normalises_text() -> None:
    assert clean("  a\r\nb \n") == "a\nb"
    assert clean("   ") is None
    assert clean(None) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("location", "Room 4"),
        ("url", "https://zoom.us/j/123"),
        ("notes", "Bring the slides"),
        ("time_zone", "America/Chicago"),
        ("availability", Availability.FREE),
        ("all_day", True),
    ],
)
def test_mirror_equality_covers_every_copied_field(start: datetime, field: str, value: object) -> None:
    base = Mirror("id", window(start), "Standup")
    assert replace(base, **{field: value}) != base  # type: ignore[arg-type]
    assert replace(base) == base


def test_mirror_equality_covers_the_title_and_times(start: datetime) -> None:
    base = Mirror("id", window(start), "Standup")
    assert replace(base, title="Renamed") != base
    assert replace(base, window=window(start, hours=2)) != base


def test_mirror_equality_ignores_provenance_and_native_dates(start: datetime) -> None:
    base = Mirror("id", window(start), "Standup")
    assert Mirror("id", window(start), "Standup", original="x", origin="Work.Office", dates=("a", "b")) == base


def test_plan_with_only_an_adoption_is_not_empty_and_logs_it(start: datetime, caplog: pytest.LogCaptureFixture) -> None:
    target = Event(
        identifier="existing",
        title="Standup",
        account="Home",
        window=window(start),
        all_day=False,
        participation=Participation.ACCEPTED,
        status=Status.CONFIRMED,
    )
    plan = Plan(create=(), update=(), delete=(), adopt=(Adoption(Mirror("id", window(start), "Standup"), target),))
    assert not plan.empty
    assert str(plan) == "create=0 update=0 delete=0 adopt=1"
    with caplog.at_level(logging.INFO):
        plan.log()
    assert "adopting='existing'" in caplog.text
