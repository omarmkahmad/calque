"""Tests for the reconciliation core, a single mirror pass, and the both-ways synchronisation run."""

import logging
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, call

import pytest

from calque import sync
from calque.config import Config
from calque.errors import DeleteLimitError
from calque.model import Adoption, Availability, Event, Mirror, Participation, Plan, Status, Tag, Window
from calque.sync import (
    Carry,
    adoptions,
    designate,
    exceeds,
    mirror,
    overlay,
    reconcile,
    relocate,
    settle,
    staged,
    synchronise,
    to_mirrors,
)

EVERY_AVAILABILITY = frozenset(Availability) - {Availability.UNSUPPORTED}
NOTHING = Plan(create=(), update=(), delete=())


@pytest.fixture
def start() -> datetime:
    return datetime(2026, 6, 5, 9, 0, tzinfo=UTC)


def event(identifier: str, start: datetime, participation: Participation = Participation.ACCEPTED) -> Event:
    return Event(
        identifier=identifier,
        title=identifier,
        account="Client",
        window=Window(start, start + timedelta(hours=1)),
        all_day=False,
        participation=participation,
        status=Status.CONFIRMED,
    )


def block(identifier: str, start: datetime, title: str = "Busy") -> Mirror:
    return Mirror(identifier, Window(start, start + timedelta(hours=1)), title)


def test_to_mirrors_keys_every_event_by_identifier(start: datetime) -> None:
    # No participation filtering here — that is now an exclusion rule applied before this step.
    events = [event("a", start), event("b", start, Participation.DECLINED)]
    assert set(to_mirrors(events, "Busy", "Client.Calendar")) == {"a", "b"}


def test_to_mirrors_anonymises_to_the_given_template(start: datetime) -> None:
    selected = to_mirrors([event("one", start)], "Unavailable", "Client.Calendar")
    assert selected["one"].title == "Unavailable"


def test_to_mirrors_records_the_origin_calendar(start: datetime) -> None:
    selected = to_mirrors([event("one", start)], "Busy", "Client.Calendar")
    assert selected["one"].origin == "Client.Calendar"


def test_to_mirrors_fills_template_placeholders_from_the_event(start: datetime) -> None:
    selected = to_mirrors([event("standup", start)], "{account}: {title}", "Client.Calendar")
    assert selected["standup"].title == "Client: standup"


def test_reconcile_creates_missing(start: datetime) -> None:
    plan = reconcile({"one": block("one", start)}, {})
    assert plan.create == (block("one", start),)
    assert not plan.update
    assert not plan.delete


def test_reconcile_deletes_orphans(start: datetime) -> None:
    plan = reconcile({}, {"gone": block("gone", start)})
    assert plan.delete == (block("gone", start),)


def test_reconcile_updates_when_times_move(start: datetime) -> None:
    desired = {"one": block("one", start + timedelta(hours=2))}
    existing = {"one": block("one", start)}
    plan = reconcile(desired, existing)
    assert plan.update == (block("one", start + timedelta(hours=2)),)


def test_reconcile_leaves_unchanged_alone(start: datetime) -> None:
    same = {"one": block("one", start)}
    assert reconcile(same, dict(same)).empty


def deleting(count: int, start: datetime) -> Plan:
    """A plan that only deletes ``count`` blocks."""
    return Plan(create=(), update=(), delete=tuple(block(str(n), start) for n in range(count)))


def test_exceeds_never_trips_without_a_limit(start: datetime) -> None:
    assert not exceeds(deleting(100, start), None)


def test_exceeds_allows_deletes_up_to_the_limit(start: datetime) -> None:
    assert not exceeds(deleting(3, start), 3)


def test_exceeds_trips_one_over_the_limit(start: datetime) -> None:
    assert exceeds(deleting(4, start), 3)


def test_exceeds_trips_on_any_delete_with_a_zero_limit(start: datetime) -> None:
    assert exceeds(deleting(1, start), 0)
    assert not exceeds(deleting(0, start), 0)


def calendar(qualified: str) -> Mock:
    """A stand-in calendar reporting no events, no busy periods, and no existing mirrors."""
    return Mock(
        qualified=qualified,
        **{
            "events.return_value": [],
            "busy.return_value": [],
            "mirrors.return_value": {},
            "availabilities.return_value": EVERY_AVAILABILITY,
            "keeps_urls.return_value": True,
            "source.return_value.title.return_value": qualified.partition(".")[0],
        },
    )


def test_mirror_applies_the_plan_to_the_target(start: datetime) -> None:
    source, target = calendar("Personal.Home"), calendar("Work.Office")
    window = Window(start, start + timedelta(days=1))
    mirror(source, target, Config(), window)
    target.apply.assert_called_once()
    plan, applied_window = target.apply.call_args.args
    assert plan.empty
    assert applied_window is window


def test_mirror_reconciles_only_against_blocks_from_this_source(start: datetime) -> None:
    # The existing set must be scoped to the source's origin, so another source's blocks in the
    # target are never seen as orphans and deleted.
    source, target = calendar("Personal.Home"), calendar("Work.Office")
    window = Window(start, start + timedelta(days=1))
    mirror(source, target, Config(), window)
    target.mirrors.assert_called_once_with(window, "Personal.Home")


def test_mirror_writes_nothing_on_a_dry_run(start: datetime) -> None:
    source, target = calendar("Personal.Home"), calendar("Work.Office")
    mirror(source, target, Config(dry_run=True), Window(start, start + timedelta(days=1)))
    target.apply.assert_not_called()


def test_mirror_deletes_finished_blocks_when_cleanup_enabled() -> None:
    # With cleanup on, a source event that is already over drops out of the desired set, so
    # reconciliation deletes its stale mirror while an event still to come is left in place.
    # The cut-off is the wall clock, so the events are positioned relative to it.
    now = datetime.now(UTC)
    over = event("over", now - timedelta(hours=2))
    ahead = event("ahead", now + timedelta(hours=2))
    source, target = calendar("Personal.Home"), calendar("Work.Office")
    source.events.return_value = [over, ahead]
    target.mirrors.return_value = {
        "over": block("over", now - timedelta(hours=2), "over"),
        "ahead": block("ahead", now + timedelta(hours=2), "ahead"),
    }
    mirror(source, target, Config(cleanup=True), Window(now - timedelta(days=1), now))
    plan = target.apply.call_args.args[0]
    assert [removed.source for removed in plan.delete] == ["over"]
    assert not plan.create


def test_mirror_only_writes_whitelisted_events_from_a_source_with_include_patterns(start: datetime) -> None:
    listed = event("House Viewing (graham busy)", start)
    unlisted_event = event("Swimming lesson", start)
    source, target = calendar("Home.Shared"), calendar("Work.Office")
    source.events.return_value = [listed, unlisted_event]
    config = Config(
        calendar_include_patterns={"Home.Shared": (re.compile(r"graham busy"),)},
        exclude_clashes=False,
        exclude_out_of_hours=False,
    )
    mirror(source, target, config, Window(start, start + timedelta(days=1)))
    created = target.apply.call_args.args[0].create
    assert [block.source for block in created] == ["House Viewing (graham busy)"]


def orphaned(qualified: str, count: int, start: datetime) -> Mock:
    """A stand-in target holding ``count`` mirror blocks whose source events have all gone."""
    target = calendar(qualified)
    target.mirrors.return_value = {f"gone-{n}": block(f"gone-{n}", start) for n in range(count)}
    return target


def test_mirror_skips_a_pass_deleting_more_than_the_limit(start: datetime, caplog: pytest.LogCaptureFixture) -> None:
    source, target = calendar("Personal.Home"), orphaned("Work.Office", 3, start)
    plan = mirror(source, target, Config(max_deletes=2), Window(start, start + timedelta(days=1)))
    assert exceeds(plan, 2)
    target.apply.assert_not_called()
    assert "Personal.Home -> Work.Office: skipped, 3 deletes exceed the limit of 2" in caplog.text


def test_mirror_applies_a_pass_within_the_limit(start: datetime) -> None:
    source, target = calendar("Personal.Home"), orphaned("Work.Office", 2, start)
    plan = mirror(source, target, Config(max_deletes=2), Window(start, start + timedelta(days=1)))
    assert target.apply.call_args.args[0] is plan
    assert len(plan.delete) == 2


def test_mirror_warns_of_an_over_limit_pass_on_a_dry_run(start: datetime, caplog: pytest.LogCaptureFixture) -> None:
    source, target = calendar("Personal.Home"), orphaned("Work.Office", 3, start)
    config = Config(max_deletes=2, dry_run=True)
    caplog.set_level("INFO")
    mirror(source, target, config, Window(start, start + timedelta(days=1)))
    target.apply.assert_not_called()
    assert "skipped, 3 deletes exceed the limit of 2" in caplog.text
    assert "dry run: no changes written" in caplog.text


def plan_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """The captured records of the pass's ``source -> target: plan`` line."""
    return [record for record in caplog.records if record.getMessage().startswith("Personal.Home -> Work.Office: ")]


def test_mirror_logs_an_empty_pass_at_debug(start: datetime, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG")
    mirror(calendar("Personal.Home"), calendar("Work.Office"), Config(), Window(start, start + timedelta(days=1)))
    assert [record.levelno for record in plan_records(caplog)] == [logging.DEBUG]
    assert not [record for record in caplog.records if record.levelno >= logging.INFO]


def test_mirror_logs_a_pass_with_changes_at_info(start: datetime, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    source, target = calendar("Personal.Home"), orphaned("Work.Office", 1, start)
    mirror(source, target, Config(), Window(start, start + timedelta(days=1)))
    assert [record.levelno for record in plan_records(caplog)] == [logging.INFO]


def test_mirror_logs_an_empty_dry_run_pass_at_info(start: datetime, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    config = Config(dry_run=True)
    mirror(calendar("Personal.Home"), calendar("Work.Office"), config, Window(start, start + timedelta(days=1)))
    assert [record.levelno for record in plan_records(caplog)] == [logging.INFO]


def test_synchronise_collects_each_auxiliary_into_the_primary_then_fans_out(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = {title: calendar(title) for title in ("Personal", "Work", "Side")}
    monkeypatch.setattr(sync, "CalendarStore", Mock(return_value=Mock(calendar=lambda title: resolved[title])))
    passes = Mock(return_value=NOTHING)
    monkeypatch.setattr(sync, "mirror", passes)
    config = Config()
    synchronise(config, ["Personal", "Work", "Side"])
    assert passes.call_count == 4  # two auxiliaries collected into the primary, then fanned back out
    primary, window = resolved["Personal"], passes.call_args.args[3]
    passes.assert_any_call(resolved["Work"], primary, config, window, Carry())
    passes.assert_any_call(resolved["Side"], primary, config, window, Carry())
    passes.assert_any_call(primary, resolved["Work"], config, window, Carry())
    passes.assert_any_call(primary, resolved["Side"], config, window, Carry())


def test_synchronise_does_not_fan_out_to_a_muted_calendar(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = {title: calendar(title) for title in ("Personal", "Work", "Side")}
    monkeypatch.setattr(sync, "CalendarStore", Mock(return_value=Mock(calendar=lambda title: resolved[title])))
    passes = Mock(return_value=NOTHING)
    monkeypatch.setattr(sync, "mirror", passes)
    config = Config(muted=frozenset({"Side"}))
    synchronise(config, ["Personal", "Work", "Side"])
    primary, window = resolved["Personal"], passes.call_args.args[3]
    # Side is still collected into the primary, but the primary is never mirrored back out to it.
    passes.assert_any_call(resolved["Side"], primary, config, window, Carry())
    assert call(primary, resolved["Side"], config, window, Carry()) not in passes.call_args_list


def test_synchronise_muting_the_primary_skips_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = {title: calendar(title) for title in ("Personal", "Work")}
    monkeypatch.setattr(sync, "CalendarStore", Mock(return_value=Mock(calendar=lambda title: resolved[title])))
    passes = Mock(return_value=NOTHING)
    monkeypatch.setattr(sync, "mirror", passes)
    config = Config(muted=frozenset({"Personal"}))
    synchronise(config, ["Personal", "Work"])
    # Nothing is mirrored into the primary; it is only fanned out to the auxiliary.
    passes.assert_called_once_with(resolved["Personal"], resolved["Work"], config, passes.call_args.args[3], Carry())


@pytest.mark.parametrize(
    "skips",
    [
        pytest.param([6, 0, 0, 0], id="collecting Work into Personal"),
        pytest.param([0, 0, 0, 6], id="fanning Personal out to Side"),
    ],
)
def test_synchronise_runs_every_pass_before_failing_on_a_skipped_one(
    monkeypatch: pytest.MonkeyPatch, skips: list[int], start: datetime
) -> None:
    resolved = {title: calendar(title) for title in ("Personal", "Work", "Side")}
    monkeypatch.setattr(sync, "CalendarStore", Mock(return_value=Mock(calendar=lambda title: resolved[title])))
    # Exactly one pass is over the limit, either while collecting into the primary or while fanning out.
    passes = Mock(side_effect=[deleting(count, start) for count in skips])
    monkeypatch.setattr(sync, "mirror", passes)
    config = Config(max_deletes=5)
    with pytest.raises(DeleteLimitError, match="more than 5 "):
        synchronise(config, ["Personal", "Work", "Side"])
    assert passes.call_count == 4


def test_synchronise_does_not_fail_a_dry_run_over_the_delete_limit(
    monkeypatch: pytest.MonkeyPatch, start: datetime
) -> None:
    resolved = {title: calendar(title) for title in ("Personal", "Work")}
    monkeypatch.setattr(sync, "CalendarStore", Mock(return_value=Mock(calendar=lambda title: resolved[title])))
    monkeypatch.setattr(sync, "mirror", Mock(return_value=deleting(6, start)))
    synchronise(Config(max_deletes=5, dry_run=True), ["Personal", "Work"])


def detailed(identifier: str, start: datetime, **fields: object) -> Event:
    """An event titled "Clinic" carrying the given extra fields."""
    return replace(event(identifier, start), **{"title": "Clinic", **fields})  # type: ignore[arg-type]


def test_to_mirrors_copies_every_detail_of_the_event(start: datetime) -> None:
    dates = (object(), object())
    source = detailed(
        "one",
        start,
        location="Room 4",
        url="https://zoom.us/j/123",
        notes="Bring the slides",
        time_zone="America/Chicago",
        availability=Availability.TENTATIVE,
        all_day=True,
        dates=dates,
    )
    block = to_mirrors([source], "{title}", "Work.Office")["one"]
    assert block == Mirror(
        "one",
        source.window,
        "Clinic",
        location="Room 4",
        url="https://zoom.us/j/123",
        notes="Bring the slides",
        time_zone="America/Chicago",
        availability=Availability.TENTATIVE,
        all_day=True,
    )
    assert block.dates is dates  # the native all-day dates travel with the block, uncopied


def test_reconcile_updates_when_a_title_changes(start: datetime) -> None:
    plan = reconcile({"one": block("one", start, "Renamed")}, {"one": block("one", start, "Original")})
    assert [changed.title for changed in plan.update] == ["Renamed"]


@pytest.mark.parametrize(
    "change",
    [
        {"location": "Room 5"},
        {"url": "https://zoom.us/j/456"},
        {"notes": "Bring the laptop"},
        {"time_zone": "Europe/London"},
        {"availability": Availability.FREE},
        {"all_day": True},
    ],
    ids=lambda change: next(iter(change)),
)
def test_reconcile_updates_when_any_copied_field_changes(start: datetime, change: dict[str, object]) -> None:
    existing = block("one", start)
    plan = reconcile({"one": replace(existing, **change)}, {"one": existing})  # type: ignore[arg-type]
    assert len(plan.update) == 1


def test_designate_keeps_the_earliest_created_event_and_marks_the_rest_as_copies(start: datetime) -> None:
    original = detailed("original", start, created=start - timedelta(days=9))
    first_copy = detailed("copy-1", start, created=start - timedelta(days=8))
    second_copy = detailed("copy-2", start, created=start - timedelta(days=7))
    assert designate([[first_copy], [original], [second_copy]]) == {"copy-1", "copy-2"}


def test_designate_prefers_the_earliest_listed_calendar_when_creation_times_are_unknown(start: datetime) -> None:
    assert designate([[detailed("hub", start)], [detailed("spoke", start)]]) == {"spoke"}


def test_designate_ranks_an_unknown_creation_time_after_a_known_one(start: datetime) -> None:
    assert designate([[detailed("undated", start)], [detailed("dated", start, created=start)]]) == {"undated"}


def test_designate_keeps_an_invitation_as_the_original_over_an_older_plain_event(start: datetime) -> None:
    invitation = detailed("invitation", start, invited=True, created=start)
    plain = detailed("plain", start, created=start - timedelta(days=30))
    assert designate([[plain], [invitation]]) == {"plain"}


def test_designate_marks_a_second_copy_of_one_invitation_so_it_is_not_mirrored_twice(start: datetime) -> None:
    first = detailed("work-invite", start, invited=True, created=start - timedelta(days=2))
    second = detailed("side-invite", start, invited=True, created=start - timedelta(days=1))
    assert designate([[], [first], [second]]) == {"side-invite"}


def test_designate_ignores_events_that_differ_in_title_or_time(start: datetime) -> None:
    clinic = detailed("clinic", start)
    renamed = replace(detailed("renamed", start), title="Clinic (moved)")
    later = detailed("later", start + timedelta(minutes=30))
    assert designate([[clinic], [renamed, later]]) == frozenset()


def test_designate_ignores_lookalikes_within_a_single_calendar(start: datetime) -> None:
    assert designate([[detailed("one", start), detailed("two", start)], []]) == frozenset()


def test_designate_never_marks_an_event_a_mirror_block_was_copied_from(start: datetime) -> None:
    # The hub already holds our block for the Work event: Work's event is the original however it ranks.
    source = detailed("work-event", start)
    hub_block = detailed("hub-block", start, tag=Tag("Work.Calendar", "work-event"))
    stray = detailed("side-copy", start, created=start - timedelta(days=99))
    assert designate([[hub_block], [source], [stray]]) == {"side-copy"}


def test_designate_treats_an_untagged_twin_of_a_mirror_block_as_a_copy(start: datetime) -> None:
    # A block of ours stands for an original elsewhere, so the lone untagged look-alike cannot be that original.
    hub_block = detailed("hub-block", start, tag=Tag("Work.Calendar", "work-event"))
    assert designate([[hub_block], [detailed("side-copy", start)]]) == {"side-copy"}


def desired(start: datetime, *identifiers: str) -> dict[str, Mirror]:
    return {identifier: block(identifier, start, "Clinic") for identifier in identifiers}


def test_adoptions_tags_a_matching_copy_instead_of_creating_a_duplicate(start: datetime) -> None:
    copy = detailed("copy", start)
    adopt, present = adoptions(desired(start, "source"), {}, [copy], {"copy"})
    assert adopt == (Adoption(block("source", start, "Clinic"), copy),)
    assert not present


@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param({"title": "Clinic B"}, id="different title"),
        pytest.param(
            {"window": Window(datetime(2026, 6, 5, 9, 30, tzinfo=UTC), datetime(2026, 6, 5, 10, tzinfo=UTC))},
            id="different start",
        ),
        pytest.param(
            {"window": Window(datetime(2026, 6, 5, 9, tzinfo=UTC), datetime(2026, 6, 5, 11, tzinfo=UTC))},
            id="different end",
        ),
        pytest.param({"tag": Tag("Other.Calendar", "x")}, id="already a mirror block"),
    ],
)
def test_adoptions_needs_an_untagged_event_with_the_same_title_start_and_end(
    start: datetime, candidate: dict[str, object]
) -> None:
    lookalike = detailed("copy", start, **candidate)
    assert adoptions(desired(start, "source"), {}, [lookalike], {"copy"}) == ((), frozenset())


def test_adoptions_leaves_a_block_that_already_exists_alone(start: datetime) -> None:
    existing = desired(start, "source")
    assert adoptions(desired(start, "source"), existing, [detailed("copy", start)], {"copy"}) == ((), frozenset())


def test_adoptions_never_tags_an_invitation_but_does_not_duplicate_it_either(start: datetime) -> None:
    invitation = detailed("invite", start, invited=True)
    assert adoptions(desired(start, "source"), {}, [invitation], {"invite"}) == ((), frozenset({"source"}))


def test_adoptions_never_tags_an_original_but_does_not_duplicate_it_either(start: datetime) -> None:
    assert adoptions(desired(start, "source"), {}, [detailed("original", start)], set()) == ((), frozenset({"source"}))


def test_adoptions_gives_each_copy_to_one_block_only(start: datetime) -> None:
    adopt, present = adoptions(desired(start, "first", "second"), {}, [detailed("copy", start)], {"copy"})
    assert [(adoption.mirror.source, adoption.target.identifier) for adoption in adopt] == [("first", "copy")]
    assert not present


def test_mirror_adopts_a_copy_and_creates_only_what_is_missing(start: datetime) -> None:
    source, target = calendar("Work.Calendar"), calendar("Home.Personal")
    source.events.return_value = [detailed("clinic", start, notes="Bring the slides"), event("standup", start)]
    copy = detailed("fantastical-copy", start)
    target.events.return_value = [copy]
    plan = mirror(
        source, target, Config(), Window(start, start + timedelta(days=1)), Carry(frozenset({copy.identifier}))
    )
    assert [(adoption.mirror.source, adoption.target) for adoption in plan.adopt] == [("clinic", copy)]
    assert plan.adopt[0].mirror.notes == "Bring the slides"
    assert [created.source for created in plan.create] == ["standup"]
    assert target.apply.call_args.args[0] is plan


def test_mirror_does_not_send_a_copy_back_as_if_it_were_a_real_event(start: datetime) -> None:
    source, target = calendar("Home.Personal"), calendar("Work.Calendar")
    source.events.return_value = [detailed("fantastical-copy", start)]
    plan = mirror(
        source, target, Config(), Window(start, start + timedelta(days=1)), Carry(frozenset({"fantastical-copy"}))
    )
    assert plan.empty


def test_mirror_does_not_duplicate_an_event_the_target_already_shows(
    start: datetime, caplog: pytest.LogCaptureFixture
) -> None:
    source, target = calendar("Home.Personal"), calendar("Work.Calendar")
    source.events.return_value = [detailed("clinic", start)]
    target.events.return_value = [detailed("same-invitation", start, invited=True)]
    caplog.set_level("DEBUG")
    assert mirror(source, target, Config(), Window(start, start + timedelta(days=1))).empty
    assert "1 already in the target, left alone" in caplog.text


def test_mirror_with_adoption_off_creates_alongside_a_lookalike(start: datetime) -> None:
    source, target = calendar("Work.Calendar"), calendar("Home.Personal")
    source.events.return_value = [detailed("clinic", start)]
    target.events.return_value = [detailed("fantastical-copy", start)]
    plan = mirror(source, target, Config(adopt=False), Window(start, start + timedelta(days=1)))
    assert not plan.adopt
    assert [created.source for created in plan.create] == ["clinic"]


def test_relocate_leaves_a_block_without_a_url_as_it_is(start: datetime) -> None:
    wanted = replace(block("one", start), notes="Agenda")
    assert relocate(wanted) is wanted


def test_relocate_moves_the_url_to_the_first_line_of_the_notes(start: datetime) -> None:
    moved = relocate(replace(block("one", start), url="https://zoom.us/j/123", notes="Agenda"))
    assert (moved.url, moved.notes) == (None, "https://zoom.us/j/123\n\nAgenda")


def test_settle_leaves_a_new_block_as_it_is(start: datetime) -> None:
    wanted = replace(block("one", start), url="https://zoom.us/j/123")
    assert settle(wanted, None, EVERY_AVAILABILITY) is wanted


def test_settle_leaves_a_block_whose_url_survived(start: datetime) -> None:
    wanted = replace(block("one", start), url="https://zoom.us/j/123", notes="Agenda")
    assert settle(wanted, wanted, EVERY_AVAILABILITY) is wanted


@pytest.mark.parametrize(
    ("notes", "expected"),
    [
        pytest.param("Agenda", "https://zoom.us/j/123\n\nAgenda", id="ahead of the notes"),
        pytest.param(None, "https://zoom.us/j/123", id="as the only note"),
        pytest.param("Join: https://zoom.us/j/123 now", "Join: https://zoom.us/j/123 now", id="already in the notes"),
    ],
)
def test_settle_moves_a_dropped_url_to_the_first_line_of_the_notes(
    start: datetime, notes: str | None, expected: str
) -> None:
    wanted = replace(block("one", start), url="https://zoom.us/j/123", notes=notes)
    stored = replace(block("one", start), notes=notes)  # the target came back without the URL
    settled = settle(wanted, stored, EVERY_AVAILABILITY)
    assert settled.url is None
    assert settled.notes == expected


def test_settle_does_not_repeat_a_link_the_location_already_carries(start: datetime) -> None:
    wanted = replace(block("one", start), url="https://zoom.us/j/123", location="https://zoom.us/j/123")
    settled = settle(wanted, replace(wanted, url=None), EVERY_AVAILABILITY)
    assert (settled.url, settled.notes) == (None, None)


def test_settle_stops_rewriting_once_the_link_is_in_the_notes(start: datetime) -> None:
    wanted = replace(block("one", start), url="https://zoom.us/j/123", notes="Agenda")
    stored = replace(block("one", start), notes="https://zoom.us/j/123\n\nAgenda")
    assert settle(wanted, stored, EVERY_AVAILABILITY) == stored


def test_settle_leaves_a_location_the_target_moved_below_the_marker(start: datetime) -> None:
    wanted = replace(block("one", start), location="https://zoom.us/j/123")
    stored = replace(block("one", start), trailer="Join Zoom Meeting\nhttps://zoom.us/j/123")
    assert settle(wanted, stored, EVERY_AVAILABILITY) == stored


@pytest.mark.parametrize(
    ("trailer", "location"),
    [
        pytest.param(None, None, id="nothing below the marker"),
        pytest.param("Join Zoom Meeting", None, id="trailer without the location"),
        pytest.param("https://zoom.us/j/123", "Elsewhere", id="target still has a location"),
    ],
)
def test_settle_restores_a_location_the_target_merely_lost(
    start: datetime, trailer: str | None, location: str | None
) -> None:
    wanted = replace(block("one", start), location="https://zoom.us/j/123")
    stored = replace(block("one", start), trailer=trailer, location=location)
    assert settle(wanted, stored, EVERY_AVAILABILITY).location == wanted.location


def test_settle_keeps_the_targets_availability_when_it_cannot_store_the_sources(start: datetime) -> None:
    wanted = replace(block("one", start), availability=Availability.TENTATIVE)
    stored = block("one", start)
    assert settle(wanted, stored, frozenset({Availability.BUSY, Availability.FREE})) == stored
    assert settle(wanted, stored, EVERY_AVAILABILITY).availability is Availability.TENTATIVE


def test_mirror_does_not_keep_rewriting_a_block_whose_url_the_target_dropped(start: datetime) -> None:
    source, target = calendar("Work.Calendar"), calendar("Exchange.Calendar")
    source.events.return_value = [detailed("clinic", start, url="https://zoom.us/j/123", notes="Agenda")]
    window = Window(start, start + timedelta(days=1))
    target.mirrors.return_value = {"clinic": replace(block("clinic", start, "Clinic"), notes="Agenda")}
    plan = mirror(source, target, Config(), window)
    assert [(changed.url, changed.notes) for changed in plan.update] == [(None, "https://zoom.us/j/123\n\nAgenda")]
    target.mirrors.return_value = {"clinic": plan.update[0]}
    assert mirror(source, target, Config(), window).empty


def test_mirror_writes_the_link_into_the_notes_of_a_target_that_never_keeps_urls(start: datetime) -> None:
    # Exchange reads the URL back from the local store although the server dropped it, so nothing can be
    # learned from the block that comes back: the link goes in the notes from the first write.
    source, target = calendar("Work.Calendar"), calendar("Exchange.Calendar")
    target.keeps_urls.return_value = False
    source.events.return_value = [detailed("clinic", start, url="https://zoom.us/j/123", notes="Agenda")]
    window = Window(start, start + timedelta(days=1))
    plan = mirror(source, target, Config(), window)
    assert [(created.url, created.notes) for created in plan.create] == [(None, "https://zoom.us/j/123\n\nAgenda")]
    target.mirrors.return_value = {"clinic": plan.create[0]}
    assert mirror(source, target, Config(), window).empty


def test_staged_previews_what_a_plan_would_leave_in_its_target(start: datetime) -> None:
    created = replace(block("new", start, "Clinic"), origin="Work.Calendar", notes="Agenda")
    adopted = replace(block("old", start, "Rounds"), origin="Work.Calendar")
    plan = Plan(create=(created,), update=(), delete=(), adopt=(Adoption(adopted, detailed("copy", start)),))
    first, second = staged(plan, "iCloud")
    assert (first.identifier, first.title, first.notes, first.account) == ("pending:new", "Clinic", "Agenda", "iCloud")
    assert first.tag == Tag("Work.Calendar", "new")
    assert (second.identifier, second.title, second.tag) == ("copy", "Rounds", Tag("Work.Calendar", "old"))


def test_overlay_adds_pending_events_and_replaces_those_they_stand_for(start: datetime) -> None:
    untouched, copy = detailed("untouched", start), detailed("copy", start)
    adopted = replace(copy, tag=Tag("Work.Calendar", "old"))
    added = detailed("pending:new", start)
    assert list(overlay([untouched, copy], [adopted, added])) == [untouched, adopted, added]


def test_synchronise_dry_run_reports_the_fan_out_of_what_it_would_have_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Work's event has stray copies in both other calendars. A dry run writes nothing, yet must still report
    # that the hub's copy is adopted and then fanned out to adopt Side's copy, never returning to Work.
    now = datetime.now(UTC)
    clinic = replace(detailed("clinic", now, created=now - timedelta(days=3)), account="Work")
    hub_copy = detailed("hub-copy", now, created=now - timedelta(days=2))
    side_copy = detailed("side-copy", now, created=now - timedelta(days=1))
    fresh = event("standup", now + timedelta(hours=3))
    resolved = {title: calendar(title) for title in ("Personal", "Work", "Side")}
    resolved["Personal"].events.return_value = [hub_copy]
    resolved["Work"].events.return_value = [clinic, fresh]
    resolved["Side"].events.return_value = [side_copy]
    monkeypatch.setattr(sync, "CalendarStore", Mock(return_value=Mock(calendar=lambda title: resolved[title])))
    plans: list[Plan] = []
    real = sync.mirror

    def recording(*arguments: object) -> Plan:
        plans.append(real(*arguments))  # type: ignore[arg-type]
        return plans[-1]

    monkeypatch.setattr(sync, "mirror", recording)
    synchronise(Config(dry_run=True), ["Personal", "Work", "Side"])
    into_hub, from_side, out_to_work, out_to_side = plans
    assert [adoption.target.identifier for adoption in into_hub.adopt] == ["hub-copy"]
    assert [created.source for created in into_hub.create] == ["standup"]
    assert from_side.empty  # Side's copy is not mirrored into the hub as if it were a real event
    assert out_to_work.empty  # nothing that came from Work goes back to it
    assert [adoption.target.identifier for adoption in out_to_side.adopt] == ["side-copy"]
    assert [created.source for created in out_to_side.create] == ["pending:standup"]
    for unwritten in resolved.values():
        unwritten.apply.assert_not_called()


def test_synchronise_with_adoption_off_designates_no_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    resolved = {title: calendar(title) for title in ("Personal", "Work")}
    monkeypatch.setattr(sync, "CalendarStore", Mock(return_value=Mock(calendar=lambda title: resolved[title])))
    designating = Mock()
    monkeypatch.setattr(sync, "designate", designating)
    synchronise(Config(adopt=False), ["Personal", "Work"])
    designating.assert_not_called()
