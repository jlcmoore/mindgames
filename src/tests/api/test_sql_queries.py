"""
Author: Jared Moore
Date: September, 2024

Tests for the sql database queries.
"""

from collections import Counter
from datetime import datetime, timezone

import pytest


###

from mindgames.game import Game, TURN_LIMIT

from mindgames.known_models import (
    SOLUTION,
)
from mindgames.conditions import (
    Roles,
    Condition,
    RATIONAL_TARGET_ROLE,
    PAIRED_HUMAN_ROLE,
)
from mindgames.utils import (
    EX_SCENARIO,
)

from api.utils import (
    DEFAULT_WAITING_ROOM_TIMEOUT,
    ServerSettings,
)

from api.sql_model import (
    Model,
    Scenario,
    Round,
    ExternalUser,
    Participant,
)

from api.sql_queries import (
    get_participant_rounds,
    get_paired_participant,
    round_types_count,
    get_round_types_remaining,
    choose_condition,
    get_bonuses,
    rounds_by_condition,
)

from api.current_round_helpers import choose_participant_conditions

from .context import session_fixture, engine_fixture


def test_participant_rounds(session):

    user_a = ExternalUser(external_id="A")
    session.add(user_a)
    session.commit()
    session.refresh(user_a)
    assert user_a.id
    sc = Scenario(**EX_SCENARIO)
    par = Participant(
        id=user_a.id,
        role="persuader",
        game_model_types_remaining=Counter({"solution": 1}),
    )
    gm = Model(data=SOLUTION.model_dump(), game_model_type="solution")

    session.add(sc)
    session.add(par)
    session.add(gm)

    session.commit()

    session.refresh(sc)
    session.refresh(par)
    session.refresh(gm)

    ###

    rds = get_participant_rounds(par, session)

    assert not rds
    ###

    game = Game(
        model=SOLUTION,
        turn_limit=TURN_LIMIT,
        is_ideal_target=True,
        **EX_SCENARIO,
    )
    gm_data = game.model_dump()

    rd = Round(
        game_data=gm_data,
        game_model_type="solution",
        persuader_id=par.id,
        scenario_id=sc.id,
        game_model_id=gm.id,
    )

    session.add(rd)

    session.commit()
    session.refresh(rd)

    rds = get_participant_rounds(par, session)

    assert rd in rds

    rd2 = Round(
        game_data=gm_data,
        game_model_type="solution",
        llm_persuader="test",
        target_id=par.id,
        scenario_id=sc.id,
        game_model_id=gm.id,
    )

    session.add(rd2)

    session.commit()
    session.refresh(rd2)

    rds = get_participant_rounds(par, session)

    assert rd in rds and rd2 in rds


def test_pair_participants(session):

    sc = Scenario(**EX_SCENARIO)
    gm = Model(data=SOLUTION.model_dump(), game_model_type="solution")

    session.add(sc)
    session.add(gm)
    session.commit()
    session.refresh(sc)
    session.refresh(gm)

    user_a = ExternalUser(external_id="A")
    session.add(user_a)
    session.commit()
    session.refresh(user_a)
    assert user_a.id

    user_b = ExternalUser(external_id="B")
    session.add(user_b)
    session.commit()
    session.refresh(user_b)
    assert user_b.id

    par_a = Participant(
        id=user_a.id, role=None, game_model_types_remaining=Counter({"solution": 1})
    )

    session.add(par_a)
    session.commit()
    session.refresh(par_a)

    settings = ServerSettings(
        participants_rematch=False, waiting_room_timeout=DEFAULT_WAITING_ROOM_TIMEOUT
    )

    par = get_paired_participant(par_a, session, settings)

    assert not par

    par_b = Participant(
        id=user_b.id, role=None, game_model_types_remaining=Counter({"solution": 1})
    )

    session.add(par_b)
    session.commit()
    session.refresh(par_b)

    par_a.entered_waiting_room = datetime.now(timezone.utc)
    par_b.entered_waiting_room = datetime.now(timezone.utc)
    par = get_paired_participant(
        par_a,
        session,
        settings,
    )

    assert par_b == par

    par_a.role = "either"
    par_b.role = "either"
    par_a.entered_waiting_room = None
    par_b.entered_waiting_room = None

    session.add(par_b)
    session.add(par_a)
    session.commit()
    session.add(par_a)
    session.refresh(par_b)

    par = get_paired_participant(
        par_a,
        session,
        settings,
    )

    assert not par

    par_b.entered_waiting_room = datetime.now(timezone.utc)

    session.add(par_b)
    session.commit()
    session.refresh(par_b)

    par = get_paired_participant(
        par_a,
        session,
        settings,
    )

    assert not par

    par_a.entered_waiting_room = datetime.now(timezone.utc)

    session.add(par_a)
    session.commit()
    session.refresh(par_a)

    par = get_paired_participant(
        par_a,
        session,
        settings,
    )

    assert par_b == par

    par_b.game_model_types_remaining = Counter({"can-win": 1})

    session.add(par_b)
    session.commit()
    session.refresh(par_b)

    par = get_paired_participant(
        par_a,
        session,
        settings,
    )

    assert not par

    par_b.entered_waiting_room = (
        datetime.now(timezone.utc) - 2 * DEFAULT_WAITING_ROOM_TIMEOUT
    )
    par_b.game_model_types_remaining = Counter({"solution": 1})

    session.add(par_b)
    session.commit()
    session.refresh(par_b)

    par = get_paired_participant(
        par_a,
        session,
        settings,
    )

    assert not par

    par_b.entered_waiting_room = datetime.now(timezone.utc)
    par_b.role = "persuader"

    session.add(par_b)
    session.commit()
    session.refresh(par_b)

    par = get_paired_participant(
        par_a,
        session,
        settings,
    )

    assert par_b == par

    par_b.role = "target"

    session.add(par_b)
    session.commit()
    session.refresh(par_b)

    par = get_paired_participant(
        par_a,
        session,
        settings,
    )

    assert par_b == par

    par_a.role = "persuader"

    session.add(par_a)
    session.commit()
    session.refresh(par_a)

    par = get_paired_participant(
        par_a,
        session,
        settings,
    )

    assert par_b == par

    par_b.role = "persuader"

    session.add(par_b)
    session.commit()
    session.refresh(par_b)

    par = get_paired_participant(
        par_a,
        session,
        settings,
    )

    assert not par

    ###

    par_b.role = "target"

    par_b.allow_lying = True
    par_b.targets_values = False
    par_b.reveal_belief = True
    par_b.reveal_motivation = True

    par_a.allow_lying = None
    par_a.targets_values = None
    par_a.reveal_belief = None
    par_a.reveal_motivation = None
    session.add(par_b)
    session.add(par_a)
    session.commit()
    session.refresh(par_b)
    session.refresh(par_a)

    # Not checking for the extra conditions
    par = get_paired_participant(
        par_a,
        session,
        settings,
    )
    assert par_b == par

    # Now checking for the extra ones

    force_condition = Condition(
        roles=PAIRED_HUMAN_ROLE,
        allow_lying=True,
        targets_values=False,
        reveal_belief=True,
        reveal_motivation=True,
    )

    par = get_paired_participant(
        par_a,
        session,
        settings,
    )
    # We should still get par_b b/c par_a is unassigned
    assert par_b == par

    choose_participant_conditions(par_a, par_b, force_condition, True)

    assert par_a.allow_lying
    assert not par_a.targets_values
    assert par_a.reveal_belief
    assert par_a.reveal_motivation

    par_a.allow_lying = False
    par_a.targets_values = False
    par_a.reveal_belief = False
    par_a.reveal_motivation = False
    par = get_paired_participant(
        par_a,
        session,
        settings,
    )
    # We should not get par_b b/c par_a is assigned
    assert not par

    par_b.allow_lying = None
    par_b.targets_values = None
    par_b.reveal_belief = None
    par_b.reveal_motivation = None
    session.add(par_b)
    session.commit()

    par_a.allow_lying = None
    par_a.targets_values = None
    par_a.reveal_belief = None
    par_a.reveal_motivation = None

    par = get_paired_participant(
        par_a,
        session,
        settings,
    )
    # We should get par_b b/c both are unassigned
    assert par == par_b

    choose_participant_conditions(par_a, par_b, force_condition, True)
    assert isinstance(par_b.allow_lying, bool)
    assert isinstance(par_b.targets_values, bool)
    assert isinstance(par_b.reveal_belief, bool)
    assert isinstance(par_b.reveal_motivation, bool)

    assert isinstance(par_a.allow_lying, bool)
    assert isinstance(par_a.targets_values, bool)
    assert isinstance(par_a.reveal_belief, bool)
    assert isinstance(par_a.reveal_motivation, bool)

    # Cleaning up
    session.add(par_a)
    session.add(par_b)
    session.commit()

    ###

    game = Game(
        model=SOLUTION,
        turn_limit=TURN_LIMIT,
        is_ideal_target=True,
        **EX_SCENARIO,
    )
    gm_data = game.model_dump()

    rd = Round(
        game_data=gm_data,
        game_model_type="solution",
        persuader_id=par_a.id,
        target_id=par_b.id,
        scenario_id=sc.id,
        game_model_id=gm.id,
    )

    session.add(rd)
    session.commit()
    session.refresh(rd)

    # now add the two in a round together and check for rematch

    par = get_paired_participant(
        par_a,
        session,
        settings,
    )
    assert not par

    settings.participants_rematch = True
    par = get_paired_participant(
        par_a,
        session,
        settings,
    )

    assert par == par_b

    session.delete(par_a)
    session.delete(par_b)
    session.delete(rd)
    session.delete(sc)
    session.delete(gm)
    session.commit()


def test_round_methods(session):

    rational_target_condition = Condition(roles=RATIONAL_TARGET_ROLE)
    paired_human_condition = Condition(roles=PAIRED_HUMAN_ROLE)

    user_a = ExternalUser(external_id="A")
    session.add(user_a)
    session.commit()
    session.refresh(user_a)
    assert user_a.id

    user_b = ExternalUser(external_id="B")
    session.add(user_b)
    session.commit()
    session.refresh(user_b)
    assert user_b.id

    sc = Scenario(**EX_SCENARIO)
    gm = Model(data=SOLUTION.model_dump(), game_model_type="solution")

    par_a = Participant(
        id=user_a.id, role="either", game_model_types_remaining=Counter({"solution": 1})
    )
    par_b = Participant(
        id=user_b.id, role="either", game_model_types_remaining=Counter({"solution": 1})
    )

    session.add(par_a)
    session.add(par_b)
    session.add(sc)
    session.add(gm)
    session.commit()

    session.refresh(par_a)
    session.refresh(par_b)
    session.refresh(sc)
    session.refresh(gm)

    # A not in waiting room
    with pytest.raises(ValueError):
        role = choose_condition(
            par_a,
            chosen_participant=None,
            paired_rounds_remaining_set=set([paired_human_condition]),
            non_paired_rounds_remaining_set=set([rational_target_condition]),
            overassign_non_paired_conditions=True,
            enforce_player_round_condition=False,
            condition_num_rounds=Counter({rational_target_condition}),
            waiting_room_timeout=DEFAULT_WAITING_ROOM_TIMEOUT,
        )

    # No rounds to assign
    with pytest.raises(ValueError):
        role = choose_condition(
            par_a,
            chosen_participant=None,
            paired_rounds_remaining_set=set(),
            non_paired_rounds_remaining_set=set(),
            overassign_non_paired_conditions=True,
            enforce_player_round_condition=False,
            condition_num_rounds=Counter({rational_target_condition}),
            waiting_room_timeout=DEFAULT_WAITING_ROOM_TIMEOUT,
        )

    par_a.entered_waiting_room = datetime.now(timezone.utc)
    session.add(par_a)
    session.commit()
    session.refresh(par_a)

    # A has not been in the waiting room long enough to force a role assignment
    # and no paired rounds
    role = choose_condition(
        par_a,
        chosen_participant=None,
        paired_rounds_remaining_set=set([paired_human_condition]),
        non_paired_rounds_remaining_set=set([rational_target_condition]),
        overassign_non_paired_conditions=True,
        enforce_player_round_condition=False,
        condition_num_rounds=Counter({rational_target_condition}),
        waiting_room_timeout=DEFAULT_WAITING_ROOM_TIMEOUT,
    )

    assert role is None

    par_a.entered_waiting_room = (
        datetime.now(timezone.utc) - 2 * DEFAULT_WAITING_ROOM_TIMEOUT
    )
    session.add(par_a)
    session.commit()
    session.refresh(par_a)

    # Overassign
    role = choose_condition(
        par_a,
        chosen_participant=None,
        paired_rounds_remaining_set=set([paired_human_condition]),
        non_paired_rounds_remaining_set=set(),
        overassign_non_paired_conditions=True,
        enforce_player_round_condition=False,
        condition_num_rounds=Counter({rational_target_condition}),
        waiting_room_timeout=DEFAULT_WAITING_ROOM_TIMEOUT,
    )

    assert role.is_rational_target()

    role = choose_condition(
        par_a,
        chosen_participant=None,
        paired_rounds_remaining_set=set(),
        non_paired_rounds_remaining_set=set([rational_target_condition]),
        overassign_non_paired_conditions=False,
        enforce_player_round_condition=False,
        condition_num_rounds=Counter({rational_target_condition}),
        waiting_room_timeout=DEFAULT_WAITING_ROOM_TIMEOUT,
    )

    assert role.is_rational_target()

    par_a.role = "target"

    # There are no roles to assign to the participant
    with pytest.raises(ValueError):
        role = choose_condition(
            par_a,
            chosen_participant=None,
            paired_rounds_remaining_set=set(),
            non_paired_rounds_remaining_set=set([rational_target_condition]),
            overassign_non_paired_conditions=False,
            enforce_player_round_condition=False,
            condition_num_rounds=Counter({rational_target_condition}),
            waiting_room_timeout=DEFAULT_WAITING_ROOM_TIMEOUT,
        )

    rc = Condition(roles=Roles(llm_persuader="test", human_target=True))
    role = choose_condition(
        par_a,
        chosen_participant=None,
        paired_rounds_remaining_set=set([paired_human_condition]),
        non_paired_rounds_remaining_set=set([]),
        overassign_non_paired_conditions=True,
        enforce_player_round_condition=False,
        condition_num_rounds=Counter([rc]),
        waiting_room_timeout=DEFAULT_WAITING_ROOM_TIMEOUT,
    )

    assert role == rc

    with pytest.raises(ValueError):
        role = choose_condition(
            par_a,
            chosen_participant=None,
            paired_rounds_remaining_set=set([paired_human_condition]),
            non_paired_rounds_remaining_set=set([]),
            overassign_non_paired_conditions=True,
            enforce_player_round_condition=False,
            condition_num_rounds=Counter(
                {Condition(roles=Roles(llm_target="test", human_persuader=True))}
            ),
            waiting_room_timeout=DEFAULT_WAITING_ROOM_TIMEOUT,
        )

    # with pytest.raises(WaitingTooLong):
    #     test_p = Participant(
    #         entered_waiting_room=datetime(2024, 9, 12, 9, 59, 27, 174221),
    #         id=user_b.id,
    #         initial_survey_responses=None,
    #         current_round=None,
    #         created_at=datetime(2024, 9, 12, 16, 59, 21),
    #         role="target",
    #         game_model_types_remaining={"solution": 1},
    #         updated_at=datetime(2024, 9, 12, 16, 59, 27),
    #     )

    #     role = choose_condition(
    #         test_p,
    #         chosen_participant=None,
    #         paired_rounds_remaining_set=set([paired_human_condition]),
    #         non_paired_rounds_remaining_set=set([]),
    #         overassign_non_paired_conditions=True,
    #         enforce_player_round_condition=False,
    #         condition_num_rounds=Counter(
    #             {rational_target_condition: 1, paired_human_condition: 2}
    #         ),
    #         waiting_room_timeout=timedelta(seconds=2),
    #     )

    par_a.role = "either"

    ##

    counts = round_types_count(session)

    assert counts.total() == 0

    paired_remaining, non_paired_remaining = get_round_types_remaining(
        session, condition_num_rounds=Counter({})
    )

    assert not paired_remaining
    assert not non_paired_remaining

    paired_remaining, non_paired_remaining = get_round_types_remaining(
        session, condition_num_rounds=Counter({rational_target_condition: 1})
    )

    assert not paired_remaining
    assert rational_target_condition in non_paired_remaining

    paired_remaining, non_paired_remaining = get_round_types_remaining(
        session,
        condition_num_rounds=Counter(
            {rational_target_condition: 1, paired_human_condition: 1}
        ),
    )

    assert paired_remaining
    assert rational_target_condition in non_paired_remaining

    game = Game(
        model=SOLUTION,
        turn_limit=TURN_LIMIT,
        is_ideal_target=True,
        **EX_SCENARIO,
    )
    gm_data = game.model_dump()

    rd = Round(
        game_data=gm_data,
        game_model_type="solution",
        persuader_id=par_a.id,
        scenario_id=sc.id,
        game_model_id=gm.id,
    )

    session.add(rd)
    session.commit()
    session.refresh(rd)

    counts = round_types_count(session)

    assert counts.total() == 0

    rd.persuader_choice = "A"

    session.add(rd)
    session.commit()
    session.refresh(rd)

    counts = round_types_count(session)
    assert counts.total() == 0

    rd.target_choice = "B"

    session.add(rd)
    session.commit()
    session.refresh(rd)

    counts = round_types_count(session)
    assert counts.total() == 1

    assert counts[rational_target_condition] == 1

    paired_remaining, non_paired_remaining = get_round_types_remaining(
        session, condition_num_rounds=Counter({rational_target_condition: 1})
    )

    assert not paired_remaining
    assert not non_paired_remaining

    rd2 = Round(
        game_data=gm_data,
        game_model_type="solution",
        persuader_id=par_b.id,
        scenario_id=sc.id,
        game_model_id=gm.id,
        persuader_choice="A",
        target_choice="B",
    )

    session.add(rd2)
    session.commit()
    session.refresh(rd2)

    counts = round_types_count(session)

    assert counts.total() == 2
    assert counts[rational_target_condition] == 2

    rd3 = Round(
        game_data=gm_data,
        game_model_type="solution",
        persuader_id=par_a.id,
        target_id=par_b.id,
        scenario_id=sc.id,
        game_model_id=gm.id,
        persuader_choice="A",
        target_choice="B",
    )

    session.add(rd3)
    session.commit()
    session.refresh(rd3)

    counts = round_types_count(session)

    assert counts.total() == 3
    assert counts[rational_target_condition] == 2
    assert counts[paired_human_condition] == 1

    rd4 = Round(
        game_data=gm_data,
        game_model_type="solution",
        persuader_id=par_a.id,
        llm_target="test",
        scenario_id=sc.id,
        game_model_id=gm.id,
        persuader_choice="A",
        target_choice="B",
    )

    rd5 = Round(
        game_data=gm_data,
        game_model_type="solution",
        llm_persuader="test",
        target_id=par_b.id,
        scenario_id=sc.id,
        game_model_id=gm.id,
        persuader_choice="A",
        target_choice="B",
    )

    session.add(rd4)
    session.add(rd5)
    session.commit()

    counts = round_types_count(session)

    assert counts.total() == 5
    assert counts[rational_target_condition] == 2
    assert counts[paired_human_condition] == 1
    assert counts[Condition(roles=Roles(llm_persuader="test", human_target=True))] == 1
    assert counts[Condition(roles=Roles(llm_target="test", human_persuader=True))] == 1

    session.delete(par_a)
    session.delete(par_b)
    session.delete(rd)
    session.delete(rd2)
    session.delete(rd3)
    session.delete(rd4)
    session.delete(rd5)
    session.delete(sc)
    session.delete(gm)

    session.commit()


def test_get_bonuses(session):

    # No rounds completed
    user_a = ExternalUser(external_id="A")
    session.add(user_a)
    session.commit()
    session.refresh(user_a)

    par_a = Participant(
        id=user_a.id,
        role="persuader",
        game_model_types_remaining=Counter({"solution": 1}),
        work_approved=False,
    )
    session.add(par_a)
    session.commit()

    bonuses = get_bonuses(session=session)
    assert bonuses["A"] == 0

    # Round completed participant already approved
    par_a.work_approved = True
    session.add(par_a)
    session.commit()

    game = Game(
        model=SOLUTION,
        turn_limit=TURN_LIMIT,
        is_ideal_target=True,
        **EX_SCENARIO,
    )
    gm = Model(data=SOLUTION.model_dump(), game_model_type="solution")
    session.add(gm)
    session.commit()
    session.refresh(gm)

    sc = Scenario(**EX_SCENARIO)
    session.add(sc)
    session.commit()
    session.refresh(sc)

    rd = Round(
        game_data=game.model_dump(),
        persuader_id=par_a.id,
        target_choice="B",
        persuader_choice="A",
        scenario_id=sc.id,
        game_model_id=gm.id,
    )
    session.add(rd)
    session.commit()

    bonuses = get_bonuses(session=session)
    assert bonuses["A"] == 0

    # Round completed as persuader; did not convince target
    par_a.work_approved = None
    session.add(par_a)
    session.commit()

    bonuses = get_bonuses(session=session)
    assert bonuses["A"] == 0

    # Round completed as persuader; convinced target
    rd.persuader_choice = "A"
    rd.target_choice = "A"
    session.add(rd)
    session.commit()

    bonuses = get_bonuses(session=session)
    assert bonuses["A"] == 1

    # Round completed as target; chose the perfect info choice
    rd2 = Round(
        game_data=game.model_dump(),
        llm_persuader="test",
        target_id=par_a.id,
        persuader_choice="B",
        target_choice="A",
        target_perfect_info_choice="A",
        scenario_id=sc.id,
        game_model_id=gm.id,
    )
    session.add(rd2)
    session.commit()

    bonuses = get_bonuses(session=session)
    assert bonuses["A"] == 2

    # Round completed as target; did not choose the perfect info choice
    rd2.target_choice = "B"
    session.add(rd2)
    session.commit()

    bonuses = get_bonuses(session=session)
    assert bonuses["A"] == 1

    # Round completed as target; perfect info choice not set (Experiment 3)
    rd2.target_perfect_info_choice = None
    session.add(rd2)
    session.commit()

    bonuses = get_bonuses(session=session)
    assert bonuses["A"] == 1

    # Clean up
    session.delete(par_a)
    session.delete(rd)
    session.delete(rd2)
    session.delete(sc)
    session.delete(gm)
    session.commit()


def test_rounds_by_condition(session):

    # Fetch rounds by condition
    condition_to_games = rounds_by_condition(session)
    assert len(condition_to_games) == 0

    # Setup users, participants, and models for test
    user_a = ExternalUser(external_id="A")
    user_b = ExternalUser(external_id="B")
    session.add(user_a)
    session.add(user_b)
    session.commit()
    session.refresh(user_a)
    session.refresh(user_b)

    gm = Model(data=SOLUTION.model_dump(), game_model_type="solution")
    sc = Scenario(**EX_SCENARIO)
    session.add(gm)
    session.add(sc)
    session.commit()
    session.refresh(gm)
    session.refresh(sc)

    # Create participants
    par_a = Participant(
        id=user_a.id,
        role="persuader",
        game_model_types_remaining=Counter({"solution": 1}),
        work_approved=False,
    )
    par_b = Participant(
        id=user_b.id,
        role="target",
        game_model_types_remaining=Counter({"solution": 1}),
        work_approved=False,
    )
    session.add(par_a)
    session.add(par_b)
    session.commit()

    # Define some conditions
    condition_1 = Condition(
        roles=Roles(human_persuader=True, human_target=True),
        targets_values=False,
        reveal_motivation=True,
        reveal_belief=False,
        allow_lying=True,
    )

    # Create a game instance
    game = Game(
        model=SOLUTION,
        turn_limit=TURN_LIMIT,
        is_ideal_target=True,
        **EX_SCENARIO,
    )
    game.persuader_choice = "A"
    game.target_choice = "A"
    game.messages = [{}] * TURN_LIMIT * 2
    assert game.game_over()
    gm_data = game.model_dump()

    # Add rounds to the session with different conditions
    round_1 = Round(
        persuader_id=par_a.id,
        target_id=par_b.id,
        llm_persuader=None,
        llm_target=None,
        targets_values=condition_1.targets_values,
        reveal_motivation=condition_1.reveal_motivation,
        reveal_belief=condition_1.reveal_belief,
        allow_lying=condition_1.allow_lying,
        game_data=gm_data,
        scenario_id=sc.id,
        game_model_id=gm.id,
        persuader_choice="A",
        target_choice="A",
        game_model_type="solution",
    )

    session.add(round_1)
    session.commit()

    # Fetch rounds by condition
    condition_to_games = rounds_by_condition(session)

    # Assert the rounds are grouped correctly
    assert len(condition_to_games) == 1

    conditions = set(
        condition.as_non_id_role() for condition in condition_to_games.keys()
    )
    assert condition_1 in conditions
    assert len(list(condition_to_games.values())[0]) == 1

    # Clean up
    session.delete(par_a)
    session.delete(par_b)
    session.delete(round_1)
    session.delete(sc)
    session.delete(gm)
    session.commit()
