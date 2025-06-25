"""
Author: Jared Moore
Date: September, 2024

Tests for the sql database models.
"""

from collections import Counter
import time

import pytest


###

from mindgames.game import Game, TURN_LIMIT
from mindgames.model import GameModel

from mindgames.known_models import (
    SOLUTION,
)

from mindgames.utils import (
    EX_SCENARIO,
    EXPERIMENT_CONDITIONS,
)

from api.sql_model import (
    ExternalUser,
    Model,
    Scenario,
    Round,
    Participant,
    SentMessage,
)
from api.current_round_helpers import assign_participants
from api.sql_queries import populate_tables, get_last_sent_message


from .context import session_fixture, engine_fixture


def test_game_model():

    with pytest.raises(ValueError):
        Model()

    with pytest.raises(ValueError):
        Model(data=SOLUTION.model_dump())

    with pytest.raises(ValueError):
        Model(data=SOLUTION.model_dump(), game_model_type="blah")

    with pytest.raises(ValueError):
        Model(data=SOLUTION.model_dump(), game_model_type="can-win")

    gm = Model(data=SOLUTION.model_dump(), game_model_type="solution")
    assert GameModel(**gm.data) == SOLUTION


def test_scenario():

    sc = Scenario(**EX_SCENARIO)
    scenario = EX_SCENARIO.copy()
    del scenario["proposals"]
    assert sc.model_dump() == scenario


def test_external_user(session):

    user = ExternalUser(external_id="A")
    session.add(user)
    session.commit()
    session.refresh(user)
    assert user.id

    par = Participant(
        id=user.id, role="target", game_model_types_remaining=Counter({"solution": 1})
    )

    session.add(par)
    session.commit()

    session.delete(par)
    session.delete(user)
    session.commit()


def test_participant():

    with pytest.raises(ValueError):
        Participant(
            id=0, role="test", game_model_types_remaining=Counter({"solution": 1})
        )

    with pytest.raises(ValueError):
        Participant(
            id=0, role="target", game_model_types_remaining=Counter({"test": 1})
        )

    Participant(
        id=0, role="target", game_model_types_remaining=Counter({"solution": 1})
    )


def test_assign_participants():
    par_a = Participant(id=0, role="either")
    par_b = None

    p_id, t_id = assign_participants(par_a, par_b)

    assert p_id == par_a.id
    assert t_id is None

    par_a = Participant(id=0, role="target")
    par_b = None
    p_id, t_id = assign_participants(par_a, par_b)

    assert t_id == par_a.id
    assert p_id is None

    par_a = Participant(id=0, role="persuader")
    par_b = None
    p_id, t_id = assign_participants(par_a, par_b)

    assert p_id == par_a.id
    assert t_id is None

    par_a = Participant(id=0, role="persuader")
    par_b = Participant(id=1, role="target")
    p_id, t_id = assign_participants(par_a, par_b)

    assert p_id == par_a.id
    assert t_id == par_b.id

    par_a = Participant(id=0, role="target")
    par_b = Participant(id=1, role="persuader")
    p_id, t_id = assign_participants(par_a, par_b)

    assert t_id == par_a.id
    assert p_id == par_b.id

    par_a = Participant(id=0, role="either")
    par_b = Participant(id=1, role="persuader")
    p_id, t_id = assign_participants(par_a, par_b)

    assert t_id == par_a.id
    assert p_id == par_b.id

    par_a = Participant(id=0, role="either")
    par_b = Participant(id=1, role="target")
    p_id, t_id = assign_participants(par_a, par_b)

    assert p_id == par_a.id
    assert t_id == par_b.id

    par_a = Participant(id=0, role="either")
    par_b = Participant(id=1, role="either")
    p_id, t_id = assign_participants(par_a, par_b)

    assert p_id == par_a.id
    assert t_id == par_b.id


def test_round_and_send_messages(session):
    sc = Scenario(**EX_SCENARIO)

    user = ExternalUser(external_id="A")
    session.add(user)
    session.commit()
    session.refresh(user)

    par = Participant(
        id=user.id,
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

    assert sc.id is not None
    assert sc.id is not None
    assert par.id is not None

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

    assert rd.id is not None

    ##

    with pytest.raises(ValueError):
        Round()

    with pytest.raises(ValueError):
        # Cannot have a game with just a human target
        # and no llm
        rd = Round(
            game_data=gm_data,
            game_model_type="solution",
            target_id=par.id,
            scenario_id=sc.id,
            game_model_id=gm.id,
        )

    with pytest.raises(ValueError):
        # Cannot have a game with just an llm
        rd = Round(
            game_data=gm_data,
            game_model_type="solution",
            llm_persuader="A",
            scenario_id=sc.id,
            game_model_id=gm.id,
        )

    with pytest.raises(ValueError):
        # or just two
        rd = Round(
            game_data=gm_data,
            game_model_type="solution",
            llm_persuader="A",
            llm_target="B",
            scenario_id=sc.id,
            game_model_id=gm.id,
        )

    with pytest.raises(ValueError):
        # or just two
        rd = Round(
            game_data=gm_data,
            game_model_type="solution",
            llm_persuader="A",
            llm_target="B",
            scenario_id=sc.id,
            game_model_id=gm.id,
        )

    rd = Round(
        game_data=gm_data,
        game_model_type="solution",
        persuader_id=par.id,
        scenario_id=sc.id,
        game_model_id=gm.id,
    )

    assert rd.awaiting_persuader_response
    assert not rd.awaiting_target_response

    assert game == Game(**rd.game_data)

    session.add(rd)

    session.commit()

    session.refresh(rd)

    assert rd.id is not None

    par.current_round = rd.id

    session.add(par)
    session.commit()
    session.refresh(par)

    msg = SentMessage(is_target=False, content="A", flagged=False, round_id=rd.id)

    session.add(msg)
    session.commit()
    session.refresh(msg)

    assert msg.id

    last_msg = get_last_sent_message(session, is_target=False, round_id=rd.id)

    assert last_msg
    assert last_msg == msg

    msg2 = SentMessage(is_target=False, content="B", flagged=False, round_id=rd.id)
    time.sleep(2)
    session.add(msg2)
    session.commit()
    session.refresh(msg2)

    assert msg2.id

    last_msg2 = get_last_sent_message(session, is_target=False, round_id=rd.id)
    assert last_msg2
    assert last_msg2 == msg2

    session.delete(par)
    session.delete(rd)
    session.delete(sc)
    session.delete(gm)
    session.delete(msg)
    session.delete(msg2)
    session.commit()


def test_populate_tables(session):

    populate_tables(session=session, difficulty_conditions=EXPERIMENT_CONDITIONS)
