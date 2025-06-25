from collections import Counter
import os

import pytest
from sqlmodel import Session

from mindgames.game import Game
from mindgames.utils import EX_SCENARIO, DEFAULT_PROPOSALS
from mindgames.known_models import SOLUTION
from mindgames.conditions import RATIONAL_TARGET_ROLE, PAIRED_HUMAN_ROLE, Roles

from api.message_processing import (
    process_sent_message,
    receieve_response_and_choose_proposal,
)
from api.sql_model import Scenario, Model, Participant, Round, ExternalUser
from api.sql_queries import get_last_sent_message


from .context import engine_fixture


@pytest.mark.skipif(
    os.getenv("RUN_QUERY_TESTS", "False") != "True", reason="Skipping query test case"
)
def test_receieve_response_and_choose_proposal():
    game = Game(
        model=SOLUTION,
        turn_limit=2,
        is_ideal_target=True,
        **EX_SCENARIO,
    )

    msg = {"role": "persuader", "content": "1"}
    game.messages.append(msg)
    msg = {"role": "target", "content": "2"}
    game.messages.append(msg)
    content, prop = receieve_response_and_choose_proposal(
        is_target=True, role=RATIONAL_TARGET_ROLE, game=game
    )

    assert content == msg["content"]
    assert not prop

    msg = {"role": "persuader", "content": "3"}
    game.messages.append(msg)
    msg = {"role": "target", "content": "4"}
    game.messages.append(msg)
    content, prop = receieve_response_and_choose_proposal(
        is_target=True, role=RATIONAL_TARGET_ROLE, game=game
    )

    assert content == msg["content"]
    assert prop == game.choose_proposal(is_target=True)

    # Test a game in which we get LLM responses
    game = Game(
        model=SOLUTION,
        turn_limit=2,
        is_ideal_target=False,
        **EX_SCENARIO,
    )

    role = Roles(llm_persuader="gpt-4o", human_target=True)
    content, prop = receieve_response_and_choose_proposal(
        is_target=False, role=role, game=game
    )
    assert content
    assert not prop

    game.messages.append({"role": "target", "content": "response 1"})
    content, prop = receieve_response_and_choose_proposal(
        is_target=False, role=role, game=game
    )
    assert content
    assert prop == game.choose_proposal(is_target=False)

    # Now have the llm choose a proposal
    game = Game(
        model=SOLUTION,
        turn_limit=2,
        is_ideal_target=False,
        **EX_SCENARIO,
    )
    game.messages.append({"role": "persuader", "content": "statement 1"})
    game.chain_of_thought.append({"role": "persuader", "content": None})

    role = Roles(llm_target="gpt-4o", human_persuader=True)

    content, prop = receieve_response_and_choose_proposal(
        is_target=True, role=role, game=game
    )
    assert content
    assert not prop

    game.messages.append({"role": "persuader", "content": "statement 2"})
    game.chain_of_thought.append({"role": "persuader", "content": None})

    content, prop = receieve_response_and_choose_proposal(
        is_target=True, role=role, game=game
    )
    assert content
    assert prop in DEFAULT_PROPOSALS

    # Now Together
    game = Game(
        model=SOLUTION,
        turn_limit=2,
        is_ideal_target=False,
        **EX_SCENARIO,
    )
    game.messages.append({"role": "persuader", "content": "statement 1"})
    game.chain_of_thought.append({"role": "persuader", "content": None})

    role = Roles(
        llm_target="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", human_persuader=True
    )

    content, prop = receieve_response_and_choose_proposal(
        is_target=True, role=role, game=game
    )
    assert content
    assert not prop

    game.messages.append({"role": "persuader", "content": "statement 2"})
    game.chain_of_thought.append({"role": "persuader", "content": None})

    content, prop = receieve_response_and_choose_proposal(
        is_target=True, role=role, game=game
    )
    assert content
    assert prop in DEFAULT_PROPOSALS

    # Now anthropic
    game = Game(
        model=SOLUTION,
        turn_limit=2,
        is_ideal_target=False,
        **EX_SCENARIO,
    )
    game.messages.append({"role": "persuader", "content": "statement 1"})
    game.chain_of_thought.append({"role": "persuader", "content": None})

    role = Roles(llm_target="claude-3-5-sonnet-20240620", human_persuader=True)

    content, prop = receieve_response_and_choose_proposal(
        is_target=True, role=role, game=game
    )
    assert content
    assert not prop

    game.messages.append({"role": "persuader", "content": "statement 2"})
    game.chain_of_thought.append({"role": "persuader", "content": None})

    content, prop = receieve_response_and_choose_proposal(
        is_target=True, role=role, game=game
    )
    assert content
    assert prop in DEFAULT_PROPOSALS


@pytest.mark.skipif(
    os.getenv("RUN_QUERY_TESTS", "False") != "True", reason="Skipping query test case"
)
def test_process_sent_message(engine):
    game = Game(
        model=SOLUTION,
        turn_limit=2,
        is_ideal_target=True,
        **EX_SCENARIO,
    )
    gm_data = game.model_dump()

    round_id = None
    par_a_id = None
    par_b_id = None
    sc_id = None
    gm_id = None
    with Session(engine) as session:
        sc = Scenario(**EX_SCENARIO)
        gm = Model(data=SOLUTION.model_dump(), game_model_type="solution")

        user_a = ExternalUser(external_id="A")
        session.add(user_a)
        user_b = ExternalUser(external_id="B")
        session.add(user_b)
        session.commit()
        session.refresh(user_b)
        session.refresh(user_a)
        assert user_a.id
        assert user_b.id

        par_a = Participant(
            id=user_a.id,
            role="either",
            game_model_types_remaining=Counter({"solution": 1}),
        )
        par_b = Participant(
            id=user_b.id,
            role="either",
            game_model_types_remaining=Counter({"solution": 1}),
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

        par_a_id = par_a.id
        par_b_id = par_b.id
        sc_id = sc.id
        gm_id = gm.id

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
        round_id = rd.id

        assert not rd.awaiting_target_response
        assert rd.awaiting_persuader_response

        assert not rd.persuader_choice
        assert not rd.target_choice

        assert not rd.processing_target_response
        assert not rd.processing_persuader_response

        assert not game.messages

        session.add(rd)
        session.commit()
        session.refresh(rd)

    process_sent_message(
        "test",
        is_target=False,
        game=game,
        role=RATIONAL_TARGET_ROLE,
        round_id=round_id,
        engine=engine,
    )

    with Session(engine) as session:

        rd = session.get(Round, round_id)

        assert not rd.awaiting_target_response
        assert rd.awaiting_persuader_response

        assert not rd.persuader_choice
        assert not rd.target_choice

        assert not rd.processing_target_response
        assert not rd.processing_persuader_response
        game = Game(**rd.game_data)

        assert game.messages[-2] == {"role": "persuader", "content": "test"}
        assert game.messages[-1]["role"] == "target"

        last = get_last_sent_message(session, is_target=False, round_id=rd.id)
        assert not last.flagged

    # now test a lie
    process_sent_message(
        "Proposal A increases housing availability by one million.",
        is_target=False,
        game=game,
        role=RATIONAL_TARGET_ROLE,
        round_id=round_id,
        engine=engine,
    )

    with Session(engine) as session:

        rd = session.get(Round, round_id)

        assert not rd.awaiting_target_response
        assert rd.awaiting_persuader_response

        assert not rd.persuader_choice
        assert not rd.target_choice

        assert not rd.processing_target_response
        assert not rd.processing_persuader_response

        game = Game(**rd.game_data)
        assert len(game.messages) == 2

        last = get_last_sent_message(session, is_target=False, round_id=rd.id)
        assert last.flagged

    process_sent_message(
        "test2",
        is_target=False,
        game=game,
        role=RATIONAL_TARGET_ROLE,
        round_id=round_id,
        engine=engine,
    )

    with Session(engine) as session:
        rd = session.get(Round, round_id)

        assert not rd.awaiting_target_response
        assert not rd.awaiting_persuader_response

        game = Game(**rd.game_data)

        assert rd.persuader_choice == game.choose_proposal(is_target=False)
        assert rd.target_choice == game.choose_proposal(is_target=True)

        assert not rd.processing_target_response
        assert not rd.processing_persuader_response

        assert game.messages[-2] == {"role": "persuader", "content": "test2"}
        assert game.messages[-1]["role"] == "target"

    ### Now a new game

    game = Game(
        model=SOLUTION,
        turn_limit=2,
        is_ideal_target=False,
        **EX_SCENARIO,
    )

    with Session(engine) as session:
        rd = Round(
            game_data=game.model_dump(),
            game_model_type="solution",
            persuader_id=par_a_id,
            target_id=par_b_id,
            scenario_id=sc_id,
            game_model_id=gm_id,
        )

        session.add(rd)
        session.commit()
        session.refresh(rd)
        round_id = rd.id

        assert not rd.awaiting_target_response
        assert rd.awaiting_persuader_response

        assert not rd.persuader_choice
        assert not rd.target_choice

        assert not rd.processing_target_response
        assert not rd.processing_persuader_response

        assert not game.messages

        session.add(rd)
        session.commit()
        session.refresh(rd)

    process_sent_message(
        "1",
        is_target=False,
        game=game,
        role=PAIRED_HUMAN_ROLE,
        round_id=round_id,
        engine=engine,
    )

    with Session(engine) as session:
        rd = session.get(Round, round_id)

        assert rd.awaiting_target_response
        assert not rd.awaiting_persuader_response

        game = Game(**rd.game_data)

        assert not rd.persuader_choice
        assert not rd.target_choice

        assert not rd.processing_target_response
        assert not rd.processing_persuader_response

        assert game.messages[-1] == {"role": "persuader", "content": "1"}
        last = get_last_sent_message(session, is_target=False, round_id=rd.id)
        assert last.content == "1"
        assert not last.flagged

    process_sent_message(
        "2",
        is_target=True,
        game=game,
        role=PAIRED_HUMAN_ROLE,
        round_id=round_id,
        engine=engine,
    )

    with Session(engine) as session:
        rd = session.get(Round, round_id)

        assert not rd.awaiting_target_response
        assert rd.awaiting_persuader_response

        game = Game(**rd.game_data)

        assert not rd.persuader_choice
        assert not rd.target_choice

        assert not rd.processing_target_response
        assert not rd.processing_persuader_response

        assert game.messages[-1] == {"role": "target", "content": "2"}
        last = get_last_sent_message(session, is_target=True, round_id=rd.id)
        assert last.content == "2"
        assert not last.flagged

    process_sent_message(
        "3",
        is_target=False,
        game=game,
        role=PAIRED_HUMAN_ROLE,
        round_id=round_id,
        engine=engine,
    )

    with Session(engine) as session:
        rd = session.get(Round, round_id)

        assert rd.awaiting_target_response
        assert not rd.awaiting_persuader_response

        game = Game(**rd.game_data)

        assert rd.persuader_choice == game.choose_proposal(is_target=False)
        assert not rd.target_choice

        assert not rd.processing_target_response
        assert not rd.processing_persuader_response

        assert game.messages[-1] == {"role": "persuader", "content": "3"}
        last = get_last_sent_message(session, is_target=False, round_id=rd.id)
        assert last.content == "3"
        assert not last.flagged

    process_sent_message(
        "4",
        is_target=True,
        game=game,
        role=PAIRED_HUMAN_ROLE,
        round_id=round_id,
        engine=engine,
    )

    with Session(engine) as session:
        rd = session.get(Round, round_id)

        assert not rd.awaiting_target_response
        assert not rd.awaiting_persuader_response

        game = Game(**rd.game_data)

        assert rd.persuader_choice == game.choose_proposal(is_target=False)
        assert not rd.target_choice

        assert not rd.processing_target_response
        assert not rd.processing_persuader_response

        assert game.messages[-1] == {"role": "target", "content": "4"}
        last = get_last_sent_message(session, is_target=True, round_id=rd.id)
        assert last.content == "4"
        assert not last.flagged

    # TODO: test interacting with an LLM with persuader and as target
