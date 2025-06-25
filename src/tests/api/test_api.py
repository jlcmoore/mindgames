from collections import Counter
import copy
from datetime import datetime, timedelta, timezone
import os
import time

import pytest


from sqlmodel import Session

from fastapi import HTTPException

##

from mindgames.game import Game
from mindgames.conditions import Condition, RATIONAL_TARGET_ROLE, PAIRED_HUMAN_ROLE
from mindgames.utils import EXPERIMENT_CONDITIONS


from api.sql_queries import (
    populate_tables,
    get_last_sent_message,
)

from api.utils import (
    ServerSettings,
)

from api.api import (
    participant_init,
    participant_ready,
    current_round,
    send_message,
    retrieve_response,
    make_choice,
    round_result,
    participant_rounds,
    ParticipantInitRequest,
    MessageRequest,
    ChoiceRequest,
    ParticipantRequest,
    RoundRequest,
    ParticipantRoundRequest,
)

from api.sql_model import Model, Scenario, Round, Participant, ExternalUser

from .test_sql_queries import test_participant_rounds

from .context import session_fixture, engine_fixture

TEST_SETTINGS = ServerSettings(
    round_conditions=Counter({"solution": 1, "can-win": 1}),
    condition_num_rounds=Counter(
        {
            Condition(roles=RATIONAL_TARGET_ROLE): 1,
            Condition(roles=PAIRED_HUMAN_ROLE): 2,
        }
    ),
    conditions=None,
    turn_limit=2,
    waiting_room_timeout=timedelta(seconds=2),
    enforce_persuader_target_roles=True,
    enforce_player_round_condition=False,
    participant_conversation_timeout=timedelta(seconds=2),
    overassign_non_paired_conditions=True,
    non_mental=False,
)


def test_participant_init(session: Session):

    settings = ServerSettings(
        round_conditions=Counter({"solution": 1}),
        enforce_persuader_target_roles=False,
    )
    response = participant_init(ParticipantInitRequest(id="A"), session, settings)

    assert response["participant_id"] == 1

    par = session.get(Participant, 1)

    assert par.game_model_types_remaining == Counter({"solution": 1})

    session.delete(par)
    session.commit()

    user = session.get(ExternalUser, 1)
    session.delete(user)
    session.commit()

    response = participant_init(ParticipantInitRequest(id="A"), session, TEST_SETTINGS)

    assert response["participant_id"] == 1

    par = session.get(Participant, 1)
    assert par.game_model_types_remaining == Counter({"solution": 1, "can-win": 1})

    response = participant_init(ParticipantInitRequest(id="B"), session, TEST_SETTINGS)

    assert response["participant_id"] == 2

    response = participant_init(ParticipantInitRequest(id="A"), session, TEST_SETTINGS)

    assert response["participant_id"] == 1


def test_participant_entered_waiting_room(session: Session):
    with pytest.raises(HTTPException):
        response = participant_ready(ParticipantRequest(id=1), session)

    test_participant_init(session)

    par = session.get(Participant, 1)

    assert par.entered_waiting_room is None

    response = participant_ready(ParticipantRequest(id=1), session)

    assert response is None

    par = session.get(Participant, 1)

    assert datetime.now(timezone.utc) - par.entered_waiting_room.replace(
        tzinfo=timezone.utc
    ) < timedelta(seconds=1)

    time.sleep(2)

    response = participant_ready(ParticipantRequest(id=1), session)

    assert response is None

    par = session.get(Participant, 1)

    assert datetime.now(timezone.utc) - par.entered_waiting_room.replace(
        tzinfo=timezone.utc
    ) < timedelta(seconds=1)


def test_current_round(session: Session):
    with pytest.raises(HTTPException):
        response = current_round(
            ParticipantRoundRequest(participant_id=1), session, TEST_SETTINGS, None
        )

    # Adds A and B, B not waiting yet
    test_participant_entered_waiting_room(session)

    par_a = session.get(Participant, 1)
    assert Counter(par_a.game_model_types_remaining).total() == 2

    par_b = session.get(Participant, 2)
    assert Counter(par_b.game_model_types_remaining).total() == 2

    populate_tables(session, EXPERIMENT_CONDITIONS)
    # waiting response, empty dict
    response = current_round(
        ParticipantRoundRequest(participant_id=1), session, TEST_SETTINGS, None
    )
    assert "waiting_time" in response

    # The participant has not yet timed out to be assigned to the rational target

    time.sleep(3)

    par_a = session.get(Participant, 1)
    assert Counter(par_a.game_model_types_remaining).total() == 2

    response = current_round(
        ParticipantRoundRequest(participant_id=1), session, TEST_SETTINGS, None
    )

    # The participant has timed out to be assigned to the rational target
    assert response["round_id"] == 1
    assert not response["is_target"]

    response = current_round(
        ParticipantRoundRequest(participant_id=1), session, TEST_SETTINGS, None
    )
    # The participant has timed out to be assigned to the rational target
    assert response["round_id"] == 1

    par_a = session.get(Participant, 1)
    assert Counter(par_a.game_model_types_remaining).total() == 1

    rd = session.get(Round, response["round_id"])

    # Force end the round
    rd.persuader_choice = "test"
    rd.target_choice = "test"
    par_a.current_round = None
    par_b.current_round = None
    gm = Game(**rd.game_data)
    gm.messages = [
        {"role": "persuader", "content": "1"},
        {"role": "target", "content": "2"},
        {"role": "persuader", "content": "3"},
        {"role": "target", "content": "4"},
    ]
    gm.target_choice = "test"
    gm.persuader_choice = "test"
    rd.game_data = gm.model_dump()

    session.add(rd)
    session.add(par_a)
    session.add(par_b)
    session.commit()

    response = participant_ready(ParticipantRequest(id=1), session)
    assert response is None

    par_a = session.get(Participant, 1)
    assert par_a.entered_waiting_room
    assert not par_a.current_round
    assert Counter(par_a.game_model_types_remaining).total() == 1

    response = current_round(
        ParticipantRoundRequest(participant_id=1), session, TEST_SETTINGS, None
    )
    assert "waiting_time" in response

    response = participant_ready(ParticipantRequest(id=2), session)
    assert response is None

    par_b = session.get(Participant, 2)
    assert par_b.entered_waiting_room
    assert not par_b.current_round

    response_1 = current_round(
        ParticipantRoundRequest(participant_id=1), session, TEST_SETTINGS, None
    )
    response_2 = current_round(
        ParticipantRoundRequest(participant_id=2), session, TEST_SETTINGS, None
    )

    assert response_1["round_id"] == response_2["round_id"]

    # A was the persuader last and should be again based on the forced
    # role setting
    assert not response_1["is_target"]
    assert response_2["is_target"]

    rd = session.get(Round, response_1["round_id"])

    # Force end the round
    rd.persuader_choice = "test"
    rd.target_choice = "test"
    par_a.current_round = None
    par_b.current_round = None
    gm = Game(**rd.game_data)
    gm.messages = [
        {"role": "persuader", "content": "1"},
        {"role": "target", "content": "2"},
        {"role": "persuader", "content": "3"},
        {"role": "target", "content": "4"},
    ]
    gm.target_choice = "test"
    gm.persuader_choice = "test"
    rd.game_data = gm.model_dump()

    session.add(rd)
    session.add(par_a)
    session.add(par_b)
    session.commit()

    par_b = session.get(Participant, 2)
    assert Counter(par_b.game_model_types_remaining).total() == 1

    response = participant_ready(ParticipantRequest(id=1), session)
    assert response is None
    # A has filled all their slots and should be done
    with pytest.raises(HTTPException):
        response = current_round(
            ParticipantRoundRequest(participant_id=1), session, TEST_SETTINGS, None
        )

    # But B still has slots left -- and the whole game does for a paired
    # game
    response = participant_ready(ParticipantRequest(id=2), session)
    assert response is None

    # Need to wait for the waiting room timeout
    time.sleep(3)

    with pytest.raises(ValueError):
        response = current_round(
            ParticipantRoundRequest(participant_id=2), session, TEST_SETTINGS, None
        )

    # Fudging it so we can overassign to a rational-target-role
    par_b = session.get(Participant, 2)
    par_b.role = "either"
    par_b = session.get(Participant, 2)
    assert Counter(par_b.game_model_types_remaining).total() == 1

    session.add(par_b)
    session.commit()

    # wait more than the timeout
    time.sleep(3)

    response = current_round(
        ParticipantRoundRequest(participant_id=2), session, TEST_SETTINGS, None
    )
    assert not response["is_target"]

    rd = session.get(Round, response["round_id"])

    session.refresh(par_b)
    par_b.role = "target"
    assert Counter(par_b.game_model_types_remaining).total() == 0

    par_b.game_model_types_remaining = Counter({"solution": 1})

    # Delete the round to now let b and C pair
    session.add(par_b)

    ## change settings overassign_non_paired_conditions = False
    session.delete(rd)
    session.commit()

    par_b.game_model_types_remaining = Counter({"solution": 1})

    par_b.current_round = None
    session.add(par_b)

    session.add(par_b)
    session.commit()

    # wait more than the timeout
    time.sleep(3)

    settings = TEST_SETTINGS
    settings.overassign_non_paired_conditions = False
    response = participant_ready(ParticipantRequest(id=2), session)
    assert response is None
    response = current_round(
        ParticipantRoundRequest(participant_id=2), session, settings, None
    )
    assert "waiting_time" in response

    par_b = session.get(Participant, 2)
    assert Counter(par_b.game_model_types_remaining).total() == 1

    ## Now another game between A and B which we allow by rematch and then delete

    # First make A think that it has rounds left
    par_a = session.get(Participant, 1)
    par_a.game_model_types_remaining = Counter({"solution": 1})

    response = participant_ready(ParticipantRequest(id=1), session)
    assert response is None
    response = participant_ready(ParticipantRequest(id=2), session)
    assert response is None

    settings = TEST_SETTINGS
    settings.participants_rematch = True
    response_1 = current_round(
        ParticipantRoundRequest(participant_id=1), session, settings, None
    )
    response_2 = current_round(
        ParticipantRoundRequest(participant_id=2), session, settings, None
    )

    assert response_1["round_id"] == response_2["round_id"]

    par_a = session.get(Participant, 1)
    par_a.game_model_types_remaining = Counter()

    par_b = session.get(Participant, 2)
    par_b.game_model_types_remaining = Counter({"solution": 1})

    rd = session.get(Round, response_1["round_id"])

    par_b.current_round = None
    par_a.current_round = None
    session.add(par_a)

    session.add(par_b)

    session.delete(rd)

    ## Now a new game
    # So now we add in participant C
    response = participant_ready(ParticipantRequest(id=2), session)
    assert response is None
    response = participant_init(ParticipantInitRequest(id="C"), session, TEST_SETTINGS)
    assert response["participant_id"] == 3
    response = participant_ready(ParticipantRequest(id=3), session)
    assert response is None

    par = session.get(Participant, 2)
    assert Counter(par.game_model_types_remaining).total() > 0
    response_1 = current_round(
        ParticipantRoundRequest(participant_id=2), session, TEST_SETTINGS, None
    )

    response_2 = current_round(
        ParticipantRoundRequest(participant_id=3), session, TEST_SETTINGS, None
    )

    assert response_1["round_id"] == response_2["round_id"]
    assert response_1["is_target"]

    par_b = session.get(Participant, 2)
    assert Counter(par_b.game_model_types_remaining).total() == 0

    par_c = session.get(Participant, 3)
    assert Counter(par_c.game_model_types_remaining).total() == 1

    rd = session.get(Round, response_1["round_id"])

    # Force end the round
    rd.persuader_choice = "test"
    rd.target_choice = "test"
    par_a.current_round = None
    par_b.current_round = None
    gm = Game(**rd.game_data)
    gm.messages = [
        {"role": "persuader", "content": "1"},
        {"role": "target", "content": "2"},
        {"role": "persuader", "content": "3"},
        {"role": "target", "content": "4"},
    ]
    gm.target_choice = "test"
    gm.persuader_choice = "test"
    rd.game_data = gm.model_dump()

    session.add(rd)
    session.add(par_a)
    session.add(par_b)
    session.commit()

    # C still has a round to play but shouldn't be allowed as the whole game is now full
    response = participant_ready(ParticipantRequest(id=3), session)
    assert response is None

    with pytest.raises(HTTPException):
        response = current_round(
            ParticipantRoundRequest(participant_id=3), session, TEST_SETTINGS, None
        )

    ##
    # Add extra parameters but not in dev
    with pytest.raises(HTTPException):
        settings = TEST_SETTINGS
        settings.dev_environment = False
        # This is a rational target game
        response = current_round(
            ParticipantRoundRequest(
                participant_id=1,
                is_target=False,
                game_model_type="solution",
            ),
            session,
            settings,
            None,
        )

    # This is not a valid game, there must be an llm persuader
    settings = TEST_SETTINGS
    settings.dev_environment = True
    with pytest.raises(HTTPException):
        response = current_round(
            ParticipantRoundRequest(
                participant_id=1,
                is_target=True,
                game_model_type="solution",
            ),
            session,
            settings,
            None,
        )

    # Test the force assignment of a round
    par_a.current_round = None
    session.add(par_a)
    session.commit()
    response = current_round(
        ParticipantRoundRequest(
            participant_id=par_a.id,
            is_target=False,
            game_model_type="solution",
            targets_values=False,
            reveal_motivation=False,
            reveal_belief=False,
            allow_lying=False,
        ),
        session,
        settings,
        None,
    )
    rd = session.get(Round, response["round_id"])
    assert rd.persuader_id == par_a.id

    session.delete(rd)
    par_a.current_round = None
    session.add(par_a)
    session.commit()

    # and tests with more settings added
    response = current_round(
        ParticipantRoundRequest(
            participant_id=par_a.id,
            is_target=False,
            game_model_type="solution",
            allow_lying=True,
            targets_values=False,
            reveal_motivation=True,
            reveal_belief=True,
        ),
        session,
        settings,
        None,
    )
    rd = session.get(Round, response["round_id"])
    assert rd.persuader_id == par_a.id
    session.delete(rd)
    par_a.current_round = None
    session.add(par_a)
    session.commit()

    # TODO: change settings to enforce_persuader_target_roles = False

    # TODO: (later) have to implement test cases for an LLM

    # TODO: Test more parameter combinations for forced round assignment

    # TDOO: Test paired participants with more of the conditions

    # TODO: (later) flags for experiment three, getting rid of this error
    with pytest.raises(ValueError):
        current_round(
            ParticipantRoundRequest(
                participant_id=par_a.id,
                is_target=False,
                game_model_type="solution",
                targets_values=True,
                allow_lying=False,
                reveal_motivation=False,
                reveal_belief=False,
            ),
            session,
            settings,
            None,
        )


def a_persuader_b_round(session: Session):
    par_a = session.get(Participant, 1)
    par_a.role = "persuader"
    session.add(par_a)
    session.commit()

    response = participant_ready(ParticipantRequest(id=1), session)
    assert response is None

    response = participant_ready(ParticipantRequest(id=2), session)
    assert response is None

    response_1 = current_round(
        ParticipantRoundRequest(participant_id=1), session, TEST_SETTINGS, None
    )
    response_2 = current_round(
        ParticipantRoundRequest(participant_id=2), session, TEST_SETTINGS, None
    )

    assert response_1["round_id"] == response_2["round_id"]

    round_id = response_1["round_id"]

    return round_id


@pytest.mark.skipif(
    os.getenv("RUN_QUERY_TESTS", "False") != "True", reason="Skipping query test case"
)
def test_send_message(session: Session):
    with pytest.raises(HTTPException):
        response = send_message(
            MessageRequest(
                participant_id=1,
                round_id=0,
                message_content="test",
                thought_content=None,
            ),
            session,
            background_tasks=None,
            settings=TEST_SETTINGS,
        )

    test_participant_init(session)

    populate_tables(session, EXPERIMENT_CONDITIONS)

    round_id = a_persuader_b_round(session)

    with pytest.raises(HTTPException):
        response = send_message(
            MessageRequest(
                participant_id=2,
                round_id=round_id,
                message_content="test wrong turn",
                thought_content=None,
            ),
            session,
            background_tasks=None,
            settings=TEST_SETTINGS,
        )

    with pytest.raises(HTTPException):
        response = send_message(
            MessageRequest(
                participant_id=3,
                round_id=round_id,
                message_content="test no participant",
                thought_content=None,
            ),
            session,
            background_tasks=None,
            settings=TEST_SETTINGS,
        )

    with pytest.raises(HTTPException):
        response = send_message(
            MessageRequest(
                participant_id=1,
                round_id=10,
                message_content="test wrong round",
                thought_content=None,
            ),
            session,
            background_tasks=None,
            settings=TEST_SETTINGS,
        )

    rd = session.get(Round, round_id)
    gm = Game(**rd.game_data)
    assert gm.turn_limit == TEST_SETTINGS.turn_limit
    assert not rd.processing_persuader_response

    response = send_message(
        MessageRequest(
            participant_id=1,
            round_id=round_id,
            message_content="I hope you die",
            thought_content=None,
        ),
        session,
        background_tasks=None,
        settings=TEST_SETTINGS,
    )
    assert response is None

    rd = session.get(Round, round_id)

    msg = get_last_sent_message(session, is_target=False, round_id=round_id)
    assert msg.flagged
    assert rd.awaiting_persuader_response

    assert not rd.processing_persuader_response

    response = send_message(
        MessageRequest(
            participant_id=1,
            round_id=round_id,
            message_content="Hi how are you",
            thought_content=None,
        ),
        session,
        background_tasks=None,
        settings=TEST_SETTINGS,
    )
    assert response is None

    rd = session.get(Round, round_id)
    msg = get_last_sent_message(session, is_target=False, round_id=round_id)

    assert not msg.flagged
    assert not rd.awaiting_persuader_response
    assert not rd.processing_persuader_response

    # Pretend we are not done processing it
    rd.processing_persuader_response = True
    rd.awaiting_target_response = False

    session.add(rd)
    session.commit()

    with pytest.raises(HTTPException):
        response = send_message(
            MessageRequest(
                participant_id=2,
                round_id=round_id,
                message_content="I'm well, and you?",
            ),
            session,
            background_tasks=None,
            settings=TEST_SETTINGS,
        )
    rd.processing_persuader_response = False
    rd.awaiting_target_response = True
    session.add(rd)
    session.commit()

    response = send_message(
        MessageRequest(
            participant_id=2, round_id=round_id, message_content="I'm well, and you?"
        ),
        session,
        background_tasks=None,
        settings=TEST_SETTINGS,
    )
    assert response is None

    response = send_message(
        MessageRequest(
            participant_id=1, round_id=round_id, message_content="Nice weather, huh?"
        ),
        session,
        background_tasks=None,
        settings=TEST_SETTINGS,
    )
    assert response is None

    response = send_message(
        MessageRequest(
            participant_id=2,
            round_id=round_id,
            message_content="Maybe for you!",
            thought_content=None,
        ),
        session,
        background_tasks=None,
        settings=TEST_SETTINGS,
    )

    rd = session.get(Round, round_id)
    gm = Game(**rd.game_data)
    # test on real round that has been completed
    with pytest.raises(HTTPException):
        response = send_message(
            MessageRequest(
                participant_id=1,
                round_id=round_id,
                message_content="well the game is over",
                thought_content=None,
            ),
            session,
            background_tasks=None,
            settings=TEST_SETTINGS,
        )


def test_retreive_response(session: Session):
    # the participant, round do not exist
    with pytest.raises(HTTPException):
        response = retrieve_response(
            RoundRequest(participant_id=1, round_id=0),
            session,
            TEST_SETTINGS,
        )

    test_participant_init(session)

    populate_tables(session, EXPERIMENT_CONDITIONS)

    round_id = a_persuader_b_round(session)

    # the participant is not in the round
    with pytest.raises(HTTPException):
        response = retrieve_response(
            RoundRequest(participant_id=3, round_id=round_id),
            session,
            TEST_SETTINGS,
        )

    response = send_message(
        MessageRequest(
            participant_id=1,
            round_id=round_id,
            message_content="I hope you die",
            thought_content=None,
        ),
        session,
        background_tasks=None,
        settings=TEST_SETTINGS,
    )
    assert response is None

    # Previous message was a lie, time to send again
    response = retrieve_response(
        RoundRequest(participant_id=1, round_id=round_id),
        session,
        TEST_SETTINGS,
    )
    assert response["content"] == "I hope you die"
    assert (
        response["flagged_response"]
        == "The message content is inappropriate. Please be respectful."
    )

    # waiting response
    response = retrieve_response(
        RoundRequest(participant_id=2, round_id=round_id),
        session,
        TEST_SETTINGS,
    )
    assert response is None

    time.sleep(2)

    # The participant has waited too long and is kicked out of the round.
    with pytest.raises(HTTPException):
        response = retrieve_response(
            RoundRequest(participant_id=2, round_id=round_id),
            session,
            TEST_SETTINGS,
        )

    par_a = session.get(Participant, 1)
    par_b = session.get(Participant, 2)

    assert par_a.current_round is None
    assert par_b.current_round is None

    # Now add the two participants back to the round
    par_a.current_round = round_id
    par_b.current_round = round_id
    session.add(par_a)
    session.add(par_b)
    session.commit()

    ##
    response = send_message(
        MessageRequest(
            participant_id=1,
            round_id=round_id,
            message_content="1",
            thought_content=None,
        ),
        session,
        background_tasks=None,
        settings=TEST_SETTINGS,
    )
    assert response is None

    response = retrieve_response(
        RoundRequest(participant_id=2, round_id=round_id),
        session,
        TEST_SETTINGS,
    )
    assert response["turns_left"] == 2
    assert response["all_messages"] == [{"role": "persuader", "content": "1"}]

    ##
    response = send_message(
        MessageRequest(
            participant_id=2,
            round_id=round_id,
            message_content="2",
            thought_content=None,
        ),
        session,
        background_tasks=None,
        settings=TEST_SETTINGS,
    )
    assert response is None

    response = retrieve_response(
        RoundRequest(participant_id=1, round_id=round_id),
        session,
        TEST_SETTINGS,
    )
    assert response["turns_left"] == 1
    assert response["all_messages"] == [
        {"role": "persuader", "content": "1"},
        {"role": "target", "content": "2"},
    ]

    ##
    response = send_message(
        MessageRequest(
            participant_id=1,
            round_id=round_id,
            message_content="3",
            thought_content=None,
        ),
        session,
        background_tasks=None,
        settings=TEST_SETTINGS,
    )
    assert response is None

    response = retrieve_response(
        RoundRequest(participant_id=2, round_id=round_id),
        session,
        TEST_SETTINGS,
    )
    assert response["turns_left"] == 1
    assert response["all_messages"] == [
        {"role": "persuader", "content": "1"},
        {"role": "target", "content": "2"},
        {"role": "persuader", "content": "3"},
    ]

    ##
    response = send_message(
        MessageRequest(
            participant_id=2,
            round_id=round_id,
            message_content="4",
            thought_content=None,
        ),
        session,
        background_tasks=None,
        settings=TEST_SETTINGS,
    )
    assert response is None

    response = retrieve_response(
        RoundRequest(participant_id=1, round_id=round_id),
        session,
        TEST_SETTINGS,
    )
    assert response["turns_left"] == 0
    assert response["all_messages"] == [
        {"role": "persuader", "content": "1"},
        {"role": "target", "content": "2"},
        {"role": "persuader", "content": "3"},
        {"role": "target", "content": "4"},
    ]


def test_make_choice_round_result(session: Session):
    # Non existant round, participant
    with pytest.raises(HTTPException):
        response = make_choice(
            ChoiceRequest(participant_id=1, round_id=0, choice="A"),
            session,
            TEST_SETTINGS,
        )
    test_retreive_response(session)
    round_id = 1

    # Game not over
    with pytest.raises(HTTPException):
        response = round_result(
            RoundRequest(participant_id=1, round_id=round_id), session
        )

    # Real round, invalid choice
    with pytest.raises(HTTPException):
        response = make_choice(
            ChoiceRequest(participant_id=1, round_id=round_id, choice="Z"),
            session,
            TEST_SETTINGS,
        )

    # Persuaders cannot make a choice
    with pytest.raises(HTTPException):

        response = make_choice(
            ChoiceRequest(participant_id=1, round_id=round_id, choice="A"),
            session,
            TEST_SETTINGS,
        )

    rd = session.get(Round, round_id)
    game = Game(**rd.game_data)
    game_without_last = copy.deepcopy(game)
    del game_without_last.messages[-1]
    rd.game_data = game_without_last.model_dump()
    session.add(rd)
    session.commit()

    # Target has not yet finished its turns
    with pytest.raises(HTTPException):
        settings = TEST_SETTINGS
        settings.dev_environment = False
        response = make_choice(
            ChoiceRequest(participant_id=2, round_id=round_id, choice="A"),
            session,
            settings,
        )

    rd.game_data = game.model_dump()
    session.add(rd)
    session.commit()

    response = make_choice(
        ChoiceRequest(participant_id=2, round_id=round_id, choice="A"),
        session,
        TEST_SETTINGS,
    )

    # Choice already made
    with pytest.raises(HTTPException):
        response = make_choice(
            ChoiceRequest(participant_id=1, round_id=0, choice="A"),
            session,
            TEST_SETTINGS,
        )

    rd = session.get(Round, round_id)
    assert rd.persuader_choice
    assert rd.target_choice == "A"

    r1 = round_result(RoundRequest(participant_id=1, round_id=round_id), session)
    r2 = round_result(RoundRequest(participant_id=2, round_id=round_id), session)
    assert r1 == r2
    assert r1["target_choice"] == rd.target_choice
    assert r1["persuader_choice"] == rd.persuader_choice


def test_participant_rounds_api(session: Session):
    # Non existant participant
    with pytest.raises(HTTPException):
        response = participant_rounds(ParticipantRequest(id=1), session, TEST_SETTINGS)

    test_participant_rounds(session)

    response = participant_rounds(ParticipantRequest(id=1), session, TEST_SETTINGS)

    assert response["num_human_conversations"] == 1
    assert response["rounds_remaining"]
