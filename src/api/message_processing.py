"""
Author: Jared Moore
Date: September, 2024

Utilities for processing messages.
"""

import logging
import time

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from mindgames.conditions import Roles
from mindgames.game import Game, CHARACTERS_PER_RESPONSE
from mindgames.utils import limit_to_n_characters

from modelendpoints.query import Endpoint, get_option, find_answer
from modelendpoints.utils import split_thought_from_response

from .sql_model import Round, SentMessage


# Say 20 seconds for ten words
ASSUMED_PARTICIPANT_WORDS_PER_SECOND = 0.5
ASSUMED_RETREIVE_RESPONSE_OTHER_PARTICIPANT_ROUND_TRIP_SECONDS = 0.5

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def receieve_response_and_choose_proposal(
    is_target: bool, role: Roles, game: Game
) -> (str | None, str | None):
    """
    A helper function to get the response from either the Rational Target or an LLM.

    Parameters:
    - is_target (bool): whether *this player* the llm or rational target is the target
    - role (Roles): the condition of the game
    - game (Game): the game state which will be modified

    Returns:
    - str: The response, or None if no more turns
    - str: The chosen proposal, or None if not yet the end
    """
    logger.info("in receieve_response_and_choose_proposal")
    response_content = None
    chosen_proposal = None
    assert not role.is_paired_human()

    if role.is_rational_target():
        assert is_target
        response_content = game.last_message(is_target=True)
    elif game.turns_left(is_target=is_target) > 0:
        # Here we get the response from the relevant LLM
        model = role.llm_target if is_target else role.llm_persuader
        assert model
        messages = game.messages_for_llms(is_target=is_target)

        logger.info(f"Calling {model} for a response to {messages}")

        # Call the relevant LLM, passing in the entire game history
        with Endpoint(model=model, temperature=1) as endpoint:
            # TODO: Need to have a fail mechanism
            response = endpoint(messages=messages)
            response_attempt = response.get("text")

            if not response_attempt:
                raise ValueError("No response from model.")

            # TODO: Also collect the `thought_content` from the user. Not sure how.
            thought_content, response_content = split_thought_from_response(
                response_attempt
            )
            response_content = limit_to_n_characters(
                response_content, CHARACTERS_PER_RESPONSE
            )

            flagged_response = game.moderate_and_check_lies(
                message_content=response_content,
                thought_content=thought_content,
                is_target=is_target,
            )
            if flagged_response:
                raise ValueError("Moderated out.")

        logger.info(f"Got a response from {model}")

    # If the other player is not a participant and they have no turns left,
    # we need to ask that player (the rational target or the LLM) for their choice
    if not game.turns_left(is_target=is_target):
        if not is_target or role.is_rational_target():
            # Either the player is the persuader, in which case they do not choose,
            # or they are the rational target
            chosen_proposal = game.choose_proposal(is_target=is_target)
        else:
            # Ask the LLM-target

            # NB: it is okay that this is a background task (simply a different
            # thread pool) as opposed to a true new process because we are not
            # doing much computation, just networking
            assert is_target
            prompt = game.choose_proposal_prompt(is_target=True)
            messages = [{"role": "user", "content": prompt}]
            with Endpoint(model=model, temperature=1) as endpoint:
                response = endpoint(messages=messages)
                thought_content, chosen_proposal_content = split_thought_from_response(
                    response["text"]
                )

                chosen_option = get_option(chosen_proposal_content)
                chosen_proposal = find_answer(chosen_option, game.model.proposals)
                if not chosen_proposal:
                    raise ValueError("No chosen proposal from model.")

    return response_content, chosen_proposal


def process_sent_message(
    message_content: str | None,
    is_target: bool,
    game: Game,
    role: Roles,
    round_id: int,
    engine: Engine,
    dev_environment: bool = False,
    thought_content: str | None = None,
):
    """
    Moderates the `sent_message`, then if not `is_target`, tallies whether it reveals certain info,
    appeals, or lies.

    Queries for a response from an LLM or the rational target, if applicable. Asks for their
    chosen proposals if the game is over.

    Updates relevant database state.

    Parameters:
    - message_content (str): the message sent by the participant (always human).
        Pass `None` when starting the game with an LLM persuader
    - thought_content (str): the thought accompanying the message sent by the participant
        Can be None.
    - is_target (str): whehter or not the participant who sent this message is the target
    - game (Game): the state of the game
    - role (Roles): the kind of game this is
    - round_id (int): the id of the Round

    Should take place on a different thread.
    """
    logger.info(
        "Processing message for round %s, message content: %s",
        round_id,
        message_content,
    )
    start_time = time.time()
    unmodified_game = Game(**game.model_dump())

    # Blocking
    sent_message = None
    if message_content:
        logger.info("Moderating message: %s", message_content)
        flagged_response = game.moderate_and_check_lies(
            message_content=message_content,
            is_target=is_target,
            thought_content=thought_content,
        )
        logger.info("Flagged response: %s", flagged_response)

        sent_message = SentMessage(
            is_target=is_target,
            content=message_content,
            flagged=flagged_response is not None,
            round_id=round_id,
        )
        if sent_message.flagged:
            sent_message.flagged_response = flagged_response
            logger.info("Flagged message: %s", sent_message)

    received_message = None
    other_player_chosen_proposal = None

    # If we need to calculate a response now
    logger.info("Sent message: %s", sent_message)
    logger.info("Role: %s", role)
    if (not sent_message or not sent_message.flagged) and not role.is_paired_human():
        # Blocking
        response_content, other_player_chosen_proposal = (
            receieve_response_and_choose_proposal(not is_target, role, game)
        )
        if response_content:
            received_message = SentMessage(
                is_target=not is_target,
                content=response_content,
                flagged=False,
                round_id=round_id,
            )
            logger.info("Received message: %s", received_message)

    logger.info("Updating database for round %s", round_id)

    with Session(engine) as session:
        statement = select(Round).filter_by(id=round_id).with_for_update()
        game_round = session.exec(statement).one()

        # Unnecessary, probably, but making sure that the game hasn't changed
        # since this thread has spun off
        current_game = Game(**game_round.game_data)

        if len(unmodified_game.messages) != len(current_game.messages):
            logger.error("Game has been modified since message was sent.")

        game_round.game_data = game.model_dump()

        if not sent_message or not sent_message.flagged:
            # NB: we only set these 'awaiting' flags once we are sure the message is moderated
            # and not a lie

            # It is the participant's turn again if they have already received a response
            # and they have turns left to play
            is_participants_turn_again = (received_message is not None) and (
                game.turns_left(is_target=is_target) != 0
            )
            is_others_turn_again = (
                game.turns_left(is_target=not is_target) != 0
            ) and not is_participants_turn_again

            if is_target:  # the sent_message is from the target
                game_round.awaiting_target_response = is_participants_turn_again
                game_round.awaiting_persuader_response = is_others_turn_again
            else:  # the sent_message is not from the target
                game_round.awaiting_persuader_response = is_participants_turn_again
                game_round.awaiting_target_response = is_others_turn_again

            if received_message:
                if not dev_environment:

                    # Either we have an LLM response or from rational target

                    # Don't update the database util enough time to have written the message has elapsed
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    response_length = len(
                        received_message.content.split(" ")  # pylint: disable=no-member
                    )
                    response_time = max(
                        (
                            (response_length * ASSUMED_PARTICIPANT_WORDS_PER_SECOND)
                            - ASSUMED_RETREIVE_RESPONSE_OTHER_PARTICIPANT_ROUND_TRIP_SECONDS
                            - elapsed_time
                        ),
                        0,
                    )
                    logger.info("Sleeping for %s seconds", response_time)
                    time.sleep(response_time)

                session.add(received_message)

            # Either we are a persuader and have no turns
            # Or the other player is a persuader and has no turns
            if game.turns_left(is_target=False) == 0:
                game_round.persuader_choice = game.choose_proposal(is_target=False)

            # The other player has chosen and they cannot be a persuader (we are the persuader)
            if other_player_chosen_proposal and not is_target:
                game_round.target_choice = other_player_chosen_proposal

        # Updates the relevant state
        if is_target:
            game_round.processing_target_response = False
        else:
            game_round.processing_persuader_response = False

        if sent_message:
            # There might not be a sent message on the first message
            logger.info("Adding sent message for round %s", round_id)

            session.add(sent_message)

        session.add(game_round)
        session.commit()

    logger.info("Finished processing message for round %s", round_id)
