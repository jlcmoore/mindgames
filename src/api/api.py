"""
Author: Jared Moore
Date: September, 2024

An API and the logic to run participant experiments, possibly with just one
participant and either an LLM or a simple program or between two participants.
Stores intermediate results in a sql database.
"""

from contextlib import asynccontextmanager
from collections import Counter
import copy
from datetime import datetime, timezone
from functools import lru_cache
import random
import logging

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
import markdown
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import (
    SQLModel,
    Session,
    create_engine,
    select,
)
from typing_extensions import Annotated
from pydantic import BaseModel

import mindgames
from mindgames.game import Game
from mindgames.model import GameModel
from mindgames.utils import (
    DEFAULT_PROPOSALS,
    EXPERIMENT_CONDITIONS,
    SURVEY,
    SurveyResponse,
)


from mindgames.conditions import Condition, Roles

from .utils import ServerSettings, MAX_WAITING_TILL_END_EXPERIMENT_MULTIPLIER

from .message_processing import (
    process_sent_message,
)

from .sql_model import (
    ExternalUser,
    Model,
    Scenario,
    Round,
    Participant,
    SQLITE_URL,
    CONNECT_ARGS,
)
from .sql_queries import (
    populate_tables,
    get_participant_rounds,
    get_paired_participant,
    get_round_types_remaining,
    choose_condition,
    get_last_sent_message,
)

from .current_round_helpers import (
    assign_participants,
    choose_participant_conditions,
    choose_model_for_participants,
    get_all_rounds,
)

WAITING_TOO_LONG_MSG = "Participant {id} has waited too long."

##### App startup

logger = logging.getLogger(__name__)


@lru_cache
def get_settings():
    """Loads in the settings from disc and caches them"""
    return ServerSettings()


global_engine = create_engine(SQLITE_URL, echo=False, connect_args=CONNECT_ARGS)


def get_session():
    """A FastAPI dependency. Yields the global session on call."""
    with Session(global_engine) as session:
        yield session


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Defines the start up and clean up necessary when running the server through FastAPI.
    NB: Everything before `yield` occurs before the server starts up and after `yield`
    occurs on shutdown.
    """
    settings = get_settings()

    SQLModel.metadata.create_all(global_engine)
    with Session(global_engine) as session:
        models = session.exec(select(Model)).all()
        # Only populate the tables with scenarios and models if they are not already there
        if not models:
            scenarios = session.exec(select(Scenario)).all()
            assert not scenarios
            populate_tables(
                session=session,
                difficulty_conditions=EXPERIMENT_CONDITIONS,
                non_mental=settings.non_mental,
            )

    yield
    # NB: After this yield we clean up any resources
    # If dev_environment, overwrite the database

    if settings.dev_environment:
        SQLModel.metadata.drop_all(global_engine)


if get_settings().dev_environment:
    app = FastAPI(lifespan=lifespan)
else:
    # Don't expose the docs in production
    app = FastAPI(lifespan=lifespan, openapi_url=None, docs_url=None, redoc_url=None)


# suppress sqlalchemy logging


###############
##### API
###############


## Shared error messasges


def get_round_error(session: Session, round_id: int) -> Round:
    """A helper to get the round by round_id and raise errors"""
    game_round = session.get(Round, round_id)
    if not game_round:
        message = "Round not found"
        logger.error(message)
        raise HTTPException(status_code=400, detail=message)
    return game_round


def get_round_error_completed(session: Session, round_id: int) -> Round:
    """A helper to get the round by and raise an error if the game is over"""
    game_round = get_round_error(session, round_id)
    if game_round.persuader_choice is not None and game_round.target_choice is not None:
        message = "Round is completed"
        logger.error(message)
        raise HTTPException(status_code=400, detail=message)
    return game_round


def participant_round_error(participant: Participant, game_round: Round) -> None:
    """Raises an error if the participant is not in the round"""
    if (
        participant.id not in set([game_round.target_id, game_round.persuader_id])
        or participant.current_round != game_round.id
    ):
        message = "The participant is not in this round"
        logger.error("The participant is not in this round")
        raise HTTPException(status_code=400, detail=message)


## HTTP Request Classes


class MessageRequest(BaseModel):
    round_id: int
    participant_id: int
    message_content: str
    thought_content: str | None = None


class ChoiceRequest(BaseModel):
    round_id: int
    participant_id: int
    choice: str
    initial_choice: bool = False


class FeedbackRequest(BaseModel):
    participant_id: int
    feedback: str


class ParticipantInitRequest(BaseModel):
    id: str
    # TODO: see comment in `sql_model` about participant surveys
    # TODO: does this need to be a field?
    survey_responses: list[SurveyResponse] | None = None


class ParticipantSurveyRequest(BaseModel):
    id: int  # NB: this takes our internal id -- not the prolific ID
    survey_responses: list[SurveyResponse] | None = None


class ParticipantRoundRequest(BaseModel):
    participant_id: int

    # This request is only availble if the server is running on development.
    # They force the particpant to be assinged
    # to the following conditions, if possible.
    is_target: bool | None = None
    game_model_type: str | None = None
    llm_persuader: str | None = None
    llm_target: str | None = None
    scenario_id: str | None = None
    game_model_id: int | None = None
    targets_values: bool | None = None
    allow_lying: bool | None = None
    reveal_belief: bool | None = None
    reveal_motivation: bool | None = None

    def force_round_settings(self):
        return (
            self.is_target is not None
            or self.game_model_type is not None
            or self.llm_persuader is not None
            or self.llm_target is not None
            or self.scenario_id is not None
            or self.game_model_id is not None
            or self.targets_values
            or self.allow_lying is not None
            or self.reveal_belief is not None
            or self.reveal_motivation is not None
        )


class ParticipantRequest(BaseModel):
    id: int


class RoundRequest(BaseModel):
    round_id: int
    participant_id: int | None = None


def get_participant(session: Session, participant_id: int) -> Participant:
    """Tries to get the participant. Throws an error if they don't exist"""
    participant = session.get(Participant, participant_id)

    if participant is None:
        message = f"Participant {participant_id} does not exist."
        logger.error(message)
        raise HTTPException(status_code=400, detail=message)

    return participant


def participant_round_validate(
    request: ParticipantRoundRequest,
    session: Session,
    settings: ServerSettings,
):
    """Raises various validation errors for ParticipantRoundRequests"""
    if not settings.dev_environment and request.force_round_settings():
        raise HTTPException(
            status_code=400,
            detail="You may only change the kind of round to play in development.",
        )
    if request.force_round_settings():

        if request.scenario_id and not session.get(Scenario, request.scenario_id):
            raise HTTPException(
                status_code=400,
                detail="Requested session does not exist",
            )
        if (
            request.game_model_type
            and request.game_model_type not in EXPERIMENT_CONDITIONS
        ):
            raise HTTPException(
                status_code=400,
                detail="Requested game model type not exist",
            )
        if request.game_model_id:
            game_model = session.get(Model, request.game_model_id)
            if not game_model:
                raise HTTPException(
                    status_code=400,
                    detail="Requested game model does not exist",
                )
            if (
                request.game_model_type
                and game_model.game_model_type != request.game_model_type
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Passed game_model_id must be of the requested game_model_type",
                )
        if (
            request.is_target is None
            or request.game_model_type is None
            or request.targets_values is None
            or request.reveal_motivation is None
            or request.reveal_belief is None
            or request.allow_lying is None
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "You must pass is_target, game_model_type, targets_values, "
                    + "reveal_motivation, reveal_belief, and allow_lying"
                ),
            )
        if request.is_target and not request.llm_persuader:
            raise HTTPException(
                status_code=400,
                detail="You must give an llm persuader when the participant is the target",
            )


## API proper


@app.post("/participant_instructions/")
def participant_instructions(
    request: ParticipantRequest,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[ServerSettings, Depends(get_settings)],
):
    """
    Returns the high-level instructions for the participant.
    """
    logger.info(request)
    participant = get_participant(session, request.id)

    # TODO: Need to eliminate the optional 'either' role as showing the instructions
    # first does not support this.
    if not participant.role or participant.role == "either":
        message = "Instructions for unset role not implemented!"
        logger.error(message)
        raise ValueError(message)
    is_target = participant.role == "target"

    instructions = Game._instructions(
        turn_limit=settings.turn_limit,
        add_hint=settings.add_hint,
        include_chain_of_thought=False,
        is_target=is_target,
        include_title=False,
        include_character_limit=False,  # Could change this later
        targets_values=participant.targets_values,
        non_mental=settings.non_mental,
    )
    return markdown.markdown(instructions, extensions=["md_in_html"])


def to_participant_survey(
    survey_responses: list[dict[str, any]] | None
) -> None | list[SurveyResponse]:
    """
    Processes the participant survey responses from JSON into SurveyResponses
    """
    responses = None
    if survey_responses:
        responses = [SurveyResponse(**r) for r in survey_responses]
    return responses


def from_participant_survey(
    survey_responses: list[SurveyResponse] | None,
) -> None | list[dict[str, any]]:
    """
    Processes the participant survey responses from JSON into SurveyResponses
    """
    responses = None
    if survey_responses:
        responses = [r.model_dump() for r in survey_responses]
    return responses


@app.post("/post_game_survey/")
def post_game_survey(
    request: ParticipantSurveyRequest,
    session: Annotated[Session, Depends(get_session)],
) -> None:
    """
    Processes the final survey responses from the participant.
    Throws an error if the participant exists or if they have already supplied responses.
    """
    logger.info(request)

    participant = get_participant(session, request.id)

    if participant.final_survey_responses:
        message = f"Final survey for {participant.id} already supplied."
        logger.error(message)
        raise HTTPException(status_code=400, detail=message)

    final_responses = from_participant_survey(request.survey_responses)
    participant.final_survey_responses = final_responses

    if participant.role == "target":
        # If this is the target, add their final response to all the rounds
        # (this is effectively their decision so we need to track it)
        rds = get_participant_rounds(participant, session)
        for rd in rds:
            game = Game(**rd.game_data)
            game.final_survey_responses = final_responses
            rd.game_data = game.model_dump()

            session.add(rd)

    session.add(participant)
    session.commit()


@app.post("/participant_init/")
def participant_init(
    request: ParticipantInitRequest,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[ServerSettings, Depends(get_settings)],
) -> None:
    """
    Check if this ID already exists. if not, add in the user.
    Regardless, returns the participant id (session id, effectively) of the given user.
    """
    logger.info(request)

    user = session.exec(
        select(ExternalUser).where(ExternalUser.external_id == request.id)
    ).first()

    if user is None:
        # External User has already been added.
        user = ExternalUser(external_id=request.id)
        session.add(user)
        session.commit()
        session.refresh(user)

        participant = Participant(
            id=user.id,
            game_model_types_remaining=settings.round_conditions,
            initial_survey_responses=from_participant_survey(request.survey_responses),
        )
        session.add(participant)
        session.commit()
        session.refresh(participant)

    return {"participant_id": user.id}


@app.post("/participant_ready/")
def participant_ready(
    request: ParticipantRequest,
    session: Annotated[Session, Depends(get_session)],
) -> None:
    """
    Add the participant, `request.participant_id` to the waiting room,
    updating the time they entered on each call.
    NB: do not call repeatedly -- this timestamp is used
    to indicate if the participant is in a current round.
    Raises an exception if the participant does not exist.
    Raises an exception if the participant is already in a round.
    """
    logger.info(request)

    participant = get_participant(session, request.id)
    if participant.current_round:
        game_round = session.get(Round, participant.current_round)
        game = Game(**game_round.game_data)
        if not game.game_over():
            message = f"Participant {request.id} is in a round."
            logger.error(message)
            raise HTTPException(status_code=400, detail=message)

    participant.entered_waiting_room = datetime.now(timezone.utc)
    participant.current_round = None
    session.add(participant)
    session.commit()
    session.refresh(participant)


@app.post("/current_round/")
def current_round(
    request: ParticipantRoundRequest,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[ServerSettings, Depends(get_settings)],
    background_tasks: BackgroundTasks,
):
    """
    Returns the current round the participant is in as a dict with keys:
    - round_id (int): the current round id
    - is_target (bool): whether the participant is the target
    - prompt (str): the initial prompt to show the participant

    Returns an empty dict if the participant cannot yet be assigned.

    If the participant is not in a round attempts to assign them to one,
    possibly pairing them with another paticipant.

    Raises an exception if:

    - the participant does not exist
    - the participant has no more rounds to play (cf. settings.round_conditions)
    - there are no more rounds to fill in general (cf. settings.condition_num_rounds)
        (This shouldn't happen.)
    """
    logger.info(request)

    try:
        # NB: We are locking on the participant here and will until the transaction is completed.
        participant = session.exec(
            select(Participant)
            .where(Participant.id == request.participant_id)
            .with_for_update(skip_locked=True)
        ).first()

        if participant is None:
            message = f"Participant {request.participant_id} does not exist."
            logger.error(message)
            raise HTTPException(status_code=400, detail=message)

        participant_round_validate(request, session, settings)
        # Is the participant already in a round?
        if participant.current_round:
            logger.debug(f"Participant {participant.id} is already in a round.")
            game_round = session.get(Round, participant.current_round)
        else:
            # These vary just over each round
            chosen_model = None
            chosen_scenario = None
            chosen_game_model_type = None
            chosen_participant = None
            if request.force_round_settings():
                target_id = request.participant_id if request.is_target else None

                persuader_id = request.participant_id if not request.is_target else None

                # No human-human rounds are allowed here.
                chosen_condition = Condition(
                    roles=Roles(
                        llm_persuader=request.llm_persuader,
                        llm_target=request.llm_target,
                        human_persuader=persuader_id is not None,
                        human_target=target_id is not None,
                    ),
                    targets_values=request.targets_values,
                    reveal_motivation=request.reveal_motivation,
                    reveal_belief=request.reveal_belief,
                    allow_lying=request.allow_lying,
                )
                chosen_game_model_type = request.game_model_type
                chosen_model = (
                    session.get(Model, request.game_model_id)
                    if request.game_model_id is not None
                    else None
                )
                chosen_scenario = (
                    session.get(Scenario, request.scenario_id)
                    if request.scenario_id is not None
                    else None
                )

                # Calling this to assign the Particpant to the condition values
                choose_participant_conditions(
                    participant,
                    None,
                    chosen_condition,
                    settings.enforce_persuader_target_roles,
                )
            else:

                # make sure the participant still has more games to play
                if Counter(participant.game_model_types_remaining).total() < 1:
                    # TODO: consider failing silently here?
                    message = (
                        f"Participant {participant.id} has no more rounds to play."
                    )
                    logger.error(message)
                    raise HTTPException(
                        status_code=400,
                        detail=message,
                    )

                # Tabulate which conditions we have yet to fill
                paired_rounds_remaining_set, non_paired_rounds_remaining_set = (
                    get_round_types_remaining(session, settings.condition_num_rounds)
                )
                if (
                    not paired_rounds_remaining_set
                    and not non_paired_rounds_remaining_set
                ):
                    # TODO: consider failing silently here?
                    message = "There are no rounds to play in general"
                    logger.error(message)
                    raise HTTPException(status_code=400, detail=message)

                # Only look for another participant if we still need to fill that condition.
                if len(paired_rounds_remaining_set) > 0 and (
                    not participant.round_condition
                    or participant.round_condition
                    and Condition(**participant.round_condition).is_paired_human()
                ):
                    ## Get an existing participant, if it exists
                    chosen_participant = get_paired_participant(
                        participant,
                        session,
                        settings,
                    )
                    if participant.round_condition and participant.round_condition:
                        assert (
                            participant.round_condition == participant.round_condition
                        )

                if (
                    participant.waiting_time()
                    > MAX_WAITING_TILL_END_EXPERIMENT_MULTIPLIER
                    * settings.waiting_room_timeout
                ):
                    # The participant has been waiting too long regardless of the condition.
                    # Send a signal that the experiment should end for them.
                    participant.waited_too_long = True
                    session.add(participant)
                    session.commit()

                    logger.error(WAITING_TOO_LONG_MSG.format(id=participant.id))
                    raise HTTPException(
                        status_code=400,
                        detail=WAITING_TOO_LONG_MSG.format(id=participant.id),
                    )

                chosen_condition = choose_condition(
                    participant=participant,
                    chosen_participant=chosen_participant,
                    paired_rounds_remaining_set=paired_rounds_remaining_set,
                    non_paired_rounds_remaining_set=non_paired_rounds_remaining_set,
                    overassign_non_paired_conditions=settings.overassign_non_paired_conditions,
                    enforce_player_round_condition=settings.enforce_player_round_condition,
                    condition_num_rounds=settings.condition_num_rounds,
                    waiting_room_timeout=settings.waiting_room_timeout,
                )

                waiting_response = {
                    "waiting_time": participant.waiting_time().total_seconds()
                }
                if not chosen_condition:
                    # Tell the user to wait longer
                    return waiting_response
                ## Choose condition assignments for these participants, if necessary
                choose_participant_conditions(
                    participant,
                    chosen_participant,
                    chosen_condition,
                    settings.enforce_persuader_target_roles,
                )

                ## Choose the type of payoff model, whether a solution or some kind of
                # non-solution control
                remaining_game_model_types = Counter(
                    participant.game_model_types_remaining
                )
                if chosen_participant:
                    remaining_game_model_types &= Counter(
                        chosen_participant.game_model_types_remaining
                    )

                assert len(remaining_game_model_types) >= 1
                chosen_game_model_type = random.choice(
                    list(remaining_game_model_types.keys())
                )

                persuader_id, target_id = assign_participants(
                    participant, chosen_participant
                )

            all_rounds = get_all_rounds(participant, chosen_participant, session)

            # NB: We have to choose the cover story first as it dictates which
            # value functions in the model to choose  in experiment three
            if not chosen_scenario:
                # Choose a scenario neither has seen
                scenario_ids = [rd.scenario_id for rd in all_rounds]
                statement = select(Scenario).where(
                    ~Scenario.id.in_(scenario_ids)  # pylint: disable=no-member
                )
                possible_scenarios = session.exec(statement).all()

                if not possible_scenarios:
                    message = "No scenarios that both participants have not seen."
                    logger.error(message)
                    return waiting_response

                chosen_scenario = mindgames.utils.Scenario(
                    **random.choice(possible_scenarios).model_dump()
                )

            is_target = participant.id == target_id

            if not chosen_model:
                chosen_model = choose_model_for_participants(
                    all_rounds,
                    chosen_game_model_type,
                    session,
                    chosen_scenario,
                    target=participant if is_target else chosen_participant,
                )

            assert chosen_model

            # We only want three attributes per scenario. Later we can change the
            # payoff matrix depending on the scenario attributes as well as based
            # on the survey questions.
            if len(chosen_scenario.attributes) > 3:
                chosen_scenario.attributes = random.sample(
                    chosen_scenario.attributes, 3
                )

            chosen_proposals = copy.deepcopy(DEFAULT_PROPOSALS)
            if not settings.dev_environment:
                # If not in a dev environment, randomize the order of the proposals.
                chosen_proposals = random.sample(chosen_proposals, 3)

            # TODO: need to change this logic later if we want LLMs to be able to supply
            # their own values
            target_survey = None
            if chosen_condition.roles.human_target:
                if is_target:
                    target_survey = to_participant_survey(
                        participant.initial_survey_responses
                    )
                else:
                    assert chosen_participant
                    target_survey = to_participant_survey(
                        chosen_participant.initial_survey_responses
                    )
            assert chosen_condition.targets_values == participant.targets_values

            game = Game(
                add_hint=settings.add_hint,
                model=GameModel(**chosen_model.data),
                reveal_belief=chosen_condition.reveal_belief,
                reveal_motivation=chosen_condition.reveal_motivation,
                targets_values=chosen_condition.targets_values,
                proposals=chosen_proposals,
                allow_lying=chosen_condition.allow_lying,
                display_lists=not settings.inline_lists,
                proposals_as_html=settings.proposals_as_html,
                turn_limit=settings.turn_limit,
                include_character_limit=False,
                is_ideal_target=chosen_condition.is_rational_target(),
                initial_survey_responses=target_survey,
                non_mental=settings.non_mental,
                **chosen_scenario.model_dump(exclude={"id"}),
            )

            logger.debug("New round with the following game:")
            logger.debug(game.model)
            perfect_msg = game.perfect_message()
            logger.debug(f"Perfect message: {perfect_msg}")

            game_round = Round(
                game_data=game.model_dump(),
                persuader_id=persuader_id,
                target_id=target_id,
                llm_persuader=chosen_condition.roles.llm_persuader,
                llm_target=chosen_condition.roles.llm_target,
                game_model_type=chosen_game_model_type,
                scenario_id=chosen_scenario.id,
                game_model_id=chosen_model.id,
                targets_values=chosen_condition.targets_values,
                reveal_belief=chosen_condition.reveal_belief,
                reveal_motivation=chosen_condition.reveal_motivation,
                allow_lying=chosen_condition.allow_lying,
                target_perfect_info_choice=game.target_perfect_info_choice,
            )
            session.add(game_round)
            session.commit()
            session.refresh(game_round)

            if chosen_condition.roles.llm_persuader:
                # We need to query for a starting message by the LLM
                assert is_target
                game_round.processing_persuader_response = True

                # NB: While this function uses the engine it does not use the same session
                process_sent_args = {
                    "message_content": None,
                    "is_target": is_target,
                    "role": chosen_condition.roles,
                    "game": game,
                    "round_id": game_round.id,
                    "engine": session.get_bind(),
                    "dev_environment": settings.dev_environment,
                }
                if background_tasks and settings.background_tasks:
                    background_tasks.add_task(process_sent_message, **process_sent_args)
                else:
                    process_sent_message(**process_sent_args)

            chosen_model_cntr = Counter({chosen_game_model_type: 1})

            ## This round is ready to go. Take the participants out of the waiting room.
            # NB: Have to wait until after refresh as the ID initializes the round id
            if chosen_participant:
                chosen_participant.entered_waiting_room = None
                chosen_participant.current_round = game_round.id
                chosen_participant.game_model_types_remaining = (
                    Counter(chosen_participant.game_model_types_remaining)
                    - chosen_model_cntr
                )
                session.add(chosen_participant)

            participant.entered_waiting_room = None
            participant.current_round = game_round.id
            participant.game_model_types_remaining = (
                Counter(participant.game_model_types_remaining) - chosen_model_cntr
            )
            session.add(participant)

            session.commit()

        is_target = request.participant_id == game_round.target_id
        logger.info(
            f"round {game_round.id} particpant {participant.id} is target {is_target}"
        )
        game = Game(**game_round.game_data)
        return {
            "round_id": game_round.id,
            "is_target": is_target,
            "prompt": markdown.markdown(
                game.prompt(is_target=is_target, reveal=False),
                extensions=["md_in_html"],
            ),
            "game_data": game.model_dump(),
        }
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database error in current_round: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error occurred") from e
    except HTTPException:
        session.rollback()
        raise


@app.post("/send_message/")
def send_message(
    request: MessageRequest,
    session: Annotated[Session, Depends(get_session)],
    background_tasks: BackgroundTasks,
    settings: Annotated[ServerSettings, Depends(get_settings)],
):
    """
    Sends `request.message_content` from `request.participant_id` in round `request.round_id`.
    Returns nothing on success.

    Raises an exception if:

    - The round does not exist
    - The round is completed
    - The participant is not in the round
    - the game is over and the participant should choose
    - It is not the participant's turn
    - It is the participant's turn, but we have not finished processing their last message
    """
    logger.info(request)

    game_round = get_round_error_completed(session, request.round_id)
    participant = get_participant(session, request.participant_id)
    participant_round_error(participant, game_round)

    neither_turns_left_error(game_round)

    is_target = game_round.target_id == request.participant_id

    if (not game_round.awaiting_target_response and is_target) or (
        not game_round.awaiting_persuader_response and not is_target
    ):
        message = (
            f"Received a message in round {game_round.id} we were not waiting for."
        )
        logger.error(message)
        raise HTTPException(
            status_code=400,
            detail=message,
        )

    if (game_round.processing_target_response and is_target) or (
        game_round.processing_persuader_response and not is_target
    ):
        message = (
            f"Still processing the participant's last message in round {game_round.id}."
        )
        logger.error(message)
        raise HTTPException(
            status_code=400,
            detail=message,
        )

    role = game_round.get_roles()

    game = Game(**game_round.game_data)

    if is_target:
        game_round.processing_target_response = True
    else:
        game_round.processing_persuader_response = True

    session.add(game_round)
    session.commit()

    # NB: While this function uses the engine it does not use the same session
    process_sent_args = {
        "message_content": request.message_content,
        "thought_content": request.thought_content,
        "is_target": is_target,
        "role": role,
        "game": game,
        "round_id": request.round_id,
        "engine": session.get_bind(),
        "dev_environment": settings.dev_environment,
    }
    if background_tasks and settings.background_tasks:
        logger.info("Adding task to background tasks.")
        background_tasks.add_task(process_sent_message, **process_sent_args)
    else:
        logger.info("Processing sent message.")
        process_sent_message(**process_sent_args)


def neither_turns_left_error(rd: Round):
    if Game(**rd.game_data).neither_turns_left():
        message = f"Round {rd.id}: The game is over. The participant should choose, if relevant."
        logger.error(message)
        raise HTTPException(
            status_code=400,
            detail=message,
        )


@app.post("/retrieve_response/")
def retrieve_response(
    request: RoundRequest,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[ServerSettings, Depends(get_settings)],
):
    """
    For the given round `request.round_id` and participant, `request.participant_id`...
    returns only one of the following:

    1. the last message the participant sent and an indication it has been flagged
        and a message about why it was flagged
    2. nothing, HTTP 200 = ok, if still waiting for the other player to respond
    3. all messages, including the last response from the other player than this one

    Raises an exception if:

    - the round does not exist
    - the participant is not in the round
    - the other participant has taken too long to respond and the round has ended

    Ok to poll.
    """
    logger.info(request)

    game_round = get_round_error(session, request.round_id)

    participant = get_participant(session, request.participant_id)
    participant_round_error(participant, game_round)

    is_target = game_round.target_id == request.participant_id

    last_sent = get_last_sent_message(session, is_target, game_round.id)

    prefix = f"Participant {request.participant_id} in round {game_round.id}"

    logger.debug("last_sent: %s", last_sent)

    # 1) Check if the message the participant just sent (if it exists)
    # was flagged, if so return flagged response
    if last_sent and last_sent.flagged:
        logger.info("1. Last message was flagged.")
        # TODO: respond with the right error code
        return {
            "content": last_sent.content,
            "flagged_response": last_sent.flagged_response,
        }

    # 2a) Tell them to end the round if the other player has taken too long
    last_update = (
        game_round.updated_at if game_round.updated_at else game_round.created_at
    )
    logger.debug("last_update: %s", last_update)
    if (
        datetime.now(timezone.utc) - last_update.replace(tzinfo=timezone.utc)
        > settings.participant_conversation_timeout
    ):
        message = f"{prefix} has waited too long. Staring a new game."
        logger.error(message)
        # This participant has waited too long.
        target = session.get(Participant, game_round.target_id)
        if target:
            target.current_round = None
            session.add(target)
        persuader = session.get(Participant, game_round.persuader_id)
        if persuader:
            persuader.current_round = None
            session.add(persuader)
        session.commit()
        raise HTTPException(
            status_code=400,
            detail=message,
        )

    # 2) Tell them to wait if the other player has yet to respond
    if (is_target and game_round.awaiting_persuader_response) or (
        not is_target and game_round.awaiting_target_response
    ):
        logger.info(f"{prefix} still waiting for the other player to respond.")
        return None

    # 3) Actually give them the last sent message by the other player
    # NB: if this is flagged awaiting response should be true
    last_received = get_last_sent_message(session, not is_target, game_round.id)

    if not last_received:
        logger.info(f"{prefix}. No message received yet, keep polling.")
        return None  # No message received yet, keep polling
        # TODO: surface error to the user

    game = Game(**game_round.game_data)
    message_content = game.last_message(not is_target)
    # assert message_content == last_received.content
    # TODO: handle this case

    logger.info(f"{prefix}. Message received. Returning all messages.")
    return {
        # Send all messages so that the client view can update fully
        "all_messages": game.messages,
        "all_thoughts": game.chain_of_thought,
        "turns_left": game.turns_left(is_target=is_target),
    }


@app.post("/make_choice/")
def make_choice(
    request: ChoiceRequest,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[ServerSettings, Depends(get_settings)],
):
    """
    Sets the participant's choice as request.choice.

    Raises an exception if:

    - The round does not exist
    - The round is completed
    - The participant is not in the round
    - The choice is not a valid proposal in the game
    - The persuader has made a choice
    """
    logger.info(request)

    game_round = get_round_error_completed(session, request.round_id)
    participant = get_participant(session, request.participant_id)
    participant_round_error(participant, game_round)

    is_target = game_round.persuader_id != request.participant_id

    if not is_target and request.choice:
        message = "Persuaders cannot make a choice, only call this function."
        logger.error(message)
        raise HTTPException(
            status_code=400,
            detail=message,
        )

    game = Game(**game_round.game_data)

    if (
        not request.initial_choice
        and game.turns_left(is_target=True)
        and not settings.dev_environment
    ):
        message = "Target cannot make a choice before all turns have been taken."
        logger.error(message)
        raise HTTPException(
            status_code=400,
            detail=message,
        )

    if request.choice not in game.model.proposals:
        message = f"Invalid choice, {request.choice}"
        logger.error(message)
        raise HTTPException(status_code=400, detail=message)

    assert game_round.target_id == request.participant_id

    if request.initial_choice:
        game.target_initial_choice = request.choice

    else:
        game_round.target_choice = request.choice
        game.target_choice = request.choice

        game_round.persuader_choice = game.choose_proposal(is_target=False)

    game_round.game_data = game.model_dump()

    session.add(game_round)
    session.commit()


@app.post("/request_target_decision/")
def request_target_decision(
    request: RoundRequest,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[ServerSettings, Depends(get_settings)],
):
    """
    Ends the round, setting the target's choice using current game data.

    Raises an exception if:

    - the round does not exist
    - the round is completed
    - the participant is not the persuader
    - it is not a a dev environment
    """
    logger.info(request)

    if not settings.dev_environment:
        raise HTTPException(
            status_code=400, detail="Cannot be called in a live experiment."
        )

    game_round = get_round_error_completed(session, request.round_id)
    participant = get_participant(session, request.participant_id)
    participant_round_error(participant, game_round)

    if game_round.persuader_id != request.participant_id:
        message = "Only the persuader can end the round."
        logger.error(message)
        raise HTTPException(
            status_code=400,
            detail=message,
        )

    if game_round.target_id or game_round.llm_target:
        message = "Cannot be called on a human or llm target."
        logger.error(message)
        raise HTTPException(status_code=400, detail=message)

    game = Game(**game_round.game_data)

    game_round.target_choice = game.choose_proposal(is_target=True)
    game_round.persuader_choice = game.choose_proposal(is_target=False)

    session.commit()


@app.post("/round_result/")
def round_result(
    request: RoundRequest, session: Annotated[Session, Depends(get_session)]
):
    """
    Get information on the current round, `request.round_id`.

    Returns the

    Raises an exception if

    - the round does not exist
    - the round is not completed
    """
    logger.info(request)

    game_round = get_round_error(session, request.round_id)

    if game_round.persuader_choice is None or game_round.target_choice is None:
        message = f"Round {request.round_id} not completed"
        logger.error(message)
        raise HTTPException(status_code=400, detail=message)

    game = Game(**game_round.game_data)

    target_choice = game_round.target_choice
    persuader_choice = game_round.persuader_choice
    assert persuader_choice == game.choose_proposal(is_target=False)

    return {
        "target_choice": target_choice,
        "persuader_choice": persuader_choice,
        "perfect_target_choice": (
            game.target_perfect_info_choice if not game.targets_values else None
        ),
        "persuaded": target_choice == persuader_choice,
    }


@app.post("/send_feedback/")
def send_feedback(
    request: FeedbackRequest, session: Annotated[Session, Depends(get_session)]
):
    """
    Accepts the feedback from the participant.

    Raises an exception if

    - the participant does not exist
    - the participant is not done with their rounds
    - the participant has already answered the feedback
    """
    logger.info(request)

    participant = get_participant(session, request.participant_id)

    rounds_left = Counter(participant.game_model_types_remaining).total()
    if rounds_left and not participant.waited_too_long:
        message = "Participant has rounds left."
        logger.error(message)
        raise HTTPException(status_code=400, detail=message)

    if participant.feedback is not None:
        message = "Participant has given feedback."
        logger.error(message)
        raise HTTPException(status_code=400, detail=message)

    participant.feedback = request.feedback
    session.add(participant)
    session.commit()

    logger.info(
        "Feedback from participant %s stored: %s",
        request.participant_id,
        request.feedback,
    )


@app.post("/participant_rounds/")
def participant_rounds(
    request: ParticipantRequest,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[ServerSettings, Depends(get_settings)],
):
    """
    Get information on the rounds the participant has played in for use in disclosing
    to that participant in which conversations they interacted with other humans vs.
    LLMs.

    Returns a dict with keys

    - 'human_conversations': a list of bools indicating whether each of the conversaitons
        that the participant was were with another human
    - 'rounds_remaining': a bool indicating whether or not the participant has more rounds
        left

    Raises an exception if

    - the participant does not exist
    """
    logger.info(request)

    participant = get_participant(session, request.id)

    rounds = get_participant_rounds(participant, session)

    is_target = participant.role == "target"

    human_conversations = list(
        filter(
            lambda rd: (is_target and rd.persuader_id)
            or (not is_target and rd.target_id),
            rounds,
        )
    )

    rounds_remaining = Counter(participant.game_model_types_remaining).total()

    return {
        "num_human_conversations": len(human_conversations),
        "rounds_remaining": rounds_remaining,
        "rounds_completed": len(rounds),
        "total_rounds": len(rounds) + rounds_remaining,
        "completion_code": settings.completion_code,
    }


@app.get("/survey/")
def survey():
    """Returns the survey questions to ask."""
    return SURVEY


@app.get("/development_mode/")
def development_mode(
    settings: Annotated[ServerSettings, Depends(get_settings)],
):
    """Returns True if in a development environment and False otherwise."""
    return {"development_mode": settings.dev_environment}


@app.get("/conversation_timeout/")
def timeout(
    settings: Annotated[ServerSettings, Depends(get_settings)],
):
    """Returns the timeout"""
    return settings.participant_conversation_timeout.total_seconds()


@app.get("/chain_of_thought/")
def chain_of_thought(
    settings: Annotated[ServerSettings, Depends(get_settings)],
):
    """Returns True if we want to show a chain of thought and False otherwise."""
    return {"chain_of_thought": settings.chain_of_thought}
