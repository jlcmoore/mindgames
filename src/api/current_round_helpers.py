"""
Author: Jared Moore
Date: October, 2024

Helpers for assigning participants to rounds.
"""

import itertools
import logging

from fastapi import HTTPException
from sqlmodel import (
    Session,
    select,
)

from mindgames.conditions import Condition
from mindgames.model import GameModel
from mindgames.utils import rank_attributes, Scenario

from .sql_model import Participant, Model, Round
from .sql_queries import get_participant_rounds


logger = logging.getLogger(__name__)

### Helpers to assign participants to particular conditions


def assign_binary_condition_to_pair(
    first: Participant, second: Participant | None, field: str, condition: Condition
):
    """Assigns the `field` of `first` to the value of `field` of `second` or uses
    the value of `field` in `condition. Modifies the object `first`
    """
    chosen = getattr(second, field, None)
    if chosen is None:
        # Randomly assign the participants
        chosen = getattr(condition, field)
    setattr(first, field, chosen)


def choose_participant_conditions(
    first: Participant,
    second: Participant | None,
    chosen_condition: Condition,
    enforce_persuader_target_roles: bool,
):
    """
    Chooses roles for the participants if not already set. Modifies the objects.
    A helper for `choose_round`.
    """
    if not first.conditions_assigned():
        # First assign the roles
        if enforce_persuader_target_roles:
            if second:
                first.role = "target" if second.role == "persuader" else "persuader"
                if not second.role:
                    assert first.role == "persuader"
                    second.role = "target"
            else:
                assert not chosen_condition.is_paired_human()
                first.role = (
                    "target" if chosen_condition.roles.human_target else "persuader"
                )
        else:
            first.role = "either"
            if second and not second.role:
                second.role = "either"

    assign_binary_condition_to_pair(first, second, "allow_lying", chosen_condition)
    assign_binary_condition_to_pair(first, second, "targets_values", chosen_condition)
    assign_binary_condition_to_pair(
        first, second, "reveal_motivation", chosen_condition
    )
    assign_binary_condition_to_pair(first, second, "reveal_belief", chosen_condition)

    assert first.conditions_assigned()

    if second and not second.conditions_assigned():
        choose_participant_conditions(
            first=second,
            second=first,
            chosen_condition=chosen_condition,
            enforce_persuader_target_roles=enforce_persuader_target_roles,
        )
        assert second.conditions_assigned()
        assert (
            first.role == "either"
            and second.role == "either"
            or first.role != second.role
        )
        assert first.targets_values == second.targets_values
        assert first.reveal_belief == second.reveal_belief
        assert first.reveal_motivation == second.reveal_motivation
        assert first.allow_lying == second.allow_lying

    new_chosen_condition = Condition(
        roles=chosen_condition.roles,
        targets_values=first.targets_values,
        reveal_belief=first.reveal_belief,
        reveal_motivation=first.reveal_motivation,
        allow_lying=first.allow_lying,
    )
    assert chosen_condition == new_chosen_condition


def assign_participants(
    first: Participant, second: Participant | None
) -> (int | None, int | None):
    """
    Assigns the passed participants to roles for a round. For use in a round.
    Returns the persuader and target ids
    """
    persuader_id = None
    target_id = None

    assert first.role
    assert not second or second.role

    if first.role == "persuader":
        persuader_id = first.id
    elif first.role == "target":
        target_id = first.id

    if second:
        if second.role == "persuader":
            persuader_id = second.id
        elif second.role == "target":
            target_id = second.id

    if persuader_id is None and first.role == "either":
        persuader_id = first.id
    elif not target_id and first.role == "either":
        target_id = first.id

    if second:
        if persuader_id is None and second.role == "either":
            persuader_id = second.id
        elif not target_id and second.role == "either":
            target_id = second.id
    if first and second:
        assert target_id is not None and persuader_id is not None
    else:
        assert target_id is not None or persuader_id is not None
    return persuader_id, target_id


def filter_models_for_target(
    models: list[GameModel], participant: Participant, scenario: Scenario
) -> GameModel | None:
    """
    Returns only those `models` where the inferred value function of the `participant`
    (from their survey response) on `scenario` matches that of the target.

    Modifies the state of `scenario.attributes` to indicate the chosen attributes

    Parameters:
    - models: list[GameModel]
    - participant: Participant
    - scenario: Scenario
    Returns:
    - the chosen GameModel, with a target value function (coefficients) matching
       the participant's
    """

    attributes: list[str] = scenario.attributes
    survey_responses: list[dict[str, any]] | None = participant.initial_survey_responses
    if not survey_responses:
        return None

    ranking: dict[str, int] = rank_attributes(survey_responses)

    assert len(set(attributes) - set(ranking.keys())) == 0

    # If the scenario has more than three attributes, retry the calculation with
    # each subset of three
    attributes_subsets = itertools.combinations(attributes, 3)

    for subset in attributes_subsets:
        value_function: dict[str, int] = {
            attribute: ranking[attribute] for attribute in subset
        }
        # NB: We just compare the values here because by default the keys of
        # the models (the attributes) are x, y, z; those are only changed later
        # once we actually make a game based on this game model

        possible_models = list(
            filter(
                lambda m, vf=value_function: tuple(m.target_coefficients.values())
                == tuple(vf.values()),
                models,
            )
        )
        if possible_models:
            scenario.attributes = attributes
            return possible_models[0]
    return None


def get_all_rounds(
    first: Participant, second: Participant | None, session: Session
) -> list[Round]:
    """Returns all of the rounds `first` and `second` have played in."""
    rounds = get_participant_rounds(first, session)
    all_rounds = rounds
    if second:
        other_rounds = get_participant_rounds(second, session)
        all_rounds.extend(other_rounds)
    return all_rounds


def choose_model_for_participants(
    all_rounds: list[Round],
    chosen_game_model_type: str,
    session: Session,
    scenario: Scenario | None = None,
    target: Participant | None = None,
) -> Model:
    """
    For the given `first` and `second` participants chooses a game model of type
    `chosen_game_model_type` using `session`.

    If `target_id` is passed, also filters the game models to only allow
    those which the target could have agreed with.  Modifies the state of `scenario`
    if passed and choosing the payoff based on the target's value function.

    Returns that chosen model.
    """

    # Chosse a payoff model neither has seen
    # TODO: do we want some kind of randomization here?
    # at the moment this will just pull the first entry from the table
    game_model_ids = [rd.game_model_id for rd in all_rounds]
    statement = (
        select(Model)
        .where(~Model.id.in_(game_model_ids))  # pylint: disable=no-member
        .where(Model.game_model_type == chosen_game_model_type)
    )
    possible_models = session.exec(statement).all()

    # If experiment three in which we use the real values of the target
    # only choose payoffs that accord with the values of that target
    if target:
        assert scenario
        game_models_to_model = {
            GameModel(**model.data): model for model in possible_models
        }
        chosen_game_model = filter_models_for_target(
            game_models_to_model.keys(), target, scenario
        )
        if chosen_game_model is not None:
            return game_models_to_model[chosen_game_model]

        logger.error(
            "Could not find a matching game model for the values of participant."
        )

    if not possible_models:
        message = "No model with desired attributes found"
        logger.error(message)
        raise HTTPException(
            status_code=400,
            detail=message,
        )

    return possible_models[0]
