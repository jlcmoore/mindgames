"""
Author: Jared Moore
Date: October, 2024

Used to run the llm-llm experiments.
"""

import argparse
import asyncio
import logging
import os

from typing import Any, Collection

import pprint

from pydantic import validate_call
from tqdm.asyncio import tqdm

from mindgames.conditions import Condition
from mindgames.game import Game, CHARACTERS_PER_RESPONSE
from mindgames.utils import (
    limit_to_n_characters,
)

from modelendpoints.query import Endpoint, get_option, find_answer
from modelendpoints.utils import (
    split_thought_from_response,
    split_reasoning_from_response,
)

from .utils import (
    load_config,
    output_conditions_and_games,
    dir_to_condition,
    load_file,
    populate_games_for_conditions,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_CONFIG = "config/llmllm.yaml"


@validate_call
async def collect_llm_response(game: Game, response_text: str):
    """
    For the given llm response, `response_text`, modifies `game` according to
    whether it is a target or persuader message or a target chosen proposal.
    """
    is_target = game.target_plays_next()
    if not response_text:
        assert not is_target
        return
    reasoning_content, response_less_reasoning = split_reasoning_from_response(
        response_text
    )
    thought_content, response_content = split_thought_from_response(
        response_less_reasoning
    )
    if not game.discrete_game:
        response_content = limit_to_n_characters(
            response_content, CHARACTERS_PER_RESPONSE
        )

    if game.neither_turns_left():
        chosen_option = get_option(response_content)
        chosen_proposal = find_answer(chosen_option, game.model.proposals)
        if not chosen_proposal:
            logger.error(f"No chosen proposal from model in {response_content}.")
        elif chosen_proposal in game.model.proposals:
            game.set_target_choice(chosen_proposal)
        else:
            logger.error(f"Invalid choice, {chosen_proposal}.")
    else:
        assert game.turns_left(is_target=is_target)
        l1 = len(game.messages)
        if not is_target:
            # Blocking, query
            try:
                game.process_persuader_message(
                    response_content,
                    thought_content=thought_content,
                    reasoning_content=reasoning_content,
                )
                assert len(game.messages) > l1
            except ValueError as e:
                logging.error(e)
                # TODO: the persuader lied here. We need to prompt again
        else:
            game.process_target_message(
                response_content,
                thought_content=thought_content,
                reasoning_content=reasoning_content,
            )
            assert len(game.messages) > l1


@validate_call
async def query_llm(
    model: str, games: list[Game], model_args: dict[str, Any] | None = None
):
    """
    Batch queries `model` for the next messages in each game in `games`.
    Modifies the game state of each.
    """
    for game in games:
        if game.game_over():
            raise ValueError("Passed a finished game.")

    ## Generate prompts, not blocking
    keys_to_messages = {}

    for i, game in enumerate(games):
        i = str(i)
        messages = None  # In case it is the persuader and there is nothing to prompt

        # NB: making the assumption that we do not need to moderate the content in
        # llm-llm games

        if game.neither_turns_left():
            # the llm must choose as the target
            prompt = game.choose_proposal_prompt(is_target=True)
            messages = [{"role": "user", "content": prompt}]
            # max_tokens = 8 # We might be able to let the max_tokens vary...
        else:
            is_target = game.target_plays_next()
            assert game.turns_left(is_target=is_target)

            ## Persuader or target prompt; they have messages left to send
            messages = game.messages_for_llms(is_target=is_target)
            # max_tokens = 256
        if messages:
            keys_to_messages[i] = messages
        else:
            logger.info(f"No messages for game {i}")

    ## Make calls. Blocking.
    # TODO: we later need to force pass `source` to make vllm calls
    these_args = {"model": model, "batch_prompts": True, "async_function": True}
    if model_args:
        these_args.update(model_args)

    with Endpoint(**these_args) as endpoint:
        keys_to_responses = await endpoint(
            keys_to_messages=keys_to_messages,
        )

    ## Collect calls, storing the results in the relevant Games
    # TODO: this next block (which calls the rational target)
    # should be converted to a *batch* call as well (not just async).
    coroutines = []
    for i, game in enumerate(games):
        i = str(i)
        request = keys_to_messages.get(i)
        response = keys_to_responses.get(i)
        if not request:
            continue
        if not response:
            logger.error(f"No response from model, {model}.")
            continue

        response_text = response.get("text")
        logger.debug(f"{model}: {pprint.pformat(request)}")
        logger.debug(f"{model}: {response_text}")

        if not response_text:
            logger.error(
                f"No response text from model, {model} on. Response: {response}."
            )
            continue
        coroutines.append(collect_llm_response(game, response_text))

    # Await all tasks to complete, each game will be modified.
    # TODO: should have some error flag in each game and just check that.
    await asyncio.gather(*coroutines)


def value_lists_count(keys_to_value_lists: dict[Any, Collection[Any]]) -> int:
    """
    Returns the sum of length of the values of the passed list.
    """
    return sum(len(a_list) for a_list in keys_to_value_lists.values())


@validate_call
async def run_batch_llms(
    next_model_to_games: dict[str, list[tuple[Condition, Game]]],
    turn_limit: int,
    model_to_args: dict[str, dict[str, Any]] | None = None,
):
    """
    Query the relevant endpoints.

    The premise here is that we gather up all of the games in which a given
    endpoint plays (usually just one LLM) (either as target or persuader).
    We either batch or parallelize the calls to that endpoint and then get the updates.
    Then we go update the state of all the games.
    Repeat for each LLM until there are no more messages to send in any game.

    Parameters
    next_model_to_games: maps from the model to play next to their games
    turn_limit: how many turns each game should have
    model_to_args: Additional args for various model endpoints

    Modifies the state of each Game.
    """

    # Loop until there are no more games to play,
    max_dialogue_turns = 4 * turn_limit
    # Should just be `turn limit`, no? but flagged messages...

    game_pbar = tqdm(
        total=value_lists_count(next_model_to_games), desc="Game progress", unit="game"
    )

    turn_pbar = tqdm(total=max_dialogue_turns, desc="Dialogue turns", unit="turn")

    try:
        while value_lists_count(next_model_to_games):
            # TODO: should verify that `next_model_to_games` is in fact all
            # different objects otherwise we are in for a world of hurt.

            # First call each of the endpoints simultaneously

            # TODO: if we have two vllm endpoints we almost certainly cannot run them
            # concurrently. Schedule them synchronously.
            coroutines = []
            for model, conditions_and_games in next_model_to_games.items():
                if not conditions_and_games:
                    continue

                games = [game for _, game in conditions_and_games]
                model_args = None
                if model_to_args and model in model_to_args:
                    model_args = model_to_args[model]
                coroutines.append(query_llm(model, games, model_args))

            await asyncio.gather(*coroutines)

            # Then gather their responses
            # Create a new data structure, moving the working progress over
            new_next_model_to_games: dict[str, list[tuple[Condition, Game]]] = {}

            for model, conditions_and_games in next_model_to_games.items():
                for condition, game in conditions_and_games:
                    if game.game_over():
                        # A game has been completed
                        game_pbar.update(1)
                        continue

                    # TODO: later we should here exicse the games that have errored
                    # E.g. game.bad_message, or some such

                    # The persuader plays if the target just played or if it is the
                    # rational target, but not if the game is over.
                    # The target plays otherwise
                    if (
                        model == condition.roles.llm_target
                        or condition.roles.llm_target is None
                    ) and not game.neither_turns_left():
                        next_model = condition.roles.llm_persuader
                    else:
                        next_model = condition.roles.llm_target
                    assert next_model
                    logger.info(next_model)

                    if next_model not in new_next_model_to_games:
                        new_next_model_to_games[next_model] = []

                    # Add all of this model's games to the relevant next model
                    if not game.target_choice:
                        new_next_model_to_games[next_model].append((condition, game))

            next_model_to_games = new_next_model_to_games

            turn_pbar.update(1)

            if turn_pbar.n > max_dialogue_turns:
                # TODO: This could be more elegant.
                logger.error("We have run for too many dialogue turns.")
                break
    finally:
        turn_pbar.close()
        game_pbar.close()

    # TODO: log anything relevant in `next_model_to_games`, e.g. if we had to end early


def main():
    """
    Loads in the relevant config file.
    In batches, queries each of the LLMs for their responses.
    Save the results.
    """
    parser = argparse.ArgumentParser(description="Run LLM-LLM experiments")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--previous-condition",
        type=str,
        help=(
            "Previous run to build off of (overrides config). "
            "Only works for one condition."
        ),
    )

    args = parser.parse_args()
    config, conditions = load_config(args.config)

    # Make all of the relevant games. We get our results from this data structure
    condition_to_games: dict[Condition, list[Game]] = {}

    if args.previous_condition:
        parts = args.previous_condition.split(os.sep)
        dir_name = parts[1]
        condition = dir_to_condition(dir_name)

        if condition.roles.human_persuader or condition.roles.human_target:
            raise ValueError("Invalid condition passed. No human roles allowed.")

        results = load_file(args.previous_condition, allow_nested=False)
        condition_to_games = {condition: results}
    else:
        condition_to_games = populate_games_for_conditions(
            conditions=conditions,
            turn_limit=config.turn_limit,
            scenarios=config.scenarios,
            non_mental_scenarios=config.non_mental_scenarios,
            num_unique_payoffs_per_round_condition=config.num_unique_payoffs_per_round_condition,
            difficulty_to_payoff=config.difficulty_to_payoff,
        )

    # This is the 'working' data structure and stores a map from
    # the next player (llm) to the games in which they play next
    next_model_to_games: dict[str, list[tuple[Condition, Game]]] = {}

    for condition, games in condition_to_games.items():
        for game in games:
            if game.game_over():
                continue

            if (condition.roles.llm_target is None) and not game.neither_turns_left():
                next_model = condition.roles.llm_persuader
            else:
                next_model = condition.roles.llm_target

            if next_model not in next_model_to_games:
                next_model_to_games[next_model] = []

            # Add all of this model's games to the relevant next model
            if not game.target_choice:
                next_model_to_games[next_model].append((condition, game))

    print(f"Running all {value_lists_count(next_model_to_games)} games.")

    # Making sure we did not put the rational target in
    assert None not in next_model_to_games
    # Run the experiment.
    try:
        asyncio.run(
            run_batch_llms(next_model_to_games, config.turn_limit, config.model_to_args)
        )
    finally:
        print("Outputting games.")

        # Save the results.
        output_conditions_and_games(condition_to_games)


if __name__ == "__main__":
    main()
