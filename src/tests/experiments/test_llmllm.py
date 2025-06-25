"""
Author: Jared Moore
Date: October, 2024

Tests for the llmllm experiment.
"""

import logging
import os

import pytest

from experiments.llmllm import run_batch_llms

from mindgames.conditions import Roles, Condition
from mindgames.game import Game, DISPLAY_LISTS
from mindgames.known_models import SOLUTION
from mindgames.utils import EX_SCENARIO

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@pytest.mark.skipif(
    os.getenv("RUN_QUERY_TESTS", "False") != "True"
    and os.getenv("RUN_QUERY_TESTS", "False") != "True",
    reason="Skipping query and batch test case",
)
async def test_simple_batch():

    turn_limit = 1

    ## No errors should occur on an empty run.
    await run_batch_llms({}, turn_limit)

    ## TODO: loop this next code block, changing the model to interact with the rational target...
    ## or just do them all at once?

    all_roles = [
        # The first three are rational target conditions
        Roles(
            llm_persuader="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", llm_target=None
        ),
        Roles(llm_persuader="gpt-4o-mini", llm_target=None),
        Roles(llm_persuader="claude-3-5-sonnet-20240620", llm_target=None),
        # And then some pair-ups. We want each model to play the target
        # to test if they answer the multiple choice question ok.
        Roles(
            llm_persuader="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            llm_target="claude-3-5-sonnet-20240620",
        ),
        Roles(
            llm_persuader="claude-3-5-sonnet-20240620",
            llm_target="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        ),
        Roles(
            llm_persuader="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            llm_target="gpt-4o-mini",
        ),
    ]
    next_model_to_games: dict[str, list[tuple[Condition, Game]]] = {}

    for roles in all_roles:
        condition = Condition(
            roles=roles,
            allow_lying=False,
            targets_values=False,
            reveal_motivation=False,
            reveal_belief=False,
        )

        game = Game(
            model=SOLUTION,
            reveal_belief=condition.reveal_belief,
            reveal_motivation=condition.reveal_motivation,
            allow_lying=condition.allow_lying,
            display_lists=DISPLAY_LISTS,
            turn_limit=turn_limit,
            is_ideal_target=condition.is_rational_target(),
            **EX_SCENARIO,
        )

        if roles.llm_persuader not in next_model_to_games:
            next_model_to_games[roles.llm_persuader] = []

        next_model_to_games[roles.llm_persuader].append((condition, game))

    await run_batch_llms(next_model_to_games, turn_limit)

    for _, games in next_model_to_games.items():
        for condition, game in games:
            logger.info(condition)
            logger.debug(game)
            assert len(game.messages) == 2
            assert game.game_over()
            assert game.persuader_choice
