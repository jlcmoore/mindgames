"""
Author: Jared Moore
Date: February, 2025

Used to run the random baseline experiment.
"""

import copy
import random

from mindgames.game import TURN_LIMIT, Game, DEFAULT_GENERIC_RESPONSE
from mindgames.conditions import Condition, Roles
from mindgames.utils import DEFAULT_ATTRIBUTES

from .utils import (
    populate_games_for_conditions,
    model_type_to_models,
    output_conditions_and_games,
)

N_SAMPLES = 10
N_DRAWS = 8


def main(draws: int = N_DRAWS, samples: int = N_SAMPLES):
    """
    Runs the random baseline on the solution games choosing `draws` pieces of information
    to reveal with replacement and repeating all games `samples` number of times.
    """
    difficulty_to_payoff = {}
    difficulty_to_payoff["solution"] = model_type_to_models("solution")

    conditions = [Condition(roles=Roles(llm_persuader=f"random-baseline-{draws}"))]

    condition_to_games = populate_games_for_conditions(
        conditions=conditions,
        turn_limit=TURN_LIMIT,
        scenarios=[{"attributes": DEFAULT_ATTRIBUTES}],
        num_unique_payoffs_per_round_condition=100,
        difficulty_to_payoff=difficulty_to_payoff,
    )

    condition_to_games_more_samples = {}

    for condition, games in condition_to_games.items():
        condition_to_games_more_samples[condition] = []
        for game in games:
            for i in range(samples):
                condition_to_games_more_samples[condition].append(
                    Game(**game.model_dump())
                )

    condition_to_games = condition_to_games_more_samples

    for condition, games in condition_to_games.items():
        for game in games:
            for i in range(TURN_LIMIT):
                if i < draws:
                    p = random.choice(game.model.proposals)
                    a = random.choice(game.model.attributes)
                    utility = game.model.utilities[p][a]
                    print()
                    game.reveal_info(p, a, utility)

                    statement = game._proposal_str(p, attributes=[a])

                    game.all_disclosures.append({p: {a: utility}})

                    game.messages.extend(
                        [
                            {"role": "persuader", "content": statement},
                            {"role": "target", "content": statement},
                        ]
                    )
                else:
                    game.messages.extend(
                        [
                            {"role": "persuader", "content": "test"},
                            {"role": "target", "content": DEFAULT_GENERIC_RESPONSE},
                        ]
                    )

                game.chain_of_thought.extend(
                    [
                        {"role": "persuader", "content": None},
                        {"role": "target", "content": None},
                    ]
                )

                if game.neither_turns_left():
                    game.choose_proposal(is_target=True)
            print(game.target_choice)
            assert game.game_over()

    output_conditions_and_games(condition_to_games)


if __name__ == "__main__":
    main()
