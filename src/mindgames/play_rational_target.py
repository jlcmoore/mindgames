"""
Author: Jared Moore
Date: August, 2024

Handles the flow for a CLI-version of the game, attempting to persuade 
an ideal target.
"""

import argparse
import logging
import pprint
import random
import sys
import time

import pandas as pd

from .game import Game, TURN_LIMIT, BONUS
from .model import GameModel
from .utils import (
    DIFFICULTY_CONDITIONS,
    set_logger,
    get_payoff_file_path,
    DEFAULT_PROPOSALS,
    EX_COVER_STORY,
    EX_PERSUADER_ROLE,
    EX_TARGET_ROLE,
    EX_ATTRIBUTES,
    SCENARIOS_FILE,
    EX_SCENARIO,
    NON_MENTAL_SCENARIOS_FILE,
)

logger = logging.getLogger(__name__)

TITLE_SCREEN = r"""
 __  __  ____  _  _  ____      ___    __    __  __  ____  ___ 
(  \/  )(_  _)( \( )(  _ \    / __)  /__\  (  \/  )( ___)/ __)
 )    (  _)(_  )  (  )(_) )  ( (_-. /(__)\  )    (  )__) \__ \
(_/\/\_)(____)(_)\_)(____/    \___/(__)(__)(_/\/\_)(____)(___/
                                                          
"""

TAGLINE = "MindGames: win $ through persuasive dialogue."

H_BAR = f"\n{'-' * 50}"
H_BAR_SHORT = f"{'-' * 20}"


def ask_yes_no_question(question):
    """Defaults to no if the user enters nothing."""
    while True:
        try:
            response = input(question + " (y/n): ").strip().lower()
            # Clear the line after the user answers
            sys.stdout.write("\033[F\033[K")
            sys.stdout.flush()
            if response in ["y", "yes"]:
                return True
            if response in ["n", "no", ""]:
                return False
        except (KeyboardInterrupt, EOFError):
            return None
        print("Invalid input. Please enter 'y' or 'n'.")
        time.sleep(2)
        sys.stdout.write("\033[F\033[K")
        sys.stdout.flush()


def end_game():
    sys.exit()


def main():
    parser = argparse.ArgumentParser(description=TAGLINE)
    parser.add_argument(
        "--scenario-id",
        type=str,
        default=None,
        help="The id of the scenario (cover story) to use. If not passed chooses one randomly.",
    )
    parser.add_argument(
        "--payoff-file",
        type=str,
        default=None,
        help="The payoff file to load GameModels from",
    )
    parser.add_argument(
        "--non-solutions",
        default=False,
        action="store_true",
        help="Whether to use non solution payoffs",
    )
    parser.add_argument(
        "--difficulty",
        default=None,
        choices=DIFFICULTY_CONDITIONS,
        help="If `non-solutions`, how winnable, persuadable to make the games",
    )
    parser.add_argument(
        "--reveal-motivation",
        default=False,
        action="store_true",
        help="Whether to reveal the value function of the target to the persuader",
    )
    parser.add_argument(
        "--reveal-belief",
        default=False,
        action="store_true",
        help="Whether to reveal the hidden information of the target to the persuader",
    )
    parser.add_argument(
        "--add-hint",
        default=False,
        action="store_true",
        help="Whether to add a hint to the prompt (including the table)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="WARNING",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "--inline-lists",
        action="store_true",
        default=False,
        help="Display the info to the use as inline lists.",
    )
    parser.add_argument(
        "--allow-lying",
        action="store_true",
        default=False,
        help="Allow the persuader to lie to the target.",
    )
    parser.add_argument(
        "--discrete-game",
        action="store_true",
        default=False,
        help="Use the discrete action game with the target.",
    )
    parser.add_argument(
        "--non-mental",
        action="store_true",
        default=False,
        help="Use the non mental scenarios",
    )

    args = parser.parse_args()

    # Convert the log level string to a logging level
    set_logger(args.log)

    if args.non_mental:
        scenarios = pd.read_json(NON_MENTAL_SCENARIOS_FILE, lines=True)
    else:
        scenarios = pd.read_json(SCENARIOS_FILE, lines=True)

    if args.scenario_id and not scenarios["id"].isin([args.scenario_id]).any():
        raise ValueError(
            f"Invalid scenario ID. Valid IDs include: {scenarios['id'].tolist()}"
        )

    scenario = EX_SCENARIO

    if args.scenario_id:
        scenario = scenarios[scenarios["id"] == args.scenario_id].iloc[0].to_dict()
        del scenario["id"]
        if len(scenario["attributes"]) > 3:
            # TODO: What should we do with scenarios with more than three attributes?
            # Currently randomizing.
            scenario["attributes"] = random.sample(scenario["attributes"], 3)

    if args.payoff_file and args.non_solutions or args.difficulty:
        raise ValueError(
            f"Either specify a payoff file OR non-solutions AND difficulty"
        )

    if not args.payoff_file:
        args.payoff_file = get_payoff_file_path(
            non_solutions=args.non_solutions, difficulty=args.difficulty
        )

    df = pd.read_json(args.payoff_file, orient="records", lines=True)
    models = [GameModel(**row.to_dict()) for _, row in df.iterrows()]

    print(TITLE_SCREEN)
    print()
    print(f"Welcome to {TAGLINE}")
    print()

    model = random.choice(models)

    game = Game(
        model=model,
        reveal_belief=args.reveal_belief,
        reveal_motivation=args.reveal_motivation,
        allow_lying=args.allow_lying,
        display_lists=not args.inline_lists,
        is_ideal_target=True,
        turn_limit=TURN_LIMIT,
        add_hint=args.add_hint,
        discrete_game=args.discrete_game,
        proposals=DEFAULT_PROPOSALS,
        **scenario,  # expands the kwargs in scenario
    )

    print(game.instructions(is_target=False, include_chain_of_thought=False))

    # TODO: later we can have a loop here for multiple games
    print(H_BAR)
    print()

    persuader_choice = game.choose_proposal(is_target=False)

    persuader_ends_game = False
    # The game is over when there are no turns left or the persuader chooses to end the game
    # NB: In the production games, all turns must be taken.

    init_persuader = game.prompt(is_target=False)
    init_target = game.prompt(is_target=True)

    logger.info(game.model)

    print(init_persuader)

    logger.debug("(begin target prompt):")
    logger.debug(init_target)
    logger.debug("(end target prompt)")

    turns_left = TURN_LIMIT

    print()
    print(H_BAR_SHORT)
    while turns_left > 0 and not persuader_ends_game:
        # The persuader goes first
        try:
            message_content = input("You  > ").strip()
        except (KeyboardInterrupt, EOFError):
            end_game()

        try:
            response_content = game.process_persuader_message(message_content)
        except ValueError as e:
            print(str(e))
            continue

        # TODO: here is where we would connect another participant's responses
        print("Them > " + response_content)

        persuader_ends_game = ask_yes_no_question(
            f"    _Are you ready to let the {game.other_player_name} choose?_"
        )
        if persuader_ends_game is None:
            end_game()

        # NB: and then the target plays (not shown)
        turns_left -= 1

    print(H_BAR_SHORT)
    print()

    target_choice = game.choose_proposal(is_target=True)

    game_result = game.model_dump()
    logger.debug(pprint.pformat(game_result, indent=4))

    print(f"The target chooses proposal {target_choice}.")
    print(f"Your top choice was proposal {persuader_choice}.")
    print()

    if target_choice == persuader_choice:
        print(f"You persuaded the target! You get an extra ${BONUS}.")
    else:
        print("You lose.")

    print(H_BAR)
