"""
Author: Jared Moore
Date: October, 2024

Experiment utils.
"""

import copy
import datetime
import inspect
import itertools
import json
import logging
import os
from typing import Any, Dict

import pandas as pd
from pydantic import validate_call
import yaml

from mindgames.conditions import Roles, Condition
from mindgames.game import TURN_LIMIT, Game, DISPLAY_LISTS

from mindgames.model import GameModel
from mindgames.utils import (
    EXPERIMENT_CONDITIONS,
    DEFAULT_PROPOSALS,
    get_payoff_file_path,
    SCENARIOS_FILE,
    NON_MENTAL_SCENARIOS_FILE,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESULTS_DIR = "results"

### Configs


def populate_games_for_conditions(
    conditions: list[Condition],
    turn_limit: int,
    scenarios: list[dict[str, str]],
    non_mental_scenarios: list[dict[str, str]],
    num_unique_payoffs_per_round_condition: int,
    difficulty_to_payoff: dict[str, list[GameModel]],
) -> dict[Condition, list[Game]]:
    """Returns a dict of conditions mapping to lists of games"""
    condition_to_games: dict[Condition, list[Game]] = {}
    for condition in conditions:
        condition_to_games[condition] = []

        for _, payoffs in difficulty_to_payoff.items():
            unseen_payoff = payoffs[-1]

            for i in range(0, num_unique_payoffs_per_round_condition):
                payoff = payoffs[i]
                condition_scenarios = scenarios
                if condition.non_mental:
                    condition_scenarios = non_mental_scenarios
                for scenario in condition_scenarios:
                    game = Game(
                        model=payoff,
                        reveal_belief=condition.reveal_belief,
                        reveal_motivation=condition.reveal_motivation,
                        proposals=DEFAULT_PROPOSALS,
                        allow_lying=condition.allow_lying,
                        display_lists=DISPLAY_LISTS,
                        turn_limit=turn_limit,
                        is_ideal_target=condition.is_rational_target(),
                        add_hint=condition.add_hint,
                        discrete_game=condition.discrete_game,
                        **scenario,
                    )

                    if condition.perfect_game:
                        in_context_game = Game(
                            model=unseen_payoff,
                            proposals=DEFAULT_PROPOSALS,
                            attributes=scenario["attributes"],
                            **game.model_dump(exclude={"model"}),
                        )

                        in_context_game = in_context_game.perfect_game()
                        game.in_context_games.append(in_context_game)

                    condition_to_games[condition].append(game)
    return condition_to_games


def model_type_to_models(game_model_type: str) -> list[GameModel]:
    """Returns a list of GameModels matching the requested type"""
    non_solutions = game_model_type != "solution"
    difficulty = game_model_type if non_solutions else None
    payoff_file = get_payoff_file_path(
        non_solutions=non_solutions, difficulty=difficulty
    )
    df = pd.read_json(payoff_file, orient="records", lines=True)

    models = list(df.apply(lambda x: GameModel(**x.to_dict()), axis=1))
    return models


class LLMConfig:
    """Stores the condition information for a series of llm-llm rounds"""

    round_conditions: set[str]

    # Allow these two to be specified, but could be much higher,
    # e.g. full scenario coverage
    num_unique_payoffs_per_round_condition: int
    num_unique_scenarios_per_payoff: int
    # NB: total rounds = `Counter(round_conditions).total() *
    # (num_unique_payoffs_per_round_condition + num_unique_scenarios_per_payoff`)
    # E.g.: 6 * (3 + 3) = 54 unique rounds/conversations per condition

    temperature: int

    turn_limit: int

    difficulty_to_payoff: dict[str, list[GameModel]]
    scenarios: list[dict[str, str]]

    model_to_args: dict[str, dict[str, Any]] | None = None

    @validate_call
    def __init__(
        self,
        round_conditions: set[str] | None = None,
        turn_limit=TURN_LIMIT,
        num_unique_payoffs_per_round_condition: int = 3,
        num_unique_scenarios_per_payoff: int = 3,
        temperature: int = 0,
        model_to_args: dict[str, dict[str, Any]] | None = None,
    ):
        """
        Initializes variables to use in an llm-llm experiment. Loads in
        scenarios and game models to memory.
        """
        if round_conditions is None:
            round_conditions = set(copy.deepcopy(EXPERIMENT_CONDITIONS))
        if round_conditions - set(EXPERIMENT_CONDITIONS):
            raise ValueError(
                f"The only valid round conditions are {EXPERIMENT_CONDITIONS}."
            )
        if num_unique_payoffs_per_round_condition <= 0:
            raise ValueError("num_unique_payoffs_per_round_condition must be > 0")
        if num_unique_scenarios_per_payoff <= 0:
            raise ValueError("num_unique_scenarios_per_payoff must be > 0")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")

        if model_to_args:
            self.model_to_args = model_to_args

        self.round_conditions = round_conditions
        self.num_unique_payoffs_per_round_condition = (
            num_unique_payoffs_per_round_condition
        )
        self.num_unique_scenarios_per_payoff = num_unique_scenarios_per_payoff
        self.temperature = temperature
        self.turn_limit = turn_limit

        # TODO: make sure to use the exact same scenario and payoff pairs
        # throught out  the experiments
        # Currently doing so simply by reading in the first N from each file

        # Load in all of the scenarios and payoffs to memory
        self.difficulty_to_payoff = {}
        for game_model_type in self.round_conditions:

            models = model_type_to_models(game_model_type)
            # The plus one so we have one payoff the model never sees.
            if len(models) + 1 < self.num_unique_payoffs_per_round_condition:
                raise ValueError("Num unique payoffs too small")

            self.difficulty_to_payoff[game_model_type] = models

        # NB: Right now I'm making the assumption we want to see the
        # same scenario across different difficulty types for comparison
        # so am not mapping from GameModel to scenario

        ## Add entries for the different scenario types
        self.scenarios = []
        self.non_mental_scenarios = []

        nm_df = pd.read_json(NON_MENTAL_SCENARIOS_FILE, lines=True)
        df = pd.read_json(SCENARIOS_FILE, lines=True)

        if (
            len(df) < self.num_unique_scenarios_per_payoff
            or len(nm_df) < self.num_unique_scenarios_per_payoff
        ):
            logger.warn("Num unique scenarios too small")
        for i in range(self.num_unique_scenarios_per_payoff):
            if i < len(df):
                self.scenarios.append(df.iloc[i].to_dict())
            if i < len(nm_df):
                self.non_mental_scenarios.append(nm_df.iloc[i].to_dict())


def load_config(file_path: str) -> (LLMConfig, list[Condition]):
    """
    Loads the config file at `file_path`, returning the config arguments
    and a list of the conditions to run.
    """
    loaded_conditions: list[Condition] = []
    with open(file_path, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)
        conditions = copy.deepcopy(yaml_data["conditions"])
        del yaml_data["conditions"]
        config = LLMConfig(**yaml_data)
        for condition in conditions:
            roles = Roles(
                llm_target=condition["llm_target"],
                llm_persuader=condition["llm_persuader"],
            )
            del condition["llm_persuader"]
            del condition["llm_target"]

            loaded_conditions.append(
                Condition(roles=roles, **condition),
            )
    return (config, loaded_conditions)


### Input-output filename handling


def condition_to_dir(condition: Condition) -> str:
    """Converts the Condition to an encoded string as for a directory
    Removes the entries in the condition that are simply the default values.
    """
    # Get the roles as a dictionary string.
    roles_str = dict_to_string(condition.roles.model_dump(exclude_defaults=True))

    # Convert the Condition to a dict and remove roles since it is encoded separately.
    condition_dict = condition.model_dump(exclude_defaults=True, exclude={"roles"})

    # Convert the remaining key/value pairs to a string.
    other_conditions_str = dict_to_string(condition_dict)

    # Join the roles string and the condition string.
    dir_name = roles_str
    if other_conditions_str:
        dir_name += "&" + other_conditions_str
    return dir_name


def dir_to_condition(dir_name: str) -> Condition:
    """Converts a string in the format of an encoded directory into a Condition"""
    args_dict = string_to_dict(dir_name)

    roles_args = Roles.__fields__
    roles_dict = {}
    conditions_dict = {}
    for key, value in args_dict.items():
        if key in roles_args:
            roles_dict[key] = value
        else:
            conditions_dict[key] = value

    # Reconstruct Roles and Condition objects
    roles = Roles(**roles_dict)
    condition = Condition(roles=roles, **conditions_dict)
    return condition


@validate_call
def output_conditions_and_games(
    condition_to_games: dict[Condition, list[Game]],
    include_unfinished_surveys: bool = False,
    dry_run: bool = False,
):
    """
    Outputs each of the passed conditions in a different directory with the games
    stored as a jsonl file named by the current date.

    `include_unfinished_surveys` -- (only relevant for conditions with targets_values) If not set, will
    only return rounds in which the target has completed the pre and post survey.

    Args:
        condition_to_games (dict[Condition, list[Game]]): The dictionary to output.
    """

    conditon_to_results: dict[Condition, list[list[dict[str, Any]]]] = {}
    # Group the human conditions by ID
    for condition, games in condition_to_games.items():
        # Should already be ordered from oldedst to newest
        results = []
        for game in games:
            if (
                not include_unfinished_surveys
                and condition.targets_values
                and (
                    not game.initial_survey_responses or not game.final_survey_responses
                )
            ):
                print("\t\tIgnoring a game b/c it does not have a pre or post survey.")
            else:
                results.append(game.model_dump())

        non_id_cond = condition.as_non_id_role()
        if non_id_cond not in conditon_to_results:
            conditon_to_results[non_id_cond] = []
        conditon_to_results[non_id_cond].append(results)

    for condition, results in conditon_to_results.items():

        num_games = sum(len(result) for result in results)
        print(f"Condition, {condition}")
        print(f"\t has {len(results)} players' data")
        print(f"\t has {num_games} games total")
        dir_name = condition_to_dir(condition)

        now = datetime.datetime.now().date().isoformat()

        roles_dir = os.path.join(RESULTS_DIR, dir_name)

        if not os.path.exists(roles_dir):
            os.makedirs(roles_dir)

        output_file_path = os.path.join(roles_dir, f"{now}.jsonl")

        if dry_run:
            continue

        with open(output_file_path, mode="w", encoding="utf-8") as f:
            for list_of_dicts in results:
                json_str = json.dumps(list_of_dicts)
                f.write(json_str + "\n")

        print(f'Outputting to "{output_file_path}"')
        print()


def load_file(
    file_path: str, allow_nested: bool = True
) -> list[Game] | list[list[Game]]:
    """
    Loads the games in the file at `file_path`
    If allow_nested will allow list[list[Game]]
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]
        if isinstance(data, list):
            results = []
            for games_data in data:
                games = [Game(**game_data) for game_data in games_data]
                results.append(games)
            if allow_nested:
                return results
            return list(itertools.chain(*results))
        return [Game(**game_data) for game_data in data]


@validate_call
def load_game_results(
    min_date: str | None = None,
) -> dict[Condition, list[Game]] | dict[Condition, list[list[Game]]]:
    """
    Loads game results from the results directory, reconstructing Conditions
    and Games from the directory and file names and contents respectively.
    If the corresponding file has list of lists of games, outputs them.

    Args:
        min_date (str | None): Optional ISO formatted date string (YYYY-MM-DD).
            If provided, only loads results from this date or newer.

    Returns:
        dict[Condition, list[Game]]: A dictionary mapping Condition objects to
            lists of Game objects.
    """
    condition_to_games = {}
    min_datetime = None
    if min_date:
        min_datetime = datetime.datetime.strptime(min_date, "%Y-%m-%d")

    # Iterate through all directories in the results directory
    for dir_name in os.listdir(RESULTS_DIR):
        dir_path = os.path.join(RESULTS_DIR, dir_name)
        if os.path.isdir(dir_path):
            # Extract roles and other conditions from the directory name
            condition = dir_to_condition(dir_name)

            # Find all files that meet the date criteria
            valid_files = []
            for file_name in os.listdir(dir_path):
                # Assuming filename format is like 'YYYY-MM-DD.jsonl'
                try:
                    file_date_str = file_name.split(".")[0]  # Extract the date part
                    file_date = datetime.datetime.strptime(file_date_str, "%Y-%m-%d")

                    # Skip files older than min_date if specified
                    if min_datetime and file_date < min_datetime:
                        continue

                    valid_files.append((file_date, os.path.join(dir_path, file_name)))
                except ValueError:
                    continue  # If the conversion fails, skip this file

            # Sort files by date, newest first
            valid_files.sort(reverse=True)

            # Load games from valid files
            for _, file_path in valid_files:
                results = load_file(file_path)

                # Append games to the condition's entry in the dictionary
                if condition not in condition_to_games:
                    condition_to_games[condition] = results
                else:
                    condition_to_games[condition].extend(results)

                # Right now we just want the most recent file
                break

    return condition_to_games


def escape_string(value: str) -> str:
    """
    Escape characters in a string that could be problematic in filenames or directory names.

    Args:
        value (str): The string to escape.

    Returns:
        str: The escaped string.
    """
    if "&" in value or "\\" in value:
        raise TypeError("Values must not contain '&' nor '\\'.")
    return value.replace("/", "__")


def unescape_string(value: str) -> str:
    """
    Unescape characters in a string that were previously escaped.

    Args:
        value (str): The string to unescape.

    Returns:
        str: The original string with escaped characters restored.
    """
    return value.replace("__", "/")


def dict_to_string(d: Dict[str, Any]) -> str:
    """
    Convert a dictionary into a string of key=value pairs separated by underscores,
    sorted by keys. Handles integers, booleans, strings, and None.

    Args:
        d (Dict[str, Any]): The dictionary to convert.

    Returns:
        str: A directory safe string representation of the dictionary.

    Raises:
        TypeError: If a value in the dictionary is not an int, bool, str, or None.
        Also if a string contains a `&`
    """
    # Validate each value type
    for key, value in d.items():
        if not isinstance(value, (int, bool, str, type(None))):
            raise TypeError(f"Unsupported type for key '{key}': {type(value).__name__}")

    # Ensure the dictionary is sorted by keys
    sorted_items = sorted(d.items())

    # Create a keyword=value formatted string, converting `None` to a string
    kv_pairs = [
        f"{key}={escape_string(str(value)) if value is not None else 'None'}"
        for key, value in sorted_items
    ]

    # Join the pairs with underscores
    result_string = "&".join(kv_pairs)

    return result_string


def string_to_dict(s: str) -> Dict[str, Any]:
    """
    Convert a string of key=value pairs separated by underscores back into a dictionary.
    Handles integers, booleans, strings, and None.

    Args:
        s (str): The string representation of the dictionary.

    Returns:
        Dict[str, Any]: The reconstructed dictionary.
    """

    # Define a helper function to convert string to correct type
    def convert_value(value: str) -> Any:
        value = unescape_string(value)  # Unescape strings first
        if value.isdigit():
            return int(value)
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        if value == "None":
            return None
        return value

    # Split the string into key=value pairs
    kv_pairs = s.split("&")

    # Convert to dictionary with proper type handling
    return {
        key: convert_value(value)
        for key, value in (pair.split("=") for pair in kv_pairs)
    }
