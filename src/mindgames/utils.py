"""
Author: Jared Moore
Date: August, 2024

Contains utility functions used throughout the package.
"""

import importlib
from functools import wraps
import logging
import os
import sys

import inflect
import numpy as np
import pandas as pd
from pydantic import BaseModel

import data

DEFAULT_PROPOSALS = ("A", "B", "C")
DEFAULT_ATTRIBUTES = ("x", "y", "z")

SOLUTION_CONDITION = "solution"
DIFFICULTY_CONDITIONS = ["never-win", "always-win", "can-win"]
EXPERIMENT_CONDITIONS = set(DIFFICULTY_CONDITIONS) | set([SOLUTION_CONDITION])

EX_COVER_STORY = """Urban planning affects all aspects of modern life. \
In this scenario, imagine that your local city government is considering proposals to change its \
urban development policies. The city council will soon consider these proposals. \
In the meantime, an urban planning expert is discussing the proposals with a council member.\
"""

EX_PERSUADER_ROLE = "urban planning expert"

EX_TARGET_ROLE = "city council member"

EX_ATTRIBUTES = ["housing availability", "public transportation", "golf courses"]

EX_SCENARIO = {
    "id": "example",
    "attributes": EX_ATTRIBUTES,
    "proposals": DEFAULT_PROPOSALS,
    "cover_story": EX_COVER_STORY,
    "target_role": EX_TARGET_ROLE,
    "persuader_role": EX_PERSUADER_ROLE,
}

TERNARY_COEFFICIENTS = (-1, 1)

PAYOFF_FILE_FMT = os.path.join(
    "payoffs",
    "utility-range=(-1, 1)_value-function-coefficient-range={coefficients}_"
    + "max-hidden-utilities=4_non-solutions={non_solutions}_difficulty={difficulty}_"
    + "max-solutions=1000_n-games-to-save=150_"
    + "value-function-quota=5.jsonl",
)

PAYOFF_DIR = importlib.resources.files(data) / "payoffs"

####

MODEL_NAMES_SHORT = {
    # NB: assuming no base models
    "meta-llama/Llama-2-70b-chat-hf": "llama2-70b",
    "meta-llama/Llama-3.1-8B-Instruct": "llama3.1-8b",
    "meta-llama/Llama-3.1-70B-Instruct": "llama3.1-70b",
    "meta-llama/Llama-3.1-405B-Instruct": "llama3.1-405b",
    "claude-3-5-sonnet-20240620": "claude3.5-sonnet",
    # TODO: add in claude 3.5 opus when out.
    "gpt-4o-2024-08-06": "gpt-4o",
}


######## Survey Utils

RATINGS = {
    2: "Increased a lot",
    1: "Increased a little",
    0: "Stayed the same",
    -1: "Decreased a little",
    -2: "Decreased a lot",
}


class Scenario(BaseModel):
    """A table to store scenarios"""

    id: str
    cover_story: str
    persuader_role: str | None = None
    target_role: str | None = None
    attributes: list[str]


class SurveyResponse(BaseModel):
    id: str
    statement: str
    rating: int

    def __str__(self) -> str:
        return f"**{RATINGS[self.rating]}**: {self.statement}"

    def __sub__(self, before: "SurveyResponse") -> "SurveyResponseDifference":
        """
        Calculates the change in rating between this (after) and another (before) survey response.
        """
        diff = self.rating - before.rating
        return SurveyResponseDifference(
            id=self.id,
            statement=self.statement,
            before_rating=before.rating,
            after_rating=self.rating,
            difference=diff,
        )


class SurveyResponseDifference(BaseModel):
    """
    A class to store the difference in survey responses
    """

    id: str
    statement: str
    before_rating: int
    after_rating: int
    difference: int

    def __str__(self) -> str:
        """Returns a string representation of the survey response difference"""
        base = f'"{self.statement}"\n '
        base += f"Before: {RATINGS[self.before_rating]}, After: {RATINGS[self.after_rating]}"
        if self.difference == 0:
            delta = "No change"
        else:
            changeword = "Increased" if self.difference > 0 else "Decreased"
            delta = f"{changeword} by {abs(self.difference)}"
        return f"{base}\n {delta}"


def rank_attributes(statements: list[dict[str, any]]) -> dict[str, int]:
    """
    Ranks each attribute based on the user's agreement or disagreement with statements.

    Parameters:
    - statements (list of dicts): A list where each dict represents a statement with its
        supporting and opposing attributes

    Returns:
    - dict: A dictionary where keys are attributes and values are +1, -1, or 0 indicating
        support, opposition, or neutrality.
    """
    ranked_attributes = {}
    for statement in statements:
        rating = np.sign(
            statement.get("rating", 0)
        )  # Default to neutral if rating is missing
        ranked_attributes[statement["id"]] = rating

    return ranked_attributes


def values_from_survey(
    survey: list[SurveyResponse], attributes: list[str]
) -> dict[str, int]:
    """
    Turns a survey into the "coefficients" or value function as used by a game model.
    """
    ranking = rank_attributes(survey)
    assert not set(attributes) - set(ranking.keys())
    coefficients = {attribute: ranking[attribute] for attribute in attributes}
    return coefficients


def value_str_from_survey(survey: list[SurveyResponse], attributes: list[str]) -> str:
    """
    Returns a string (a list) representing the answers on the survey to questions
    which pertain to `attributes`
    """
    relevant: list[str] = []
    attributes = set(attributes)
    for response in survey:
        if response.id in attributes:
            relevant.append(str(response))
    return "".join([f"\n- {response}" for response in relevant]) + "\n"


#########


def limit_to_n_characters(message: str, n: int) -> str:
    """Returns up to n characters of the message."""
    # Split the message into words
    return message[0:n]


def model_name_short(model_name: str) -> str:
    """Shortens the passed model name, if possible"""
    if model_name in MODEL_NAMES_SHORT:
        return MODEL_NAMES_SHORT[model_name]
    return model_name


def get_data_file_path(filename):
    """
    For a given name of a package file (in src/data) returns the full file path for that file
    or errors if it does not exist.
    """
    data_path = importlib.resources.files(data) / filename
    if not os.path.exists(data_path):
        raise ValueError(f"Package file does not exist: {filename}")
    return data_path


SCENARIOS_FILE = get_data_file_path("scenarios.jsonl")

NON_MENTAL_SCENARIOS_FILE = get_data_file_path("non_mental_scenarios.jsonl")

SURVEY_FILE = get_data_file_path("survey.jsonl")


def _make_survey():
    survey = pd.read_json(SURVEY_FILE, lines=True)
    scenarios = pd.read_json(SCENARIOS_FILE, lines=True)
    cover_story_map = scenarios.set_index("id")["cover_story"]

    survey["cover_story"] = survey["scenario_id"].map(cover_story_map)
    return survey.to_dict(orient="records")


SURVEY = _make_survey()


def log_function_call(logger: logging.Logger, level: int = logging.INFO):
    """
    Returns a decorator with the given logger for logging function calls.
    """

    def decorator(func):
        """
        A decorater to log the call to the passed function, `func`, and its return.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Log function input as a callable string"""
            args_str = ", ".join(repr(arg) for arg in args)
            if args_str:
                assert args
                args_str += ", "
            kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            call_str = f"{func.__name__}({args_str}"
            if kwargs_str:
                call_str += f"{kwargs_str}"
            call_str += ")"
            logger.log(level, f"Function called: {call_str}")

            # Call the actual function
            result = func(*args, **kwargs)
            # Log function output
            logger.log(level, f"Function returned: {func.__name__}: {result!r}")

            return result

        return wrapper

    return decorator


def args_to_str(args, exclude_args=None):
    """
    Generates a str representation of the arguments in `args` excluding `exclude_args`
    """
    if exclude_args is None:
        exclude_args = []

    parts = []
    for arg, value in vars(args).items():
        if arg not in exclude_args:
            arg_with_hyphens = arg.replace("_", "-")
            parts.append(f"{arg_with_hyphens}={value}")

    return "_".join(parts)


def get_payoff_file_path(
    non_solutions=False, coefficients=TERNARY_COEFFICIENTS, difficulty=None
):
    """
    Returns the name of a relevant payoff file pertinent to the passed variables.
    """
    only_difficulty = difficulty
    if non_solutions and not difficulty or difficulty == "solution":
        only_difficulty = "can-win"
    filename = PAYOFF_FILE_FMT.format(
        non_solutions=non_solutions,
        coefficients=coefficients,
        difficulty=only_difficulty,
    )
    return get_data_file_path(filename)


def set_logger(level, logger=None):
    """Convert the log level string to a logging level, raises an error if not a valid level"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    # Only configure if not already configured to avoid no-op
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    if logger:
        logger.setLevel(numeric_level)


def prefix_with_conjunction(prefix, conjunction, lst):
    """
    Convert a list of strings into a single string with `prefix` and `conjunction`
    before the last item.

    Parameters:
        lst (list of str): The list of strings to be joined.

    Returns:
        str: A single string with items separated by `prefix` and `conjunction`
            before the last item. Returns an empty string if the list is empty.
    """
    lst = list(lst)
    if len(lst) == 0:
        return ""
    if len(lst) == 1:
        return lst[0]
    return (prefix + " ").join(lst[:-1]) + " " + conjunction + " " + lst[-1]


def comma_with_and(lst):
    """
    Like `prefix_with_and` where prefix=',' and conjunction='and'
    """
    return prefix_with_conjunction(",", "and", lst)


def int_to_words(number):
    """
    Convert an integer to its word representation.

    Parameters:
        number (int): The integer to be converted.

    Returns:
        str: The word representation of the integer.
    """
    p = inflect.engine()
    return p.number_to_words(number)


def number_to_words_ordinal(n):
    """
    Convert an integer to its ordinal word representation.

    Parameters:
        n (int): The integer to be converted.

    Returns:
        str: The ordinal word representation of the integer.
    """
    p = inflect.engine()
    return p.number_to_words(p.ordinal(n))


def replace_json_chars(text):
    """
    Replaces senstive characters used in JSON parsing.

    Parameters:
        text (str): The string to replace.

    Returns:
        str: The replacement
    """
    return (
        text.replace("“", '"')
        .replace("'", '"')
        .replace("”", '"')
        .replace("：", ":")
        .replace("，", ",")
        .replace("```json", "")
        .replace("```", "")
    )


def top_proposals_tie(proposals_and_values: list[tuple[str, int]]) -> bool:
    """True if the top two proposals tie, false otherwise"""
    _, max_value = proposals_and_values[0]
    return max_value == proposals_and_values[1][1]
