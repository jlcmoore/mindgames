"""
Author: Jared Moore
Date: August, 2024

Various known solutions and non solutions.
"""

from .model import GameModel

SOLUTION = GameModel(
    utilities={
        "a": {"x": 1, "y": -2, "z": 1},
        "b": {"x": 0, "y": -2, "z": 3},
        "c": {"x": -2, "y": -2, "z": 3},
    },
    hidden={
        "a": {"x": False, "y": False, "z": False},
        "b": {"x": False, "y": False, "z": True},
        "c": {"x": True, "y": False, "z": False},
    },
    ideal_revealed={
        "a": {"x": False, "y": False, "z": False},
        "b": {"x": False, "y": False, "z": False},
        "c": {"x": True, "y": False, "z": False},
    },
    target_coefficients={
        "x": 1,
        "y": -1,
        "z": 1,
    },
    persuader_coefficients={
        "x": 0,
        "y": 1,
        "z": -1,
    },
    proposals=["a", "b", "c"],
    attributes=["x", "y", "z"],
    max_hidden_utilities=2,
)

CAN_WIN = GameModel(
    **{
        "utilities": {
            "A": {"x": 1, "y": 0, "z": 1},
            "B": {"x": 0, "y": 1, "z": 0},
            "C": {"x": 1, "y": 1, "z": -1},
        },
        "hidden": {
            "A": {"x": False, "y": False, "z": True},
            "B": {"x": True, "y": True, "z": False},
            "C": {"x": False, "y": False, "z": True},
        },
        "ideal_revealed": {
            "A": {"x": False, "y": False, "z": False},
            "B": {"x": True, "y": True, "z": False},
            "C": {"x": False, "y": False, "z": False},
        },
        "target_coefficients": {"x": 1, "y": 1, "z": 1},
        "persuader_coefficients": {"x": 1, "y": -1, "z": 1},
        "proposals": ["A", "B", "C"],
        "attributes": ["x", "y", "z"],
        "max_hidden_utilities": 4,
    }
)

SOLUTION_TERNARY = GameModel(
    utilities={
        "a": {"x": -1, "y": -1, "z": 1},
        "b": {"x": -1, "y": 0, "z": 1},
        "c": {"x": 1, "y": 1, "z": 1},
    },
    hidden={
        "a": {"x": False, "y": True, "z": True},
        "b": {"x": False, "y": False, "z": True},
        "c": {"x": True, "y": False, "z": False},
    },
    ideal_revealed={
        "a": {"x": False, "y": False, "z": True},
        "b": {"x": False, "y": False, "z": False},
        "c": {"x": True, "y": False, "z": False},
    },
    target_coefficients={"x": -1, "y": 1, "z": 1},
    persuader_coefficients={"x": -1, "y": -1, "z": -1},
    proposals=["a", "b", "c"],
    attributes=["x", "y", "z"],
    max_hidden_utilities=4,  # num hidden
)

OTHER_SOLUTION_TERNARY = GameModel(
    utilities={
        "A": {"x": 1, "y": 0, "z": -1},
        "B": {"x": -1, "y": -1, "z": -1},
        "C": {"x": 0, "y": -1, "z": -1},
    },
    hidden={
        "A": {"x": 1, "y": 0, "z": 0},
        "B": {"x": 1, "y": 0, "z": 0},
        "C": {"x": 0, "y": 1, "z": 1},
    },
    ideal_revealed={
        "A": {"x": 0, "y": 0, "z": 0},
        "B": {"x": 0, "y": 0, "z": 0},
        "C": {"x": 0, "y": 1, "z": 1},
    },
    target_coefficients={"x": -1, "y": 1, "z": 1},
    persuader_coefficients={"x": 0, "y": 1, "z": -1},
    proposals=["A", "B", "C"],
    attributes=["x", "y", "z"],
    max_hidden_utilities=4,
)

NOT_SOLUTION_TOO_MANY_HIDDEN = GameModel(
    utilities={
        "a": {"x": 1, "y": -2, "z": 1},
        "b": {"x": 0, "y": -2, "z": 3},
        "c": {"x": -2, "y": -2, "z": 3},
    },
    hidden={
        "a": {"x": False, "y": False, "z": False},
        "b": {"x": False, "y": False, "z": True},
        "c": {"x": True, "y": False, "z": False},
    },
    ideal_revealed={
        "a": {"x": False, "y": False, "z": False},
        "b": {"x": False, "y": False, "z": False},
        "c": {"x": True, "y": False, "z": False},
    },
    target_coefficients={
        "x": 1,
        "y": -1,
        "z": 1,
    },
    persuader_coefficients={
        "x": 0,
        "y": 1,
        "z": -1,
    },
    proposals=["a", "b", "c"],
    attributes=["x", "y", "z"],
    max_hidden_utilities=1,  # too many hidden
)

NOT_SOLUTION_TOO_FEW_REVEALED = GameModel(
    utilities={
        "a": {"x": 1, "y": -2, "z": 1},
        "b": {"x": 0, "y": -2, "z": 3},
        "c": {"x": -2, "y": -2, "z": 3},
    },
    hidden={
        "a": {"x": False, "y": False, "z": False},
        "b": {"x": False, "y": False, "z": True},
        "c": {"x": True, "y": False, "z": False},
    },
    ideal_revealed={
        "a": {"x": False, "y": False, "z": False},
        "b": {"x": False, "y": False, "z": False},
        "c": {"x": False, "y": False, "z": False},
    },
    target_coefficients={
        "x": 1,
        "y": -1,
        "z": 1,
    },
    persuader_coefficients={
        "x": 0,
        "y": 1,
        "z": -1,
    },
    proposals=["a", "b", "c"],
    attributes=["x", "y", "z"],
    max_hidden_utilities=2,
)

NOT_SOLUTION_DEFAULT_A = GameModel(
    utilities={
        "a": {"x": 1, "y": -2, "z": 1},
        "b": {"x": 0, "y": -2, "z": 3},
        "c": {"x": -2, "y": -2, "z": 3},
    },
    hidden={
        "a": {"x": False, "y": False, "z": False},
        "b": {"x": False, "y": False, "z": True},
        "c": {"x": False, "y": False, "z": False},
    },
    ideal_revealed={
        "a": {"x": False, "y": False, "z": False},
        "b": {"x": False, "y": False, "z": False},
        "c": {"x": False, "y": False, "z": False},
    },
    target_coefficients={
        "x": 1,
        "y": -1,
        "z": 1,
    },
    persuader_coefficients={
        "x": 0,
        "y": 1,
        "z": -1,
    },
    proposals=["a", "b", "c"],
    attributes=["x", "y", "z"],
    max_hidden_utilities=2,
)

NOT_SOLUTION_STUCK_B = GameModel(
    utilities={
        "a": {"x": 0, "y": -2, "z": 1},
        "b": {"x": 0, "y": -2, "z": 3},
        "c": {"x": -2, "y": -2, "z": 3},
    },
    hidden={
        "a": {"x": True, "y": False, "z": False},
        "b": {"x": False, "y": False, "z": False},
        "c": {"x": False, "y": False, "z": False},
    },
    ideal_revealed={
        "a": {"x": False, "y": False, "z": False},
        "b": {"x": False, "y": False, "z": False},
        "c": {"x": False, "y": False, "z": False},
    },
    target_coefficients={
        "x": 1,
        "y": -1,
        "z": 1,
    },
    persuader_coefficients={
        "x": 0,
        "y": 1,
        "z": -1,
    },
    proposals=["a", "b", "c"],
    attributes=["x", "y", "z"],
    max_hidden_utilities=2,
)
