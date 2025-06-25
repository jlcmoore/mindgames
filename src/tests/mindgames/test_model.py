"""
Author: Jared Moore
Date: August, 2024

Tests for the Model class.
"""

import unittest

import pandas as pd

from mindgames import GameModel
from mindgames.model import _value_function
from mindgames.utils import get_payoff_file_path
from mindgames.known_models import (
    SOLUTION,
    SOLUTION_TERNARY,
    OTHER_SOLUTION_TERNARY,
    NOT_SOLUTION_TOO_MANY_HIDDEN,
    NOT_SOLUTION_TOO_FEW_REVEALED,
    NOT_SOLUTION_STUCK_B,
    NOT_SOLUTION_DEFAULT_A,
)

KNOWN_SOLUTION_STR = r"""
+---------------------------------------+--------+-------+--------+----+------------+------------+
|                                       |        |       |        |    | Tar.       | Per.       |
|                                       |   a    |   b   |   c    |    | Value      | Value      |
|                                       |        |       |        |    | Function   | Function   |
+=======================================+========+=======+========+====+============+============+
|                                     x |   1    |   0   | _(-2)_ |    | 1          | 0          |
+---------------------------------------+--------+-------+--------+----+------------+------------+
|                                     y |   -2   |  -2   |   -2   |    | -1         | 1          |
+---------------------------------------+--------+-------+--------+----+------------+------------+
|                                     z |   1    |  (3)  |   3    |    | 1          | -1         |
+---------------------------------------+--------+-------+--------+----+------------+------------+
+---------------------------------------+--------+-------+--------+----+------------+------------+
|                      Per. Preferences | **-3** |  -5   |   -5   |    |            |            |
+---------------------------------------+--------+-------+--------+----+------------+------------+
|              Tar. Initial Preferences |   4    |   2   | **5**  |    |            |            |
|             ("(#)" is a hidden value) |        |       |        |    |            |            |
+---------------------------------------+--------+-------+--------+----+------------+------------+
|              Tar. Perfect Preferences |   4    | **5** |   3    |    |            |            |
|               (with no hidden values) |        |       |        |    |            |            |
+---------------------------------------+--------+-------+--------+----+------------+------------+
|                Tar. Ideal Preferences | **4**  |   2   |   3    |    |            |            |
| (with ideal revealed info, "\_(#)\_") |        |       |        |    |            |            |
+---------------------------------------+--------+-------+--------+----+------------+------------+"""


class TestValueFunction(unittest.TestCase):

    def test_value_function(self):
        # Define a sample set of inputs
        proposals = ["a", "b", "c"]
        attributes = ["x", "y", "z"]
        utilities = {
            "a": {"x": 1, "y": 0, "z": 0},
            "b": {"x": 0, "y": 1, "z": 0},
            "c": {"x": 0, "y": 0, "z": 1},
        }
        coefficients = {"x": 1, "y": 1, "z": 1}
        hidden = {
            "a": {"x": False, "y": False, "z": False},
            "b": {"x": False, "y": False, "z": False},
            "c": {"x": False, "y": False, "z": False},
        }
        revealed = {
            "a": {"x": False, "y": False, "z": False},
            "b": {"x": False, "y": False, "z": False},
            "c": {"x": False, "y": False, "z": False},
        }

        # When nothing is hidden or revealed these should not change the VF
        for p in proposals:
            all_info = _value_function(p, attributes, utilities, coefficients)
            initial = _value_function(
                p, attributes, utilities, coefficients, hidden=hidden
            )
            ideal_revealed = _value_function(
                p, attributes, utilities, coefficients, hidden=hidden, revealed=revealed
            )
            self.assertEqual(all_info, 1)
            self.assertEqual(all_info, initial)
            self.assertEqual(initial, ideal_revealed)

        hidden = {
            "a": {"x": False, "y": False, "z": False},
            "b": {"x": False, "y": True, "z": False},
            "c": {"x": False, "y": False, "z": False},
        }
        revealed = {
            "a": {"x": False, "y": False, "z": False},
            "b": {"x": False, "y": True, "z": False},
            "c": {"x": False, "y": False, "z": False},
        }

        # Test with hidden information
        result = _value_function(
            "b", attributes, utilities, coefficients, hidden=hidden
        )
        self.assertEqual(result, 0)

        # Test with revealed information
        result = _value_function(
            "b", attributes, utilities, coefficients, hidden=hidden, revealed=revealed
        )
        self.assertEqual(result, 1)

        # Test with hidden and revealed information where revealed value is different
        revealed = {
            "a": {"x": False, "y": False, "z": False},
            "b": {"x": False, "y": 5, "z": False},
            "c": {"x": False, "y": False, "z": False},
        }
        result = _value_function(
            "b", attributes, utilities, coefficients, hidden=hidden, revealed=revealed
        )
        self.assertEqual(result, 5)

        # Test with different coefficients
        coefficients = {"x": -1, "y": 1, "z": -1}
        result = _value_function("a", attributes, utilities, coefficients)
        self.assertEqual(result, -1)

        utilities = {
            "A": {"x": 1, "y": 0, "z": -1},
            "B": {"x": -1, "y": -1, "z": -1},
            "C": {"x": 0, "y": -1, "z": -1},
        }
        hidden = {
            "A": {"x": True, "y": False, "z": False},
            "B": {"x": True, "y": False, "z": False},
            "C": {"x": False, "y": True, "z": True},
        }
        revealed = {
            "A": {"x": False, "y": False, "z": False},
            "B": {"x": False, "y": False, "z": False},
            "C": {"x": False, "y": True, "z": True},
        }
        ideal_revealed = {
            "A": {"x": 0, "y": 0, "z": 0},
            "B": {"x": 0, "y": 0, "z": 0},
            "C": {"x": 0, "y": 1, "z": 1},
        }
        target_coefficients = {"x": -1, "y": 1, "z": 1}
        proposals = ["A", "B", "C"]
        attributes = ["x", "y", "z"]

        # In this case, ideal revealed reveals all info about C so the two should be equal
        result = _value_function("C", attributes, utilities, target_coefficients)
        result_revealed = _value_function(
            "C",
            attributes,
            utilities,
            target_coefficients,
            hidden=hidden,
            revealed=revealed,
        )
        self.assertEqual(result, result_revealed)


class TestModel(unittest.TestCase):

    def test_str(self):
        # Serialize the SOLUTION model to a dictionary
        result = str(SOLUTION)
        self.assertEqual(result, KNOWN_SOLUTION_STR)

    def test_serialize(self):
        # Serialize the SOLUTION model to a dictionary
        solution_dict = SOLUTION.model_dump()

        # Deserialize the dictionary back to a GameModel instance
        deserialized_solution = GameModel(**solution_dict)

        # Check if the deserialized model is equal to the original model
        self.assertEqual(SOLUTION, deserialized_solution)

        # Check if the deserialized model is still a solution

        self.assertTrue(deserialized_solution.is_solution())

    def test_known_solutions(self):
        self.assertTrue(SOLUTION.is_solution())
        self.assertTrue(SOLUTION_TERNARY.is_solution())
        self.assertFalse(NOT_SOLUTION_TOO_MANY_HIDDEN.is_solution())
        self.assertFalse(NOT_SOLUTION_TOO_FEW_REVEALED.is_solution())
        self.assertFalse(NOT_SOLUTION_STUCK_B.is_solution())
        self.assertFalse(NOT_SOLUTION_DEFAULT_A.is_solution())

    def test_known_solutions_payoff_file(self):
        df = pd.read_json(get_payoff_file_path(), lines=True)
        models = [GameModel(**row.to_dict()) for _, row in df.iterrows()]
        for model in models:
            self.assertTrue(model.is_solution())

    def test_known_non_solutions_payoff_file(self):
        for difficulty in ["never-win", "always-win", "can-win"]:
            df = pd.read_json(
                get_payoff_file_path(non_solutions=True, difficulty=difficulty),
                lines=True,
            )
            models = [GameModel(**row.to_dict()) for _, row in df.iterrows()]
            for model in models:
                self.assertFalse(model.is_solution())

    def test_non_solutions_difficulty(self):
        df = pd.read_json(get_payoff_file_path(), lines=True)
        models = [GameModel(**row.to_dict()) for _, row in df.iterrows()]
        for model in models:
            self.assertIsNone(model.non_solution_difficulty())
        for difficulty in ["never-win", "always-win", "can-win"]:
            df = pd.read_json(
                get_payoff_file_path(non_solutions=True, difficulty=difficulty),
                lines=True,
            )
            models = [GameModel(**row.to_dict()) for _, row in df.iterrows()]
            for model in models:
                self.assertFalse(model.is_solution())
                self.assertEqual(model.non_solution_difficulty(), difficulty)

    def test_check_information(self):
        # Test truthful information
        correct_information = {
            "a": {"x": 1, "y": -2, "z": 1},
            "b": {"x": 0, "y": -2, "z": 3},
            "c": {"x": -2, "y": -2, "z": 3},
        }
        self.assertTrue(SOLUTION.check_information(correct_information))

        # Test misinformation (one wrong value)
        incorrect_information_1 = {
            "a": {"x": 1, "y": -2, "z": 1},
            "b": {"x": 1, "y": -2, "z": 3},  # Incorrect value for "x"
            "c": {"x": -2, "y": -2, "z": 3},
        }
        self.assertFalse(SOLUTION.check_information(incorrect_information_1))

        # Test misinformation (all wrong values for one proposal)
        incorrect_information_2 = {
            "a": {"x": 1, "y": -2, "z": 1},
            "b": {
                "x": 1,
                "y": -1,
                "z": 2,
            },  # Incorrect values for all attributes in "b"
            "c": {"x": -2, "y": -2, "z": 3},
        }
        self.assertFalse(SOLUTION.check_information(incorrect_information_2))

        # Test misinformation (mix of correct and wrong values across multiple proposals)
        incorrect_information_3 = {
            "a": {"x": 1, "y": -1, "z": 1},  # Incorrect value for "y"
            "b": {"x": 0, "y": -2, "z": 3},
            "c": {"x": -2, "y": -2, "z": 2},  # Incorrect value for "z"
        }
        self.assertFalse(SOLUTION.check_information(incorrect_information_3))

    def test_value_function(self):

        # In this case, ideal revealed reveals all info about C so the two should be equal
        self.assertEqual(
            OTHER_SOLUTION_TERNARY.value_function("C", is_target=True),
            OTHER_SOLUTION_TERNARY.value_function(
                "C", is_target=True, hidden=True, revealed=True
            ),
        )

    def test_set_proposals(self):
        # TODO: should test if the set_proposals_attributes works
        pass
