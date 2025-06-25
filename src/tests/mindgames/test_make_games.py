"""
Author: Jared Moore
Date: August, 2024

Tests for the making of Game Models, the CSP.
"""

import logging
from io import StringIO
import concurrent
import os
import unittest
from unittest.mock import patch

from ortools.sat.python import cp_model

from mindgames.make_games import setup_csp, main
from mindgames.known_models import SOLUTION_TERNARY
from mindgames.utils import DEFAULT_PROPOSALS, DEFAULT_ATTRIBUTES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)


class TestMakeGames(unittest.TestCase):

    @patch("sys.argv", ["make_games", "--max_hidden_utilities", "0"])
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_no_solution(self, mock_stdout):
        main()
        output = mock_stdout.getvalue().strip()
        self.assertIn("No solution found.", output)

    @patch(
        "sys.argv",
        ["make_games", "--utility_range", "-1", "1", "--max_hidden_utilities", "2"],
    )
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_ternary_no_solution(self, mock_stdout):
        main()
        output = mock_stdout.getvalue().strip()
        self.assertIn("No solution found.", output)

    @unittest.skipUnless(
        os.getenv("RUN_ALL_TESTS", "False") == "True", "Skipping extra test case"
    )
    def test_main_ternary_yes_solution(self):
        model, solution_callback = setup_csp()

        # Solving the CSP
        solver = cp_model.CpSolver()
        solver.parameters.enumerate_all_solutions = True
        solver.solve(model, solution_callback)

        self.assertEqual(len(solution_callback.solutions), 76728)
        solution = SOLUTION_TERNARY.set_proposals_attributes(
            DEFAULT_PROPOSALS, DEFAULT_ATTRIBUTES
        )
        self.assertIn(solution, solution_callback.solutions)

    @unittest.skipUnless(
        os.getenv("RUN_ALL_TESTS", "False") == "True", "Skipping extra test case"
    )
    @patch("sys.argv", ["make_games", "--print-first"])
    @patch("sys.stdout", new_callable=StringIO)
    def test_main_print_first(self, mock_stdout):
        # For whatever reason this script does not always finish; it hangs.
        timed_function_call(main)
        output = mock_stdout.getvalue().strip()
        print(output)
        self.assertIn("GameModel(utilities=", output)

    @patch("sys.argv", ["make_games", "--help"])
    @patch("sys.stdout", new_callable=StringIO)
    def test_help(self, mock_stdout):
        with self.assertRaises(SystemExit):  # Assuming --help exits the script
            main()
        output = mock_stdout.getvalue().strip()
        self.assertIn("usage:", output)


def timed_function_call(func, *args, **kwargs):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        max_retries = 10
        for attempt in range(max_retries):
            future = executor.submit(func, *args, **kwargs)
            try:
                # Wait for the function to complete with a timeout of 0.5 seconds
                future.result(timeout=1)
                return
            except concurrent.futures.TimeoutError:
                logger.info(
                    f"Attempt {attempt + 1}: Function took too long, retrying..."
                )

        raise ValueError(
            "Max retries reached. Function did not complete within the time limit."
        )
