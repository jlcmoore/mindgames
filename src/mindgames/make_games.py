"""
Author: Jared Moore
Date: August, 2024

Generates games which satisfy the below conditions. Specifically:

Let the attributes, A, have individual members i.
Let the proposals, P, have individual members p.
Let the coefficients of the value function of the target be $v_T(a)$
Let the H be a function which maps from a proposal and action to whether those are hidden
    to the target.
    $\\forall{p \\in P, a \\in A} H(p_a) \\rightarrow {0, 1}$ (False, True)
Let the R be a function which maps from a proposal and action to whether those are revealed
    in optimal play by the persuader to the target.
    $\\forall{p \\in P, a \\in A} R(p_a) \\rightarrow {0, 1}$ (False, True)

These three conditions must be satisfied:
    
    1. $V_T(y) > V_T(x), V_T(z)$       -- Given all info, the target chooses `x`
    2. $V_T^H(z) > V_T^H(x), V_T^H(y)$ -- Given only info that isn't hidden (start state),
                                         the target chooses `z`
    3. $V_T^R(x) > V_T^R(z), V_T^R(y)$ -- Given the revealed info (what a persuader should say),
                                         the target chooses `y`

The value function for the target is the sum of the coefficients of the target's value function for 
each attribute times the utility of each attribute of each proposal.

    $V_T(p) = \\forall_{i \\in A} v_T(a) U(p_a)$

And with the hidden information ($(1 - H(p_a))$ evaluates to 0 if hidden, 1 otherwise)

    $V_T^H(p) = \\forall_{i \\in A} v_T(a) U(p_a) (1 - H(p_a))$

And with the revealed information 
($(1 - H(p_a) + R(p_a) - 1)$ evaluates to 0 if hidden and not revealed, 1 if not hidden or revealed)

    $V_T^R(p) = \\forall_{i \\in A} v_T(a) U(p_a)(1 - H(p_a) + R(p_a) - 1)$

Also these conditions must be satisfied:

    1.  $|H| \\leq some constant$
    2.  $\\forall_p \\forall_i \\not H_p(a) \\implies \\not R_p(a)$

"""

import argparse
import logging
import math
import os
import pprint
import random
import warnings
from itertools import product

import pandas as pd
from ortools.sat.python import cp_model
from tqdm import tqdm

from .model import (
    GameModel,
    UTIL_FMT,
    HIDDEN_FMT,
    REVEALED_FMT,
    COEFFICIENT_FMT,
)
from .utils import (
    DEFAULT_PROPOSALS,
    DEFAULT_ATTRIBUTES,
    set_logger,
    args_to_str,
    PAYOFF_DIR,
    DIFFICULTY_CONDITIONS,
)

logger = logging.getLogger(__name__)

DEFAULT_VALUE_RANGE = (-1, 1)
DEFAULT_UTILITY_RANGE = (-1, 1)
DEFAULT_MAX_HIDDEN = 4


class GameSolutionCallback(cp_model.CpSolverSolutionCallback):
    """Saves intermediate solutions."""

    def __init__(
        self,
        utility_vars,
        hidden_vars,
        revealed_vars,
        target_coefficient_vars,
        persuader_coefficient_vars,
        proposals,
        attributes,
        max_hidden_utilities,
        max_solutions=None,
    ):
        """
        Initializes the callback with the necessary variables and parameters.

        Parameters:
            - utility_vars, dict[str, dict[str, IntVar]]: The payoff matrix
            - hidden_vars, dict[str, dict[str, BoolVar]]:
                Whether a state is initially hidden to the target
            - revealed_vars, dict[str, dict[str, BoolVar]]:
                Whether a state ought rationally be revealed to the target
            - target_coefficient_vars, dict[str, IntVar]:
                The coefficients on the target's utility function
            - persuader_coefficient_vars, dict[str, IntVar]:
                The coefficients on the persuader's utility function
            - proposals, list[str]: List of proposals
            - attributes, list[str]: List of attributes
            - max_hidden_utilities, int: Maximum number of hidden utilities
            - max_solutions (int): the number of solutions to stop after, if None finds all
        """
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.utility_vars = utility_vars
        self.hidden_vars = hidden_vars
        self.revealed_vars = revealed_vars
        self.target_coefficient_vars = target_coefficient_vars
        self.persuader_coefficient_vars = persuader_coefficient_vars
        self.proposals = proposals
        self.attributes = attributes
        self.max_hidden_utilities = max_hidden_utilities
        self.solution_count = 0
        self.solutions = set()
        self.pbar = tqdm(desc="Collecting solutions")
        self.max_solutions = max_solutions

    def OnSolutionCallback(self):
        """
        Called when a new solution is found. It extracts the solution values and stores them.
        """
        self.solution_count += 1
        logger.debug(f"Solution {self.solution_count}")
        self.pbar.update(1)

        if self.max_solutions and self.solution_count >= self.max_solutions:
            self.StopSearch()

        utilities = {}
        hidden = {}
        revealed = {}
        target_coefficients = {}
        persuader_coefficients = {}

        for p in self.proposals:
            utilities[p] = {}
            hidden[p] = {}
            revealed[p] = {}
            for a in self.attributes:
                u_val = self.Value(self.utility_vars[p][a])
                h_val = self.Value(self.hidden_vars[p][a])
                r_val = self.Value(self.revealed_vars[p][a])
                utilities[p][a] = u_val
                hidden[p][a] = bool(h_val)
                revealed[p][a] = bool(r_val)

        for a in self.attributes:
            target_coeff_val = self.Value(self.target_coefficient_vars[a])
            target_coefficients[a] = target_coeff_val

            persuader_coeff_val = self.Value(self.persuader_coefficient_vars[a])
            persuader_coefficients[a] = persuader_coeff_val

        solution = GameModel(
            utilities,
            hidden,
            revealed,
            target_coefficients,
            persuader_coefficients,
            self.proposals,
            self.attributes,
            self.max_hidden_utilities,
        )
        solution.is_solution()
        if solution in self.solutions:
            logger.debug("duplicate")
        self.solutions.add(solution)  # Store the solution


def value_function_vars(
    model,
    coefficient_vars,
    utility_vars,
    proposals,
    attributes,
    utility_range,
    value_function_coefficient_range,
    hidden_vars=None,
    revealed_vars=None,
):
    """
    Parameters:
        - model: The CP-SAT model
        - coefficient_vars, dict[str, IntVar]:
            - The coefficients on the utility function, either of the target or persuader
            - `hidden_vars` and `revealed_vars` should not be passed for the persuader
        - utility_vars, dict[str, dict[str, IntVar]]: The payoff matrix
        - proposals, list[str]: List of proposals
        - attributes, list[str]: List of attributes
        - utility_range, tuple[int, int]: Range of utility values
        - value_function_coefficient_range, tuple[int, int]: Range of value function coefficients
        - hidden_vars, dict[str, dict[str, BoolVar]]:
            - Whether a state is initially hidden to the target, or None if calculated
        - revealed_vars, dict[str, dict[str, BoolVar]]:
            - Whether a state ought rationally be revealed to the target, or None if not calculated
    Returns:
        - dict[str, IntVar]:
            The value of the proposal to the target given the hidden and revealed information,
            if passed
    """
    total_utility_vars = {}

    min_total = utility_range[0] * value_function_coefficient_range[1] * len(attributes)
    max_total = utility_range[1] * value_function_coefficient_range[1] * len(attributes)
    for p in proposals:
        terms = []
        for a in attributes:
            """
            Because of how the CP-Model works we can't actually add any multiplications
            in as constraints
            Instead we have to make a number of dummy terms which are equal to the result
            of the multiplications
            and then add a constraint on them.
            """
            base_term = model.NewIntVar(
                min_total, max_total, f"term_{p}_{a}"
            )  # Adjust the range as needed
            model.AddMultiplicationEquality(
                base_term, [utility_vars[p][a], coefficient_vars[a]]
            )

            if hidden_vars and revealed_vars:
                revealed_term = model.NewIntVar(0, 1, f"revealed_term_{p}_{a}")
                model.Add(
                    revealed_term == (1 - hidden_vars[p][a] + revealed_vars[p][a])
                )
                final_term = model.NewIntVar(
                    min_total, max_total, f"revealed_final_term_{p}_{a}"
                )
                model.AddMultiplicationEquality(final_term, [base_term, revealed_term])
                terms.append(final_term)
            elif hidden_vars:
                hidden_term = model.NewIntVar(0, 1, f"hidden_term_{p}_{a}")
                model.Add(hidden_term == (1 - hidden_vars[p][a]))
                final_term = model.NewIntVar(
                    min_total, max_total, f"hidden_final_term_{p}_{a}"
                )
                model.AddMultiplicationEquality(final_term, [base_term, hidden_term])
                terms.append(final_term)
            else:  # Neither hidden nor revealed
                terms.append(base_term)

        # Create a variable to hold the sum of terms
        sum_id = ""
        if hidden_vars and revealed_vars:
            sum_id += "revealed_"
        elif hidden_vars:
            sum_id += "hidden_"
        sum_terms = model.NewIntVar(min_total, max_total, f"{sum_id}sum_terms_{p}")
        model.Add(sum_terms == sum(terms))

        total_utility_vars[p] = sum_terms

    return total_utility_vars


def setup_csp(
    proposals=DEFAULT_PROPOSALS,
    attributes=DEFAULT_ATTRIBUTES,
    utility_range=DEFAULT_UTILITY_RANGE,
    value_function_coefficient_range=DEFAULT_VALUE_RANGE,
    max_hidden_utilities=DEFAULT_MAX_HIDDEN,
    non_solutions=False,
    max_solutions=None,
    difficulty=None,
    fixed_target_coefficients=None,
):
    """
    Sets up the CSP and returns it and a callback.
    """
    model = cp_model.CpModel()

    utility_vars = {}
    hidden_vars = {}
    list_of_hidden = []
    revealed_vars = {}
    target_coefficient_vars = {}
    persuader_coefficient_vars = {}

    for p in proposals:
        utility_vars[p] = {}
        hidden_vars[p] = {}
        revealed_vars[p] = {}
        for a in attributes:
            util_id = UTIL_FMT.format(p, a)
            util_var = model.NewIntVar(utility_range[0], utility_range[1], util_id)
            utility_vars[p][a] = util_var

            hidden_id = HIDDEN_FMT.format(p, a)
            hidden_var = model.NewBoolVar(hidden_id)
            hidden_vars[p][a] = hidden_var
            list_of_hidden.append(hidden_var)

            revealed_id = REVEALED_FMT.format(p, a)
            revealed_var = model.NewBoolVar(revealed_id)
            revealed_vars[p][a] = revealed_var

            # Constraint: Only utilities that were hidden can be revealed
            model.AddImplication(hidden_var.Not(), revealed_var.Not())

    # Constraint: Limit the number of utilities that are hidden
    model.Add(sum(list_of_hidden) <= max_hidden_utilities)

    for a in attributes:
        target_coefficient_range = value_function_coefficient_range

        if fixed_target_coefficients:
            coeff_value = fixed_target_coefficients[a]
            target_coefficient_range = (coeff_value, coeff_value)

        target_coeff_var = model.NewIntVar(
            target_coefficient_range[0],
            target_coefficient_range[1],
            COEFFICIENT_FMT.format("T", a),
        )
        target_coefficient_vars[a] = target_coeff_var

        persuader_coeff_var = model.NewIntVar(
            value_function_coefficient_range[0],
            value_function_coefficient_range[1],
            COEFFICIENT_FMT.format("P", a),
        )
        persuader_coefficient_vars[a] = persuader_coeff_var

    # Given all the information, the persuader chooses the first proposal
    persuader_values = value_function_vars(
        model,
        persuader_coefficient_vars,
        utility_vars,
        proposals,
        attributes,
        utility_range,
        value_function_coefficient_range,
    )
    # Conditon 0) Peruader chooses "A"
    model.Add(persuader_values["A"] > persuader_values["B"])
    model.Add(persuader_values["A"] > persuader_values["C"])

    none_hidden_values = value_function_vars(
        model,
        target_coefficient_vars,
        utility_vars,
        proposals,
        attributes,
        utility_range,
        value_function_coefficient_range,
    )

    initial_values = value_function_vars(
        model,
        target_coefficient_vars,
        utility_vars,
        proposals,
        attributes,
        utility_range,
        value_function_coefficient_range,
        hidden_vars,
    )

    revealed_values = value_function_vars(
        model,
        target_coefficient_vars,
        utility_vars,
        proposals,
        attributes,
        utility_range,
        value_function_coefficient_range,
        hidden_vars,
        revealed_vars,
    )

    if non_solutions:
        if difficulty == "always-win":
            # No matter what the target chooses A

            # Condition 1) -- none hidden
            model.Add(none_hidden_values["A"] > none_hidden_values["B"])
            model.Add(none_hidden_values["A"] > none_hidden_values["C"])

            # Condition 2) -- all hidden
            model.Add(initial_values["A"] > initial_values["B"])
            model.Add(initial_values["A"] > initial_values["C"])

            # Condition 3) -- some revealed
            model.Add(revealed_values["A"] > revealed_values["B"])
            model.Add(revealed_values["A"] > revealed_values["C"])

        elif difficulty == "never-win":
            # - move from C to B and stay there no matter what

            # Condition 1) -- none hidden
            model.Add(none_hidden_values["A"] < none_hidden_values["B"])
            model.Add(none_hidden_values["C"] < none_hidden_values["B"])

            # Condition 2) -- all hidden
            model.Add(initial_values["A"] < initial_values["C"])
            model.Add(initial_values["B"] < initial_values["C"])

            # Condition 3) -- some revealed
            model.Add(revealed_values["A"] < revealed_values["B"])
            model.Add(revealed_values["C"] < revealed_values["B"])

        else:  # difficulty == "can-win":
            # A.k.a. only win when they reveal all the info

            # Condition 1) -- none hidden
            model.Add(none_hidden_values["A"] > none_hidden_values["B"])
            model.Add(none_hidden_values["A"] > none_hidden_values["C"])

            # Condition 2) -- all hidden
            model.Add(initial_values["C"] > initial_values["A"])
            model.Add(initial_values["C"] > initial_values["B"])

            # Condition 3) -- some revealed
            model.Add(revealed_values["A"] < revealed_values["C"])

    else:  # these are the constraints for a solution
        # Condition 1) -- none hidden
        model.Add(none_hidden_values["B"] > none_hidden_values["A"])
        model.Add(none_hidden_values["B"] > none_hidden_values["C"])

        # Condition 2) -- all hidden
        model.Add(initial_values["C"] > initial_values["A"])
        model.Add(initial_values["C"] > initial_values["B"])

        # Condition 3) -- some revealed
        model.Add(revealed_values["A"] > revealed_values["B"])
        model.Add(revealed_values["A"] > revealed_values["C"])

    solution_callback = GameSolutionCallback(
        utility_vars,
        hidden_vars,
        revealed_vars,
        target_coefficient_vars,
        persuader_coefficient_vars,
        proposals,
        attributes,
        max_hidden_utilities,
        max_solutions,
    )

    return model, solution_callback


def main():
    parser = argparse.ArgumentParser(
        description="CP-SAT Model for Game Payoff Matrices"
    )
    # Currently only set up for three proposals
    parser.add_argument(
        "--proposals", nargs="+", default=DEFAULT_PROPOSALS, help="List of proposals"
    )
    parser.add_argument(
        "--attributes", nargs="+", default=DEFAULT_ATTRIBUTES, help="List of attributes"
    )
    parser.add_argument(
        "--utility_range",
        type=int,
        nargs=2,
        default=DEFAULT_UTILITY_RANGE,
        help="Range of utility values",
    )
    parser.add_argument(
        "--value_function_coefficient_range",
        type=int,
        nargs=2,
        default=DEFAULT_VALUE_RANGE,
        help="Range of value function coefficients",
    )
    parser.add_argument(
        "--max_hidden_utilities",
        type=int,
        default=DEFAULT_MAX_HIDDEN,
        help="Maximum number of hidden utilities",
    )
    parser.add_argument(
        "--print-first",
        default=False,
        action="store_true",
        help="Whether to simply print the first solution found.",
    )
    parser.add_argument(
        "--save-games",
        default=False,
        action="store_true",
        help="Whether to save a subset of size `n-games-to-save` of the found solutions.",
    )
    parser.add_argument(
        "--non-solutions",
        default=False,
        action="store_true",
        help="Whether to find non-solutions.",
    )
    parser.add_argument(
        "--difficulty",
        default=None,
        choices=DIFFICULTY_CONDITIONS,
        help="If `non-solutions`, how winnable, persuadable to make the games",
    )
    parser.add_argument(
        "--max-solutions",
        default=None,
        required=False,
        type=int,
        help="The max number of solutions to find.",
    )
    parser.add_argument(
        "--n-games-to-save",
        type=int,
        default=100,
        help="How many games to save: bounded by `max_solutions`",
    )
    parser.add_argument(
        "--value-function-quota",
        type=int,
        default=None,
        help=(
            "If set, enforces that at least this number of games per "
            "target coefficient assignment are saved when using --save-games."
        ),
    )
    parser.add_argument(
        "--log",
        type=str,
        default="WARNING",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    args = parser.parse_args()
    set_logger(args.log, logger=logger)

    if args.print_first:
        args.max_solutions = 1

    if args.max_solutions:
        args.n_games_to_save = min(args.n_games_to_save, args.max_solutions)

    if args.non_solutions and not args.difficulty:
        args.difficulty = "can-win"

    if args.value_function_quota:
        # Adjust the script to enforce at least `n` games per target coefficient assignment
        # Get all possible target coefficient assignments
        possible_coefficients = [-1, 1, 0]
        attributes = args.attributes
        all_possible_assignments = list(
            product(possible_coefficients, repeat=len(attributes))
        )

        sampled_solutions = []
        total_count = 0
        for coeffs in all_possible_assignments:
            fixed_target_coefficients = dict(zip(attributes, coeffs))
            logger.info(f"Processing target coefficients: {fixed_target_coefficients}")

            model, solution_callback = setup_csp(
                args.proposals,
                attributes,
                tuple(args.utility_range),
                tuple(args.value_function_coefficient_range),
                args.max_hidden_utilities,
                non_solutions=args.non_solutions,
                max_solutions=args.max_solutions,
                difficulty=args.difficulty,
                fixed_target_coefficients=fixed_target_coefficients,
            )

            # Solving the CSP
            solver = cp_model.CpSolver()
            status = solver.SearchForAllSolutions(model, solution_callback)

            solutions = list(solution_callback.solutions)
            num_solutions_found = len(solutions)
            if num_solutions_found < args.value_function_quota:
                print(
                    f"Warning: Only found {num_solutions_found} solutions "
                    f"for target coefficients {fixed_target_coefficients}. "
                    f"Required: {args.value_function_quota}"
                )
            else:
                logger.info(
                    f"Found {num_solutions_found} solutions for {fixed_target_coefficients}"
                )

            total_count += num_solutions_found

            if solutions:
                max_to_choose = math.ceil(
                    args.n_games_to_save / len(all_possible_assignments)
                )
                sampled_items = random.sample(solutions, max_to_choose)
                sampled_solutions.extend(sampled_items)

        if not sampled_solutions:
            print("No solutions found.")
            return

        data = [solution.model_dump() for solution in sampled_solutions]

    else:

        model, solution_callback = setup_csp(
            args.proposals,
            args.attributes,
            tuple(args.utility_range),
            tuple(args.value_function_coefficient_range),
            args.max_hidden_utilities,
            non_solutions=args.non_solutions,
            max_solutions=args.max_solutions,
            difficulty=args.difficulty,
        )

        # Solving the CSP
        solver = cp_model.CpSolver()

        # NB: A warning here tells us to use the next line but it leads to a memory leak...
        # solver.enumerate_all_solutions = True
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="search_for_all_solutions is deprecated",
        )
        status = solver.SearchForAllSolutions(model, solution_callback)

        if status != cp_model.OPTIMAL and status != cp_model.FEASIBLE:
            print("No solution found.")

        if args.print_first:
            first = solution_callback.solutions.pop()
            print(pprint.pformat(first, indent=4))

        # Randomly sample N items
        solutions = list(solution_callback.solutions)
        sampled_items = random.sample(
            solutions, min(len(solutions), args.n_games_to_save)
        )
        data = [solution.model_dump() for solution in sampled_items]

    if args.save_games:

        # Define the directory and file name
        file_name = args_to_str(
            args,
            exclude_args=[
                "log",
                "save_games",
                "print_first",
                "proposals",
                "attributes",
            ],
        )
        file_name += ".jsonl"
        file_path = os.path.join(PAYOFF_DIR, file_name)

        # Ensure the directory exists
        os.makedirs(PAYOFF_DIR, exist_ok=True)

        # Convert the list of JSON strings to a DataFrame and save to a file
        df = pd.DataFrame(data)
        df.to_json(file_path, orient="records", lines=True)
        print(f"Saved {len(data)} games to {file_path}")
