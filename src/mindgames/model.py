"""
Author: Jared Moore
Date: August, 2024

Contains the GameModel class and a known solution instance.
"""

import copy
from dataclasses import dataclass
import logging
from typing import Dict, Tuple

from pydantic import BaseModel, ConfigDict, validate_call

from tabulate import tabulate

from .utils import DEFAULT_PROPOSALS, DEFAULT_ATTRIBUTES

logger = logging.getLogger(__name__)

UTIL_FMT = "U({0}_{1})"
HIDDEN_FMT = "H({0}_{1})"
REVEALED_FMT = "R({0}_{1})"
COEFFICIENT_FMT = "v_{0}({1})"


@validate_call
def _value_function(
    proposal: str,
    attributes: list[str],
    utilities: dict[str, dict[str, int]],
    coefficients: dict[str, int],
    hidden: dict[str, dict[str, bool]] | None = None,
    revealed: dict[str, dict[str, bool | int]] | None = None,
    return_computation: bool = False,
) -> int | tuple[int, str]:
    """
    Parameters:
        - proposal (str): The proposal to calculate the value of
        - attributes (list[str]): The attributes of each proposal
        - utilities (dict[dict[str, int]]): The real utilities of each proposal and attribute
        - coefficients (dict[str, int]): The coefficient on each attribute for this player
        - hidden (dict[dict[str, bool]]): Whether a state is initially hidden to the target,
            or None if not passed
        - revealed (dict[dict[str, <bool, int>]]): Whether a state ought rationally be
            revealed to the target, or the actual value revealed (in case of deception),
            or None if not passed
        - return_computation: Whether to return a string of the computation of the result
    Returns:
        - int: The value of the proposal to the target given the hidden and revealed information,
            if set to True
        - str: The computation necessary for the result (optional)
    """
    total = 0
    computation_str = ""
    for attribute in attributes:
        utility = utilities[proposal][attribute]
        # If calculating hidden states and this state is hidden
        if hidden and hidden[proposal][attribute]:
            # If not calculating revealed states or this state is not revealed
            if not revealed or (
                revealed
                and revealed[proposal][attribute] is False
                or revealed[proposal][attribute] is None
            ):
                continue
            if not isinstance(revealed[proposal][attribute], bool):
                utility = revealed[proposal][attribute]
        term = utility * coefficients[attribute]
        computation_str += f"({coefficients[attribute]} * {utility}) + "
        total += term

    if return_computation:
        # Remove trailing " + " and add total value
        computation_str = computation_str.rstrip(" + ") + f" = {total}"
        return (total, computation_str)
    return total


@dataclass(frozen=True)
class GameModel(BaseModel):
    """
    This is a model of the payoff matrices, value functions, and such of the game.
    It is also a solution to the CSP in `make_games.py`.
    """

    utilities: Dict[str, Dict[str, int]]
    hidden: Dict[str, Dict[str, bool]]
    ideal_revealed: Dict[str, Dict[str, bool]]
    target_coefficients: Dict[str, int]
    persuader_coefficients: Dict[str, int]
    proposals: list[str]  # These two cannot be sets because the order matters!
    attributes: list[str]
    max_hidden_utilities: int

    # This config lets you make a GameModel from a list of properties as with SQL
    model_config = ConfigDict(from_attributes=True)

    def __post_init__(self):
        object.__setattr__(self, "proposals", list(self.proposals))
        object.__setattr__(self, "attributes", list(self.attributes))

        hidden_bools = {}

        modified_ideal_revealed = {}
        for p in self.proposals:
            modified_ideal_revealed[p] = {}
            hidden_bools[p] = {}
            for a in self.attributes:
                # Ensure ideal_revealed contains boolean values
                modified_ideal_revealed[p][a] = bool(self.ideal_revealed[p][a])
                hidden_bools[p][a] = bool(self.hidden[p][a])
        object.__setattr__(self, "hidden", hidden_bools)

        object.__setattr__(self, "ideal_revealed", modified_ideal_revealed)

    def utilities_ternary(self):
        # Are the utilities in the set of {-1, 0, 1}? If so, we can change how we report them
        utilities_set = set()
        for p in self.proposals:
            for a in self.attributes:
                utilities_set.add(self.utilities[p][a])
        return utilities_set == {-1, 0, 1}

    def __repr__(self):
        return (
            f"GameModel(utilities={self.utilities}, hidden={self.hidden}, "
            f"ideal_revealed={self.ideal_revealed}, target_coefficients={self.target_coefficients},"
            f" persuader_coefficients={self.persuader_coefficients}, proposals={self.proposals}, "
            f"attributes={self.attributes}, max_hidden_utilities={self.max_hidden_utilities})"
        )

    def __str__(self):
        return self.as_table()

    def as_generic(self) -> "GameModel":
        """Returns a copy of this GameModel in which the attributes and proposals
        have generic names"""
        return self.set_proposals_attributes(DEFAULT_PROPOSALS, DEFAULT_ATTRIBUTES)

    def as_table(
        self,
        target_pronoun: str = "Tar.",
        persuader_pronoun: str = "Per.",
        persuader_value_function: bool = True,
        target_none_hidden_value_function: bool = True,
        target_some_revealed_value_function: bool = True,
        return_computation: bool = False,
        as_html: bool = False,
    ) -> str:
        """Returns this GameModel represented as a table
        Parameters:
            target_pronoun: the pronoun to use for the target
            persuader_pronoun: the pronoun to use for the persuader
            persuader_value_function: Whether to show the persuader VF
            target_none_hidden_value_function: Whether to show the VF of the target
                with none hidden
            target_some_revealed_value_function: Whether to show the VF of the target
                with the perfect info revealed
            return_computation: Whether to show the intermediary calculations.
            as_html: Whether to return the table as html
        """
        # Set up headers
        result = ""
        table = []
        headers = (
            [""]
            + ["\n" + p for p in self.proposals]
            + ["", f"{target_pronoun}\nValue\nFunction"]
        )
        alignment = ("right",) + ("center",) * len(self.proposals) + ("center", "left")
        if persuader_value_function:
            headers.append(f"{persuader_pronoun}\nValue\nFunction")
            alignment += ("left",)

        # Add the actual game state info
        for a in self.attributes:
            row = [a]
            for p in self.proposals:
                util = self.utilities[p][a]
                if self.hidden[p][a]:
                    util = f"({util})"
                if self.ideal_revealed[p][a] and target_some_revealed_value_function:
                    util = f"_{util}_"
                row.append(util)
            row += ["", self.target_coefficients[a]]
            if persuader_value_function:
                row.append(self.persuader_coefficients[a])
            table.append(row)
        # This is a separting line
        table.append([])

        names = []
        value_function_params = []

        if persuader_value_function:
            value_function_params.append({"is_target": False})
            names.append(f"{persuader_pronoun} Preferences")
        value_function_params.append({"is_target": True, "hidden": True})
        names.append(f'{target_pronoun} Initial Preferences\n("(#)" is a hidden value)')
        if target_none_hidden_value_function:
            value_function_params.append({"is_target": True})
            names.append(
                f"{target_pronoun} Perfect Preferences\n(with no hidden values)"
            )
        if target_some_revealed_value_function:
            value_function_params.append(
                {"is_target": True, "hidden": True, "revealed": True}
            )
            names.append(
                f'{target_pronoun} Ideal Preferences\n(with ideal revealed info, "\\_(#)\\_")'
            )

        # Add the derived info, the preferred proposals and sums
        for name, params in zip(names, value_function_params):
            values = []
            results = []
            for p in self.proposals:
                computation_str = ""
                value = self.value_function(
                    p, **params, return_computation=return_computation
                )
                if return_computation:
                    value, computation_str = value
                    computation_strs = computation_str.split("+")

                    results.append("+\n".join(computation_strs))
                else:
                    results.append(value)
                values.append(value)
            # Bold the max
            idx = values.index(max(values))
            results[idx] = f"**{results[idx]}**"
            table.append([name] + results)

        fmt = "html" if as_html else "grid"

        # Make the table
        result = tabulate(
            table,
            headers=headers,
            tablefmt=fmt,
            colalign=alignment,
        )
        return "\n" + result

    def __hash__(self):
        return hash(
            (
                frozenset(
                    (p, a, self.utilities[p][a])
                    for p in self.proposals
                    for a in self.attributes
                ),
                frozenset(
                    (p, a, self.hidden[p][a])
                    for p in self.proposals
                    for a in self.attributes
                ),
                frozenset(
                    (p, a, self.ideal_revealed[p][a])
                    for p in self.proposals
                    for a in self.attributes
                ),
                frozenset((a, self.target_coefficients[a]) for a in self.attributes),
                frozenset((a, self.persuader_coefficients[a]) for a in self.attributes),
                frozenset(self.proposals),
                frozenset(self.attributes),
            )
        )

    def is_solution(self):
        """
        Tests for the constraints we define in the csp. See `make_games.py`.
        """
        num_hidden = 0
        for p in self.proposals:
            for a in self.attributes:
                num_hidden += self.hidden[p][a]

                # not hidden implies not revealed
                if not self.hidden[p][a] and self.ideal_revealed[p][a]:
                    logger.error("Revealed and not hidden")
                    return False

        # These are "A" "B" and "C", effectively
        first_p, second_p, third_p = self.proposals[:3]

        # The persuader chooses the first proposal
        condition0 = (
            self.value_function(first_p, is_target=False)
            > self.value_function(second_p, is_target=False)
        ) and (
            self.value_function(first_p, is_target=False)
            > self.value_function(third_p, is_target=False)
        )

        if not condition0:
            logger.debug(f"Persuader does not choose {first_p}")

        # The target chooses the second proposal when all is revealed
        condition1 = (
            self.value_function(second_p, is_target=True)
            > self.value_function(first_p, is_target=True)
        ) and (
            self.value_function(second_p, is_target=True)
            > self.value_function(third_p, is_target=True)
        )

        if not condition1:
            logger.debug(f"Target does not choose {second_p} when all info known.")

        # Given only info that isn't hidden (start state),
        #                                the target chooses `z`
        condition2 = (
            self.value_function(third_p, hidden=True)
            > self.value_function(first_p, hidden=True)
        ) and (
            self.value_function(third_p, hidden=True)
            > self.value_function(second_p, hidden=True)
        )

        if not condition2:
            logger.debug(f"Target does not choose {third_p} when all info hidden.")

        # Given the revealed info (what a persuader should say),
        #                                 the target chooses `y`
        condition3 = (
            self.value_function(first_p, hidden=True, revealed=True)
            > self.value_function(third_p, hidden=True, revealed=True)
        ) and (
            self.value_function(first_p, hidden=True, revealed=True)
            > self.value_function(second_p, hidden=True, revealed=True)
        )
        if not condition3:
            logger.debug(f"Target does not choose {first_p} when ideal info revealed.")

        result = (
            num_hidden <= self.max_hidden_utilities
            and condition0
            and condition1
            and condition2
            and condition3
        )

        if not result:
            logger.debug(self)
        return result

    def non_solution_difficulty(self):
        """
        If this is not a solution, returns the difficulty level, whether the persuader
        will 'always-win', 'can-win', or will 'never-win' against an ideal target.
        Returns None if this is a solution or of the conditions for the difficulty levels
        are not met.
        """

        # These are "A" "B" and "C", effectively
        first_p, second_p, third_p = self.proposals[:3]

        if self.is_solution():
            return None

        persuader_chooses_a = (
            self.value_function(first_p, is_target=False)
            > self.value_function(second_p, is_target=False)
        ) and (
            self.value_function(first_p, is_target=False)
            > self.value_function(third_p, is_target=False)
        )

        if not persuader_chooses_a:
            return None

        b_first_none_hidden = (
            self.value_function(first_p) < self.value_function(second_p)
        ) and (self.value_function(third_p) < self.value_function(second_p))

        b_first_some_revealed = (
            self.value_function(third_p, hidden=True, revealed=True)
            < self.value_function(second_p, hidden=True, revealed=True)
        ) and (
            self.value_function(first_p, hidden=True, revealed=True)
            < self.value_function(second_p, hidden=True, revealed=True)
        )

        ###

        a_first_none_hidden = (
            self.value_function(first_p) > self.value_function(second_p)
        ) and (self.value_function(first_p) > self.value_function(third_p))

        a_first_all_hidden = (
            self.value_function(first_p, hidden=True)
            > self.value_function(third_p, hidden=True)
        ) and (
            self.value_function(first_p, hidden=True)
            > self.value_function(second_p, hidden=True)
        )

        a_first_some_revealed = (
            self.value_function(first_p, hidden=True, revealed=True)
            > self.value_function(third_p, hidden=True, revealed=True)
        ) and (
            self.value_function(first_p, hidden=True, revealed=True)
            > self.value_function(second_p, hidden=True, revealed=True)
        )

        ###

        c_first_all_hidden = (
            self.value_function(third_p, hidden=True)
            > self.value_function(first_p, hidden=True)
        ) and (
            self.value_function(third_p, hidden=True)
            > self.value_function(second_p, hidden=True)
        )

        c_beats_a_some_revealed = self.value_function(
            third_p, hidden=True, revealed=True
        ) > self.value_function(first_p, hidden=True, revealed=True)

        if b_first_none_hidden and c_first_all_hidden and b_first_some_revealed:
            return "never-win"
        if a_first_none_hidden and a_first_all_hidden and a_first_some_revealed:
            return "always-win"
        if a_first_none_hidden and c_first_all_hidden and c_beats_a_some_revealed:
            return "can-win"
        return None

    def can_reveal_one_incorrect_info(self, just_one=False):
        """
        True if it is still possible to win while revealing one of the incorrect pieces
        of information and False otherwise.
        If just_one, returns True if Any of the incorrect pieces can be revealed. Otherwise requires
        All (independently) to be able to be revealed.
        """

        # These are "A" "B" and "C", effectively
        first_p, second_p, third_p = self.proposals[:3]

        incorrect_reveals: list[Tuple[str, str]] = []
        for p, attrs in self.hidden.items():
            for a, hidden in attrs.items():
                if hidden and not self.ideal_revealed[p][a]:
                    incorrect_reveals.append((p, a))

        if just_one:
            can_reveal_one = False
        else:
            can_reveal_one = True

        for p, a in incorrect_reveals:
            revealed = copy.deepcopy(self.ideal_revealed)
            revealed[p][a] = True

            v_a = _value_function(
                first_p,
                attributes=self.attributes,
                utilities=self.utilities,
                coefficients=self.target_coefficients,
                hidden=self.hidden,
                revealed=revealed,
                return_computation=False,
            )

            v_b = _value_function(
                second_p,
                attributes=self.attributes,
                utilities=self.utilities,
                coefficients=self.target_coefficients,
                hidden=self.hidden,
                revealed=revealed,
                return_computation=False,
            )

            v_c = _value_function(
                third_p,
                attributes=self.attributes,
                utilities=self.utilities,
                coefficients=self.target_coefficients,
                hidden=self.hidden,
                revealed=revealed,
                return_computation=False,
            )

            chooses_first_proposal = (v_a >= v_c) and (v_a >= v_b)
            if just_one:
                can_reveal_one |= chooses_first_proposal
            else:
                can_reveal_one &= chooses_first_proposal
        return can_reveal_one

    def value_function(
        self,
        proposal,
        is_target=True,
        hidden=False,
        revealed=False,
        return_computation: bool = False,
    ) -> int | tuple[int, str]:
        """
        Parameters:
            - proposal, str: The proposal to calculate the value of
            - is_target, bool: Whether this is the "target" or "persuader"
            - hidden, bool:
                - Whether a state is initially hidden to the target
            - revealed, bool:
                - Whether a state ought rationally be revealed to the target
            - actual_revealed (dict[dict[str, int]]): The actual states revealed to the target.
            - return_computation: Whether to return a string of the computation of the result
        Returns:
            - int:
                The value of the proposal to the target given the hidden and revealed information,
                if set to True
            - str: A string representing the computation of the result.
        """
        coefficients = (
            self.target_coefficients if is_target else self.persuader_coefficients
        )

        return _value_function(
            proposal,
            self.attributes,
            self.utilities,
            coefficients,
            self.hidden if hidden else None,
            self.ideal_revealed if revealed else None,
            return_computation=return_computation,
        )

    def get_hidden_states(self, ideal_revealed_only: bool = False):
        """
        Returns the hidden states to the target as a dict of the proposals mapping
        to the hidden attributes
        If `ideal_revealed_only`, only returns the states to ideally reveal.
        """
        hidden = {}
        for p in self.proposals:
            hidden[p] = []
            for a in self.attributes:
                if self.hidden[p][a] and (
                    not ideal_revealed_only
                    or ideal_revealed_only
                    and self.ideal_revealed[p][a]
                ):
                    hidden[p].append(a)
        return hidden

    def check_information(self, information):
        """
        Parameters:
        - information (dict[str, dict[str, int]]): Reported utility values.
        Returns:
        - (bool): True if the information is truthful, False if not
        """
        if len(set(information.keys()) - set(self.proposals)) > 0:
            logging.error(f"Information, {information}")
            raise ValueError("Requested nonexistant information.")
        for p in information:
            if len(set(information[p].keys()) - set(self.attributes)) > 0:
                logging.error(f"Information, {information}")
                raise ValueError("Requested nonexistant information.")
            for a in information[p]:
                reported_utility = information[p][a]
                actual_utility = self.utilities[p][a]

                # If the reported utility does not match the actual utility, information is false
                if reported_utility != actual_utility:
                    logger.debug(
                        f"Discrepancy found for {p}, {a}: "
                        f"Reported: {reported_utility}, Actual: {actual_utility}"
                    )
                    return False

        return True

    def set_proposals_attributes(self, proposals, attributes):
        """
        Returns a new GameModel which the proposals and attributes swapped
        """
        if (len(proposals) != len(self.proposals)) or (
            len(attributes) != len(self.attributes)
        ):
            raise ValueError("Proposals and attributes must be of the same length.")

        utilities = {}
        hidden = {}
        ideal_revealed = {}

        for old_p, p in zip(self.proposals, proposals):
            ideal_revealed[p] = {}
            hidden[p] = {}
            utilities[p] = {}
            for old_a, a in zip(self.attributes, attributes):
                ideal_revealed[p][a] = self.ideal_revealed[old_p][old_a]
                hidden[p][a] = self.hidden[old_p][old_a]
                utilities[p][a] = self.utilities[old_p][old_a]

        target_coefficients = {}
        persuader_coefficients = {}
        for old_a, a in zip(self.attributes, attributes):
            target_coefficients[a] = self.target_coefficients[old_a]
            persuader_coefficients[a] = self.persuader_coefficients[old_a]

        return GameModel(
            utilities=utilities,
            hidden=hidden,
            ideal_revealed=ideal_revealed,
            target_coefficients=target_coefficients,
            persuader_coefficients=persuader_coefficients,
            proposals=proposals,
            attributes=attributes,
            max_hidden_utilities=self.max_hidden_utilities,
        )

    def __eq__(self, other):
        if not isinstance(other, GameModel):
            return NotImplemented
        return self.model_dump() == other.model_dump()
