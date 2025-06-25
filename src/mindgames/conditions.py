"""
Author: Jared Moore
Date: October, 2024

Contains objects to operate on experimental conditions.
"""

from pydantic import BaseModel, ConfigDict, model_validator

from .utils import model_name_short


class Roles(BaseModel):
    """
    A class to store the roles of a round, e.g. a human target and LLM persuader
    """

    # These fields can be an int if they store an (internal) id of an actual particpant
    human_persuader: bool | int = False
    human_target: bool | int = False
    llm_persuader: str | None = None
    llm_target: str | None = None

    # freeze the model so no post‐creation mutation is allowed
    model_config = ConfigDict(frozen=True)

    # pylint: disable=no-self-argument
    @model_validator(mode="after")
    def check_roles(cls, m: "Roles") -> "Roles":
        """Validate the inputs"""
        if m.human_persuader and m.llm_persuader:
            raise ValueError("Two persuaders passed")
        if m.human_target and m.llm_target:
            raise ValueError("Two targets passed")
        if not (m.human_persuader or m.llm_persuader):
            raise ValueError("You must specify either an llm or human as persuader")
        return m

    def is_rational_target(self):
        """Whether this condition is the rational target condition."""
        return (
            (self.human_persuader or self.llm_persuader)
            and not self.llm_target
            and not self.human_target
        )

    def is_paired_human(self):
        """Whether this condition involves two human participants."""
        return self.human_persuader and self.human_target

    def persuader_type(self) -> str:
        """Returns a string description of the persuader type"""
        if self.human_persuader:
            result = "Human"
            if isinstance(self.human_persuader, int) and not isinstance(
                self.human_persuader, bool
            ):
                result += f" {self.human_persuader}"
            return result
        return model_name_short(self.llm_persuader)

    def target_type(self) -> str:
        """Returns a string description of the persuader type"""
        if self.human_target:
            result = "Human"
            if isinstance(self.human_target, int):
                result += f" {self.human_target}"
            return result
        if self.llm_target:
            return model_name_short(self.llm_target)
        return "Rational"

    def as_non_id_role(self, no_target_id: bool | None = None) -> "Roles":
        """
        Returns a copy of this role without the participant and persuader ids

        no_target_id (bool | None) if None returns without both participant ids.
        If True returns with just the target id and no persuader id, if False returns
        with just the persuader id and no target id

        """
        return Roles(
            human_persuader=(
                bool(self.human_persuader)
                if no_target_id is None or no_target_id
                else self.human_persuader
            ),
            human_target=(
                bool(self.human_target)
                if no_target_id is None or not no_target_id
                else self.human_target
            ),
            llm_persuader=self.llm_persuader,
            llm_target=self.llm_target,
        )

    def __str__(self) -> str:
        """Returns a readable string for these roles"""
        return f"{self.persuader_type()} Persuader, {self.target_type()} Target"


RATIONAL_TARGET_ROLE = Roles(human_persuader=True)
PAIRED_HUMAN_ROLE = Roles(human_persuader=True, human_target=True)


class Condition(BaseModel):
    """
    A class to store an experimental condition; a kind of round to have been played.
    """

    roles: Roles
    # These flags control whether we turn on the other condition flags
    allow_lying: bool = False
    # TODO: Use target's values is currently not implemented
    targets_values: bool = False
    reveal_motivation: bool = False
    reveal_belief: bool = False
    add_hint: bool = False
    perfect_game: bool = False  # Only relevant for LLMs
    # Whether to show a perfect in context game
    # (same scenario, different payoff of the same type) to persuaders in
    # rational target games

    # Whether to discretize the action space of the game with JSON
    discrete_game: bool = False
    non_mental: bool = False

    model_config = ConfigDict(frozen=True)

    def is_rational_target(self):
        """Whether this condition is the rational target condition."""
        return self.roles.is_rational_target()

    def is_paired_human(self):
        """Whether this condition involves two human participants."""
        return self.roles.is_paired_human()

    def is_control(self) -> bool:
        """Whether this is a control condition (reveal motivation and belief)"""
        return self.reveal_motivation and self.reveal_belief

    # pylint: disable=no-self-argument
    @model_validator(mode="after")
    def check_consistency(cls, m: "Condition") -> "Condition":
        """Validate the inputs"""
        if m.targets_values and not m.roles.human_target:
            raise ValueError("Use targets' own values only implemented for humans.")
        if m.perfect_game and not m.roles.is_rational_target():
            raise ValueError("Perfect game only for rational target.")
        if m.targets_values and (not m.roles.human_target or m.roles.llm_target):
            raise ValueError("Cannot use the target's values when there is no target.")
        return m

    def as_non_id_role(self, **kwargs) -> "Condition":
        """Returns a copy of this condition with the role as a non id role"""
        return Condition(
            roles=self.roles.as_non_id_role(**kwargs),
            allow_lying=self.allow_lying,
            targets_values=self.targets_values,
            reveal_motivation=self.reveal_motivation,
            reveal_belief=self.reveal_belief,
            add_hint=self.add_hint,
            perfect_game=self.perfect_game,
            discrete_game=self.discrete_game,
            non_mental=self.non_mental,
        )

    def __str__(self) -> str:
        """Returns a readable version of this Condition"""
        result = str(self.roles)

        value_type = "Real" if self.targets_values else "Provided"
        result += f", {value_type} values"
        if (not self.reveal_motivation and self.reveal_belief) or (
            self.reveal_motivation and not self.reveal_belief
        ):
            raise NotImplementedError(
                "Assuming both reveal belief and motivation or neither"
            )

        if self.is_control():
            result += ", Revealed"
        else:
            result += ", Hidden"
        if self.allow_lying:
            result += " (Lying allowed)"
        if self.add_hint:
            result += " (Added hint)"
        if self.perfect_game:
            result += " (Perfect Game in Context)"
        if self.discrete_game:
            result += " (Discrete Action Space)"
        if self.discrete_game:
            result += " (Non Mental Scenario)"
        return result
