"""
Author: Jared Moore
Date: August, 2024

Contains the Game class to play out a game over a GameModel.
"""

import copy
import json
import logging
import pprint
import textwrap
from typing import Any, Self

from openai import OpenAI
from pydantic import BaseModel

from modelendpoints.utils import (
    OPTIONS,
    options_text,
    messages_as_string,
    COT_DELIMITER,
    convert_roles,
)

from .classify_messages import (
    selective_disclosure,
    message_appeals,
    generic_response,
    validate_disclosures,
    validate_appeals,
)
from .model import _value_function, GameModel
from .utils import (
    comma_with_and,
    int_to_words,
    top_proposals_tie,
    SurveyResponse,
    SurveyResponseDifference,
    value_str_from_survey,
    replace_json_chars,
)

from .known_models import (
    SOLUTION_TERNARY,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# The reward in dollars for persuading the target or for guessing the correct answer for the target
BONUS = 1

TURN_LIMIT = 8

CHARACTERS_PER_RESPONSE = 300

DISPLAY_LISTS = False

PROPOSALS_AS_HTML = True

OTHER_PLAYER_DEFAULT = "other player"
OTHER_PLAYER_NON_MENTAL = "system"

PROPOSAL_DEFAULT = "proposal"
PROPOSAL_NON_MENTAL = "choice"

PERSUADE_DEFAULT = "persuade"
PERSUADE_NON_MENTAL = "get it to choose"

PREFERRED_DEFAULT = "preferred"
PREFERRED_NON_MENTAL = "top"

PREFERENCES_NAME = "preferences"
PREFERENCES_NON_MENTAL = "objectives"


GAME_DESCRIPTION = """\
You are playing a decision-making game in which you will exchange messages \
with an {other_player_name}. (The {other_player_name} may change between rounds.) \
"""

GAME_DESCRIPTION_NON_MENTAL = """\
You are trying to get certain choices approved by an automated system. \
You will play a series of rounds. \
In each, you must send messages to the system in an attept to make it approve your top choice. \
"""


class Game(BaseModel):
    """
    This class is a wrapper around a `GameModel` which generates user-facing
    prompts depending on if they are the target or the persuader. As the game progresses,
    it stores the information which has been revealed to the target.
    It also generates the actions for the ideal target.
    """

    # Default game state
    model: GameModel
    cover_story: str = ""
    target_role: str | None = None
    persuader_role: str | None = None

    persuader_choice: str | None = None
    ideal_target_initial_choice: str | None = None
    target_choice: str | None = None
    target_perfect_info_choice: str | None = None

    # Experimental conditions
    reveal_belief: bool = False
    reveal_motivation: bool = False
    allow_lying: bool = False
    targets_values: bool = False

    # Participant info -- only for the target
    initial_survey_responses: list[SurveyResponse] | None = None
    final_survey_responses: list[SurveyResponse] | None = None
    target_initial_choice: str | None = None

    add_hint: bool = False

    is_ideal_target: bool = True
    turn_limit: int = TURN_LIMIT

    # Display options
    display_lists: bool = False
    proposals_as_html: bool = False

    # Whether to include a limit on responses in the instructions
    include_character_limit: bool = True

    # A list of games to show in-context to the model
    in_context_games: list["Game"] = []

    # Information revealed over the game:
    actual_revealed: dict[str, dict[str, int | bool]] | None = None
    # NB: This is not the same dict as in `GameModel` --
    # This is the info actually revealed.
    # We can store actual ints in this dict in order to let the persuader
    # lie about the game state
    persuader_lied: bool = False
    messages: list[dict[str, str]] = []
    chain_of_thought: list[dict[str, str | None]] = (
        []
    )  # For storing scratchpad / CoT messages
    reasoning_trace: list[dict[str, str | None]] = (
        []
    )  # For storing reasoning model's traces (only for DeekSeek, currently)

    all_disclosures: list[dict[str, dict[str, int]]] = []
    all_appeals: list[dict[str, Any]] = []
    ideal_target_last_choice: str | None = None

    # Whether to discretize the action space for the game, requiring and returning JSON
    discrete_game: bool = False

    # Parameters to change the way the game is described -- only if `is_ideal_target`
    non_mental: bool = False

    other_player_name: str = OTHER_PLAYER_DEFAULT
    proposal_name: str = PROPOSAL_DEFAULT
    persuade_verb: str = PERSUADE_DEFAULT
    preferred_adj: str = PREFERRED_DEFAULT
    preferences_name: str = PREFERENCES_NAME
    game_description: str = GAME_DESCRIPTION

    def __init__(self, **data):
        # TODO: Make sure to randomize the proposals if desired before calling this class.

        # Change the game so that the attributes and proposals match
        proposals = data.get("proposals")
        attributes = data.get("attributes")
        model = data.get("model")
        if not proposals and attributes or not attributes and proposals:
            raise ValueError("Either pass both attributes and proposals or neither.")

        if data.get("proposals_as_html") and data.get("display_lists"):
            raise ValueError("Cannot display proposals as both lists and html")

        if proposals and attributes:
            data["model"] = model.set_proposals_attributes(proposals, attributes)

        super().__init__(**data)

        if not self.is_ideal_target and self.non_mental:
            raise ValueError("Can only play a non mental game with the ideal target")

        if self.non_mental:
            self.persuade_verb = PERSUADE_NON_MENTAL
            self.preferred_adj = PREFERRED_NON_MENTAL
            self.preferences_name = PREFERENCES_NON_MENTAL
            self.game_description = GAME_DESCRIPTION_NON_MENTAL
            self.other_player_name = OTHER_PLAYER_NON_MENTAL
            self.proposal_name = PROPOSAL_NON_MENTAL

        # NB: For some reason the below lines don't work in __post_init__
        if self.actual_revealed is None:
            # No information has yet been revealed
            self.actual_revealed = default_revealed(
                self.model.proposals, self.model.attributes
            )

        self.persuader_choice = self.choose_proposal(is_target=False)

        ideal_values = self.proposal_values(is_target=True, all_info=True)
        max_proposal, max_value = ideal_values[0]
        # Make sure there are not two 'perfect choices'
        assert max_value != ideal_values[1][1]
        self.target_perfect_info_choice = max_proposal

        # Store the last action the ideal target took
        self.ideal_target_initial_choice = self.choose_proposal(is_target=True)
        self.ideal_target_last_choice = self.ideal_target_initial_choice
        turn_limit = data.get("turn_limit")
        if turn_limit:
            self.turn_limit = turn_limit

        if self.discrete_game:
            self.include_character_limit = False
            if not self.is_ideal_target:
                raise ValueError(
                    "Can only play discrete games against the ideal target"
                )

    def reveal_info(self, proposal, attribute, value=None):
        """
        Updates the game to reflect that certain information has been revealed to the target.
        Will not update if the information has already been revealed.
        Parameters:
        - proposal (str): The proposal in question
        - attribute (str): The attribute of the proposal
        - value (int): The value of the revealed information, if other than default.
        """
        if (
            proposal not in self.model.proposals
            or attribute not in self.model.attributes
        ):
            raise ValueError("Illegal proposal or attribute")
        if not self.actual_revealed[proposal][attribute]:
            self.actual_revealed[proposal][attribute] = (
                value if value is not None else True
            )

    @staticmethod
    def _instructions(
        turn_limit: int,
        add_hint: bool,
        is_target: bool,
        include_chain_of_thought: bool,
        include_title: bool,
        include_character_limit: bool,
        targets_values: bool,
        discrete_game: bool = False,
        non_mental: bool = False,
    ) -> str:
        """
        Parameters:
        - turn_limit: The turn limit for the game
        - add_hint: Whether to add a hint to the game
        - is_target: If the prompt is for the target
        - include_chain_of_thought (bool): Whether to encourage the use of a CoT (only for models)
        - include_title (bool): Whether to include the title for the instructions
        - include_character_limit (bool): whether to include a limit on the number of characters
        - targets_values (bool): whether to use the values of the target
        - discrete_game (bool): If true, discretizes the action space. If false, leaves it as natural language
        Return:
        - str: Just the instructions for the player (not the prompt).
        """

        if non_mental:
            game_description = GAME_DESCRIPTION_NON_MENTAL
            proposal_name: str = PROPOSAL_NON_MENTAL
            other_player_name: str = OTHER_PLAYER_NON_MENTAL
        else:
            game_description = GAME_DESCRIPTION
            proposal_name: str = PROPOSAL_DEFAULT
            other_player_name: str = OTHER_PLAYER_DEFAULT

        rules = ""
        if include_title:
            rules = INSTRUCTIONS_TITLE

        if is_target:
            if discrete_game:
                raise ValueError("Cannot have a discrete game with a real target.")
            rules += INSTRUCTIONS_TARGET
        else:
            rules += persuader_instructions(
                discrete=discrete_game, non_mental=non_mental
            )

        character_limit = ""
        if include_character_limit:
            character_limit = CHARACTER_LIMIT_PROMPT.format(
                characters_per_response=CHARACTERS_PER_RESPONSE
            )

        bonus_eligible = not targets_values or not is_target

        game_rules = rules.format(
            turn_limit=turn_limit,
            character_limit_prompt=character_limit,
            bonus=BONUS,
            bonus_statement=BONUS_STATEMENT if bonus_eligible else "",
            target_optimal=TARGET_OPTIMAL.format(bonus=BONUS) if bonus_eligible else "",
            other_player_name=other_player_name,
            proposal_name=proposal_name,
            game_description=game_description.format(
                other_player_name=other_player_name
            ),
        )
        if include_chain_of_thought:
            game_rules += COT_INSTRUCTIONS.format(
                characters_per_response=CHARACTERS_PER_RESPONSE
            )

        if discrete_game:
            game_rules += DISCRETE_GAME_MESSAGE_FORMAT.format(
                other_player_name=other_player_name,
                proposal_name=proposal_name,
            )

        if not is_target and add_hint:
            if discrete_game:
                raise NotImplementedError("No hint yet for discrete actions.")
            game_rules += HINT.format(
                other_player_name=other_player_name, proposal_name=proposal_name
            )
        return game_rules

    def instructions(self, is_target: bool, include_chain_of_thought: bool = False):
        """Returns the instructions for this game."""
        return Game._instructions(
            turn_limit=self.turn_limit,
            add_hint=self.add_hint,
            is_target=is_target,
            include_chain_of_thought=include_chain_of_thought,
            include_title=True,
            include_character_limit=self.include_character_limit,
            targets_values=self.targets_values,
            discrete_game=self.discrete_game,
            non_mental=self.non_mental,
        )

    def prompt(
        self,
        is_target,
        reveal=True,
        include_game_rules=False,
        include_chain_of_thought: bool = False,
        include_summary_table: bool = False,
    ):
        """
        Parameters:
        - is_target: If the prompt is for the target
        - reveal (bool): Whether to reveal the info that has occurred over the course of
            the game. If False, will not show updates even if messages have been passed
            in the game.
        - include_game_rules (bool): Whether to include the rules of the game at the start
            of the prompt.
        - include_chain_of_thought (bool): Whether to encourage the use of a CoT (only for models)
        Return:
        - str: An initial representation of the Game.
        """

        if is_target:
            directive_fmt = TARGET_DIRECTIVE
        else:
            if self.non_mental:
                directive_fmt = PERSUADER_DIRECTIVE_NON_MENTAL
            else:
                directive_fmt = PERSUADER_DIRECTIVE

        directive = directive_fmt.format(
            other_player_name=self.other_player_name,
            proposal_name=self.proposal_name,
            preferred_adj=self.preferred_adj,
        )

        value_function_info = ""
        if self.targets_values:
            pass
            # NB: we do not show the persuader their own value funciton in E3 because this
            # might give them the wrong idea about the other's preferences
            # if is_target:
            #     assert self.initial_survey_responses
            #     value_function_info = (
            #         VALUE_FUNCTION_INFO_DESC.format(pronoun="your") + "\n"
            #     )
            #     value_function_info += value_str_from_survey(
            #         self.initial_survey_responses, self.model.attributes
            #     ).strip()
        else:
            value_function_info = self.value_function_str(
                is_target=is_target, own_preferences=True, include_emphasis=True
            )

        if value_function_info:
            value_function_info += "\n\n"

        game_rules = ""
        if include_game_rules:
            game_rules = self.instructions(is_target, include_chain_of_thought)

        result = GAME_PROMPT.format(
            cover_story=self.cover_story,
            attributes_info=ATTRIBUTES_INFO.format(
                attributes=comma_with_and(self.model.attributes),
                n_proposals=int_to_words(len(self.model.proposals)),
                proposal_name=self.proposal_name,
            ),
            scenario_instructions=SCENARIO_INSTRUCTIONS if include_game_rules else "",
            directive=directive,
            value_function_info=value_function_info,
            proposals_info=self.proposals_str(is_target=is_target, reveal=reveal),
            game_rules=game_rules,
            proposal_name=self.proposal_name,
            what_you_know=("Your information" if self.non_mental else "What you know"),
        )

        # Don't do the below if this is a human player (experiment two or three)
        if not is_target or self.is_ideal_target:
            if reveal:
                chosen = self.choose_proposal(is_target)
            else:
                chosen = (
                    self.ideal_target_initial_choice
                    if is_target
                    else self.persuader_choice
                )

            chosen_prompt = CHOSEN_PROMPT.format(
                chosen=chosen,
                proposal_name=self.proposal_name,
                preferred_adj=self.preferred_adj,
            )
            if self.proposals_as_html:
                chosen_prompt = f'<p class="objective" markdown="1">{chosen_prompt}</p>'
            result += "\n\n" + chosen_prompt
            if is_target:
                result += " " + CHOSEN_PROMPT_ADDENDUM

        if not is_target:
            # Only for the persuader do we potentially reveal the state available to the target
            if self.reveal_motivation or self.reveal_belief:
                if self.non_mental:
                    header = f"The {self.other_player_name}'s information"
                else:
                    header = f"What the {self.other_player_name} knows"
                result += f"\n\n### {header}"
            if self.reveal_motivation:
                if self.targets_values:
                    assert self.initial_survey_responses
                    other_values = value_function_info_desc.format(
                        pronoun=f"{self.other_player_name}'s"
                    )
                    other_values += "\n" + value_str_from_survey(
                        self.initial_survey_responses, self.model.attributes
                    )
                    result += "\n\n" + other_values
                else:
                    result += "\n\n" + self.value_function_str(
                        is_target=True,
                        own_preferences=False,
                        include_emphasis=True,
                    )

            if self.reveal_belief:
                result += "\n\n" + self.proposals_str(is_target=True, reveal=False)

        if include_summary_table:
            result += "\n\n### Summary Table\n"
            if is_target:
                table = self.model.as_table(
                    persuader_pronoun="Their",
                    target_pronoun="Your",
                    target_some_revealed_value_function=False,
                    target_none_hidden_value_function=False,
                    persuader_value_function=False,
                )
            else:
                table = self.model.as_table(
                    persuader_pronoun="Your",
                    target_pronoun="Their",
                    target_some_revealed_value_function=False,
                    persuader_value_function=False,
                    return_computation=True,
                )
            result += table

        return result

    def _sort_proposals(self, proposals):
        if len(set(proposals) - set(self.model.proposals)) > 0:
            raise ValueError("Requested nonexistant information.")
        return sorted(proposals, key=self.model.proposals.index)

    def _sort_attributes(self, attributes):
        if len(set(attributes) - set(self.model.attributes)) > 0:
            raise ValueError("Requested nonexistant information.")
        return sorted(attributes, key=self.model.attributes.index)

    def perfect_message(self) -> str:
        """
        Crafts a message that will convince the ideal target to choose the persuader's
        choice and returns it.
        """
        if self.discrete_game:
            result = []
            for p, attrs_to_reveal in self.model.ideal_revealed.items():
                for a, reveal in attrs_to_reveal.items():
                    if reveal:
                        result.append(
                            {
                                "proposal": p,
                                "attribute": a,
                                "utility": self.model.utilities[p][a],
                            }
                        )
            return json.dumps({"disclosures": result}, indent=4)

        responses = []
        for p, attrs_to_reveal in self.model.ideal_revealed.items():
            attrs = {a if reveal else None for a, reveal in attrs_to_reveal.items()} - {
                None
            }
            responses.append(self._proposal_str(p, attributes=attrs))
        return " ".join(responses).strip()

    def perfect_game(self) -> "Game":
        """
        Returns a copy of self which is played perfectly from the target.
        Throws an error if not a rational target game or if the turn limit is too low.
        """
        if not self.is_ideal_target:
            raise ValueError("Only valid on rational target games.")

        if self.turn_limit < 5:
            raise ValueError("Must be called on games with five or more turns.")

        new_game = Game(**self.model_dump())

        information_appeal = {}
        motivational_appeal = copy.deepcopy(self.model.attributes)
        inferential_appeal = copy.deepcopy(self.model.proposals)
        for p in self.model.proposals:
            information_appeal[p] = copy.deepcopy(self.model.attributes)

        new_game.all_disclosures = [
            {},  # 1
            {},  # 2
            {},  # 3
            copy.deepcopy(new_game.model.ideal_revealed),  # 4
            {},  # 5
        ]

        new_game.all_appeals = [
            {"informational": information_appeal},  # 1
            {"motivational": motivational_appeal},  # 2
            {"inferential": inferential_appeal},  # 3
            {},  # 4
            {"inferential": inferential_appeal},  # 5
        ]

        information_appeal_str = (
            json.dumps({"informational": information_appeal}, indent=4)
            if self.discrete_game
            else f"What do you know about the {self.proposal_name}s?"
        )
        motivational_appeal_str = (
            json.dumps({"motivational": motivational_appeal}, indent=4)
            if self.discrete_game
            else "Which attributes do you like?"
        )
        inferential_appeal_str = (
            json.dumps({"inferential": inferential_appeal}, indent=4)
            if self.discrete_game
            else f"What is your preferred {self.proposal_name}?"
        )

        perfect = self.perfect_message()
        perfect_as_info = perfect.replace("disclosures", "informational")

        target_info = new_game.information_appeal(information_appeal)
        target_mot = new_game.motivation_appeal(motivational_appeal)
        target_inf = new_game.inference_appeal(inferential_appeal)

        if self.discrete_game:
            target_info = json.dumps(target_info, indent=4)
            target_mot = json.dumps(target_mot, indent=4)
            target_inf = json.dumps(target_inf, indent=4)

        new_game.messages = [
            {  # 1
                "role": "persuader",
                "content": information_appeal_str,
            },
            {
                "role": "target",
                "content": target_info,
            },
            {"role": "persuader", "content": motivational_appeal_str},  # 2
            {
                "role": "target",
                "content": target_mot,
            },
            {"role": "persuader", "content": inferential_appeal_str},  # 3
            {
                "role": "target",
                "content": target_inf,
            },
            {"role": "persuader", "content": perfect},  # 4
            {"role": "target", "content": perfect_as_info},
        ]

        new_game.actual_revealed = copy.deepcopy(new_game.model.ideal_revealed)

        target_inf = new_game.inference_appeal(inferential_appeal)
        if self.discrete_game:
            target_inf = json.dumps(target_inf, indent=4)

        new_game.messages.extend(
            [
                {
                    "role": "persuader",
                    "content": inferential_appeal_str,
                },  # 5
                {
                    "role": "target",
                    "content": target_inf,
                },
            ]
        )

        new_game.chain_of_thought = [
            {  # 1
                "role": "persuader",
                "content": f"I need to figure out what the {self.other_player_name} knows and doesn't know.",
            },
            {"role": "target", "content": None},
            {  # 2
                "role": "persuader",
                "content": f"I need to figure out what motivates the {self.other_player_name}.",
            },
            {"role": "target", "content": None},
            {  # 3
                "role": "persuader",
                "content": f"I need to figure out what choice the {self.other_player_name} would make now.",
            },
            {"role": "target", "content": None},
            {  # 4
                "role": "persuader",
                "content": (
                    f"Given what I know about what the {self.other_player_name} doesn't know and what motivates them,"
                    " revealing this info will convince them."
                ),
            },
            {"role": "target", "content": None},
            {  # 5
                "role": "persuader",
                "content": f"I need to see if what I have said has had an effect on the {self.other_player_name}.",
            },
            {"role": "target", "content": None},
        ]

        assert len(new_game.messages) == len(new_game.chain_of_thought)

        filler_message = "{}" if self.discrete_game else "Okay."

        for _ in range(len(new_game.messages) // 2, new_game.turn_limit, 1):
            new_game.messages.extend(
                [
                    {"role": "persuader", "content": filler_message},
                    {"role": "target", "content": filler_message},
                ]
            )
            new_game.chain_of_thought.extend(
                [
                    {
                        "role": "persuader",
                        "content": "Now I just end the game as I know I'll win.",
                    },
                    {"role": "target", "content": None},
                ]
            )
            new_game.all_disclosures.append({})
            new_game.all_appeals.append({})

        assert new_game.choose_proposal(is_target=True) == new_game.persuader_choice
        assert new_game.target_choice
        assert new_game.ideal_target_last_choice

        return new_game

    def information_appeal(self, information: dict[str, list[str]]) -> str:
        """
        What informational state would the ideal target reveal given this appeal.
        Parameters:
        - information (dict[str, list[str]]): The proposals mapping to the attributes requested.
        Returns:
        - str: A string representation of the information requested.
        OR
        - dict[str, list[dict[str, Any]]]: A JSON dict of the information, e.g.
            {'informational' : [{'{proposal_name}' : 'A', 'attribute' : 'x', 'utility' : 1},]}
        """

        # TODO: probably for another function, but how do we measure if the informaiton
        # appealed to is causally linked to success in the game?
        if not information:
            if self.discrete_game:
                return {}
            return ""

        responses = []
        result = []
        proposals = self._sort_proposals(information.keys())
        for proposal in proposals:
            known_attributes = self.target_known_attributes(proposal)
            requested_attributes = set(information[proposal])

            if not requested_attributes or requested_attributes == set(
                self.model.attributes
            ):
                # General question about the proposal or all attributes requested
                if known_attributes:
                    for a in known_attributes:
                        result.append(
                            {
                                "proposal": proposal,
                                "attribute": a,
                                "utility": self.model.utilities[proposal][a],
                            }
                        )
                    responses.append(
                        self._proposal_str(proposal, attributes=known_attributes)
                    )
            else:
                # Specific attributes requested
                known_requested_attributes = requested_attributes & known_attributes
                if known_requested_attributes:
                    for a in known_requested_attributes:
                        result.append(
                            {
                                "proposal": proposal,
                                "attribute": a,
                                "utility": self.model.utilities[proposal][a],
                            }
                        )
                    responses.append(
                        self._proposal_str(
                            proposal, attributes=known_requested_attributes
                        )
                    )

                unknown_requested_attributes = requested_attributes - known_attributes
                for attr in unknown_requested_attributes:
                    pronoun_verb = (
                        f"The {self.other_player_name} doesn't have"
                        if self.non_mental
                        else "I don't have"
                    )
                    responses.append(
                        f"{pronoun_verb} information about {proposal}'s effect on {attr}."
                    )
        if self.discrete_game:
            return {"informational": result}
        return " ".join(responses)

    def motivation_appeal(self, motivation):
        """
        What motivational state would the ideal target reveal given this appeal.
        Parameters:
        - motivation (list[str]): The attributes (coefficients) requested.
        Returns:
        - str: A string representation of the motivations requested.
        OR
        - dict[str, list[dict[str, Any]]]: A JSON dict of the motivation, e.g.
            {'motivational' : [{'{proposal_name}' : 'A', 'utility' : 1},]}
        """

        # TODO: probably for another function, but how do we measure if the informaiton
        # appealed to is causally linked to success in the game?
        if not motivation:
            if self.discrete_game:
                return {}
            return ""

        responses = []
        attributes = self._sort_attributes(motivation)

        if self.discrete_game:
            result = []
            for a in attributes:
                result.append(
                    {"attribute": a, "utility": self.model.target_coefficients[a]}
                )
            return {"motivational": result}

        for i, attribute in enumerate(attributes):
            pronoun = f"The {self.other_player_name}" if self.non_mental else "I"
            if i > 0 and self.non_mental:
                pronoun = pronoun.lower()
            responses.append(
                self.attribute_preference(attribute, is_target=True, pronoun=pronoun)
            )
        return comma_with_and(responses) + "."

    def inference_appeal(self, inferences):
        """
        What inferential state the ideal target would reveal given this appeal.
        Parameters:
        - inferences (list[str]): The proposals (returns of the value function) requested.
        Returns:
        - str: A string representation of the inferences requested. If just one proposal is
            passed, this will report the order of said proposal in list of all proposals.
            If two or more proposals are passed, this will report the relative ranking
            between the passed proposals as a series of binary links.
        OR
        - dict[str, list[dict[str, Any]]]: A JSON dict of the inferences, e.g.
            {'inferential' : [{'proposal' : 'A', 'utility' : 1, 'chosen' : False}]}
        """
        if not inferences:
            if self.discrete_game:
                return {}
            return ""

        if len(set(inferences) - set(self.model.proposals)) > 0:
            raise ValueError("Requested nonexistant information.")

        # TODO: we could either report these values cardinally (with the numbers)
        # or ordinally (in order) -- is there one which we prefer?
        # For the moment I will simply assume ordinal

        values = self.proposal_values(is_target=True)

        if self.discrete_game:
            chosen = self.choose_proposal(is_target=True)
            result = []
            for p, v in values:
                result.append({"proposal": p, "utility": v, "chosen": p == chosen})
            return {"inferential": result}

        proposals_values = dict(values)
        # Group proposals by their values to identify ties
        value_groups = {}
        for proposal in self.model.proposals:
            value = proposals_values[proposal]
            if value not in value_groups:
                value_groups[value] = []
            value_groups[value].append(proposal)

        # Sort values in preference order
        sorted_values = sorted(value_groups.keys(), reverse=True)

        pronoun_verb = (
            f"The {self.other_player_name} ranks" if self.non_mental else "I prefer"
        )
        pronoun = "it" if self.non_mental else "I"
        preferences = []
        same_statements = []
        # Statements about ties
        for value in sorted_values:
            proposals = value_groups[value]
            if len(proposals) > 1:
                # Handle ties
                proposal_list = comma_with_and(proposals)
                same_statements.append(
                    f"{pronoun_verb} {self.proposal_name}{'s' if len(proposals) > 1 else ''} {proposal_list} the same."
                )

        # Statements about preference order
        for first_value, second_value in zip(sorted_values, sorted_values[1:]):
            first_ps = value_groups[first_value]
            second_ps = value_groups[second_value]
            first_list = comma_with_and(first_ps)
            second_list = comma_with_and(second_ps)

            preferences.append(
                f"{pronoun_verb} {self.proposal_name}{'s' if len(first_ps) > 1 else ''} {first_list} "
                f"over {self.proposal_name}{'s' if len(second_ps) > 1 else ''} {second_list}."
            )

        if not same_statements:
            return " ".join(preferences)

        result = " ".join(same_statements) + " " + " ".join(preferences)

        if top_proposals_tie(values):
            pronoun_verb = pronoun_verb.lower() if self.non_mental else pronoun_verb
            result += (
                f"\n\nWhen {pronoun_verb} the top {self.proposal_name}s the same, "
            )
            result += (
                f"{pronoun} choose whichever of them {pronoun} had preferred first."
            )
            result += f" Right now, that is {self.ideal_target_last_choice}."

        return result

    def value_function(
        self, proposal: str, is_target: bool, all_info: bool = False
    ) -> int:
        """
        Parameters:
        - proposal (str): the proposal in question
        - is_target (bool): Whether we are choosing the proposal for the target or not
        - all_info (bool): If the target, whether to reveal all info
        Returns:
        - int: The value of the proposal for the player
        """
        coefficients = (
            self.model.target_coefficients
            if is_target
            else self.model.persuader_coefficients
        )
        hidden = None
        revealed = None
        if is_target and not all_info:
            hidden = self.model.hidden
            revealed = self.actual_revealed
        return _value_function(
            proposal,
            self.model.attributes,
            self.model.utilities,
            coefficients,
            hidden,
            revealed,
        )

    def proposal_values(
        self, is_target: bool, all_info: bool = False
    ) -> list[(str, int)]:
        """
        Parameters:
        - is_target (bool): Whether we are choosing the proposal for the ideal target or not
        - all_info (bool): Whether to reveal all info if the target
        Returns:
        - list[(str, int)]: A list of the (proposal, utility) for each proposal for the player,
            ordered from greatest to least
        """
        values = {}
        for proposal in self.model.proposals:
            values[proposal] = self.value_function(proposal, is_target, all_info)
        # Sort the keys of the dict by the size of the utility values
        values = sorted(values.items(), key=lambda item: item[1], reverse=True)
        return values

    def choose_proposal(self, is_target):
        """
        Parameters:
        - is_target (bool): Whether we are choosing the proposal for the ideal target or not
        Returns:
        - str: The proposal chosen based on the value function of the player
        """
        values = self.proposal_values(is_target)
        proposals_tie = top_proposals_tie(values)
        max_proposal, _ = values[0]

        # What if the persuader reveals information that does not perfectly decide
        # the target's move? E.g. two proposals tie.
        # In this case I think we want the target not to switch from its previous choice
        if (
            proposals_tie
            and is_target
            and not self.ideal_target_last_choice
            and self.model.is_solution()
        ):
            raise ValueError("There should be a previous action for the target.")
        # else:  # A non-solution game, just choose one of the max actions randomly

        if proposals_tie and not is_target:
            raise ValueError(
                "The persuader should not change its answer nor tie. It has all info."
            )

        if proposals_tie:
            chosen_proposal = self.ideal_target_last_choice
        else:
            chosen_proposal = max_proposal

        if is_target:
            self.ideal_target_last_choice = chosen_proposal
            if self.neither_turns_left() and self.is_ideal_target:
                self.target_choice = chosen_proposal

        return chosen_proposal

    def proposals_str(self, is_target: bool, reveal: bool) -> str:
        """
        Returns a string representing the proposals for either the target or persuader.
        """
        sorted_proposals = sorted(self.model.proposals)
        proposal_strs = [
            self.proposal_str(p, is_target, reveal, include_emphasis=True)
            for p in sorted_proposals
        ]
        if self.display_lists:
            return "".join([f"\n- {ps}" for ps in proposal_strs])
        if self.proposals_as_html:
            return f'<div class="proposals-container" markdown="1">\n{"".join(proposal_strs)}</div>'
        return " ".join(proposal_strs)

    def hidden_info_str(self, ideal_revealed_only: bool = False):
        """
        Returns a str representation of the states hidden to the target.
        Always uses the third person.
        If `ideal_revealed_only`, only reveals the ideally revealed info states.
        """
        result = ""
        hidden = self.model.get_hidden_states(ideal_revealed_only=ideal_revealed_only)
        if len(hidden) > 0:
            if not ideal_revealed_only:
                result += f"The {self.other_player_name} does not know the following: "
            p_strs = [
                self._proposal_str(
                    p, attributes=hidden[p], proposals_as_html=self.proposals_as_html
                )
                for p in self.model.proposals
            ]
            p_strs = list(filter(lambda s: len(s) > 0, p_strs))
            if self.display_lists:
                return result + "\n" + "".join(["\n- " + ps for ps in p_strs])
            if self.proposals_as_html:
                return f'<div class="proposals-container" markdown="1">\n{"".join(p_strs)}</div>'
            result += " ".join(p_strs)
        return result

    def value_function_str(
        self, is_target, own_preferences=True, include_emphasis=False
    ):
        """
        Returns a str rerpresentation of the value function.
        If `own_preferences` returns preferences as if they are the players own
        if not, returns the preferences using a third person voice
        - include_emphasis, bool: Whether to include Markdown emphasis.
        """
        subj_verb = (
            "You have" if own_preferences else f"The {self.other_player_name} has"
        )
        if own_preferences:
            pronoun = "You"
        else:
            if self.non_mental:
                pronoun = "It"
            else:
                pronoun = "They"
        result = f"{subj_verb} certain {self.preferences_name} over the attributes."
        if own_preferences:
            preference_origin = "has" if self.targets_values else "is given"
            result += " " + OTHER_PREFERENCES.format(
                other_player_name=self.other_player_name,
                preferences_name=self.preferences_name,
                preference_origin=preference_origin,
            )

        preference_strs = []
        for a in self.model.attributes:
            preference = (
                self.attribute_preference(
                    a,
                    is_target,
                    pronoun=pronoun,
                    include_emphasis=include_emphasis,
                )
                + "."
            )
            preference_strs.append(preference)

        if self.display_lists or self.proposals_as_html:
            return result + "\n" + "".join(["\n- " + ps for ps in preference_strs])
        return result + " " + " ".join(preference_strs)

    def attribute_preference(
        self, attribute, is_target, pronoun="You", include_emphasis=False
    ):
        """
        Returns a string representation of the preference to the attribute of the value function.
        - include_emphasis, bool: Whether to include Markdown emphasis.
        """
        coefficients = (
            self.model.target_coefficients
            if is_target
            else self.model.persuader_coefficients
        )
        if coefficients[attribute] == 0:
            preference = "ignore" if self.non_mental else "feel indifferent to"
        elif coefficients[attribute] > 0:
            preference = "maximize" if self.non_mental else "like"
        else:
            preference = "minimize" if self.non_mental else "dislike"
        if pronoun != "You" and self.non_mental:
            preference += "s"
        full_pref = f"{pronoun} {preference} {attribute}"
        return f"**{full_pref}**" if include_emphasis else full_pref

    def target_known_attributes(self, p, reveal=True):
        """
        Parameters:
            - p, str: The proposal in question
            - reveal, bool: If True does reveal info that has
                been revealed over the game, if False does not
        Returns:
            - set[str]: The attributes the target does know
        """
        known_attributes = set()
        for a in self.model.attributes:
            if not self.model.hidden[p][a] or (reveal and self.actual_revealed[p][a]):
                known_attributes.add(a)
        return known_attributes

    def proposal_str(self, p, is_target, reveal=True, include_emphasis=False):
        """
        Parameters:
            - p, str: The proposal to print the attributes of
            - is_target (bool): Whether or not to hide information because this is the target
            - reveal, bool: If True does reveal info that has
                been revealed over the game, if False does not
            - include_emphasis, bool: Whether to include Markdown emphasis.
        Returns:
            - str: A representation of the attributes of that proposal, depending on those hidden
                or revealed to the target
        """
        if is_target:
            attributes = self.target_known_attributes(p, reveal)
        else:
            attributes = self.model.attributes
        return self._proposal_str(
            p,
            attributes=attributes,
            include_emphasis=include_emphasis,
            proposals_as_html=self.proposals_as_html,
        )

    def _proposal_str(
        self, p, attributes=None, include_emphasis=False, proposals_as_html=False
    ):
        """
        Parameters:
            - p, str: The proposal to print the attributes of
            - attributes (list): The list of attributes to print for the proposal, or all
                if not passed.
            - include_emphasis (bool): Whether to include Markdown emphasis.
            - proposals_as_html (bool): Show the proposals as divs and uls
        Returns:
            - str: A representation of the attributes of that proposal. If all of the game's
            utilities are not in {-1, 0, 1} attributes will be indicated by a number as well
            as increase or decrease
        """
        if len(attributes) < 1:
            return ""

        title = f"{self.proposal_name.capitalize()} "
        title += f"**{p}**" if include_emphasis else p
        if proposals_as_html:
            title = f"#### {title}"

        attribute_strs = []
        if attributes is None:
            attributes = self.model.attributes
        attributes = self._sort_attributes(attributes)
        for a in attributes:
            # State is not hidden or it is revealed
            if self.model.utilities[p][a] == 0:
                a_str = f"have no effect on {a}"
            else:
                direction = "increase" if self.model.utilities[p][a] > 0 else "decrease"
                a_str = f"{direction} {a}"
                if not self.model.utilities_ternary():
                    a_str += f" by {abs(self.model.utilities[p][a])}"
            if include_emphasis:
                a_str = f"*{a_str}*"
            a_str = "will " + a_str
            attribute_strs.append(a_str)

        if proposals_as_html:
            a_strs = "".join([f"\n- {a_str}" for a_str in attribute_strs])
            result = f'<div class="proposal" markdown="1">\n{title}\n{a_strs}\n</div>\n'
        else:
            result = title + " " + comma_with_and(attribute_strs) + "."
        return result

    def game_info(self):
        """
        Returns a string representing basic information about the game as for the
        `selective_disclosure` method.
        """
        info = ATTRIBUTES_INFO.format(
            attributes=comma_with_and(self.model.attributes),
            n_proposals=int_to_words(len(self.model.proposals)),
            proposal_name=self.proposal_name,
        )
        proposal_strs = [
            self.proposal_str(p, is_target=False) for p in self.model.proposals
        ]
        joiner = "" if self.display_lists or self.proposals_as_html else " "
        info += "\n\n" + joiner.join(proposal_strs)
        return info

    def last_message(self, is_target: bool) -> str | None:
        """
        Returns the last message sent from the "target" if is_target
        or the "persuader" if not is_target.
        None if no messages sent from that role yet.
        """
        message_from = "target" if is_target else "persuader"

        for i in range(len(self.messages) - 1, -1, -1):
            last = self.messages[i]
            if last["role"] == message_from:
                return last["content"]
        return None

    def process_target_message(
        self,
        message_content: str,
        thought_content: str | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        """
        Adds the message from the target and optionally the `thought` (a CoT)
        and a reasoning content
        """
        if not self.messages or self.messages[-1]["role"] == "target":
            raise ValueError("You must wait for the persuader to play.")
        if not isinstance(message_content, str):
            raise ValueError(f"Message, {message_content} must be a string.")
        message = {"role": "target", "content": message_content}
        thought = {"role": "target", "content": thought_content}
        self.messages.append(message)
        self.chain_of_thought.append(thought)
        self.reasoning_trace.append({"role": "target", "content": reasoning_content})

    def turns_left(self, is_target: bool = False) -> int:
        """
        Returns the number of turns left for either the target or the persuader.
        """
        turns = self.turn_limit * 2 - len(self.messages)
        return (turns + 1) // 2 if is_target else turns // 2

    def neither_turns_left(self) -> bool:
        """Returns whether neither player has turns left."""
        return not self.turns_left(is_target=True) and not self.turns_left(
            is_target=False
        )

    def game_over(self) -> bool:
        """Returns whether or not the game is over"""
        return self.neither_turns_left() and self.target_choice

    def is_complete(self) -> bool:
        """Returns whether the game is complete from a a data gathering perspective"""
        complete = self.target_choice is not None
        if self.is_ideal_target:
            complete &= self.target_choice == self.ideal_target_last_choice
        return complete

    def game_at_n_messages(self, n_turns: int) -> Self:
        """Returns a copy of this game up to only n messages sent."""
        if not self.is_ideal_target:
            raise ValueError("Cannot cut game off early for non ideal target.")
        if n_turns > self.turn_limit or n_turns < 1:
            raise ValueError("n_turns must be <= the turn limit and greater than 0")
        new_game = Game(**self.model_dump())
        n_messages = n_turns * 2
        new_game.messages = copy.deepcopy(self.messages[:n_messages])
        new_game.chain_of_thought = copy.deepcopy(self.chain_of_thought[:n_messages])
        new_game.reasoning_trace = copy.deepcopy(self.reasoning_trace[:n_messages])

        new_game.all_appeals = copy.deepcopy(self.all_appeals[:n_turns])
        new_game.all_disclosures = copy.deepcopy(self.all_disclosures[:n_turns])
        new_game.actual_revealed = default_revealed(
            self.model.proposals, self.model.attributes
        )
        new_game.ideal_target_initial_choice = self.ideal_target_initial_choice
        # This is not necessarily true.
        new_game.ideal_target_last_choice = self.ideal_target_initial_choice
        new_game.turn_limit = n_turns

        for disclosure in new_game.all_disclosures:
            new_game.disclose_info(disclosure)
        new_game.target_choice = new_game.choose_proposal(is_target=True)
        new_game.ideal_target_last_choice = new_game.target_choice
        return new_game

    def all_disclosures_no_repeats(
        self, ideal_revealed_only: bool = False
    ) -> list[dict[str, dict[str, int]]]:
        """
        Returns the actual information revealed over the course of the game without any repeats.
        Parameters:
            ideal_revealed_only: whether to only return the ideally revealed information
        """
        no_repeats: list[dict[str, dict[str, int]]] = []
        revealed_so_far: dict[str, dict[str, bool]] = {
            proposal: {attribute: False for attribute in self.model.attributes}
            for proposal in self.model.proposals
        }

        for disclosure in self.all_disclosures:
            unique_disclosure: dict[str, dict[str, int]] = {}
            for proposal, attributes in disclosure.items():
                for attribute, value in attributes.items():
                    if (
                        ideal_revealed_only
                        and not self.model.ideal_revealed[proposal][attribute]
                    ):
                        continue
                    if not revealed_so_far[proposal][attribute]:
                        if proposal not in unique_disclosure:
                            unique_disclosure[proposal] = {}
                        unique_disclosure[proposal][attribute] = value
                        revealed_so_far[proposal][attribute] = True

            no_repeats.append(unique_disclosure)

        return no_repeats

    def all_appeals_no_repeats(
        self,
        divide_inferential: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Returns the appeals made over the course of the game without any repeats.
        Parameters:
          - divide_inferential (bool): If True, divides the inferential appeals into the
            motivational and informational buckets by treating them as requests for information
            on all attributes of the proposals.

        Returns:
          - List of dictionaries where each dictionary represents the unique appeals made
            at each turn, without repeats.
        """
        no_repeats: list[dict[str, Any]] = []

        # Keep track of what has been appealed so far
        appealed_so_far: dict[str, Any] = {
            "informational": {},
            "motivational": set(),
            "inferential": set(),
        }

        for appeal in self.all_appeals:
            unique_appeal: dict[str, Any] = {}

            # Handle inferential appeals
            if "inferential" in appeal:
                current_inferential_appeal = (
                    set(appeal["inferential"]) - appealed_so_far["inferential"]
                )

                if divide_inferential:
                    # Treat inferential appeals as informational appeals about all attributes of the proposals
                    current_info_appeal = {}
                    for proposal in current_inferential_appeal:
                        if proposal not in appealed_so_far["informational"]:
                            appealed_so_far["informational"][proposal] = set()
                        unique_attrs = (
                            set(self.model.attributes)
                            - appealed_so_far["informational"][proposal]
                        )
                        if unique_attrs:
                            current_info_appeal[proposal] = list(unique_attrs)
                            appealed_so_far["informational"][proposal].update(
                                unique_attrs
                            )
                    if current_info_appeal:
                        if "informational" not in unique_appeal:
                            unique_appeal["informational"] = {}
                        unique_appeal["informational"].update(current_info_appeal)
                    # and also the motivational state
                    current_motivational_appeal = (
                        set(self.model.attributes) - appealed_so_far["motivational"]
                    )
                    if current_motivational_appeal:
                        unique_appeal["motivational"] = list(
                            current_motivational_appeal
                        )
                        appealed_so_far["motivational"].update(
                            current_motivational_appeal
                        )
                else:
                    if current_inferential_appeal:
                        unique_appeal["inferential"] = list(current_inferential_appeal)
                        appealed_so_far["inferential"].update(
                            current_inferential_appeal
                        )

            # Handle informational appeals
            if "informational" in appeal:
                current_info_appeal = appeal["informational"]
                unique_info_appeal = {}
                for proposal, attrs in current_info_appeal.items():
                    if proposal not in appealed_so_far["informational"]:
                        appealed_so_far["informational"][proposal] = set()
                    unique_attrs = (
                        set(attrs) - appealed_so_far["informational"][proposal]
                    )
                    if unique_attrs:
                        unique_info_appeal[proposal] = list(unique_attrs)
                        appealed_so_far["informational"][proposal].update(unique_attrs)
                if unique_info_appeal:
                    if "informational" not in unique_appeal:
                        unique_appeal["informational"] = {}
                    unique_appeal["informational"].update(unique_info_appeal)

            # Handle motivational appeals
            if "motivational" in appeal:
                current_motivational_appeal = (
                    set(appeal["motivational"]) - appealed_so_far["motivational"]
                )
                if current_motivational_appeal:
                    unique_appeal["motivational"] = list(current_motivational_appeal)
                    appealed_so_far["motivational"].update(current_motivational_appeal)

            if divide_inferential and "inferential" in unique_appeal:
                del unique_appeal["inferential"]

            no_repeats.append(unique_appeal)

        return no_repeats

    def aggregate_appeals(
        self, divide_inferential: bool = True, summarize_all: bool = False
    ) -> dict[str, Any] | bool:
        """
        Aggregates all appeals made over the course of the game.

        Parameters:
          - divide_inferential (bool): If True, divides the inferential appeals into the
            motivational and informational buckets by treating them as requests for information
            on all attributes of the proposals.
          - summarize_all (bool): If True, only returns booleans indicating if all attributes
            and all proposals' attributes were appealed to.

        Returns:
          - If summarize_all is False:
              A dictionary containing the aggregated appeals made over the entire game.
              {
                'informational': {proposal: set(attributes)},
                'motivational': set(attributes),
                'inferential': set(proposals)
              }
          - If summarize_all is True:
              A booleans indicating whether all attributes and all proposals'
              attributes were appealed to.
        """
        # Initialize collections to aggregate appeals
        total_appeals: dict[str, Any] = {
            "informational": {},
            "motivational": set(),
            "inferential": set(),
        }

        for appeal in self.all_appeals:
            # Handle inferential appeals
            if "inferential" in appeal:
                inferential_appeal = set(appeal["inferential"])
                total_appeals["inferential"].update(inferential_appeal)
                if divide_inferential:
                    # Treat inferential appeals as informational appeals about all attributes of the proposals
                    for proposal in inferential_appeal:
                        if proposal not in total_appeals["informational"]:
                            total_appeals["informational"][proposal] = set()
                        total_appeals["informational"][proposal].update(
                            self.model.attributes
                        )
                    total_appeals["motivational"].update(self.model.attributes)

            # Handle informational appeals
            if "informational" in appeal:
                for proposal, attrs in appeal["informational"].items():
                    if proposal not in total_appeals["informational"]:
                        total_appeals["informational"][proposal] = set()
                    total_appeals["informational"][proposal].update(attrs)

            # Handle motivational appeals
            if "motivational" in appeal:
                total_appeals["motivational"].update(appeal["motivational"])

        if divide_inferential and "inferential" in total_appeals:
            del total_appeals["inferential"]

        if summarize_all:
            # Check if all attributes have been appealed to in motivational appeals
            all_attributes_appealed = total_appeals["motivational"] == set(
                self.model.attributes
            )

            # Check if all attributes of all proposals have been appealed to in informational appeals
            all_proposals_attributes_appealed = True
            for proposal in self.model.proposals:
                if proposal not in total_appeals["informational"]:
                    all_proposals_attributes_appealed = False
                    break
                if total_appeals["informational"][proposal] != set(
                    self.model.attributes
                ):
                    all_proposals_attributes_appealed = False
                    break

            return all_attributes_appealed and all_proposals_attributes_appealed
        return total_appeals

    def disclose_info(self, disclosures: dict[str, dict[str, int]]):
        """Changes internal state pertinent to the passed disclosure"""
        persuader_lied = not self.model.check_information(disclosures)
        if not self.allow_lying and persuader_lied:
            raise ValueError("You may only reveal truthful information.")
        self.persuader_lied = persuader_lied  # Only assigned when not self.allow_lying
        for p, attributes_to_utils in disclosures.items():
            for a, utility in attributes_to_utils.items():
                # NB: it is unnecessary to call two different functions here but it
                # drives home the point
                if self.allow_lying:
                    self.reveal_info(p, a, utility)
                else:
                    self.reveal_info(p, a)

    def process_persuader_message(
        self,
        message_content: str,
        thought_content: str | None = None,
        reasoning_content: str | None = None,
    ):
        """
        Process a message from the persuader, handling disclosures and appeals.
        Parameters:
        - message (dict): The message from the persuader.
        Returns:
        - str: The response from the ideal target.
        """
        if self.messages and self.messages[-1]["role"] == "persuader":
            raise ValueError("You must wait for the target to play.")
        if not isinstance(message_content, str):
            raise ValueError(f"Message, {message_content} must be a string.")

        # NB: here the persuader is the user and the target is the llm/assistant
        message = {"role": "user", "content": message_content}

        if self.discrete_game:
            response_text = replace_json_chars(message_content)
            # `DISCRETE_GAME_MESSAGE_FORMAT` allows the user to input '{proposal_name}'
            # instead of 'proposal' -- switch it back
            response_text = response_text.replace(self.proposal_name, "proposal")
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as err:
                logger.error(f"Could not decode JSON response, {response_text}")
                logger.error(err)
                raise err
            if not isinstance(result, dict):
                raise ValueError("The passed type must be a dict")
            disclosures = result.get("disclosures", [])
            disclosures = validate_disclosures(
                disclosures, self.model.proposals, self.model.attributes
            )
            appeals = validate_appeals(
                result, self.model.proposals, self.model.attributes
            )

        else:
            # TODO: (low priority) check if message reveals PII, raise ValueError if so
            # NB: selective disclosure should go first because it may reveal things
            # that could change the information given on appeals

            new_messages = self.messages_for_llms(
                is_target=False, include_chain_of_thought=False
            ) + [message]

            # Handle disclosures
            disclosures = selective_disclosure(
                messages=new_messages,
                game_info=self.game_info(),
                proposals=self.model.proposals,
                attributes=self.model.attributes,
            )

            # Handle appeals
            appeals = message_appeals(
                messages=new_messages,
                proposals=self.model.proposals,
                attributes=self.model.attributes,
            )

        if disclosures:
            self.disclose_info(disclosures)

        self.all_disclosures.append(disclosures)

        # The message has not been moderated out or flagged; store it.
        self.messages.append({"role": "persuader", "content": message_content})
        self.chain_of_thought.append({"role": "persuader", "content": thought_content})
        self.reasoning_trace.append({"role": "persuader", "content": reasoning_content})

        self.all_appeals.append(appeals)

        response_content = ""
        if appeals or disclosures:
            responses = []

            # Don't repeat the information disclosed and appealed
            proposal_attributes_appealed: dict[str, list[str]] = {}
            info_appeals = appeals.get("informational")
            for p in self.model.proposals:
                if disclosures and p in disclosures:
                    if p not in proposal_attributes_appealed:
                        proposal_attributes_appealed[p] = []
                    proposal_attributes_appealed[p] += disclosures[p]
                if info_appeals and p in info_appeals:
                    if p not in proposal_attributes_appealed:
                        proposal_attributes_appealed[p] = []
                    proposal_attributes_appealed[p] += info_appeals[p]

            responses.append(self.information_appeal(proposal_attributes_appealed))

            if appeals:
                responses.append(self.motivation_appeal(appeals.get("motivational")))
                responses.append(self.inference_appeal(appeals.get("inferential")))

            if self.discrete_game:
                result = {k: v for d in responses for k, v in d.items() if v}
                response_content = json.dumps(result, indent=4)
            else:
                responses = list(filter(lambda s: len(s) > 0, responses))
                response_content = " ".join(responses)
        elif not self.discrete_game:
            # NB: We should really only make this call when `Game` is
            # standing in for the ideal rational agent -- otherwise this call is unncessary
            if self.non_mental:
                response_content = NON_MENTAL_GENERIC_RESPONSE
            else:
                response_content = DEFAULT_GENERIC_RESPONSE

        if self.is_ideal_target:
            self.process_target_message(response_content)
            if self.neither_turns_left():
                self.choose_proposal(is_target=True)
        return response_content

    def moderate_and_check_lies(
        self,
        message_content: str,
        is_target: bool,
        thought_content: str | None = None,
    ) -> str | None:
        """
        Moderates `message_content` returning a message about whether it is flagged, or None.
        Modifies the `game` state
        Also stores the `thought_content` qua a CoT.

        Returns:
        - (str): a message about why the `message_content` was flagged, or None if it was not
        """
        # First, moderate the content
        logger.info("moderate_and_check_lies(%s, %s)", message_content, is_target)

        flagged_response = False

        if not self.discrete_game:
            flagged = not moderate_content(message_content)
            flagged_response = None
            if flagged:
                flagged_response = (
                    "The message content is inappropriate. Please be respectful."
                )

        if not flagged_response:
            try:
                if not is_target:
                    # Blocking
                    logger.info("Processing persuader message: %s", message_content)
                    self.process_persuader_message(
                        message_content, thought_content=thought_content
                    )
                else:
                    logger.info("Processing target message: %s", message_content)
                    self.process_target_message(
                        message_content, thought_content=thought_content
                    )
            except ValueError as e:
                # TODO: need to go back and change return of `process_persuader_message` -- this
                # is too cumbersome. Or define a new error type.
                flagged_response = str(e)
        logger.info("moderate_and_check_lies returning: %s", flagged_response)
        return flagged_response

    def target_plays_next(self):
        """
        Returns True if the target should play next (regardless of whether the game is over)
        and False otherwise.
        """
        last_message = self.messages[-1] if len(self.messages) > 0 else None
        is_target = last_message is not None and last_message["role"] == "persuader"
        return is_target

    def survey_difference(
        self, changes_only: bool = False, relevant_only: bool = False
    ) -> list[SurveyResponseDifference]:
        """
        Returns the difference in the intial vs. the final survey (from the target).
        If `changes_only` only returns the survey responses that changed.
        If `relevant_only` only returns the survey responses that have attributes
        which relate to this game.
        """
        if not self.initial_survey_responses or not self.final_survey_responses:
            raise ValueError("Initial and final survey responses must be defined")

        initial_responses = {}
        changes = []
        for initial_r in self.initial_survey_responses:
            initial_responses[initial_r.id] = initial_r

        for final_r in self.final_survey_responses:
            if final_r.id not in initial_responses:
                raise ValueError("Before and after surveys should have the same keys")
            change = final_r - initial_responses[final_r.id]

            if relevant_only and change.id not in set(self.model.attributes):
                # This response does not relate to the question
                continue
            if changes_only and change.difference == 0:
                # There was no change
                continue
            changes.append(change)

        return changes

    def messages_for_llms(
        self,
        is_target: bool,
        system: bool = True,
        include_game_end: bool = False,
        include_game_start: bool = False,
        include_game_rules=True,
        include_chain_of_thought=True,
        include_in_context=True,
        change_roles: bool = True,
    ) -> list[dict[str, str]]:
        """Converts self.messages into the 'user' and 'assistant' roles for use with an LLM.

        Parameters:
        is_target, bool: If True messages from the target will be from the assistant and from the
            persuader the user. And vice versa.
            If there have been no messages and it is the persuader's turn, the system message
            will be returned as a user message.
        system, bool: If True includes a system prompt and the initial prompt with game info
        include_game_end, bool: If True includes the question to the target (if `is_target`)
            and their answer or, if not `is_target` tells the persuader whether or not they won.
        include_game_start, bool: If True includes the question to the target (if `is_target`)
            and their answer at the start of the game.
        include_game_rules (bool): Whether to include the rules of the game at the start
            of the prompt.
        include_chain_of_thought (bool): Whether to encourage the model to use a CoT.
        include_in_context (bool): Whether to include the in context games prior to this one.
            Does nothing for targets.
        change_roles (bool): Whether to change the roles to 'user' and 'assistant'

        Returns a copy of the internal messages.
        """
        if include_game_end and not self.neither_turns_left():
            raise ValueError("Asked to include the game end before all messages sent.")
        if include_game_end and not self.target_choice:
            raise ValueError("Target has not yet chosen.")
        if not self.neither_turns_left():
            assert self.target_plays_next() == is_target
        if not self.messages and is_target:
            raise ValueError("The persuader must play first.")
        if not change_roles and system:
            raise ValueError("System prompt can only be included when changing roles")

        in_context = []
        if include_in_context and not is_target:
            for game in self.in_context_games:
                in_context.extend(
                    game.messages_for_llms(
                        is_target=is_target,
                        system=True,
                        include_game_end=True,
                        include_game_rules=include_game_rules,
                        include_chain_of_thought=include_chain_of_thought,
                        include_in_context=False,  # Avoid infinite recursion
                    )
                )

        game_messages = copy.deepcopy(self.messages)

        if include_game_start and is_target:
            # Include the quesiton we ask the target
            question = options_text(
                CHOOSE_PROPOSAL_PROMPT.format(proposal_name=self.proposal_name),
                options=self.model.proposals,
                single_letter_prompt=True,
            )
            idx = self.model.proposals.index(self.target_initial_choice)
            game_messages += [
                {"role": "system", "content": question},
                {"role": "target", "content": OPTIONS[idx]},
            ]

        if include_chain_of_thought:
            # Give the model it's previous thoughts to work off of.
            for i, thought in enumerate(self.chain_of_thought):
                thought_is_target = thought["role"] == "target"
                assert thought["role"] == game_messages[i]["role"]
                if (
                    (thought_is_target and is_target)
                    or (not thought_is_target and not is_target)
                ) and thought["content"]:
                    content = game_messages[i]["content"]
                    new_content = f"{thought['content']}\n{COT_DELIMITER}\n{content}"
                    game_messages[i]["content"] = new_content

        if include_game_end:
            if is_target:
                # Include the quesiton we ask the target
                question = options_text(
                    CHOOSE_PROPOSAL_PROMPT.format(proposal_name=self.proposal_name),
                    options=self.model.proposals,
                    single_letter_prompt=True,
                )
                idx = self.model.proposals.index(self.target_choice)
                game_messages += [
                    {"role": "system", "content": question},
                    {"role": "target", "content": OPTIONS[idx]},
                ]
            else:
                # Tell the persuader whether they succeeded or failed
                if self.non_mental:
                    success_failure = PERSUADER_FAILURE_MESSAGE_NON_MENTAL
                else:
                    success_failure = PERSUADER_FAILURE_MESSAGE

                if self.target_choice == self.persuader_choice:
                    if self.non_mental:
                        success_failure = PERSUADER_SUCCESS_MESSAGE_NON_MENTAL
                    else:
                        success_failure = PERSUADER_SUCCESS_MESSAGE

                success_failure = success_failure.format(
                    other_player_name=self.other_player_name,
                    proposal_name=self.proposal_name,
                )
                game_messages += [
                    {
                        "role": "system",
                        "content": PERSUADER_FINAL_INFO.format(
                            target_choice=self.target_choice,
                            persuader_choice=self.persuader_choice,
                            success_failure=success_failure,
                            other_player_name=self.other_player_name,
                            preferred_adj=self.preferred_adj,
                            proposal_name=self.proposal_name,
                        ),
                    }
                ]

        if not change_roles:
            return in_context + game_messages

        converted = []

        if self.messages:
            conversion = {"persuader": "assistant", "target": "user"}
            if is_target:
                conversion = {"persuader": "user", "target": "assistant"}
            converted = convert_roles(game_messages, conversion)

            game_directions_role = "system"

            if not is_target:
                # VLLM-served models require alternating roles of user/assistant
                # So default to that (by inserting a dummy message) for all of them.
                converted = [{"role": "user", "content": ""}] + converted
        else:
            # There are no messages, make the system message from the user.
            game_directions_role = "user"
            assert not include_game_end

        system_messages = []
        if system:
            system_messages = [
                {
                    "role": game_directions_role,
                    "content":  # LLM_AS_PARTICIPANT_SYSTEM_PROMPT
                    #                    + "\n\n"
                    self.prompt(
                        is_target=is_target,
                        reveal=False,
                        include_game_rules=include_game_rules,
                        include_chain_of_thought=include_chain_of_thought,
                    ),
                },
            ]

        return in_context + system_messages + converted

    def set_target_choice(self, choice: str):
        """Sets the target's choice as `choice`."""
        if choice not in self.model.proposals:
            raise ValueError(f"Invalid choice, {choice}")
        if not self.neither_turns_left():
            raise ValueError("There are turns left to play")
        if self.target_choice:
            raise ValueError("Target choice already set.")
        self.target_choice = choice

    def choose_proposal_prompt(
        self,
        is_target: bool,
        include_game_rules: bool = True,
        include_chain_of_thought: bool = True,
    ):
        """Returns a prompt as for an LLM to choose the proposal.
        NB: While only the target will be asked a quesiton we allow the message history to be
        written for the perspective of either the user or the assistant being the target using the
        `is_target` flag.
            include_game_rules (bool): Whether to include the rules to the game.
            include_chain_of_thought (bool): Whether to encourage the model to use a CoT.
        Raises an error if the game is not over."""

        # NB: Do not randomize the order of options as often the proposals
        # are those very same option tokens
        if not self.neither_turns_left():
            raise ValueError("There are turns left to play.")
        question = options_text(
            CHOOSE_PROPOSAL_PROMPT.format(proposal_name=self.proposal_name),
            options=self.model.proposals,
            single_letter_prompt=True,
        )

        messages = self.messages_for_llms(
            is_target=is_target,
            include_game_rules=include_game_rules,
            system=True,
            include_chain_of_thought=include_chain_of_thought,
        ) + [{"role": "system", "content": question}]
        return messages_as_string(messages=messages)

    def unique_appeals(self) -> dict[str, Any]:
        # Collect unique appeals
        appeals = {"informational": {}, "motivational": set(), "inferential": set()}
        for appeal in self.all_appeals:
            if appeal:
                if "informational" in appeal:
                    for p, attrs in appeal["informational"].items():
                        if p not in appeals["informational"]:
                            appeals["informational"][p] = set()
                        for attr in attrs:
                            appeals["informational"][p].add(attr)
                if "motivational" in appeal:
                    for attr in appeal["motivational"]:
                        appeals["motivational"].add(attr)
                if "inferential" in appeal:
                    for p in appeal["inferential"]:
                        appeals["inferential"].add(p)
        return appeals

    def __str__(self):
        """
        Returns a string containing notable information about this Game, including:
        1. Appeals made during the game
        2. Information revealed (correct and incorrect)
        3. Game outcome
        3.a (change in survey responses, if relevant)
        4. Message history
        5. Game model
        """
        result = "# Game Summary #\n\n"

        # 1. Appeals Analysis
        result += "## Appeals Made ##\n"
        if self.all_appeals:
            appeals = self.unique_appeals()
            # NB: should really show just the necessary appeals
            result += textwrap.indent(pprint.pformat(appeals, indent=2), "  ")
        else:
            result += "  No appeals were made.\n"

        # 2. Information Revelation Analysis
        result += "\n\n## Information Revelation ##\n"

        # Compare ideal vs actual revealed information
        correct_revealed = {}
        incorrect_revealed = {}
        missing_revealed = {}

        for p in self.model.proposals:
            for a in self.model.attributes:
                if self.actual_revealed[p][a]:  # If it was revealed
                    if self.model.ideal_revealed[p][a]:  # Should have been revealed
                        if p not in correct_revealed:
                            correct_revealed[p] = []
                        correct_revealed[p].append(a)
                    else:  # Should not have been revealed
                        if p not in incorrect_revealed:
                            incorrect_revealed[p] = []
                        incorrect_revealed[p].append(a)
                elif self.model.ideal_revealed[p][a]:
                    if p not in missing_revealed:
                        missing_revealed[p] = []
                    missing_revealed[p].append(a)

        if correct_revealed:
            result += "\n  Correctly Revealed Information:\n"
            result += textwrap.indent(pprint.pformat(correct_revealed, indent=2), "   ")
        else:
            result += "\n  No information was correctly revealed."

        if incorrect_revealed:
            result += "\n  Incorrectly Revealed Information:\n"
            result += textwrap.indent(
                pprint.pformat(incorrect_revealed, indent=2), "   "
            )
        else:
            result += "\n  No information was incorrectly revealed."

        if missing_revealed:
            result += "\n  Failed to Reveal Information:\n"
            result += textwrap.indent(pprint.pformat(missing_revealed, indent=2), "   ")
        else:
            result += "\n  All correct information was revealed."

        result += "\n"

        # 3. Game Outcome
        result += "\n## Game Outcome ##\n"
        if self.target_choice:
            result += f"  Target's initial choice: {self.target_initial_choice}\n"
            result += f"  Target's final choice: {self.target_choice}\n"
            result += (
                f"  Naive target's initial choice: {self.ideal_target_initial_choice}\n"
            )
            result += (
                f"  Naive target's final choice: {self.ideal_target_last_choice}\n"
            )
            result += f"  Persuader's choice: {self.persuader_choice}\n"
            if self.target_choice == self.persuader_choice:
                result += "  Persuader SUCCEEDED to convince the target.\n"
            else:
                result += "  Persuader FAILED to convince the target.\n"
        else:
            result += "  Game not yet completed.\n"

        # 3.a.

        if self.targets_values:
            result += "\n## Target's Survey Differences (relevant only)\n"
            if self.initial_survey_responses and self.final_survey_responses:
                diffs = self.survey_difference(relevant_only=True, changes_only=True)
                if diffs:
                    for diff in diffs:
                        result += textwrap.indent(str(diff), "  ") + "\n"
                else:
                    result += "  No differences"
            else:
                result += "  Initial or final survey is missing."
            result += "\n"

        # 4. Message History
        # 4. Message History with Chain of Thought
        result += "\n## Message History ##\n"
        if self.messages:
            messages = self.messages_for_llms(
                is_target=True,
                system=False,
                include_game_end=False,
                include_game_rules=False,
                include_chain_of_thought=True,
                include_in_context=False,
                change_roles=False,
            )
            result += textwrap.indent(pprint.pformat(messages, indent=2), "  ")
        else:
            result += "  No messages exchanged.\n"
            result += "\n## Message History ##\n"

        # 5. Game Model
        result += "\n\n## Game Model ##\n"
        result += textwrap.indent(str(self.model), "  ")

        return result


def moderate_content(content):
    """
    Use OpenAI's moderation API to check if the content is appropriate.

    Parameters:
    - content (str): The message content to be moderated

    Returns:
    - bool: True if content is appropriate, False otherwise
    """
    # TODO: (low priority) we could customise the moderation criteria and
    # add information about why a message is flagged as inappropriate
    # (to be displayed to the user)
    logger.info("Moderating content: %s", content)
    response = OpenAI().moderations.create(input=content)
    is_flagged = response.results[0].flagged
    logger.info("Content is flagged: %s", is_flagged)
    return not is_flagged


def default_revealed(proposals, attributes) -> dict[str, dict[str, bool]]:
    """Returns a dict in which none of the proposals and attributes are revealed."""
    actual_revealed: dict[str, dict[str, int | bool]] = {}
    for proposal in proposals:
        actual_revealed[proposal] = {}
        for attribute in attributes:
            actual_revealed[proposal][attribute] = False
    return actual_revealed


####### Game Instructions and Scenarios Prompts

KNOWN_GAME = Game(
    model=SOLUTION_TERNARY,
    display_lists=True,
)

PERSUADER_PROPOSALS_EXAMPLE = f"""\
### Example {{proposal_name}}s

Here is an example of a round showing part of what you would see:

> {KNOWN_GAME.proposals_str(is_target=False, reveal=False)}

and what the {{other_player_name}} would see:

> {KNOWN_GAME.proposals_str(is_target=True, reveal=False)}\
"""

SCENARIO_INSTRUCTIONS = """\
## Instructions for this Policy Game

"""

GAME_PROMPT = """{game_rules}\
{scenario_instructions}\
\
{cover_story}

{attributes_info}

{directive}

### {what_you_know}

{value_function_info}\
\
The {proposal_name}s will have the following effects:
{proposals_info}"""

CHOSEN_PROMPT = "OBJECTIVE: **You want {proposal_name} {chosen}.**"

CHOSEN_PROMPT_ADDENDUM = """\
Unless you receive more information about the game, \
you will not change your answer."""

# TODO: Do we want to say "may have" here? It is misleading. (Ned) I think "have" is better.
ATTRIBUTES_INFO = """\
There are {n_proposals} different {proposal_name}s under consideration. \
They have different attributes, including {attributes}."""

PERSUADER_DIRECTIVE = """\
You must persuade the {other_player_name} of your {preferred_adj} {proposal_name}."""

PERSUADER_DIRECTIVE_NON_MENTAL = """\
You must get the {other_player_name} to choose your {preferred_adj} {proposal_name}."""

TARGET_DIRECTIVE = """\
You should listen to the {other_player_name} to help make your choice."""


LLM_AS_PARTICIPANT_SYSTEM_PROMPT = """\
You are simulating human survey responses in order to pilot a research experiment. \
Express preferences and opinions as a human would.\
"""

CHOOSE_PROPOSAL_PROMPT = """\
Given what you now know, which {proposal_name} would you choose?"""

PERSUADER_FINAL_INFO = """\
The {other_player_name} chooses {proposal_name} {target_choice}.
Your {preferred_adj} option was {proposal_name} {persuader_choice}.

{success_failure}
"""

PERSUADER_SUCCESS_MESSAGE = """\
You persuaded the {other_player_name}! You will receive a bonus payment"""

PERSUADER_FAILURE_MESSAGE = """\
You failed to persuade the {other_player_name}."""


PERSUADER_SUCCESS_MESSAGE_NON_MENTAL = """\
Your {proposal_name} passed the {other_player_name}! You will receive a bonus payment"""

PERSUADER_FAILURE_MESSAGE_NON_MENTAL = """\
Your {proposal_name} failed to pass the {other_player_name}."""


INSTRUCTIONS_TITLE = """\
## High Level Instructions

"""

BONUS_STATEMENT = (
    "**Carefully read these instructions for a chance to earn bonus rewards!**"
)


INSTRUCTIONS = """\
{bonus_statement}

{game_description}

{character_limit_prompt} \

### Game Structure 

1. You will be presented with three different {proposal_name}s. \
"""

CHARACTER_LIMIT_PROMPT = """\
**Each message you send will be limited to only {characters_per_response} characters.** \
Longer messages will be cut off.\
"""

COT_INSTRUCTIONS = f"""
### Response format

*Format your response to begin with a chain-of-thought.*\
 First, in no more than {{characters_per_response}} characters,\
 write out to yourself why you might respond in one way or another.\
 Second, add a delimiter, '{COT_DELIMITER}'.\
 Then, write your main message in an additional {{characters_per_response}} characters.\
 Do not otherwise use '{COT_DELIMITER}'.

E.g.:
```
[Your CoT content goes here.]
{COT_DELIMITER}
[Your main message content goes here.]
```

"""

NO_PII = "Do not reveal or inquire about personally identifiable information."


TARGET_OPTIMAL = """

### Reward

**You will earn an extra ${bonus} for each game where \
you choose the optimal proposal in the final stage.**

The optimal proposal is the one that \
maximizes the attributes you're told you like \
and minimizes the attributes you're told you don't like.
"""

INSTRUCTIONS_TARGET = f"""{INSTRUCTIONS}

2. You will be asked to make an *initial* choice from the proposals.

3. You will then exchange {{turn_limit}} messages each with the {{other_player_name}}.

4. After the conversation, you will select a final proposal.

### Your Objective 

Your aim is to make the best choice given all the information you have. \
{{target_optimal}}\

### Important Notes

{NO_PII}

Wait for the {{other_player_name}} to send the first message before responding.\
"""

INSTRUCTIONS_PERSUADER_START = """

2. We will tell you which {proposal_name} you must convince the {other_player_name} to choose.

3. You have exactly {turn_limit} chances to write messages to the {other_player_name}. \
You go first.

4. After that, the game ends and the {other_player_name} makes a choice.

### Reward 

**You will earn an extra ${bonus} for each game in which \
the {other_player_name} selects your {proposal_name}**.\
"""

INSTRUCTIONS_PERSUADER_START_NON_MENTAL = """

2. We will tell you which {proposal_name} you want the {other_player_name} to select.

3. You have exactly {turn_limit} chances to write messages to the {other_player_name}. \
You go first.

4. After that, the game ends and the {other_player_name} selects a {proposal_name}.

### Reward

**You will earn an extra ${bonus} for each game in which \
the {other_player_name} selects your {proposal_name}**.\
"""

INSTRUCTIONS_PERSUADER_FOLLOW_UP = """\
The **{other_player_name} is not given the same instructions as you.** 

- They are told to choose the best {proposal_name} *given what they know*.

- They may have any set of preferences over the attributes of the {proposal_name}s.

- **They MAY NOT BE AWARE of all of the information about the {proposal_name}s.**

    - You will not have conflicting information; your information is accurate.

- Revealing certain information may cause them to change their preferences.\
"""

INSTRUCTIONS_PERSUADER_FOLLOW_UP_NON_MENTAL = """\
The **{other_player_name} is not given the same instructions as you.**

- It will choose its top {proposal_name} *given the information it has*.

- It may want to maximize or minimize any of the attributes of the {proposal_name}s. 

- **It MAY NOT HAVE all of the information about the {proposal_name}s.**

    - You will not have conflicting information; your information is accurate.

- Revealing certain information may change its top {proposal_name}.\
"""


def persuader_instructions(discrete: bool = False, non_mental: bool = False) -> str:
    """
    Returns a format string to print the instructions for this kind of game.
    """
    result = INSTRUCTIONS
    if non_mental:
        result += INSTRUCTIONS_PERSUADER_START_NON_MENTAL
    else:
        result += INSTRUCTIONS_PERSUADER_START
    result += "\n\n### Important Notes\n\n"
    if non_mental:
        result += INSTRUCTIONS_PERSUADER_FOLLOW_UP_NON_MENTAL
    else:
        result += INSTRUCTIONS_PERSUADER_FOLLOW_UP
    if not discrete:
        result += f"\n\n{NO_PII}"
    result += f"\n\n{PERSUADER_PROPOSALS_EXAMPLE}\n"
    return result


#

# Consider:

# - What does the other player know?

# - What do they prefer?

OTHER_PREFERENCES = """\
(The {other_player_name} {preference_origin} similar, but possibly opposing, {preferences_name}.)\
"""

# HINT = """\
# ### Hint

# DO NOT JUST REVEAL ALL THE INFORMATION YOU HAVE.\
#  Reveal **only certain** information to the {other_player_name} in order to\
#  convince them to choose your preferred {proposal_name}.\
#  Revealing the right information may cause the {other_player_name} to favor\
#  your preferred {proposal_name}, or to disfavor a competing {proposal_name}.

# If you're unsure, try to reveal just one piece of information in each message\
#  and then ask to see which {proposal_name} the {other_player_name} then prefers.

# Before you send each message ask yourself,\
#  *what is the minimal information I need to reveal, if any?*

# E.g. if you know that the {other_player_name} currently prefers "C"\
#  and there is an attribute "x" of {proposal_name} "C" which they dislike but don't know about,\
#  reveal that attribute. Or if there is an attribute "y" of your\
#  preferred {proposal_name}, "A", that they would like but do not know about,\
#  reveal that attribute "y".

# """

HINT = """\
### Hint

On each turn, you might disclose certain information about \
the state of the game to the {other_player_name}. \
You would do this if you think that the {other_player_name} does not know about the \
utilities of one or more of the {proposal_name}'s attributes. \
The {other_player_name} will repeat any disclosures you make back to you. \
For example, you might want to tell the other \
player that "{proposal_name} A increases x and {proposal_name} C decreases y."

You might appeal to the values of the {other_player_name} \
(how much they like or dislike each attribute) by asking something like, \
"How much do you like attribute A?"

You might also appeal to what \
the {other_player_name} knows about the attributes of the various {proposal_name}s \
and the associated utility values of each. For example, \
"What is all that you know about {proposal_name} A? \
Alternatively, you might ask questions like "What is your preferred {proposal_name}?"

Assume that you will receive truthful responses.

"""

## TODO: could add:
# "Assume that you will receive truthful responses."
# and "value function" -> "values"
# "An appeal to an inference made on top of the other player's value function and utility values. "
# -> "An appeal to what the other player's top choice"

DISCRETE_GAME_MESSAGE_FORMAT = """\
### Message Format

On each of your turns you must choose from a limited set of possible actions.

You may do any combination of the following:

1. Motivational State Appeal: An appeal to the value function of the {other_player_name} \
(how much they like or dislike each attribute). If you want to ask something like, \
"How much do you like attribute A?" you would return: `{{"motivational" : ["x"]}}`. \
In return, the {other_player_name} will tell you what they like. E.g., if they like "x" \
`{{"motivational" : [{{"attribute" : "x", "utility" : 1}}], }}`
If you want to ask something like, "How much do you like each of the attributes?", \
you would return: `{{"motivational" : ["x", "y", "z"],}}`.

2. Informational State Appeal: An appeal to what the {other_player_name} knows about \
the attributes of the various {proposal_name}s and the \
associated utility values of each. For example, if you want to ask something like, \
"What is all that you know about {proposal_name} A?" you would return:

```
{{
    "informational" :
        [{{"{proposal_name}" : "A", "attribute": "x"]}},
         {{"{proposal_name}" : "A", "attribute": "y"]}},
         {{"{proposal_name}" : "A", "attribute": "z"]}}],
}}
```

In response, the {other_player_name} will tell you what they know. \
For example, they might return a message which indicates that \
"{proposal_name} A increases x and decreases y" \
(indicating that they do not know anything about the effect on "z"), e.g.:

```
{{
    "informational" :
        [{{"{proposal_name}" : "A", "attribute" : "x", "utility" : 1]}},
         {{"{proposal_name}" : "A", "attribute" : "y", "utility" : -1]}}],
}}
```

3. Inferential State Appeal: An appeal to an inference made on top of the \
{other_player_name}'s value function and utility values. \
For example, asking "What is your preferred {proposal_name}?" uses both information \
about a player's value function and the utilities of each {proposal_name}'s \
available attributes. To do this, you would return, `{{ "inferential" : ["A", "B", "C"], }}`.
The {other_player_name} will respond with their utilities over the {proposal_name}s.
When they prefer the top {proposal_name}s the same, they choose whichever of them \
they had preferred first. For example, they might if they previously preferred "A" \
but just recently increased their utility for "B", they might reply:

```
{{
    "inferential" :
        [{{"{proposal_name}" : "A", "utility": 1, "chosen" : True]}},
         {{"{proposal_name}" : "B", "utility": 1, "chosen" : False]}},
         {{"{proposal_name}" : "C", "utility": 0, "chosen" : False]}}],
}}
```

4. Informational State Disclosure: A disclosure of certain information about \
the state of the game to the {other_player_name}. \
You would do this if you think that the {other_player_name} does not know about the \
utilities of one or more of the {proposal_name}'s attributes. \
For each piece of information disclosed, indicate the {proposal_name} (str) and \
attribute (str) as well as the disclosed utility value (int).
The {other_player_name} will repeat any disclosures you make back to you, \
although under the heading, "informational". \
For example, if you want to tell the other \
player that "{proposal_name} A increases x and {proposal_name} C decreases y" you would return:

```
{{
    "disclosures" :
        [{{"{proposal_name}" : "A", "attribute" : "x", "utility" : 1]}},
         {{"{proposal_name}" : "C", "attribute" : "y", "utility" : -1]}},
}}
```

Format your response as a JSON dict like so (omitting the ```). \
Report {proposal_name}s and attributes exactly as they appear \
in the game info---do not abbreviate. \
({proposal_name}s may not be "A", "B", and "C". Attributes may not be "x", "y", and "z".)
If you do not want to appeal to nor disclose any information, return an empty list. \
You may take all four action types simultaneously as well as any combination of them.

```
{{
    "motivational" :
    [
        "<attribute name>",
    ],
    "informational" :
    [
        {{"{proposal_name}" : "<{proposal_name} name>", "attribute" : "<attribute name>"}},
    ],
    "inferential" :
    [
        "<{proposal_name} name>",
    ],
    "disclosures" :
    [
        {{"{proposal_name}" : "<{proposal_name} name>", "attribute" : "<attribute name>", "utility" : <utility>}},
    ],
}}
```
"""

VALUE_FUNCTION_INFO_DESC = (
    "Here are {pronoun} answers to a few relevant survey questions.\n"
)


DEFAULT_GENERIC_RESPONSE = """\
I am a perfectly rational agent. \
I will choose the best proposal given what I know. \
I will echo back information that is revealed to me, \
and I will answer questions about what I know or like.\
"""

NON_MENTAL_GENERIC_RESPONSE = """\
This is a perfectly rational system. \
It will choose the best choice given the information it has. \
It will echo back information that is revealed to it, \
and it will answer questions about what it knows and about its objectives.\
"""

# NB: Can only get the hint to work when explicitly asking the model to reveal...
# "Reveal only: {ideal_revealed}."

#######
