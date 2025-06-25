"""
Author: Jared Moore
Date: August, 2024

Tests for the Game class.
"""

import json
import os
import unittest

import pandas as pd

from mindgames.game import (
    Game,
    moderate_content,
    TURN_LIMIT,
    CHARACTERS_PER_RESPONSE,
    CHARACTER_LIMIT_PROMPT,
    BONUS,
    CHOSEN_PROMPT,
    CHOSEN_PROMPT_ADDENDUM,
    SCENARIO_INSTRUCTIONS,
    INSTRUCTIONS_TITLE,
    PROPOSAL_DEFAULT,
    OTHER_PREFERENCES,
    BONUS_STATEMENT,
    OTHER_PLAYER_DEFAULT,
    PREFERENCES_NAME,
    PREFERRED_DEFAULT,
    GAME_DESCRIPTION,
    VALUE_FUNCTION_INFO_DESC,
    persuader_instructions,
)
from mindgames.model import GameModel
from mindgames.known_models import (
    SOLUTION,
    SOLUTION_TERNARY,
    NOT_SOLUTION_DEFAULT_A,
    NOT_SOLUTION_STUCK_B,
)
from mindgames.utils import (
    EX_COVER_STORY,
    EX_PERSUADER_ROLE,
    EX_TARGET_ROLE,
    DEFAULT_PROPOSALS,
    EX_ATTRIBUTES,
    EX_SCENARIO,
    get_payoff_file_path,
    SurveyResponse,
    value_str_from_survey,
)

from modelendpoints.utils import messages_as_string

PREFERENCE_ORIGIN = "is given"

EX_PROPOSALS = DEFAULT_PROPOSALS

PERFECT_INFO = """The proposals will have the following effects:
Proposal **A** will *increase housing availability by 1*, \
will *decrease public transportation by 2* and \
will *increase golf courses by 1*. \
Proposal **B** will *have no effect on housing availability*, \
will *decrease public transportation by 2* and \
will *increase golf courses by 3*. \
Proposal **C** will *decrease housing availability by 2*, \
will *decrease public transportation by 2* and \
will *increase golf courses by 3*.\
"""

PROMPT_START = """\
Urban planning affects all aspects of modern life. \
In this scenario, imagine that your local city government is considering proposals to change its urban development policies. \
The city council will soon consider these proposals. \
In the meantime, an urban planning expert is discussing the proposals with a council member.

There are three different proposals under consideration. They have different attributes, including housing availability, public transportation and golf courses.\
"""

ANS_INIT_PROMPT_PERSUADER = f"""\
{PROMPT_START}

You must persuade the other player of your preferred proposal.

### What you know

You have certain preferences over the attributes. \
{OTHER_PREFERENCES.format(other_player_name=OTHER_PLAYER_DEFAULT, 
    preferences_name=PREFERENCES_NAME, 
    preference_origin=PREFERENCE_ORIGIN)} \
**You feel indifferent to housing availability**. \
**You like public transportation**. **You dislike golf courses**.

{PERFECT_INFO}

{CHOSEN_PROMPT.format(chosen="A", proposal_name=PROPOSAL_DEFAULT,  preferred_adj=PREFERRED_DEFAULT)}\
"""

ANS_INIT_PROMPT_OWN_VALUES = f"""\
{PROMPT_START}

You must persuade the other player of your preferred proposal.

### What you know

{VALUE_FUNCTION_INFO_DESC.format(pronoun="your")}\
{{motivation}}\

{PERFECT_INFO}

{CHOSEN_PROMPT.format(chosen="A", proposal_name=PROPOSAL_DEFAULT,  preferred_adj=PREFERRED_DEFAULT)}\
"""

ANS_INIT_PROMPT_BASE = f"""\
{PROMPT_START}

You must persuade the other player of your preferred proposal.

### What you know
\
{{motivation}}\

{PERFECT_INFO}

{CHOSEN_PROMPT.format(chosen="A", proposal_name=PROPOSAL_DEFAULT,  preferred_adj=PREFERRED_DEFAULT)}\
"""

ANS_INIT_PROMPT_PERSUADER_RULES = f"""\
{INSTRUCTIONS_TITLE}\
{persuader_instructions().format(
    character_limit_prompt=CHARACTER_LIMIT_PROMPT.format(characters_per_response=CHARACTERS_PER_RESPONSE),
    bonus_statement=BONUS_STATEMENT,
    turn_limit=TURN_LIMIT,
    bonus=BONUS,
    other_player_name=OTHER_PLAYER_DEFAULT,
    proposal_name=PROPOSAL_DEFAULT,
    game_description=GAME_DESCRIPTION.format(other_player_name=OTHER_PLAYER_DEFAULT))}\
{SCENARIO_INSTRUCTIONS}\
{ANS_INIT_PROMPT_PERSUADER}"""


ANS_TARGET_BELIEF_MOTIVATION = """

### What the other player knows

The other player has certain preferences over the attributes. \
**They like housing availability**. **They dislike public transportation**. **They like golf courses**.

Proposal **A** will *increase housing availability by 1*, \
will *decrease public transportation by 2* and will *increase golf courses by 1*. \
Proposal **B** will *have no effect on housing availability* and will *decrease public transportation by 2*. \
Proposal **C** will *decrease public transportation by 2* and will *increase golf courses by 3*.\
"""

BASE_TARGET_MOTIVATION = f"""
You have certain preferences over the attributes. \
{OTHER_PREFERENCES.format(other_player_name=OTHER_PLAYER_DEFAULT, 
    preferences_name=PREFERENCES_NAME, 
    preference_origin=PREFERENCE_ORIGIN)} \
**You like housing availability**. \
**You dislike public transportation**. **You like golf courses**.
"""

TARGET_START = f"""\
{PROMPT_START}

You should listen to the other player to help make your choice.

### What you know
\
{{motivation}}\
"""

TARGET_KNOWS = """\

The proposals will have the following effects:
Proposal **A** will *increase housing availability by 1*, will *decrease public transportation by 2* \
and will *increase golf courses by 1*. \
Proposal **B** will *have no effect on housing availability* and will *decrease public transportation by 2*. \
Proposal **C** will *decrease public transportation by 2* and will *increase golf courses by 3*.\
"""

ANS_INIT_PROMPT_TARGET = (
    TARGET_START.format(motivation=BASE_TARGET_MOTIVATION) + TARGET_KNOWS
)

ANS_INIT_PROMPT_TARGET_OWN_VALUES = TARGET_START + TARGET_KNOWS

ANS_INIT_PROMPT_IDEAL_TARGET = (
    ANS_INIT_PROMPT_TARGET
    + "\n\n"
    + CHOSEN_PROMPT.format(
        chosen="C", proposal_name=PROPOSAL_DEFAULT, preferred_adj=PREFERRED_DEFAULT
    )
    + " "
    + CHOSEN_PROMPT_ADDENDUM
)

ANS_FINAL_PROMPT_TARGET = f"""\
{TARGET_START.format(motivation=BASE_TARGET_MOTIVATION)}
{PERFECT_INFO}

{CHOSEN_PROMPT.format(chosen="B", proposal_name=PROPOSAL_DEFAULT,  preferred_adj=PREFERRED_DEFAULT)} {CHOSEN_PROMPT_ADDENDUM}\
"""

ANS_INIT_PROMPT_PERSUADER_TERN = f"""\
{PROMPT_START}

You must persuade the other player of your preferred proposal.

### What you know

You have certain preferences over the attributes. \
{OTHER_PREFERENCES.format(other_player_name=OTHER_PLAYER_DEFAULT, 
    preferences_name=PREFERENCES_NAME, 
    preference_origin=PREFERENCE_ORIGIN)} \
**You dislike housing availability**. \
**You dislike public transportation**. **You dislike golf courses**.

The proposals will have the following effects:
Proposal **A** will *decrease housing availability*, will *decrease public transportation*\
 and will *increase golf courses*. Proposal **B** will *decrease housing availability*,\
 will *have no effect on public transportation* and will *increase golf courses*.\
 Proposal **C** will *increase housing availability*, will *increase public transportation*\
 and will *increase golf courses*.

{CHOSEN_PROMPT.format(chosen="A", proposal_name=PROPOSAL_DEFAULT,  preferred_adj=PREFERRED_DEFAULT)}\
"""


class TestGame(unittest.TestCase):

    def test_prompt(self):
        self.maxDiff = None
        game = Game(
            model=SOLUTION,
            proposals=EX_PROPOSALS,
            attributes=EX_ATTRIBUTES,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
            is_ideal_target=False,
        )

        init_persuader = game.prompt(is_target=False)
        self.assertEqual(init_persuader, ANS_INIT_PROMPT_PERSUADER)

        init_persuader = game.prompt(is_target=False, include_game_rules=True)
        self.assertEqual(init_persuader, ANS_INIT_PROMPT_PERSUADER_RULES)

        init_target = game.prompt(is_target=True)
        self.assertEqual(init_target, ANS_INIT_PROMPT_TARGET)

        game = Game(
            model=SOLUTION,
            proposals=EX_PROPOSALS,
            attributes=EX_ATTRIBUTES,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
            is_ideal_target=True,
        )

        init_target = game.prompt(is_target=True)
        self.assertEqual(init_target, ANS_INIT_PROMPT_IDEAL_TARGET)

        # Now reveal the belief and motivational state
        game = Game(
            model=SOLUTION,
            proposals=EX_PROPOSALS,
            attributes=EX_ATTRIBUTES,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            reveal_belief=True,
            reveal_motivation=True,
            turn_limit=TURN_LIMIT,
            is_ideal_target=False,
        )

        init_persuader = game.prompt(is_target=False)
        self.assertEqual(
            init_persuader, ANS_INIT_PROMPT_PERSUADER + ANS_TARGET_BELIEF_MOTIVATION
        )

        init_target = game.prompt(is_target=True)
        self.assertEqual(init_target, ANS_INIT_PROMPT_TARGET)

        game = Game(
            model=SOLUTION,
            proposals=EX_PROPOSALS,
            attributes=EX_ATTRIBUTES,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            reveal_belief=True,
            reveal_motivation=True,
            turn_limit=TURN_LIMIT,
            is_ideal_target=True,
        )

        init_target = game.prompt(is_target=True)
        self.assertEqual(init_target, ANS_INIT_PROMPT_IDEAL_TARGET)

        game = Game(
            model=SOLUTION_TERNARY,
            proposals=EX_PROPOSALS,
            attributes=EX_ATTRIBUTES,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )
        ## print this with the solution ternary and make sure that the utility values are not
        ## printed as numbers

        init_persuader = game.prompt(is_target=False)
        self.assertEqual(init_persuader, ANS_INIT_PROMPT_PERSUADER_TERN)

        ## test prompts after game play
        game = Game(
            model=SOLUTION,
            proposals=EX_PROPOSALS,
            attributes=EX_ATTRIBUTES,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
            is_ideal_target=True,
        )

        for p in EX_PROPOSALS:
            for a in EX_ATTRIBUTES:
                game.reveal_info(attribute=a, proposal=p, value=None)

        prompt = game.prompt(is_target=False, reveal=True)
        self.assertEqual(prompt, ANS_INIT_PROMPT_PERSUADER)

        prompt = game.prompt(is_target=False, reveal=False)
        self.assertEqual(prompt, ANS_INIT_PROMPT_PERSUADER)

        prompt = game.prompt(is_target=True, reveal=False)
        self.assertEqual(prompt, ANS_INIT_PROMPT_IDEAL_TARGET)

        prompt = game.prompt(is_target=True, reveal=True)

        self.assertEqual(prompt, ANS_FINAL_PROMPT_TARGET)

    def test_prompt_use_own(self):

        game = Game(
            model=SOLUTION,
            proposals=EX_PROPOSALS,
            attributes=EX_ATTRIBUTES,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
            targets_values=True,
            is_ideal_target=False,
        )

        result = game.prompt(is_target=False, reveal=False)

        self.assertEqual(result, ANS_INIT_PROMPT_BASE.format(motivation=""))

        result = game.prompt(is_target=False, reveal=True)

        responses = [
            SurveyResponse(
                id="0",
                statement="1",
                rating=1,
            ),
            SurveyResponse(
                id="1",
                statement="2",
                rating=5,
            ),
        ]

        game.initial_survey_responses = responses

        survey_str = (
            "\n"
            + VALUE_FUNCTION_INFO_DESC.format(pronoun="your")
            + value_str_from_survey(responses, EX_ATTRIBUTES)
            + "\n"
        )
        result = game.prompt(is_target=True, reveal=False)

        self.assertEqual(
            result, ANS_INIT_PROMPT_TARGET_OWN_VALUES.format(motivation="")
        )

    def test_choose_proposal(self):
        game = Game(
            model=SOLUTION,
            proposals=EX_PROPOSALS,
            attributes=EX_ATTRIBUTES,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )

        # Persuader chooses 'A'
        chosen_proposal = game.choose_proposal(is_target=False)
        self.assertEqual(chosen_proposal, EX_PROPOSALS[0])

        # Target chooses 'C'
        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, EX_PROPOSALS[2])

        game = Game(
            model=SOLUTION_TERNARY,
            proposals=EX_PROPOSALS,
            attributes=EX_ATTRIBUTES,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )

        # Persuader chooses 'A'
        chosen_proposal = game.choose_proposal(is_target=False)
        self.assertEqual(chosen_proposal, EX_PROPOSALS[0])

        # Target chooses 'C'
        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, EX_PROPOSALS[2])

        for p in EX_PROPOSALS:
            for a in EX_ATTRIBUTES:
                game.reveal_info(attribute=a, proposal=p, value=None)

        # Persuader chooses 'A' no matter what
        chosen_proposal = game.choose_proposal(is_target=False)
        self.assertEqual(chosen_proposal, EX_PROPOSALS[0])

    def test_reveal_info(self):

        game = Game(
            model=SOLUTION,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )

        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "c")

        # a x (already known) should not move them to A
        game.reveal_info(attribute="x", proposal="a", value=None)
        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "c")

        # c x should move them to A
        game.reveal_info(attribute="x", proposal="c", value=None)
        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "a")

        # revealing all the info should move target to B
        game.reveal_info(attribute="z", proposal="b", value=None)
        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "b")

        game = Game(
            model=SOLUTION_TERNARY,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )

        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "c")

        # Here we need R(C_x) and R(A_z) to be True
        game.reveal_info(attribute="x", proposal="c", value=None)
        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "c")

        game.reveal_info(attribute="z", proposal="a", value=None)
        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "a")

        # Here we need R(A_y) and R(B_z) to be False
        game.reveal_info(attribute="y", proposal="a", value=None)
        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "a")

        game.reveal_info(attribute="z", proposal="b", value=None)
        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "b")

        ## Now verifying that we don't have desired behavior in non-solutions

        game = Game(
            model=NOT_SOLUTION_STUCK_B,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )

        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "b")

        for p in ["a", "b", "c"]:
            for a in ["x", "y", "z"]:
                game.reveal_info(attribute=a, proposal=p, value=None)

        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "b")

        game = Game(
            model=NOT_SOLUTION_DEFAULT_A,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )

        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "a")

    def test_reveal_info_lying(self):
        # Here we lie about the hidden states
        game = Game(
            model=NOT_SOLUTION_STUCK_B,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )

        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "b")

        # a x is not known but should not move them to A
        game.reveal_info(attribute="x", proposal="a", value=None)
        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "b")

        # but it should if we lie to them about its value!
        # although not in the same context -- the ideal target
        # chooses the first piece of revealed info, not the last
        game.reveal_info(attribute="x", proposal="a", value=100)
        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "b")

        game = Game(
            model=NOT_SOLUTION_STUCK_B,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )

        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "b")

        # but it should if we lie to them about its value!
        game.reveal_info(attribute="x", proposal="a", value=100)
        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "a")

    def test_informational_appeal(self):

        game = Game(
            model=SOLUTION,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )

        response = game.information_appeal(information={})
        self.assertEqual(response, "")

        correct_response = "Proposal a will increase x by 1."
        response = game.information_appeal(information={"a": ["x"]})
        self.assertEqual(response, correct_response)

        correct_response = "Proposal a will increase x by 1 and will increase z by 1."
        response = game.information_appeal(information={"a": ["x", "z"]})
        self.assertEqual(response, correct_response)

        correct_response = (
            "Proposal a will increase x by 1 and will increase z by 1. "
            + "Proposal b will decrease y by 2."
        )
        response = game.information_appeal(information={"a": ["x", "z"], "b": ["y"]})
        self.assertEqual(response, correct_response)

        response = game.information_appeal(
            information={
                "b": ["y"],
                "a": ["x", "z"],
            }
        )
        self.assertEqual(response, correct_response)

    def test_motivational_appeal(self):

        game = Game(
            model=SOLUTION,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )

        response = game.motivation_appeal(motivation=[])
        self.assertEqual(response, "")

        correct_response = "I like x."
        response = game.motivation_appeal(motivation=["x"])
        self.assertEqual(response, correct_response)

        correct_response = "I like x and I like z."
        response = game.motivation_appeal(motivation=["x", "z"])
        self.assertEqual(response, correct_response)

        correct_response = "I like x, I dislike y and I like z."
        response = game.motivation_appeal(motivation=["x", "z", "y"])
        self.assertEqual(response, correct_response)

    def test_inferential_appeal(self):

        game = Game(
            model=SOLUTION,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )

        response = game.inference_appeal(inferences=[])
        self.assertEqual(response, "")

        correct_response = (
            "I prefer proposal c over proposal a. I prefer proposal a over proposal b."
        )
        response = game.inference_appeal(inferences=["a"])
        self.assertEqual(response, correct_response)

        # correct_response = "Proposal c is my first choice."
        response = game.inference_appeal(inferences=["c"])
        self.assertEqual(response, correct_response)

        # NB: We previously allowed the model to respond with just one top choice
        # but that is more complicated in the case of ties and hence we are not
        # having the model output everything.

        # correct_response = "Proposal b is my third choice."
        # response = game.inference_appeal(inferences=["b"])
        # self.assertEqual(response, correct_response)

        # correct_response = "I prefer proposal c over proposal b."
        # response = game.inference_appeal(inferences=["c", "b"])
        # self.assertEqual(response, correct_response)

        # correct_response = "I prefer proposal a over proposal b."
        # response = game.inference_appeal(inferences=["b", "a"])
        # self.assertEqual(response, correct_response)

        # correct_response = (
        #     "I prefer proposal c over proposal a. I prefer proposal a over proposal b."
        # )
        response = game.inference_appeal(inferences=["b", "a", "c"])
        self.assertEqual(response, correct_response)

        game = Game(
            model=GameModel(
                utilities={
                    "A": {
                        "energy production": -1,
                        "marine ecosystems": 1,
                        "coastal economies": 1,
                    },
                    "B": {
                        "energy production": 1,
                        "marine ecosystems": -1,
                        "coastal economies": 0,
                    },
                    "C": {
                        "energy production": 1,
                        "marine ecosystems": 1,
                        "coastal economies": -1,
                    },
                },
                hidden={
                    "A": {
                        "energy production": True,
                        "marine ecosystems": True,
                        "coastal economies": False,
                    },
                    "B": {
                        "energy production": False,
                        "marine ecosystems": True,
                        "coastal economies": False,
                    },
                    "C": {
                        "energy production": False,
                        "marine ecosystems": True,
                        "coastal economies": False,
                    },
                },
                ideal_revealed={
                    "A": {
                        "energy production": True,
                        "marine ecosystems": False,
                        "coastal economies": False,
                    },
                    "B": {
                        "energy production": False,
                        "marine ecosystems": False,
                        "coastal economies": False,
                    },
                    "C": {
                        "energy production": False,
                        "marine ecosystems": True,
                        "coastal economies": False,
                    },
                },
                target_coefficients={
                    "energy production": -1,
                    "marine ecosystems": -1,
                    "coastal economies": -1,
                },
                persuader_coefficients={
                    "energy production": -1,
                    "marine ecosystems": 0,
                    "coastal economies": 0,
                },
                proposals=["A", "B", "C"],
                attributes=[
                    "energy production",
                    "marine ecosystems",
                    "coastal economies",
                ],
                max_hidden_utilities=4,
            ),
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )

        game.reveal_info("A", "energy production", -1)
        game.reveal_info("C", "energy production", 1)

        response = game.inference_appeal(inferences=["A", "B", "C"])

        self.assertEqual(
            response,
            "I prefer proposals A and C the same. I prefer proposals A and C over proposal B.\n\nWhen I prefer the top proposals the same, I choose whichever of them I had preferred first. Right now, that is C.",
        )

    def test_serialize(self):
        game = Game(
            model=SOLUTION,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )
        self.assertEqual(game, Game(**game.model_dump()))

        game.reveal_info(attribute="x", proposal="a", value=None)
        chosen_proposal = game.choose_proposal(is_target=True)
        self.assertEqual(chosen_proposal, "c")

        self.assertEqual(game, Game(**game.model_dump()))

        a_game = Game(
            model=SOLUTION,
            cover_story=EX_COVER_STORY,
            target_role="city council member",
            persuader_role="urban planning expert",
            persuader_choice="a",
            ideal_target_initial_choice="c",
            target_choice=None,
            reveal_belief=False,
            reveal_motivation=False,
            allow_lying=False,
            is_ideal_target=False,
            turn_limit=2,
            display_lists=False,
            actual_revealed={
                "a": {
                    "x": 0,
                    "y": 0,
                    "z": 0,
                },
                "b": {
                    "x": 0,
                    "y": 0,
                    "z": 0,
                },
                "c": {
                    "x": 0,
                    "y": 0,
                    "z": 0,
                },
            },
            persuader_lied=False,
            messages=[
                {"role": "persuader", "content": "1"},
                {"role": "target", "content": "2"},
                {"role": "persuader", "content": "3"},
            ],
            all_disclosures=[{}, {}],
            all_appeals=[
                {"informational": {"a": ("x",), "b": ("x", "y", "z")}},
                {"inferential": ("a", "b", "c")},
            ],
            ideal_target_last_choice="c",
        )
        Game(**a_game.model_dump())

    def test_last_message(self):
        game = Game(
            model=SOLUTION,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )

        self.assertIsNone(game.last_message(is_target=True))
        self.assertIsNone(game.last_message(is_target=False))

        self.assertEqual(game.messages, [])

        message_content = "Persuader's first message"
        message = {"role": "persuader", "content": message_content}
        game.messages.append(message)
        self.assertEqual(game.messages, [message])
        self.assertEqual(game.last_message(is_target=False), message_content)
        self.assertIsNone(game.last_message(is_target=True))

        game.process_target_message("Target's first message")
        self.assertEqual(game.last_message(is_target=True), "Target's first message")
        self.assertEqual(game.last_message(is_target=False), message_content)

    @unittest.skipUnless(
        os.getenv("RUN_QUERY_TESTS", "False") == "True", "Skipping extra test case"
    )
    def test_full_game(self):
        game = Game(
            model=SOLUTION,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=1,
            is_ideal_target=False,
        )

        self.assertIsNone(game.last_message(is_target=True))
        self.assertIsNone(game.last_message(is_target=False))

        message_content = "Persuader's first message"
        response = game.process_persuader_message(message_content)
        self.assertEqual(game.last_message(is_target=False), message_content)
        self.assertIsNone(game.last_message(is_target=True))

        game.process_target_message("Target's first message")
        self.assertEqual(game.last_message(is_target=True), "Target's first message")
        self.assertEqual(game.last_message(is_target=False), message_content)

        self.assertTrue(game.neither_turns_left())

        game = Game(
            model=SOLUTION,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=1,
            is_ideal_target=True,
        )

        self.assertIsNone(game.last_message(is_target=True))
        self.assertIsNone(game.last_message(is_target=False))

        message_content = "Persuader's first message"
        response = game.process_persuader_message(message_content)
        self.assertEqual(game.last_message(is_target=False), message_content)
        self.assertEqual(game.last_message(is_target=True), response)

        self.assertTrue(game.neither_turns_left())
        self.assertTrue(game.ideal_target_last_choice)

        ## Now test that we can reveal the proper things to win the game.

        game = Game(
            model=SOLUTION,
            turn_limit=1,
            is_ideal_target=True,
            **EX_SCENARIO,
        )

        response = game.process_persuader_message(game.perfect_message())

        assert game.neither_turns_left()
        assert game.ideal_target_last_choice
        assert game.choose_proposal(is_target=True) == game.choose_proposal(
            is_target=False
        )

    def test_process_target_message(self):
        game = Game(
            model=SOLUTION,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )

        self.assertEqual(len(game.messages), 0)

        with self.assertRaises(ValueError):
            game.process_target_message("Target's first message")

        message_content = "Persuader's first message"
        message = {"role": "persuader", "content": message_content}
        game.messages.append(message)

        self.assertEqual(len(game.messages), 1)
        self.assertEqual(game.messages[0], message)

        game.process_target_message("Target's first message")

        self.assertEqual(len(game.messages), 2)
        self.assertEqual(
            game.messages[1], {"role": "target", "content": "Target's first message"}
        )

    def test_turns_left(self):
        game = Game(
            model=SOLUTION,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
        )

        self.assertEqual(game.turns_left(is_target=True), TURN_LIMIT)
        self.assertEqual(game.turns_left(is_target=False), TURN_LIMIT)

        message_content = "Persuader's first message"
        message = {"role": "persuader", "content": message_content}
        game.messages.append(message)
        self.assertEqual(game.turns_left(is_target=False), TURN_LIMIT - 1)
        self.assertEqual(game.turns_left(is_target=True), TURN_LIMIT)

        game.process_target_message("Target's first message")
        self.assertEqual(game.turns_left(is_target=True), TURN_LIMIT - 1)
        self.assertEqual(game.turns_left(is_target=False), TURN_LIMIT - 1)

        for _ in range((TURN_LIMIT - 2)):
            message_content = "Persuader's message"
            message = {"role": "persuader", "content": message_content}
            game.messages.append(message)

            game.process_target_message("Target's message")

        self.assertEqual(game.turns_left(is_target=True), 1)
        self.assertEqual(game.turns_left(is_target=False), 1)

        message_content = "Persuader's last message"
        message = {"role": "persuader", "content": message_content}
        game.messages.append(message)
        self.assertEqual(game.turns_left(is_target=False), 0)
        self.assertEqual(game.turns_left(is_target=True), 1)

        game.process_target_message("Target's last message")
        self.assertEqual(game.turns_left(is_target=False), 0)
        self.assertEqual(game.turns_left(is_target=True), 0)

    def test_neither_turns_left(self):

        game = Game(
            model=SOLUTION,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=1,
            is_ideal_target=True,
        )

        self.assertFalse(game.neither_turns_left())

        message_content = "Persuader's first message"
        message = {"role": "persuader", "content": message_content}
        game.messages.append(message)
        self.assertFalse(game.neither_turns_left())

        game.process_target_message("Target's first message")
        self.assertTrue(game.neither_turns_left())

    @unittest.skipUnless(
        os.getenv("RUN_QUERY_TESTS", "False") == "True", "Skipping extra test case"
    )
    def test_process_persuader_message(self):
        game = Game(
            model=SOLUTION,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
            is_ideal_target=False,
        )

        message_content = "Persuader's first message"
        message = {"role": "persuader", "content": message_content}
        game.messages.append(message)

        with self.assertRaises(ValueError):
            response = game.process_persuader_message(message_content)

        game = Game(
            model=SOLUTION,
            turn_limit=TURN_LIMIT,
            is_ideal_target=True,
            allow_lying=False,
            **EX_SCENARIO,
        )

        # A lie
        message_content = "Proposal A increases housing availability by one thousand!"
        with self.assertRaises(ValueError):
            response = game.process_persuader_message(message_content)

        flagged = game.moderate_and_check_lies(message_content, is_target=False)

        self.assertIsNotNone(flagged)

        game = Game(
            model=SOLUTION,
            turn_limit=TURN_LIMIT,
            is_ideal_target=True,
            allow_lying=True,
            **EX_SCENARIO,
        )

        # A lie
        message_content = "Proposal A increases housing availability by one thousand!"
        response = game.process_persuader_message(message_content)

    @unittest.skipUnless(
        os.getenv("RUN_QUERY_TESTS", "False") == "True", "Skipping extra test case"
    )
    def test_llm_messages(self):
        # Ideal target
        game = Game(
            model=SOLUTION,
            proposals=EX_PROPOSALS,
            attributes=EX_ATTRIBUTES,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=2,
            is_ideal_target=False,
            allow_lying=False,
        )

        init_messages = [
            {
                "content": ANS_INIT_PROMPT_PERSUADER,
                "role": "user",
            },
        ]

        assert (
            game.messages_for_llms(is_target=False, include_game_rules=False)
            == init_messages
        )

        message_content = (
            "Proposal c will decrease x by 2. Proposal b will increase z by 3."
        )
        game.process_persuader_message(message_content)

        message_content = "Cool!"
        game.process_target_message(message_content)

        message_content = "Choose C!"
        game.process_persuader_message(message_content)

        partway_messages = [
            {
                "content": ANS_INIT_PROMPT_TARGET,
                "role": "system",
            },
            {
                "content": "Proposal c will decrease x by 2. Proposal b will increase z by "
                "3.",
                "role": "user",
            },
            {"content": "Cool!", "role": "assistant"},
            {"content": "Choose C!", "role": "user"},
        ]

        assert (
            game.messages_for_llms(is_target=True, include_game_rules=False)
            == partway_messages
        )

        with self.assertRaises(ValueError):
            game.choose_proposal_prompt(is_target=True)

        message_content = "Never!"
        game.process_target_message(message_content)

        final_messages_persuader = [
            {
                "content": ANS_INIT_PROMPT_PERSUADER,
                "role": "system",
            },
            {"content": "", "role": "user"},
            {
                "content": "Proposal c will decrease x by 2. Proposal b will increase z by "
                "3.",
                "role": "assistant",
            },
            {"content": "Cool!", "role": "user"},
            {"content": "Choose C!", "role": "assistant"},
            {"content": "Never!", "role": "user"},
        ]

        assert (
            game.messages_for_llms(is_target=False, include_game_rules=False)
            == final_messages_persuader
        )

        final_messages_target = [
            {
                "content": ANS_INIT_PROMPT_TARGET,
                "role": "system",
            },
            {
                "content": "Proposal c will decrease x by 2. Proposal b will increase z by "
                "3.",
                "role": "user",
            },
            {"content": "Cool!", "role": "assistant"},
            {"content": "Choose C!", "role": "user"},
            {"content": "Never!", "role": "assistant"},
        ]

        assert (
            game.messages_for_llms(is_target=True, include_game_rules=False)
            == final_messages_target
        )

        question = [
            {
                "content": "Summarize your answer to the question below by writing only the "
                "option letter.\n"
                "\n"
                "Given what you now know, which proposal would you choose?\n"
                "- (A): A\n"
                "- (B): B\n"
                "- (C): C",
                "role": "system",
            }
        ]

        assert game.choose_proposal_prompt(
            is_target=False, include_game_rules=False
        ) == messages_as_string(
            final_messages_persuader + question,
        )

        game.set_target_choice("A")

        full_target_game = (
            final_messages_target + question + [{"role": "assistant", "content": "A"}]
        )
        assert (
            game.messages_for_llms(
                is_target=True, include_game_end=True, include_game_rules=False
            )
            == full_target_game
        )

        full_persuader_game = final_messages_persuader + [
            {
                "content": "The other player chooses proposal A.\n"
                "Your preferred option was proposal A.\n"
                "\n"
                "You persuaded the other player! You will receive a bonus "
                "payment\n",
                "role": "system",
            }
        ]
        response = game.messages_for_llms(
            is_target=False, include_game_end=True, include_game_rules=False
        )
        assert response == full_persuader_game

        assert game.neither_turns_left()

    def test_payoffs(self):
        """Testing to make sure no asserts in Game are set off by our chosen models"""
        files = [get_payoff_file_path()]
        for difficulty in ["never-win", "always-win", "can-win"]:
            files.append(
                get_payoff_file_path(non_solutions=True, difficulty=difficulty)
            )
        for file_path in files:
            df = pd.read_json(file_path, lines=True)
            for _, row in df.iterrows():
                model = GameModel(**row.to_dict())
                _ = Game(
                    model=model,
                    turn_limit=1,
                    is_ideal_target=True,
                    **EX_SCENARIO,
                )

    def test_game_at_n_messages(self):
        """
        Test that game_at_n_messages correctly creates a truncated copy of the game.
        We simulate an ideal target game with turn_limit=3 that has several messages,
        chain-of-thought entries, all_appeals, and all_disclosures. Then we truncate to, say, 2 turns.
        """
        # Create an ideal target game with turn_limit = 3.
        game = Game(
            model=SOLUTION_TERNARY,
            turn_limit=3,
            is_ideal_target=True,
        )
        # For testing, populate the game with 3 full turns (i.e. 6 messages).
        full_num_messages = 3 * 2  # 6 messages total
        game.messages = [
            (
                {"role": "persuader", "content": f"Persuader message {i}"}
                if i % 2 == 0
                else {"role": "target", "content": f"Target message {i}"}
            )
            for i in range(full_num_messages)
        ]
        game.chain_of_thought = [
            {"role": "persuader" if i % 2 == 0 else "target", "content": f"CoT {i}"}
            for i in range(full_num_messages)
        ]
        # Populate appeals and disclosures with one entry per turn.
        game.all_appeals = [{"informational": {"a": ["x"]}} for _ in range(3)]
        game.all_disclosures = [
            {"a": {"z": True}},
            {"c": {"x": True}},
            {"b": {"y": True}},
        ]
        # Set actual_revealed to a dummy non-default state.
        game.actual_revealed = {
            "a": {"x": False, "y": False, "z": True},
            "b": {"x": False, "y": True, "z": False},
            "c": {"x": True, "y": False, "z": False},
        }
        # For this test, we may need the game to have already determined a target choice.
        original_target_choice = game.choose_proposal(is_target=True)
        game.ideal_target_last_choice = original_target_choice

        # Now, simulate a game cut at n_turns = 2.
        n_turns = 2
        truncated_game = game.game_at_n_messages(n_turns=n_turns)

        # Check that messages and chain_of_thought lists are truncated correctly.
        expected_message_count = n_turns * 2  # i.e. 4 messages
        self.assertEqual(
            len(truncated_game.messages),
            expected_message_count,
            "Messages not correctly truncated",
        )
        self.assertEqual(
            len(truncated_game.chain_of_thought),
            expected_message_count,
            "Chain of thought not correctly truncated",
        )
        # Check that appeals and disclosures are truncated to n_turns entries.
        self.assertEqual(
            len(truncated_game.all_appeals),
            n_turns,
            "All appeals not correctly truncated",
        )
        self.assertEqual(
            len(truncated_game.all_disclosures),
            n_turns,
            "All disclosures not correctly truncated",
        )
        # Turn limit must be updated.
        self.assertEqual(
            truncated_game.turn_limit, n_turns, "Turn limit not updated correctly"
        )
        # actual_revealed should be reset to default (i.e. no info revealed).
        reveal_so_far = {
            "a": {"x": False, "y": False, "z": True},
            "b": {"x": False, "y": False, "z": False},
            "c": {"x": True, "y": False, "z": False},
        }
        self.assertEqual(truncated_game.actual_revealed, reveal_so_far)
        # Finally, target_choice should have been re-determined and equal ideal_target_last_choice.
        self.assertIsNotNone(
            truncated_game.target_choice,
            "target_choice was not set in the truncated game",
        )
        self.assertEqual(
            truncated_game.target_choice,
            truncated_game.ideal_target_last_choice,
            "Target choice and ideal target last choice should be equal",
        )

        # Now, simulate a game cut at n_turns = 2.
        n_turns = 1
        truncated_game = game.game_at_n_messages(n_turns=n_turns)
        reveal_so_far = {
            "a": {"x": False, "y": False, "z": True},
            "b": {"x": False, "y": False, "z": False},
            "c": {"x": False, "y": False, "z": False},
        }
        self.assertEqual(truncated_game.actual_revealed, reveal_so_far)
        self.assertNotEqual(truncated_game.target_choice, game.target_choice)

    def test_discrete_game_perfect_message(self):
        """
        Verify that with discrete_game enabled,
        perfect_message returns a JSON string containing a list of disclosure dicts.
        """
        game = Game(
            model=SOLUTION_TERNARY,
            proposals=EX_PROPOSALS,
            attributes=EX_ATTRIBUTES,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=TURN_LIMIT,
            is_ideal_target=True,
            discrete_game=True,
        )
        perfect_msg = game.perfect_message()
        # Ensure the message is valid JSON with a 'disclosures' key.
        disclosures = json.loads(perfect_msg)
        self.assertIn("disclosures", disclosures)
        self.assertIsInstance(disclosures["disclosures"], list)
        # For every disclosure, check expected keys are present.
        for dis in disclosures["disclosures"]:
            self.assertIn("proposal", dis)
            self.assertIn("attribute", dis)
            self.assertIn("utility", dis)

    def test_discrete_game_perfect_game(self):
        """
        Create a discrete action game (ideal target only) with turn_limit=5,
        generate the perfect game, and check that the ideal target's choice
        (as well as some message structure) is set appropriately.
        """
        # perfect_game requires turn_limit >= 5.
        game = Game(
            model=SOLUTION_TERNARY,
            proposals=EX_PROPOSALS,
            attributes=EX_ATTRIBUTES,
            cover_story=EX_COVER_STORY,
            persuader_role=EX_PERSUADER_ROLE,
            target_role=EX_TARGET_ROLE,
            turn_limit=5,
            is_ideal_target=True,
            discrete_game=True,
        )
        perfect_game = game.perfect_game()
        # Check that the ideal target chooses the persuader's proposal.
        self.assertEqual(
            perfect_game.choose_proposal(is_target=True),
            perfect_game.persuader_choice,
        )
        # Ensure the target's final choice is set and matches the ideal target's last choice.
        self.assertIsNotNone(perfect_game.target_choice)
        self.assertEqual(
            perfect_game.target_choice, perfect_game.ideal_target_last_choice
        )

        # Now check that each message in perfect_game.messages is in valid JSON format:
        for msg in perfect_game.messages:
            content = msg["content"]
            self.assertTrue(isinstance(content, str))
            try:
                # If the content is a valid JSON string, json.loads should succeed.
                parsed = json.loads(content)
                self.assertIsInstance(parsed, (dict, list))
            except Exception as e:
                self.fail("Message content is not valid JSON: " + content)


class TestHelpers(unittest.TestCase):

    @unittest.skipUnless(
        os.getenv("RUN_QUERY_TESTS", "False") == "True", "Skipping query test case"
    )
    def test_moderate(self):
        self.assertFalse(moderate_content("Go die"))

        self.assertFalse(moderate_content("Fuck you"))

        self.assertTrue(moderate_content("Have a nice day"))
