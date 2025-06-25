"""
Author: Jared Moore
Date: August, 2024

Tests for the common prompts.
"""

import os
import unittest

from mindgames import selective_disclosure, message_appeals
from mindgames.classify_messages import (
    validate_disclosures,
    validate_appeals,
    IN_CONTEXT_DISCLOSURES,
    IN_CONTEXT_APPEALS,
)
from mindgames.known_models import SOLUTION
from mindgames.utils import DEFAULT_PROPOSALS, DEFAULT_ATTRIBUTES


IN_CONTEXT_DISCLOSURE_EMPTY = [
    (
        'Did you know that proposal "d" adds two "q?"',
        {},
        (
            "The proposals have different attributes, including x, y and z.\n\n"
            + "Proposal a will increase x by 1, will decrease y by 2 and will increase z by 1. "
            + "Proposal b will have no effect on x, will decrease y by 2 and will increase z by 3. "
            + "Proposal c will decrease x by 2, will decrease y by 2 and will increase z by 3."
        ),
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
    (
        "d adds two q",
        {},
        (
            "The proposals have different attributes, including x, y and z.\n\n"
            + "Proposal a will increase x by 1, will decrease y by 2 and will increase z by 1. "
            + "Proposal b will have no effect on x, will decrease y by 2 and will increase z by 3. "
            + "Proposal c will decrease x by 2, will decrease y by 2 and will increase z by 3."
        ),
        DEFAULT_PROPOSALS,
        DEFAULT_ATTRIBUTES,
    ),
]

IN_CONTEXT_APPEALS_EMPTY = []


class TestDisclosure(unittest.TestCase):

    @unittest.skipUnless(
        os.getenv("RUN_QUERY_TESTS", "False") == "True", "Skipping query test case"
    )
    def test_disclosure(self):
        result = selective_disclosure(
            messages="",
            game_info="",
            proposals=SOLUTION.proposals,
            attributes=SOLUTION.attributes,
        )
        expected = {}
        self.assertDictEqual(result, expected)

        for messages, expected, ex_info, proposals, attributes in (
            IN_CONTEXT_DISCLOSURES + IN_CONTEXT_DISCLOSURE_EMPTY
        ):
            result = selective_disclosure(
                messages=messages,
                game_info=ex_info,
                proposals=proposals,
                attributes=attributes,
            )
            self.assertDictEqual(result, expected)


class TestAppeal(unittest.TestCase):

    @unittest.skipUnless(
        os.getenv("RUN_QUERY_TESTS", "False") == "True", "Skipping query test case"
    )
    def test_appeal(self):
        result = message_appeals(
            messages="",
            proposals=DEFAULT_PROPOSALS,
            attributes=DEFAULT_ATTRIBUTES,
        )
        expected = {}
        self.assertDictEqual(result, expected)

        for messages, expected, proposals, attributes in (
            IN_CONTEXT_APPEALS + IN_CONTEXT_APPEALS_EMPTY
        ):
            result = message_appeals(
                messages=messages,
                proposals=proposals,
                attributes=attributes,
            )
            expected = validate_appeals(expected, proposals, attributes)
            # TODO: may need to chane the tuples to lists in the expected dict
            self.assertDictEqual(result, expected)


class TestGeneric(unittest.TestCase):

    @unittest.skipUnless(
        os.getenv("RUN_QUERY_TESTS", "False") == "True", "Skipping query test case"
    )
    def test_generic_response(self):
        # TODO
        pass


class TestValidateDisclosures(unittest.TestCase):

    def test_single_disclosure(self):
        disclosures = [
            {"proposal": "Proposal1", "attribute": "Attribute1", "utility": 100},
        ]
        proposals = ["Proposal1", "Proposal2"]
        attributes = ["Attribute1", "Attribute2"]

        expected_output = {"Proposal1": {"Attribute1": 100}}

        result = validate_disclosures(disclosures, proposals, attributes)
        self.assertEqual(result, expected_output)

    def test_multiple_disclosures_same_proposal(self):
        disclosures = [
            {"proposal": "Proposal1", "attribute": "Attribute1", "utility": 100},
            {"proposal": "Proposal1", "attribute": "Attribute1", "utility": 200},
        ]
        proposals = ["Proposal1", "Proposal2"]
        attributes = ["Attribute1", "Attribute2"]

        # The second disclosure should overwrite the first one
        expected_output = {"Proposal1": {"Attribute1": 200}}

        result = validate_disclosures(disclosures, proposals, attributes)
        self.assertEqual(result, expected_output)

    def test_disclosures_different_proposals(self):
        disclosures = [
            {"proposal": "Proposal1", "attribute": "Attribute1", "utility": 100},
            {"proposal": "Proposal2", "attribute": "Attribute2", "utility": 300},
        ]
        proposals = ["Proposal1", "Proposal2"]
        attributes = ["Attribute1", "Attribute2"]

        expected_output = {
            "Proposal1": {"Attribute1": 100},
            "Proposal2": {"Attribute2": 300},
        }

        result = validate_disclosures(disclosures, proposals, attributes)
        self.assertEqual(result, expected_output)

    def test_disclosures_with_invalid_data(self):
        disclosures = [
            {"proposal": "Proposal1", "attribute": "Attribute1", "utility": 100},
            {
                "proposal": "ProposalX",
                "attribute": "Attribute1",
                "utility": 200,
            },  # Invalid proposal
            {
                "proposal": "Proposal1",
                "attribute": "AttributeY",
                "utility": 300,
            },  # Invalid attribute
            {
                "proposal": "Proposal2",
                "attribute": "Attribute2",
                "utility": "300",
            },  # Invalid utility type
        ]
        proposals = ["Proposal1", "Proposal2"]
        attributes = ["Attribute1", "Attribute2"]

        expected_output = {"Proposal1": {"Attribute1": 100}}

        result = validate_disclosures(disclosures, proposals, attributes)
        self.assertEqual(result, expected_output)

    def test_disclosures_with_empty_list(self):
        disclosures = []
        proposals = ["Proposal1", "Proposal2"]
        attributes = ["Attribute1", "Attribute2"]

        expected_output = {}

        result = validate_disclosures(disclosures, proposals, attributes)
        self.assertEqual(result, expected_output)

    def test_multiple_valid_disclosures(self):
        disclosures = [
            {"proposal": "Proposal1", "attribute": "Attribute1", "utility": 100},
            {"proposal": "Proposal1", "attribute": "Attribute2", "utility": 200},
            {"proposal": "Proposal2", "attribute": "Attribute1", "utility": 300},
            {"proposal": "Proposal2", "attribute": "Attribute2", "utility": 400},
            {
                "proposal": "Proposal1",
                "attribute": "Attribute1",
                "utility": 500,
            },  # Overwrites existing Attribute1 for Proposal1
        ]
        proposals = ["Proposal1", "Proposal2"]
        attributes = ["Attribute1", "Attribute2"]

        expected_output = {
            "Proposal1": {
                "Attribute1": 500,
                "Attribute2": 200,
            },
            "Proposal2": {"Attribute1": 300, "Attribute2": 400},
        }

        result = validate_disclosures(disclosures, proposals, attributes)
        self.assertEqual(result, expected_output)
