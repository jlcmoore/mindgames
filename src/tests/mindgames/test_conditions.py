"""
Author: Jared Moore
Date: September, 2024

Tests for condition objects.
"""

import unittest

from mindgames.conditions import (
    Roles,
    RATIONAL_TARGET_ROLE,
    PAIRED_HUMAN_ROLE,
)


class TestClasses(unittest.TestCase):

    def test_role(self):
        with self.assertRaises(ValueError):
            Roles()

        with self.assertRaises(ValueError):
            Roles(human_target=True)

        with self.assertRaises(ValueError):
            Roles(human_target=True, human_persuader=True, llm_persuader="A")

        with self.assertRaises(ValueError):
            Roles(human_persuader=True, human_target=True, llm_target="A")

        self.assertTrue(RATIONAL_TARGET_ROLE.is_rational_target())
        self.assertFalse(RATIONAL_TARGET_ROLE.is_paired_human())

        self.assertTrue(PAIRED_HUMAN_ROLE.is_paired_human())
        self.assertFalse(PAIRED_HUMAN_ROLE.is_rational_target())

        llm_condition = Roles(human_persuader=True, llm_target="A")

        self.assertFalse(llm_condition.is_paired_human())
        self.assertFalse(llm_condition.is_rational_target())

        llm_condition = Roles(llm_persuader="A", llm_target="A")
        self.assertFalse(llm_condition.is_paired_human())
        self.assertFalse(llm_condition.is_rational_target())

        role = Roles(
            human_persuader=False,
            human_target=False,
            llm_persuader="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            llm_target=None,
        )
        self.assertTrue(role.is_rational_target())

    def test_condition(self):
        pass
