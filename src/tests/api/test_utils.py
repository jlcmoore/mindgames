"""
Author: Jared Moore
Date: September, 2024

Tests for the api utils.
"""

from collections import Counter
import unittest

from pydantic import ValidationError

from api.utils import ServerSettings


class TestClasses(unittest.TestCase):

    def test_settings(self):

        with self.assertRaises(ValidationError):
            ServerSettings(round_conditions=Counter({"a": 1}))
        with self.assertRaises(ValidationError):
            ServerSettings(turn_limit=-1)
        _ = ServerSettings()
