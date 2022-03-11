import re
import unittest
import cp_app

from tests.constants import F1_PATH, F2_PATH
class TestIterTogether(unittest.TestCase):
    """Tests for cp_app"""

    def setUp(self) -> None:
        self.constant1 = 5
        return super().setUp()

    def test_iter_together(self):
        """Test that two files can iterate together"""
        expected = [
            ('a','a_1','a_2'),
            ('b','b_1','b_2'),
            ('c','c_1','c_2'),
            ('d','d_1','d_2')
        ]
        result = list(cp_app.iter_together(F1_PATH, F2_PATH))
        print(result)
        self.assertEqual(expected, result)