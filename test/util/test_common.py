"""Test common utilities"""

import unittest

import numpy as np


class CommonTest(unittest.TestCase):
    """Common Test"""

    def test_pc__average_pc_of_parts_is_not_equal_to_pc_of_all(self):
        one = list(range(1, 11))
        two = [1, 2, 2, 2, 2, 2, 2, 2, 2, 0]

        single = np.corrcoef(one, two)[0, 1]
        avg = np.mean(
            [np.corrcoef(one[:5], two[:5])[0, 1], np.corrcoef(one[6:], two[6:])[0, 1]]
        )

        self.assertNotAlmostEqual(single, avg)
        self.assertAlmostEqual(-0.2447, single, places=3)
        self.assertAlmostEqual(-0.0337, avg, places=3)
