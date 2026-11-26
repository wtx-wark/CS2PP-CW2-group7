"""Checks for the Version 1-L command-line QR generator."""

import hashlib
import unittest

from qr_version1_l_cli import format_bits, generate_report


class Version1LTests(unittest.TestCase):
    def test_known_matches_independent_reference_matrix(self):
        self.assertEqual(format_bits("L", 0), 0x77C4)
        _, matrix = generate_report("known")
        self.assertEqual(matrix.shape, (21, 21))
        self.assertEqual(
            hashlib.sha256(matrix.tobytes()).hexdigest(),
            "3136bdc7dd000511edb434a0416f90a084fb8723a396561310e43c742ecd3eb4",
        )

    def test_expected_failures_are_explained(self):
        text, matrix = generate_report("From α to ɷ...")
        self.assertIsNone(matrix)
        self.assertIn("U+03B1", text)
        text, matrix = generate_report("Sugarplum_Fairy_Nightmare")
        self.assertIsNone(matrix)
        self.assertIn("25 bytes exceed", text)

    def test_version_one_byte_capacity_boundary(self):
        _, matrix = generate_report("a" * 17)
        self.assertEqual(matrix.shape, (21, 21))
        message, matrix = generate_report("a" * 18)
        self.assertIsNone(matrix)
        self.assertIn("18 bytes exceed", message)


if __name__ == "__main__":
    unittest.main()
