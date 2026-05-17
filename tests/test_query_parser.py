from __future__ import annotations

import unittest

from src.query_parser import parse_query


class QueryParserTests(unittest.TestCase):
    def test_query_parser_handles_location_company_and_skill_normalization(self) -> None:
        founder_query = parse_query("Find founders in San Francisco")
        google_query = parse_query("Show me people who worked at Google")
        mentor_query = parse_query("someone who can mentor me for my startup for vison stuff")
        ml_query = parse_query("someone from blr good with ml")

        self.assertIn("san_francisco", founder_query["locations"])
        self.assertIn("founder", founder_query["roles"])
        self.assertIn("google", google_query["companies"])
        self.assertIn("mentor", mentor_query["signals"])
        self.assertIn("startup", mentor_query["signals"])
        self.assertIn("computer_vision", mentor_query["signals"])
        self.assertIn("vision_ai", mentor_query["signals"])
        self.assertIn("bangalore", ml_query["locations"])
        self.assertIn("machine_learning", ml_query["signals"])


if __name__ == "__main__":
    unittest.main()
