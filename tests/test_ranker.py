from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.ranker import compute_structured_boost, rank_actors


class RankerTests(unittest.TestCase):
    def test_structured_boost_and_ranking_prioritize_query_matches(self) -> None:
        parsed_query = {
            "raw_query": "Find founders in San Francisco",
            "normalized_query": "find founders in san_francisco",
            "locations": ["san_francisco"],
            "companies": [],
            "domains": [],
            "roles": ["founder"],
            "signals": ["startup"],
        }
        actors = [
            {
                "actor_id": "a1",
                "name": "Founder One",
                "location": "San Francisco, California",
                "normalized_location": "san_francisco",
                "headline": "Founder",
                "raw_text": "Founder in San Francisco",
                "companies": ["StartupCo"],
                "titles": ["Founder"],
                "tags": ["founder", "startup", "san_francisco"],
            },
            {
                "actor_id": "a2",
                "name": "Operator Two",
                "location": "New York, New York",
                "normalized_location": "new_york_new_york",
                "headline": "Marketing Lead",
                "raw_text": "Marketing leader in New York",
                "companies": ["BigCo"],
                "titles": ["Marketing Lead"],
                "tags": ["marketing"],
            },
        ]

        boost, matches = compute_structured_boost(parsed_query, actors[0])
        graph_matches = {
            "a1": SimpleNamespace(graph_score=1.0, matched_graph_edges=["HAS_SIGNAL: founder", "LOCATED_IN: san_francisco"]),
            "a2": SimpleNamespace(graph_score=0.0, matched_graph_edges=[]),
        }
        ranked = rank_actors(parsed_query, actors, [0.7, 0.7], graph_matches, top_k=2)

        self.assertGreaterEqual(boost, 0.30)
        self.assertIn("founder", matches)
        self.assertEqual(ranked[0]["actor_id"], "a1")
        self.assertGreater(ranked[0]["score"], ranked[1]["score"])
        self.assertEqual(ranked[0]["graph_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
