from __future__ import annotations

import unittest

from src.transform import transform_actor


class TransformActorTests(unittest.TestCase):
    def test_transform_actor_flattens_profile_and_generates_tags(self) -> None:
        actor = {
            "resolved_actor_id": "actor-123",
            "profile": {
                "name": "Sam Founder",
                "headline": "Co-Founder building vision AI products",
                "bio": "Raised seed funding from YC and mentors early founders in San Francisco.",
                "location": "San Francisco, California, United States",
            },
            "professional": {
                "current_position": {"title": "Co-Founder & CEO", "company": "VisionFlow"},
                "work_experience": [
                    {
                        "title": "Co-Founder & CEO",
                        "company_name": "VisionFlow",
                        "description": "Building computer vision systems for retail analytics.",
                    },
                    {
                        "title": "Engineer",
                        "company_name": "Stripe",
                        "description": "Worked on payments infrastructure.",
                    },
                ],
                "education": [
                    {
                        "school": "Stanford University",
                        "degree": "MS",
                        "field_of_study": "Computer Science",
                    }
                ],
                "accomplishments": {
                    "certifications": [{"name": "Google Analytics Certified", "authority": "Google"}]
                },
            },
        }

        transformed = transform_actor(actor)

        self.assertEqual(transformed["actor_id"], "actor-123")
        self.assertEqual(transformed["current_company"], "VisionFlow")
        self.assertIn("Stripe", transformed["companies"])
        self.assertEqual(transformed["normalized_location"], "san_francisco")
        self.assertIn("Stanford University", transformed["schools"])
        self.assertIn("Signals: ", transformed["search_text"])
        self.assertIn("founder", transformed["tags"])
        self.assertIn("fundraising", transformed["tags"])
        self.assertIn("computer_vision", transformed["tags"])
        self.assertIn("san_francisco", transformed["tags"])


if __name__ == "__main__":
    unittest.main()
