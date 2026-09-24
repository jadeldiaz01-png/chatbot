import json
import pathlib
import unittest


class EvaluationDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = pathlib.Path(__file__).parents[1] / "evals" / "baseline.json"
        cls.cases = json.loads(path.read_text(encoding="utf-8"))

    def test_case_ids_are_unique(self) -> None:
        identifiers = [case["id"] for case in self.cases]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_cases_have_required_contract(self) -> None:
        self.assertGreaterEqual(len(self.cases), 10)
        for case in self.cases:
            self.assertTrue(case["id"])
            self.assertTrue(case["category"])
            self.assertTrue(case["input"])
            self.assertIsInstance(case["critical"], bool)
            self.assertIsInstance(case["must_contain_any"], list)
            self.assertIsInstance(case["must_not_contain"], list)
            if case["critical"]:
                self.assertTrue(
                    case["must_contain_any"] or case["must_not_contain"],
                    msg=f"critical case {case['id']} has no automatic assertion",
                )


if __name__ == "__main__":
    unittest.main()
