import json
import pathlib
import unittest


class ManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = pathlib.Path(__file__).parents[1] / "production-manifest.json"
        cls.manifest = json.loads(path.read_text(encoding="utf-8"))

    def test_autonomous_write_actions_are_disabled(self) -> None:
        tools = self.manifest["capabilities"]["autonomous_tools"]
        self.assertEqual(tools["state"], "disabled")
        self.assertFalse(tools["write_actions"])

    def test_response_storage_is_disabled(self) -> None:
        self.assertFalse(self.manifest["ai"]["response_store"])

    def test_release_requires_human_approval(self) -> None:
        self.assertEqual(
            self.manifest["release_gates"]["human_merge_approval"], "required"
        )

    def test_advanced_capabilities_are_gated(self) -> None:
        capabilities = self.manifest["capabilities"]
        for name in (
            "multimodal_image_input",
            "rag",
            "ml_ranking",
            "deep_learning_or_finetuning",
        ):
            self.assertIn(capabilities[name]["state"], {"gated", "research_only"})
            self.assertFalse(capabilities[name]["default_enabled"])


if __name__ == "__main__":
    unittest.main()
