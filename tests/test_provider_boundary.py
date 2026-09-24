from pathlib import Path
import unittest

ROOT = Path(__file__).parents[1]
RUNTIME_PROVIDER_FILES = (
    "jadel_chatbot/config.py",
    "jadel_chatbot/service.py",
    "streamlit_app.py",
    ".github/workflows/runtime-container-ci.yml",
    ".github/workflows/model-eval.yml",
)


class ProviderBoundaryTests(unittest.TestCase):
    def test_openai_platform_runtime_references_are_absent(self) -> None:
        forbidden = ("OPENAI_API_KEY", "OPENAI_MODEL", "api.openai.com")
        for filename in RUNTIME_PROVIDER_FILES:
            content = (ROOT / filename).read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(
                    token,
                    content,
                    msg=f"{token} unexpectedly present in runtime provider file {filename}",
                )

    def test_nvidia_runtime_contract_is_present(self) -> None:
        config = (ROOT / "jadel_chatbot" / "config.py").read_text(encoding="utf-8")
        service = (ROOT / "jadel_chatbot" / "service.py").read_text(encoding="utf-8")
        workflow = (
            ROOT / ".github" / "workflows" / "model-eval.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("NVIDIA_API_KEY", config)
        self.assertIn("https://integrate.api.nvidia.com/v1", config)
        self.assertIn("nvidia/nemotron-3-ultra-550b-a55b", config)
        self.assertIn("nvidia/nemotron-3.5-content-safety", config)
        self.assertIn("base_url=config.api_base_url", service)
        self.assertIn("NVIDIA_API_KEY", workflow)


if __name__ == "__main__":
    unittest.main()
