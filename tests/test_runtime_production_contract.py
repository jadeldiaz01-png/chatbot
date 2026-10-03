import ast
import hashlib
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "streamlit_app.py"
DOCKERFILE_PATH = ROOT / "Dockerfile"
EXPECTED_SYSTEM_INSTRUCTIONS_SHA256 = (
    "79f6e3eb01e7edaaed0fd63628a5c40fe2209211631db3bf357490ee2e7c843a"
)


def literal_assignments(path: pathlib.Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            try:
                values[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


class RuntimeProductionContractTests(unittest.TestCase):
    def test_runtime_configuration_is_pinned(self) -> None:
        values = literal_assignments(APP_PATH)
        self.assertEqual(values["MODEL"], "gpt-5.6-luna")
        self.assertEqual(values["MAX_INPUT_CHARS"], 4000)
        self.assertEqual(values["MAX_HISTORY_MESSAGES"], 20)
        self.assertEqual(values["MAX_OUTPUT_TOKENS"], 512)
        self.assertEqual(values["REQUEST_TIMEOUT_SECONDS"], 20.0)
        self.assertEqual(values["MAX_RETRIES"], 1)

    def test_promoted_system_instructions_match_v3_candidate(self) -> None:
        values = literal_assignments(APP_PATH)
        instructions = values["SYSTEM_INSTRUCTIONS"]
        self.assertIsInstance(instructions, str)
        digest = hashlib.sha256(instructions.encode("utf-8")).hexdigest()
        self.assertEqual(digest, EXPECTED_SYSTEM_INSTRUCTIONS_SHA256)

    def test_runtime_fails_closed_on_model_drift(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('CONFIGURED_MODEL = os.getenv("OPENAI_MODEL", MODEL)', source)
        self.assertIn("if CONFIGURED_MODEL != MODEL:", source)

    def test_openai_request_has_bounded_output_and_no_storage(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("timeout=REQUEST_TIMEOUT_SECONDS", source)
        self.assertIn("max_retries=MAX_RETRIES", source)
        self.assertIn("max_output_tokens=MAX_OUTPUT_TOKENS", source)
        self.assertIn("store=False", source)
        self.assertNotIn("NVIDIA_API_KEY", source)
        self.assertNotIn("integrate.api.nvidia.com", source)

    def test_container_uses_hash_locked_minimal_runtime(self) -> None:
        dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
        self.assertIn("COPY requirements.lock ./", dockerfile)
        self.assertIn("--require-hashes -r requirements.lock", dockerfile)
        self.assertIn("COPY streamlit_app.py ./", dockerfile)
        self.assertNotIn("COPY . .", dockerfile)
        self.assertIn("USER 10001:10001", dockerfile)


if __name__ == "__main__":
    unittest.main()
