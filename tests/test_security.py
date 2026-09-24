import unittest

from jadel_chatbot.security import (
    allow_session_request,
    image_to_data_url,
    redact_likely_secrets,
)


class SecurityTests(unittest.TestCase):
    def test_openai_key_is_redacted(self) -> None:
        value = "use sk-abcdefghijklmnopqrstuvwxyz123456 for testing"
        result = redact_likely_secrets(value)
        self.assertNotIn("sk-abcdefghijklmnopqrstuvwxyz123456", result.text)
        self.assertIn("openai_key", result.detected_types)

    def test_nvidia_key_is_redacted(self) -> None:
        value = "use nvapi-abcdefghijklmnopqrstuvwxyz123456 for testing"
        result = redact_likely_secrets(value)
        self.assertNotIn("nvapi-abcdefghijklmnopqrstuvwxyz123456", result.text)
        self.assertIn("nvidia_api_key", result.detected_types)

    def test_generic_secret_assignment_is_redacted(self) -> None:
        result = redact_likely_secrets("token=supersecretvalue123")
        self.assertEqual(result.text, "token=[REDACTED]")

    def test_session_rate_limit(self) -> None:
        allowed, timestamps = allow_session_request([1.0, 2.0], 3.0, limit=3)
        self.assertTrue(allowed)
        allowed, timestamps = allow_session_request(timestamps, 4.0, limit=3)
        self.assertFalse(allowed)

    def test_image_validation(self) -> None:
        data_url = image_to_data_url(b"abc", "image/png", max_bytes=10)
        self.assertTrue(data_url.startswith("data:image/png;base64,"))
        with self.assertRaises(ValueError):
            image_to_data_url(b"abc", "image/svg+xml", max_bytes=10)


if __name__ == "__main__":
    unittest.main()
