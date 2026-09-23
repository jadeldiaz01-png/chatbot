import os
import unittest
from unittest.mock import patch

from jadel_chatbot.config import AppConfig


class ConfigTests(unittest.TestCase):
    def test_secure_defaults(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=True):
            config = AppConfig.from_env()
        self.assertFalse(config.response_store)
        self.assertTrue(config.moderation_enabled)
        self.assertFalse(config.multimodal_enabled)
        self.assertEqual(config.max_retries, 2)

    def test_invalid_boolean_fails_closed(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test", "ENABLE_MODERATION": "sometimes"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                AppConfig.from_env()


if __name__ == "__main__":
    unittest.main()
