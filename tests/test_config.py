import os
import unittest
from unittest.mock import patch

from jadel_chatbot.config import (
    AppConfig,
    NVIDIA_DEFAULT_BASE_URL,
    NVIDIA_DEFAULT_MODEL,
    NVIDIA_DEFAULT_SAFETY_MODEL,
)


class ConfigTests(unittest.TestCase):
    def test_secure_nvidia_defaults(self) -> None:
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test"}, clear=True):
            config = AppConfig.from_env()

        self.assertEqual(config.provider, "nvidia_nim")
        self.assertEqual(config.api_base_url, NVIDIA_DEFAULT_BASE_URL)
        self.assertEqual(config.model, NVIDIA_DEFAULT_MODEL)
        self.assertEqual(config.safety_model, NVIDIA_DEFAULT_SAFETY_MODEL)
        self.assertTrue(config.moderation_enabled)
        self.assertTrue(config.enable_thinking)
        self.assertFalse(config.multimodal_enabled)
        self.assertEqual(config.max_retries, 2)

    def test_invalid_boolean_fails_closed(self) -> None:
        with patch.dict(
            os.environ,
            {"NVIDIA_API_KEY": "test", "ENABLE_MODERATION": "sometimes"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                AppConfig.from_env()

    def test_non_https_nvidia_endpoint_fails_closed(self) -> None:
        with patch.dict(
            os.environ,
            {
                "NVIDIA_API_KEY": "test",
                "NVIDIA_BASE_URL": "http://integrate.api.nvidia.com/v1",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError):
                AppConfig.from_env()

    def test_multimodal_cannot_be_enabled_for_text_only_ultra(self) -> None:
        with patch.dict(
            os.environ,
            {"NVIDIA_API_KEY": "test", "ENABLE_MULTIMODAL": "true"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                AppConfig.from_env()


if __name__ == "__main__":
    unittest.main()
