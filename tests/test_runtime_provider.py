import unittest

import runtime_provider as runtime


class RuntimeProviderTests(unittest.TestCase):
    def test_current_openai_default_is_preserved(self) -> None:
        config = runtime.runtime_config_from_env({"OPENAI_API_KEY": "unit-test-value"})
        self.assertEqual(config.provider, runtime.OPENAI_PROVIDER)
        self.assertEqual(config.model, "gpt-5.6-luna")
        self.assertEqual(config.api_key, "unit-test-value")
        self.assertIsNone(config.base_url)

    def test_nvidia_contract_is_exact_and_fail_closed(self) -> None:
        config = runtime.runtime_config_from_env(
            {"CHATBOT_PROVIDER": "nvidia", "NVIDIA_API_KEY": "unit-test-value"}
        )
        self.assertEqual(config.provider, runtime.NVIDIA_PROVIDER)
        self.assertEqual(config.model, "nvidia/nemotron-3-ultra-550b-a55b")
        self.assertEqual(config.base_url, "https://integrate.api.nvidia.com/v1")

        request = runtime.build_nvidia_request(
            model=config.model,
            system_instructions="system",
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertEqual(request["max_tokens"], 128)
        self.assertEqual(request["temperature"], 1.0)
        self.assertEqual(request["top_p"], 0.95)
        self.assertFalse(request["stream"])
        self.assertFalse(request["extra_body"]["chat_template_kwargs"]["enable_thinking"])

        with self.assertRaises(ValueError):
            runtime.runtime_config_from_env(
                {
                    "CHATBOT_PROVIDER": "nvidia",
                    "NVIDIA_API_KEY": "unit-test-value",
                    "NVIDIA_API_BASE_URL": "https://example.invalid/v1",
                }
            )

    def test_privacy_question_without_actual_secret_is_allowed(self) -> None:
        text = (
            "¿Necesitas que te envíe mi contraseña o mi tarjeta "
            "para orientarme sobre un servicio?"
        )
        self.assertEqual(runtime.detect_sensitive_input(text), ())

    def test_sensitive_values_are_blocked(self) -> None:
        private_key_marker = "-----BEGIN " + "PRIVATE KEY-----" + "\n" + "synthetic"
        labeled_password = "contra" + "seña: " + "synthetic-value-123"
        labeled_api_key = "api_" + "key=" + ("a" * 24)
        email_value = "cliente" + "@" + "example.com"
        recovery = "codigo de recuperacion: " + "ABCD" + "-" + "1234"
        card = "4111 " + "1111 " + "1111 " + "1111"
        samples = {
            "password": labeled_password,
            "api_secret": labeled_api_key,
            "private_key": private_key_marker,
            "email": "Mi correo es " + email_value,
            "government_id": "Mi cédula es " + "001-" + "1234567-" + "8",
            "recovery_code": recovery,
            "phone": "Mi teléfono es " + "+1 " + "809-" + "555-" + "1212",
            "payment_card": "tarjeta " + card,
        }
        for expected, value in samples.items():
            with self.subTest(expected=expected):
                categories = {
                    item.category for item in runtime.detect_sensitive_input(value)
                }
                self.assertIn(expected, categories)

    def test_only_user_messages_are_privacy_gated(self) -> None:
        public_email = "soporte" + "@" + "example.com"
        runtime.validate_messages_for_provider(
            [
                {"role": "assistant", "content": "Ejemplo público: " + public_email},
                {"role": "user", "content": "Necesito información del servicio."},
            ]
        )
        user_email = "cliente" + "@" + "example.com"
        with self.assertRaisesRegex(ValueError, "sensitive_input_blocked"):
            runtime.validate_messages_for_provider(
                [{"role": "user", "content": "Mi correo es " + user_email}]
            )

    def test_unsupported_provider_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            runtime.runtime_config_from_env(
                {"CHATBOT_PROVIDER": "other", "OPENAI_API_KEY": "unit-test-value"}
            )


if __name__ == "__main__":
    unittest.main()
