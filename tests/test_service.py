import unittest
from types import SimpleNamespace

from jadel_chatbot.config import AppConfig
from jadel_chatbot.service import AIService


class FakeChatCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["model"] == "nvidia/nemotron-3.5-content-safety":
            has_assistant = any(
                item.get("role") == "assistant" for item in kwargs["messages"]
            )
            content = "User Safety: safe"
            if has_assistant:
                content += "\nResponse Safety: safe"
            return SimpleNamespace(
                id="safety_test",
                model=kwargs["model"],
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=content))
                ],
                usage=SimpleNamespace(
                    prompt_tokens=8,
                    completion_tokens=3,
                    total_tokens=11,
                ),
            )

        return SimpleNamespace(
            id="resp_test",
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Respuesta segura.")
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=4,
                total_tokens=14,
            ),
        )


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


def make_config() -> AppConfig:
    return AppConfig(
        api_key="test",
        api_base_url="https://integrate.api.nvidia.com/v1",
        provider="nvidia_nim",
        model="nvidia/nemotron-3-ultra-550b-a55b",
        safety_model="nvidia/nemotron-3.5-content-safety",
        max_input_chars=4000,
        max_history_messages=20,
        max_output_tokens=1024,
        timeout_seconds=45.0,
        max_retries=2,
        session_requests_per_minute=10,
        moderation_enabled=True,
        multimodal_enabled=False,
        max_image_bytes=1024,
        enable_thinking=True,
        temperature=1.0,
        top_p=0.95,
    )


class ServiceTests(unittest.TestCase):
    def make_service(self) -> tuple[AIService, FakeClient]:
        service = AIService.__new__(AIService)
        service.config = make_config()
        client = FakeClient()
        service.client = client
        return service, client

    def test_nvidia_safety_wraps_main_inference(self) -> None:
        service, client = self.make_service()

        result = service.generate(
            [{"role": "user", "content": "Ayúdame con mi proyecto"}],
            current_user_text="Ayúdame con mi proyecto",
        )

        self.assertEqual(result.text, "Respuesta segura.")
        calls = client.chat.completions.calls
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["model"], "nvidia/nemotron-3.5-content-safety")
        self.assertEqual(calls[1]["model"], "nvidia/nemotron-3-ultra-550b-a55b")
        self.assertEqual(calls[2]["model"], "nvidia/nemotron-3.5-content-safety")
        self.assertTrue(
            calls[1]["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        )
        self.assertEqual(calls[1]["max_tokens"], 1024)
        self.assertEqual(calls[1]["messages"][0]["role"], "system")

    def test_text_only_boundary_fails_before_any_api_call(self) -> None:
        service, client = self.make_service()

        with self.assertRaises(ValueError):
            service.generate(
                [{"role": "user", "content": "hola"}],
                current_user_text="hola",
                image_data_url="data:image/png;base64,YWJj",
            )

        self.assertEqual(client.chat.completions.calls, [])

    def test_unrecognized_safety_verdict_fails_closed(self) -> None:
        with self.assertRaises(RuntimeError):
            AIService._safety_verdict("unknown", "User Safety")


if __name__ == "__main__":
    unittest.main()
