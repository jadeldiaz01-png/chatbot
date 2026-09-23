import unittest
from types import SimpleNamespace

from jadel_chatbot.config import AppConfig
from jadel_chatbot.service import AIService


class FakeModerations:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(results=[SimpleNamespace(flagged=False)])


class FakeResponses:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp_test",
            model="gpt-test",
            output_text="Respuesta segura.",
            usage=SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14),
        )


class FakeClient:
    def __init__(self) -> None:
        self.moderations = FakeModerations()
        self.responses = FakeResponses()


def make_config(*, multimodal_enabled: bool) -> AppConfig:
    return AppConfig(
        api_key="test",
        model="gpt-test",
        moderation_model="omni-moderation-latest",
        max_input_chars=4000,
        max_history_messages=20,
        max_output_tokens=800,
        timeout_seconds=30.0,
        max_retries=2,
        session_requests_per_minute=10,
        response_store=False,
        moderation_enabled=True,
        multimodal_enabled=multimodal_enabled,
        max_image_bytes=1024,
    )


class ServiceTests(unittest.TestCase):
    def make_service(self, *, multimodal_enabled: bool) -> tuple[AIService, FakeClient]:
        service = AIService.__new__(AIService)
        service.config = make_config(multimodal_enabled=multimodal_enabled)
        client = FakeClient()
        service.client = client
        return service, client

    def test_multimodal_input_is_moderated_before_inference(self) -> None:
        service, client = self.make_service(multimodal_enabled=True)
        image = "data:image/png;base64,YWJj"

        result = service.generate(
            [{"role": "user", "content": "Analiza esta imagen"}],
            current_user_text="Analiza esta imagen",
            image_data_url=image,
        )

        self.assertEqual(result.text, "Respuesta segura.")
        first_moderation = client.moderations.calls[0]["input"]
        self.assertEqual(first_moderation[0]["type"], "text")
        self.assertEqual(first_moderation[1]["type"], "image_url")
        self.assertEqual(first_moderation[1]["image_url"]["url"], image)
        request = client.responses.calls[0]
        self.assertFalse(request["store"])
        self.assertEqual(request["max_output_tokens"], 800)
        self.assertEqual(request["input"][-1]["content"][1]["type"], "input_image")

    def test_disabled_multimodal_fails_before_any_api_call(self) -> None:
        service, client = self.make_service(multimodal_enabled=False)

        with self.assertRaises(ValueError):
            service.generate(
                [{"role": "user", "content": "hola"}],
                current_user_text="hola",
                image_data_url="data:image/png;base64,YWJj",
            )

        self.assertEqual(client.moderations.calls, [])
        self.assertEqual(client.responses.calls, [])


if __name__ == "__main__":
    unittest.main()
