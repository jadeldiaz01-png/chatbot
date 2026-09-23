from __future__ import annotations

import logging
import time

import streamlit as st

from jadel_chatbot.config import AppConfig
from jadel_chatbot.security import (
    allow_session_request,
    image_to_data_url,
    redact_likely_secrets,
)
from jadel_chatbot.service import AIService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

st.set_page_config(page_title="Jadel Tech RD Assistant", page_icon="🤖")
st.title("🤖 Jadel Tech RD Assistant")
st.caption("Asistente informativo con acciones externas bloqueadas por diseño.")

try:
    config = AppConfig.from_env()
except (TypeError, ValueError):
    st.error("La configuración del servicio es inválida. Contacta al administrador.")
    st.stop()

if not config.api_key:
    st.error("El asistente está temporalmente fuera de servicio por configuración del servidor.")
    st.stop()


@st.cache_resource
def get_service(cfg: AppConfig) -> AIService:
    return AIService(cfg)


service = get_service(config)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "request_timestamps" not in st.session_state:
    st.session_state.request_timestamps = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

uploaded_image = None
if config.multimodal_enabled:
    uploaded_image = st.file_uploader(
        "Imagen opcional",
        type=["png", "jpg", "jpeg", "webp", "gif"],
        accept_multiple_files=False,
    )

if prompt := st.chat_input("¿En qué proyecto o servicio de Jadel Tech RD necesitas ayuda?"):
    prompt = prompt.strip()
    if not prompt:
        st.stop()
    if len(prompt) > config.max_input_chars:
        st.warning(f"El mensaje supera el límite de {config.max_input_chars} caracteres.")
        st.stop()

    allowed, active = allow_session_request(
        st.session_state.request_timestamps,
        time.monotonic(),
        limit=config.session_requests_per_minute,
    )
    st.session_state.request_timestamps = active
    if not allowed:
        st.warning(
            "Se alcanzó el límite temporal de solicitudes de esta sesión. "
            "Inténtalo nuevamente en un momento."
        )
        st.stop()

    redaction = redact_likely_secrets(prompt)
    safe_prompt = redaction.text
    if redaction.detected_types:
        st.warning(
            "Se detectó y ocultó información que parecía una credencial "
            "antes de enviarla al modelo."
        )

    image_data_url = None
    if uploaded_image is not None:
        try:
            image_data_url = image_to_data_url(
                uploaded_image.getvalue(),
                uploaded_image.type,
                max_bytes=config.max_image_bytes,
            )
        except ValueError:
            st.warning("La imagen no cumple los límites de tipo o tamaño configurados.")
            st.stop()

    st.session_state.messages.append({"role": "user", "content": safe_prompt})
    st.session_state.messages = st.session_state.messages[-config.max_history_messages :]

    with st.chat_message("user"):
        st.markdown(safe_prompt)

    try:
        result = service.generate(
            [
                {"role": message["role"], "content": message["content"]}
                for message in st.session_state.messages
            ],
            current_user_text=safe_prompt,
            image_data_url=image_data_url,
        )
        answer = result.text
    except Exception as exc:
        logging.getLogger("jadel_chatbot").error(
            "llm_request_failed error_type=%s", type(exc).__name__
        )
        answer = (
            "El servicio de IA no está disponible temporalmente. "
            "No se realizó ninguna acción externa."
        )

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.messages = st.session_state.messages[-config.max_history_messages :]
