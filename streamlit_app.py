import streamlit as st

from runtime_provider import (
    ChatRuntime,
    GENERIC_UNAVAILABLE_MESSAGE,
    SENSITIVE_INPUT_MESSAGE,
    detect_sensitive_input,
    runtime_config_from_env,
)

MAX_INPUT_CHARS = 4000
MAX_HISTORY_MESSAGES = 20

SYSTEM_INSTRUCTIONS = """You are the Jadel Tech RD website assistant. Help visitors understand Jadel Tech RD services, scope work, and identify when a human should follow up. Do not claim that a service is production-ready, profitable, connected, or authorized unless the provided conversation establishes that fact. Never request passwords, API keys, private keys, payment credentials, recovery codes, or government identity documents. You have no tools and cannot publish content, place trades, submit marketplace proposals, charge money, change accounts, or make contractual commitments. For those actions, state that human approval is required."""


@st.cache_resource
def get_runtime() -> ChatRuntime:
    return ChatRuntime(runtime_config_from_env())


st.set_page_config(page_title="Jadel Tech RD Assistant", page_icon="🤖")
st.title("🤖 Jadel Tech RD Assistant")
st.caption("Asistente informativo con acciones externas bloqueadas por diseño.")

try:
    runtime = get_runtime()
except Exception:
    st.error("El asistente está temporalmente fuera de servicio por configuración del servidor.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿En qué proyecto o servicio de Jadel Tech RD necesitas ayuda?"):
    prompt = prompt.strip()
    if not prompt:
        st.stop()
    if len(prompt) > MAX_INPUT_CHARS:
        st.warning(f"El mensaje supera el límite de {MAX_INPUT_CHARS} caracteres.")
        st.stop()

    if detect_sensitive_input(prompt):
        st.warning(SENSITIVE_INPUT_MESSAGE)
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages = st.session_state.messages[-MAX_HISTORY_MESSAGES:]

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        answer = runtime.generate(
            system_instructions=SYSTEM_INSTRUCTIONS,
            messages=[
                {"role": message["role"], "content": message["content"]}
                for message in st.session_state.messages
            ],
        )
        if not answer:
            answer = "No pude generar una respuesta útil. Inténtalo nuevamente o solicita seguimiento humano."
    except Exception:
        answer = GENERIC_UNAVAILABLE_MESSAGE

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.messages = st.session_state.messages[-MAX_HISTORY_MESSAGES:]
