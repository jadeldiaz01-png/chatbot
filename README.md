# Jadel Tech RD Assistant

Production-oriented Streamlit assistant for Jadel Tech RD using the OpenAI Responses API.

## Current baseline

- Server-side OpenAI credential only.
- Stateless API calls with `store=false` by default.
- Input/output moderation enabled by default.
- Likely credential redaction before model calls.
- Bounded input, history, output tokens, timeout and retries.
- Session-level soft throttling.
- Non-root container with healthcheck.
- Policy/config/manifest tests, Ruff and dependency audit in CI.
- Release-evidence workflow with SHA-pinned GitHub Actions and provenance attestation.
- Advanced capabilities are fail-closed: multimodal is implemented behind a disabled feature flag; RAG, ML/DL and autonomous tools require explicit evidence gates.

The canonical machine-readable state is [`production-manifest.json`](production-manifest.json). `production_status` remains `NOT_CERTIFIED` until the blocking evidence in that manifest exists for an exact commit SHA.

## Local run

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY='set-this-in-your-shell-or-secret-manager'
streamlit run streamlit_app.py
```

Do not commit `.env`, API keys or other credentials.

## Runtime configuration

| Variable | Default | Purpose |
|---|---:|---|
| `OPENAI_MODEL` | `gpt-5.6-luna` | Evaluated production model; change only with model-quality evidence |
| `OPENAI_MODERATION_MODEL` | `omni-moderation-latest` | Input/output moderation model |
| `OPENAI_RESPONSE_STORE` | `false` | Whether API response state may be stored |
| `ENABLE_MODERATION` | `true` | Fail-safe content moderation switch |
| `ENABLE_MULTIMODAL` | `false` | Gated image-input capability |
| `MAX_INPUT_CHARS` | `4000` | Per-turn input bound |
| `MAX_HISTORY_MESSAGES` | `20` | In-session context bound |
| `MAX_OUTPUT_TOKENS` | `800` | Per-response output budget |
| `OPENAI_TIMEOUT_SECONDS` | `30` | Request timeout |
| `OPENAI_MAX_RETRIES` | `2` | SDK retry bound |
| `SESSION_REQUESTS_PER_MINUTE` | `10` | Per-session soft throttle |
| `MAX_IMAGE_BYTES` | `5242880` | Image bound when multimodal is enabled |

For architecture, risk, evaluation, data/ML and operations details, see `docs/`.
