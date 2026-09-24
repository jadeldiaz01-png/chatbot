# Jadel Tech RD Assistant

Production-oriented Streamlit assistant for Jadel Tech RD using NVIDIA NIM with Nemotron 3 Ultra.

## Current baseline

- Main model: `nvidia/nemotron-3-ultra-550b-a55b`.
- NVIDIA hosted NIM endpoint: `https://integrate.api.nvidia.com/v1`.
- Dedicated input/output safety model: `nvidia/nemotron-3.5-content-safety`.
- Server-side `NVIDIA_API_KEY` only; the runtime has no `OPENAI_API_KEY` dependency.
- The Python `openai` package is retained only as an OpenAI-compatible protocol client for NVIDIA NIM; the runtime does not target `api.openai.com`.
- Reasoning is disabled by default for the isolated latency candidate; reasoning traces remain unavailable to the UI and application logs.
- Likely credential redaction before model calls, including NVIDIA `nvapi-` keys.
- Bounded input, history, output tokens, timeout and retries.
- Session-level soft throttling.
- Multimodal input is fail-closed because this Nemotron Ultra baseline is text-only.
- Non-root container with healthcheck.
- Linux/amd64 CPython 3.12 runtime and CI dependency graphs are versioned with SHA-256 hashes.
- Docker production/CI installation uses `pip --require-hashes --only-binary=:all:`.
- A read-only lock-drift workflow regenerates both graphs and fails on byte differences.
- Policy/config/manifest tests, Ruff and dependency audit in CI.
- Release-evidence workflow with SHA-pinned GitHub Actions, CycloneDX SBOM and provenance/SBOM attestation.
- RAG, ML/DL and autonomous tools remain fail-closed behind independent evidence gates.

The canonical machine-readable state is [`production-manifest.json`](production-manifest.json). `production_status` remains `NOT_CERTIFIED` until the blocking evidence in that manifest exists for an exact commit SHA.

## Provider data-handling gate

The NVIDIA Build trial endpoint may record API inputs and outputs. Therefore this repository treats the hosted trial endpoint as **not approved for sensitive production data**. Production promotion requires a privacy/data-handling review and, when appropriate, a partner or self-hosted NIM endpoint with acceptable retention terms.

## Local run

For ordinary cross-platform development, install the direct requirements:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export NVIDIA_API_KEY='set-this-in-your-shell-or-secret-manager'
streamlit run streamlit_app.py
```

The committed `requirements.lock` and `requirements-ci.lock` are intentionally platform-specific production locks for Linux/amd64 CPython 3.12 and are enforced by the container/CI path.

Do not commit `.env`, API keys or other credentials.

## Runtime configuration

| Variable | Default | Purpose |
|---|---:|---|
| `NVIDIA_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b` | Main evaluated model |
| `NVIDIA_SAFETY_MODEL` | `nvidia/nemotron-3.5-content-safety` | Input/output content-safety model |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA NIM API endpoint |
| `NVIDIA_ENABLE_THINKING` | `false` | Isolated latency candidate: disable Nemotron reasoning internally |
| `NVIDIA_TEMPERATURE` | `1.0` | Main-model sampling temperature |
| `NVIDIA_TOP_P` | `0.95` | Main-model nucleus sampling |
| `ENABLE_MODERATION` | `true` | Fail-safe NVIDIA safety moderation |
| `ENABLE_MULTIMODAL` | `false` | Must remain false for the text-only Ultra baseline |
| `MAX_INPUT_CHARS` | `4000` | Per-turn input bound |
| `MAX_HISTORY_MESSAGES` | `20` | In-session context bound |
| `MAX_OUTPUT_TOKENS` | `512` | Isolated latency candidate: per-response output budget |
| `NVIDIA_TIMEOUT_SECONDS` | `45` | Request timeout |
| `NVIDIA_MAX_RETRIES` | `2` | SDK retry bound |
| `SESSION_REQUESTS_PER_MINUTE` | `10` | Per-session soft throttle |

For architecture, risk, evaluation, data/ML and operations details, see `docs/`.
