# Jadel Chatbot — Production Architecture 2026

## Decision

The system remains a small, stateless-by-default Streamlit application. The inference provider is NVIDIA NIM, with `nvidia/nemotron-3-ultra-550b-a55b` as the main text model and `nvidia/nemotron-3.5-content-safety` as the dedicated input/output safety model.

The NVIDIA endpoint is OpenAI-compatible, so the Python `openai` package is retained strictly as a wire-protocol client. Runtime configuration points it to `https://integrate.api.nvidia.com/v1`; no runtime path targets `api.openai.com`, and `OPENAI_API_KEY` is not used.

The current workload still does not justify Kubernetes, a service mesh, a vector database, a graph database, or a multi-agent runtime. Those components become valid only when measured requirements demonstrate the need.

## Trust boundaries

1. Browser/session: untrusted user input.
2. Application process: deterministic validation, credential redaction, session throttling and prompt assembly.
3. NVIDIA Content Safety NIM: evaluates user input and generated output; unrecognized safety verdicts fail closed.
4. NVIDIA Nemotron 3 Ultra NIM: text inference only.
5. External systems: no tools are connected in the production baseline. Any future MCP/function integration must be allowlisted and pass explicit human approval for sensitive reads/writes.

## Runtime path

`Browser -> Streamlit -> input limits -> secret redaction -> NVIDIA input safety -> Nemotron 3 Ultra -> NVIDIA output safety -> browser`

No prompt, answer, or reasoning trace is intentionally written to application logs. Operational logs contain only metadata such as response ID, provider, model, prompt version and token counts.

## Provider privacy boundary

The NVIDIA Build trial endpoint may record API inputs and outputs. For that reason, sensitive production data is prohibited on the trial endpoint. Production certification requires a provider data-handling review and, where necessary, migration to a partner or self-hosted NIM endpoint with acceptable retention terms.

## Capability progression

- Text LLM: enabled, advisory-only.
- Internal reasoning: disabled in the isolated latency candidate; reasoning traces are not exposed to users or application logs.
- Multimodal image input: disabled because the Nemotron 3 Ultra baseline is text-only. A separate multimodal model and eval are required before enabling it.
- RAG: gated. A trusted, versioned corpus with lineage is required before retrieval is connected.
- ML ranking: research-only until a labeled dataset and held-out evaluation exist.
- Deep learning/fine-tuning: research-only until it beats the prompt-only baseline on a held-out set and data rights are documented.
- Autonomous tools: disabled. Future tools require structured I/O, least privilege, prompt-injection testing, audit logs and human approval for sensitive actions.

## ADR-001 — Avoid premature distributed architecture

**Status:** accepted.

The repository is currently a small web assistant. Introducing queues, Redis, PostgreSQL, Kubernetes, a vector DB or a graph DB without measured throughput/reliability needs would add failure modes and cost without evidence of benefit. Scale-out components must be driven by SLO measurements and concrete use cases.

## ADR-002 — NVIDIA NIM as the sole active model provider

**Status:** accepted.

OpenAI Platform is not an active runtime provider. The main and safety model calls both use NVIDIA NIM. No automatic provider fallback is allowed because an unevaluated fallback could change safety, privacy, cost and quality properties.

## ADR-003 — Human authority above model authority

**Status:** accepted.

The model never receives direct production write authority in the baseline. Sensitive future tools must be mediated by deterministic policy and explicit human approval.
