# Jadel Chatbot — Production Architecture 2026

## Decision

The system remains a small, stateless-by-default Streamlit application backed by the OpenAI Responses API. The current workload does not justify Kubernetes, a service mesh, a vector database, a graph database, or a multi-agent runtime. Those components become valid only when measured requirements demonstrate the need.

## Trust boundaries

1. Browser/session: untrusted user input.
2. Application process: deterministic validation, credential redaction, session throttling, prompt assembly and moderation.
3. OpenAI API: model inference only; response storage is disabled by default.
4. External systems: no tools are connected in the production baseline. Any future MCP/function integration must be allowlisted and pass explicit human approval for sensitive reads/writes.

## Runtime path

`Browser -> Streamlit -> input limits -> secret redaction -> input moderation -> Responses API -> output moderation -> browser`

No prompt or response body is intentionally written to application logs. Operational logs contain only metadata such as response ID, model, prompt version and token counts.

## Capability progression

- Text LLM: enabled, advisory-only.
- Multimodal image input: implementation exists behind `ENABLE_MULTIMODAL=false`; activation requires privacy, image-safety, cost and quality evaluation.
- RAG: gated. A trusted, versioned corpus with lineage is required before retrieval is connected.
- ML ranking: research-only until a labeled dataset and held-out evaluation exist.
- Deep learning/fine-tuning: research-only until it beats the prompt-only baseline on a held-out set and data rights are documented.
- Autonomous tools: disabled. Future tools require structured I/O, least privilege, prompt-injection testing, audit logs and human approval for sensitive actions.

## ADR-001 — Avoid premature distributed architecture

**Status:** accepted.

The repository is currently a small web assistant. Introducing queues, Redis, PostgreSQL, Kubernetes, a vector DB or a graph DB without measured throughput/reliability needs would add failure modes and cost without evidence of benefit. Scale-out components must be driven by SLO measurements and concrete use cases.

## ADR-002 — Stateless Responses API by default

**Status:** accepted.

Set `store=false` and manage bounded conversation history in the application session. This reduces retained application state and makes deletion/retention semantics simpler.

## ADR-003 — Human authority above model authority

**Status:** accepted.

The model never receives direct production write authority in the baseline. Sensitive future tools must be mediated by deterministic policy and explicit human approval.
