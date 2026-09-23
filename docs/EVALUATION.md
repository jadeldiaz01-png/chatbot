# Evaluation Plan

## Current evidence

CI can establish deterministic properties: configuration fails closed, credentials are redacted, multimodal is disabled by default, autonomous write actions are disabled, the container runs as a non-root user, dependencies pass audit, and the Streamlit health endpoint responds.

These tests do **not** certify model quality.

## Model-quality gate

Before changing the production model or enabling RAG, multimodal or tools, create a versioned held-out dataset covering:

- service discovery and scope clarification;
- uncertainty and refusal to invent production/revenue claims;
- secret-handling requests;
- prompt-injection attempts;
- requests for external actions the assistant is not authorized to perform;
- Spanish/English language quality;
- long-context truncation cases;
- safety false positives and false negatives;
- latency and token-cost distributions.

Each case should have machine-checkable criteria where possible and human review for subjective quality. Compare a candidate against the current baseline; do not promote solely because a newer model exists.

## Required metrics before a capability promotion

- Task success on a held-out set.
- Safety-policy pass rate and critical-failure count.
- Prompt-injection/tool-abuse success rate for agentic features.
- p50/p95 latency.
- input/output tokens and cost per successful task.
- moderation block rate with sampled human review.
- regression count versus the production baseline.

Promotion requires exact dataset version, exact code SHA, exact model ID, evaluator version and an immutable result artifact.
