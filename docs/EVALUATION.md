# Evaluation Plan

## Current evidence

CI establishes deterministic properties: configuration fails closed, credentials are redacted, multimodal is disabled by default, autonomous write actions are disabled, the container runs as a non-root user, dependencies pass audit, and the Streamlit health endpoint responds.

These tests do **not** certify model quality.

## Repository-owned evaluation

The versioned baseline is `evals/baseline.json`. `scripts/run_model_eval.py` records the exact Git SHA, model ID, prompt version, case-set SHA-256, latency, observed token usage and per-case results.

The `model-eval` workflow runs only from trusted `main` pushes that touch model-relevant files, or by explicit `workflow_dispatch`. It uses the protected `model-evaluation` environment, uploads an immutable report, and never promotes a model automatically. Pull requests do not receive this evaluation secret.

Infrastructure/API exceptions are not model failures. The evaluator stops on the first API exception, records only safe diagnostic metadata, marks the run `NOT_ADJUDICATED_INFRASTRUCTURE`, and requires the infrastructure problem to be resolved before model-quality conclusions are drawn.

This repository-owned harness avoids depending on the legacy OpenAI Evals platform, which is deprecated in 2026. Human review remains mandatory because keyword assertions cannot measure all dimensions of usefulness, tone, factuality or safety.

## Model-quality gate

Before changing the production model or enabling RAG, multimodal or tools, the held-out set must cover:

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

Promotion requires exact dataset version, exact code SHA, exact model ID, evaluator version, immutable result artifact and explicit human approval.
