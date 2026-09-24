# Evaluation Plan

## Current evidence

CI establishes deterministic properties: configuration fails closed, credentials are redacted, multimodal is disabled, autonomous write actions are disabled, the container runs as a non-root user, dependencies pass audit, and the Streamlit health endpoint responds.

These tests do **not** certify model quality.

## Repository-owned evaluation

The versioned baseline is `evals/baseline.json`. `scripts/run_model_eval.py` records the exact Git SHA, provider, NVIDIA API base URL, main model ID, safety-model ID, reasoning setting, prompt version, case-set SHA-256, latency, observed token usage and per-case results.

The `model-eval` workflow runs only from trusted `main` pushes that touch model-relevant files, or by explicit `workflow_dispatch`. It uses the protected `model-evaluation` environment, reads `NVIDIA_API_KEY`, uploads an immutable report, and never promotes a model automatically. Pull requests do not receive this evaluation secret.

Infrastructure/API exceptions are not model failures. The evaluator stops on the first API exception, records only safe diagnostic metadata, marks the run `NOT_ADJUDICATED_INFRASTRUCTURE`, and requires the infrastructure problem to be resolved before model-quality conclusions are drawn.

Human review remains mandatory because keyword assertions cannot measure all dimensions of usefulness, tone, factuality or safety.

## Nemotron migration baseline

The OpenAI production baseline has been retired as an active provider. `nvidia/nemotron-3-ultra-550b-a55b` is a new model/provider baseline and therefore requires a fresh held-out run; prior GPT results cannot certify Nemotron.

The safety path also changes: each evaluated request passes through `nvidia/nemotron-3.5-content-safety` before main inference and again after generation. Safety-model failures fail closed and must be separated from main-model quality failures.

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

Each case should have machine-checkable criteria where possible and human review for subjective quality. Compare a candidate against the current evaluated baseline; do not promote solely because a newer model exists.

## Required metrics before a capability promotion

- Task success on a held-out set.
- Safety-policy pass rate and critical-failure count.
- Prompt-injection/tool-abuse success rate for agentic features.
- p50/p95 latency.
- input/output tokens and cost per successful task.
- safety block rate with sampled human review.
- regression count versus the production baseline.

Promotion requires exact dataset version, exact code SHA, exact provider/model IDs, evaluator version, immutable result artifact and explicit human approval.

## Stage latency instrumentation

The evaluator records end-to-end latency and separates the NVIDIA request path into `input_safety`, `main_model`, and `output_safety`. For each observed stage it records latency, HTTP attempt count, derived retry count, status codes, provider request IDs, model ID, output-token limit, configured timeout, and configured retry budget.

HTTP request hooks count actual SDK attempts, including transparent retries. The instrumentation does not record request bodies, prompts, generated answers, authorization headers, API keys, or arbitrary response headers. Per-stage p50/p95, total HTTP attempts, total retries, and retried-case counts are aggregated into the immutable evaluation report. Infrastructure exceptions retain only the same safe stage metadata plus the existing sanitized provider error fields.

This instrumentation is observational only: it must not change model IDs, safety policy, prompt version, sampling, output limits, dataset, quality threshold, or the production SLO while a latency experiment is being adjudicated.
