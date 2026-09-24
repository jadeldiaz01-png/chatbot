# NVIDIA Provider Latency-Floor Probe

This probe is intentionally separate from the model-quality evaluation.

## Purpose

Measure the hosted NVIDIA endpoint's minimum practical latency envelope using synthetic, non-sensitive, deterministic requests before changing chatbot behavior again.

The probe targets:

- `nvidia/nemotron-3-ultra-550b-a55b`
- `nvidia/nemotron-3.5-content-safety`
- `https://integrate.api.nvidia.com/v1`

It uses the same protected `NVIDIA_API_KEY` environment secret as trusted model evaluation.

## Method

The default run executes 20 sequential pairs:

1. one minimal streaming request to Nemotron Ultra;
2. one minimal streaming request to Content Safety.

No requests are executed concurrently.

For each sample the report records:

- time until the streaming response is opened;
- time until the first response event;
- time until first non-empty content;
- total request duration;
- actual HTTP attempts and derived retries;
- HTTP status codes;
- provider request IDs when present;
- returned model identifier;
- output character count and chunk count.

The report never records API keys, authorization headers, arbitrary response headers, or generated content.

## Probe inputs

Ultra uses:

- reasoning disabled;
- `max_tokens=8`;
- `temperature=0`;
- `top_p=1`;
- synthetic prompt: `Reply exactly with: OK`.

Content Safety uses:

- reasoning disabled;
- `max_tokens=16`;
- the same safety sampling settings used by the application;
- synthetic prompt: `Hello.`.

These are latency-floor requests, not quality-evaluation cases.

## Interpretation

The production SLO remains p95 <= 8 seconds.

For each provider component the probe reports p50/p95/min/max/mean for first-content latency and total latency.

`provider_floor_necessary_condition_pass=true` only when both Ultra and Content Safety complete the full probe without infrastructure errors and each has total-latency p95 <= 8 seconds.

This is only a necessary condition. Passing it does not certify the end-to-end chatbot, whose real path includes input safety, main inference, output safety, application processing, and network overhead.

If either provider component exceeds 8 seconds by itself, the current hosted endpoint cannot support the declared end-to-end SLO without an architectural/provider change.

## Security and privacy

The live probe runs only from trusted `main` or explicit workflow dispatch and uses the protected `model-evaluation` environment. Pull requests run offline validation only and receive no NVIDIA secret.

NVIDIA's trial endpoint may record inputs and outputs, so the probe uses only fixed synthetic, non-sensitive content.
