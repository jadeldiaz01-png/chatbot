# Operations Runbook

## Health

- Liveness/readiness signal: `/_stcore/health`.
- Container must run as UID/GID `10001:10001`.
- Missing `NVIDIA_API_KEY` is a fail-closed configuration condition visible to the user; no model call is attempted.
- The active upstream is NVIDIA NIM. There is no automatic fallback to OpenAI Platform or another provider.

## Incident classes

### NVIDIA NIM/API degradation

Symptoms: elevated errors, timeouts or rate limits from the NVIDIA endpoint. Keep retries bounded. Return the generic degraded-service message. Do not silently switch to an unevaluated model or provider.

### Safety-model degradation

If the Nemotron content-safety service returns an unrecognized verdict or is unavailable, the request fails closed. Do not bypass the safety model to restore availability.

### Cost anomaly

Freeze model/tool promotions, lower traffic at the edge, inspect token telemetry and provider-side budgets. Do not solve a cost incident by removing safety checks.

### Provider privacy/data-handling incident

The NVIDIA Build trial endpoint may record request/response content. Do not send sensitive production data to the trial endpoint. If sensitive content may have been transmitted, stop traffic, preserve metadata-only evidence, review provider controls/terms, and rotate credentials if compromise is suspected.

### Suspected secret exposure

Revoke/rotate the affected secret in its provider, inspect metadata-only logs, invalidate sessions where applicable and document the incident. Never paste the replacement secret into tickets, chat or source code.

### Prompt-injection/tool abuse

Because tools are disabled in the baseline, external side effects should be impossible. If future tool-enabled releases are affected, disable the relevant capability flag first, then investigate traces/evidence.

## Rollback

Deploy the last exact SHA that passed all required gates. Do not roll forward to an unreviewed model, provider, safety configuration or dependency version during an incident.
