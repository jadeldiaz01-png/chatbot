# Operations Runbook

## Health

- Liveness/readiness signal: `/_stcore/health`.
- Container must run as UID/GID `10001:10001`.
- Missing `OPENAI_API_KEY` is a fail-closed configuration condition visible to the user; no model call is attempted.

## Incident classes

### OpenAI/API degradation

Symptoms: elevated errors, timeouts or rate limits. Keep retries bounded. Return the generic degraded-service message. Do not silently switch to an unevaluated model.

### Cost anomaly

Freeze model/tool promotions, lower traffic at the edge, inspect token telemetry and account-side budgets. Do not solve a cost incident by removing safety checks.

### Suspected secret exposure

Revoke/rotate the affected secret in its provider, inspect metadata-only logs, invalidate sessions where applicable and document the incident. Never paste the replacement secret into tickets, chat or source code.

### Prompt-injection/tool abuse

Because tools are disabled in the baseline, external side effects should be impossible. If future tool-enabled releases are affected, disable the relevant capability flag first, then investigate traces/evidence.

## Rollback

Deploy the last exact SHA that passed all required gates. Do not roll forward to an unreviewed model or dependency version during an incident.
