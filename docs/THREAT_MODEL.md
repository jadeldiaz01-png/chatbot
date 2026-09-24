# Threat Model

## Assets

- NVIDIA API credential.
- User-provided content and any future uploaded media.
- Jadel Tech RD brand and service claims.
- CI/CD identity, repository integrity and release artifacts.
- Future external-system credentials and customer data.

## Primary threats and controls

| Threat | Baseline control | Residual risk / next gate |
|---|---|---|
| Direct prompt injection | Developer/system prompt, bounded input, no tools | Model can still be manipulated conversationally; keep authority advisory-only |
| Indirect prompt injection | External retrieval/tools disabled | RAG/MCP must add data isolation, structured extraction and adversarial evals |
| Secret leakage from user input | Deterministic redaction before model call, including `nvapi-` | Pattern matching is not complete; public scale should add DLP at the edge |
| Sensitive output | Dedicated NVIDIA input/output safety model; no content logs | Domain-specific policy evals still required |
| Safety-service failure | Unrecognized verdicts and safety-call errors fail closed | Availability depends on the safety endpoint |
| Excessive agency | No tools/write actions | Future tools require least privilege and explicit approvals |
| Unbounded consumption | Input/history/output limits, retries bounded, session soft throttle | Add authenticated distributed rate limiting and provider budget alerts before scale |
| Dependency compromise | Version pins, immutable base digest, CI audit, Dependabot | Release artifact still requires signed provenance/SBOM evidence |
| Model behavior drift | Model ID configurable; changes require eval gate | Build and maintain a representative held-out evaluation corpus |
| Provider data retention | No application content logs; sensitive data prohibited on trial endpoint | Review NVIDIA endpoint retention/terms; prefer partner or self-hosted NIM for production-sensitive data |

## Prohibited baseline behaviors

The assistant cannot directly transfer money, place trades, publish content, submit marketplace proposals, change account settings, deploy infrastructure, approve contracts, or operate privileged external tools.
