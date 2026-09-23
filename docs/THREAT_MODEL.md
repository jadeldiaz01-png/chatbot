# Threat Model

## Assets

- OpenAI API credential.
- User-provided content and any future uploaded media.
- Jadel Tech RD brand and service claims.
- CI/CD identity, repository integrity and release artifacts.
- Future external-system credentials and customer data.

## Primary threats and controls

| Threat | Baseline control | Residual risk / next gate |
|---|---|---|
| Direct prompt injection | Developer/system prompt, bounded input, no tools | Model can still be manipulated conversationally; keep authority advisory-only |
| Indirect prompt injection | External retrieval/tools disabled | RAG/MCP must add data isolation, structured extraction and adversarial evals |
| Secret leakage from user input | Deterministic redaction before model call | Pattern matching is not complete; public scale should add DLP at the edge |
| Sensitive output | Output moderation; no content logs | Domain-specific policy evals still required |
| Excessive agency | No tools/write actions | Future tools require least privilege and explicit approvals |
| Unbounded consumption | Input/history/output limits, retries bounded, session soft throttle | Add authenticated distributed rate limiting and account budget alerts before scale |
| Dependency compromise | Version pins, immutable base digest, CI audit, Dependabot | Release artifact still requires signed provenance/SBOM evidence |
| Model behavior drift | Model ID configurable; changes require eval gate | Build and maintain a representative held-out evaluation corpus |
| Data retention | `store=false`; no application content logs | Verify organization/project data controls and external processor policies |

## Prohibited baseline behaviors

The assistant cannot directly transfer money, place trades, publish content, submit marketplace proposals, change account settings, deploy infrastructure, approve contracts, or operate privileged external tools.
