# Production Gates

The repository may be code-ready without being production-certified. Promotion is evidence-based and exact-SHA.

## G0 — Repository governance

Human/admin action required:
- enforce a GitHub ruleset or equivalent branch protection on `main`;
- require pull requests instead of direct pushes;
- require the exact-SHA `runtime-container-ci / test-and-runtime` and dependency-lock status checks;
- require human approval and resolved review conversations;
- prevent bypass except for an explicitly governed emergency path.

## G1 — Reproducible dependencies

Implemented in the candidate branch:
- direct requirements remain human-readable inputs;
- `requirements.lock` pins the Linux/amd64 CPython 3.12 runtime graph to exact wheel SHA-256 hashes;
- `requirements-ci.lock` does the same for runtime + CI tooling;
- Docker installs both with `--only-binary=:all: --require-hashes`;
- the dependency-lock workflow regenerates both graphs and fails if either differs from the committed lock.

This gate is satisfied for a release only when the lock-drift workflow and runtime CI both pass on the same exact SHA.

## G2 — Model baseline

Run `.github/workflows/model-eval.yml` manually against the production model and preserve the resulting report. Required evidence:
- exact Git SHA;
- exact model ID;
- exact prompt version;
- exact evaluation-corpus SHA-256;
- no critical automated failures;
- reviewed latency/token metrics;
- documented human review.

A newer model is a candidate, not an automatic upgrade.

## G3 — Public-edge controls

Before public scale, add authenticated or otherwise abuse-resistant edge controls:
- distributed rate limiting rather than session-only throttling;
- request/body limits;
- TLS and secure headers;
- bot/abuse controls appropriate to the deployment;
- budget and usage alerts.

## G4 — Deployment target

Provision a production target with a server-side secret store. Never bake `OPENAI_API_KEY` into the image. Record deployment identity, environment, runtime version, exact image digest/artifact hash and rollback target.

## G5 — Release evidence

After an approved merge, `release-evidence.yml` must succeed and produce:
- exact runtime image archive built from the hash-locked dependency graph;
- CycloneDX dependency SBOM from the runtime lock;
- both committed lockfiles;
- container inspection metadata;
- SHA-256 checksums;
- GitHub/Sigstore provenance attestation;
- SBOM attestation.

## G6 — Canary and rollback

Deploy the exact attested artifact to a controlled environment first. Exercise:
- health/readiness;
- one safe LLM request;
- failure behavior with unavailable upstream;
- logging without prompt/response content;
- rollback to the previous known-good SHA.

Only then may the production status be adjudicated.

## G7 — Advanced capability promotion

RAG, multimodal, ML/DL and autonomous tools are independent promotions. Each requires its own data rights, evals, security tests, cost evidence and rollback. Tool write authority additionally requires least privilege, allowlisting, structured I/O, immutable audit evidence and explicit human approval.

No capability inherits approval merely because the base chatbot is production-certified.
