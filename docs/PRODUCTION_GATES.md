# Production Gates

The repository may be code-ready without being production-certified. Promotion is evidence-based and exact-SHA.

## G0 — Repository governance

Verified for `main`:
- active repository ruleset targets the default branch;
- pull requests are required;
- `test-and-runtime` and `verify-locks` are required;
- review conversations must be resolved;
- deletion and non-fast-forward updates are blocked;
- no bypass actors are configured.

## G1 — Reproducible dependencies

Implemented in the candidate branch:
- direct requirements remain human-readable inputs;
- `requirements.lock` pins the Linux/amd64 CPython 3.12 runtime graph to exact wheel SHA-256 hashes;
- `requirements-ci.lock` does the same for runtime + CI tooling;
- Docker installs both with `--only-binary=:all: --require-hashes`;
- the dependency-lock workflow regenerates both graphs and fails if either differs from the committed lock.

This gate is satisfied for a release only when the lock-drift workflow and runtime CI both pass on the same exact SHA.

## G2 — Nemotron model baseline

Run `.github/workflows/model-eval.yml` against `nvidia/nemotron-3-ultra-550b-a55b` using the protected `model-evaluation` environment and preserve the resulting report. Required evidence:
- exact Git SHA;
- exact NVIDIA model ID;
- exact NVIDIA safety-model ID;
- exact prompt version;
- exact evaluation-corpus SHA-256;
- no critical automated failures;
- reviewed latency/token metrics;
- documented human review.

A model/provider change is a candidate, not an automatic promotion. The previous OpenAI baseline is not a fallback.

## G3 — Provider privacy/data handling

Before sensitive production traffic:
- confirm the NVIDIA endpoint's retention and data-use terms;
- do not send sensitive data through the NVIDIA Build trial endpoint while it may record API inputs/outputs;
- prefer a partner or self-hosted NIM endpoint when the required privacy/retention guarantees cannot be met by the trial endpoint;
- document the approved endpoint and secret scope.

## G4 — Public-edge controls

Before public scale, add authenticated or otherwise abuse-resistant edge controls:
- distributed rate limiting rather than session-only throttling;
- request/body limits;
- TLS and secure headers;
- bot/abuse controls appropriate to the deployment;
- budget and usage alerts.

## G5 — Deployment target

Provision a production target with a server-side secret store. Never bake `NVIDIA_API_KEY` into the image. Record deployment identity, environment, runtime version, exact image digest/artifact hash and rollback target.

## G6 — Release evidence

After an approved merge, `release-evidence.yml` must succeed and produce:
- exact runtime image archive built from the hash-locked dependency graph;
- CycloneDX dependency SBOM from the runtime lock;
- both committed lockfiles;
- container inspection metadata;
- SHA-256 checksums;
- GitHub/Sigstore provenance attestation;
- SBOM attestation.

## G7 — Canary and rollback

Deploy the exact attested artifact to a controlled environment first. Exercise:
- health/readiness;
- one safe Nemotron request through input/output safety checks;
- failure behavior with unavailable NVIDIA upstream;
- failure behavior with an invalid safety verdict;
- logging without prompt/response/reasoning content;
- rollback to the previous known-good SHA.

Only then may the production status be adjudicated.

## G8 — Advanced capability promotion

RAG, multimodal, ML/DL and autonomous tools are independent promotions. Each requires its own data rights, evals, security tests, cost evidence and rollback. Tool write authority additionally requires least privilege, allowlisting, structured I/O, immutable audit evidence and explicit human approval.

No capability inherits approval merely because the base chatbot is production-certified.
