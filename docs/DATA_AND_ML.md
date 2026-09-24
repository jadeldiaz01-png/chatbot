# Data, RAG, ML and Deep Learning Contract

## Data classes

1. Public product/service documentation with explicit provenance.
2. Internal operational knowledge approved for model access.
3. User session content, minimized and not intentionally persisted by this application.
4. Evaluation data, versioned separately from training data.
5. Labels/outcomes used for any future ML ranking model.

## RAG admission contract

Every document must have: source URI or system-of-record identifier, content hash, acquisition timestamp, owner, rights/usage classification, sensitivity class, expiration/review date and parser version. Untrusted document text must never become developer/system instructions.

## ML/DL policy

Do not train a custom model because the technology is available. A custom model is justified only when a measurable task has enough rights-cleared labeled data and a simpler prompt/RAG baseline is insufficient. Training, validation and test sets must be separated before model selection; the held-out set remains closed until final evaluation.

## Multimodal policy

Image input is implemented but disabled by default. Activation requires accepted file types, strict size limits, image moderation, privacy review, representative image evals and cost/latency measurements. Audio/video should be separate capabilities with their own data contracts rather than implicitly inheriting image approval.
