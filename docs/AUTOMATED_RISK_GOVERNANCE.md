# Automated risk validation — no compulsory independent reviewer

The proposed CI gate classifies each PR as **LOW**, **MEDIUM**, or **HIGH** based on its changed paths. LOW and MEDIUM can pass automatically alongside existing CI tests. HIGH covers workflows, security, deployment, payments, trading execution, licensing, migrations, permissions, and dependency locks.

For HIGH, the repository owner must authorize the **exact 40-character PR HEAD SHA** through the GitHub Actions repository variable `RISK_APPROVED_SHA` (Settings → Secrets and variables → Actions → Variables). After reviewing the exact commit and changing the variable, rerun the failed `automated-risk` check. Clear or replace the variable after the decision. Any commit change invalidates the approval.

This is a GitHub merge-quality gate only. **It does not authorize production deploys, real money, trading LIVE, PayPal LIVE, external writes, paid services, or data use without permission.** Such operations retain their distinct manual approvals and existing fail-closed controls.

The workflow has read-only permissions, no credentials, no external dependencies, no self-hosted runner, a four-minute timeout, and reproducible unit tests. It runs on PRs, not automatically on production deployments.

## Enforcement prerequisite

A workflow alone is **not an enforced required status check**. Once this PR passes its tests, add `automated-risk` as a **required status check** in the protected main-branch ruleset, alongside every existing CI check. Only after confirming this enforcement should an administrator reduce **required approving reviews** to 0 and disable mandatory CODEOWNERS reviews where appropriate. Keep PR-only changes, conversation resolution, no force pushes, deletion restrictions, and required build/security tests. On GitHub Free private repos, check whether these settings are available; do not assume enforcement.

The independent reviewer is optional for LOW/MEDIUM; HIGH requires approval by the repository owner against the exact SHA, not necessarily a second reviewer.
