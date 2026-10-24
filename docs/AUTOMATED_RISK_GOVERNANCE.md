# Risk-tiered validation for Codex PRs

The independent technical reviewer is **optional** for ordinary changes. This
proposal maintains automatic tests and uses an independent, trusted CI *check*
instead of requiring another *person* on every PR.

## Classification and effects
- **LOW:** documentation and ordinary prose; automated validation plus existing CI.
- **MEDIUM:** ordinary code; automated validation plus existing CI.
- **HIGH:** workflows, security/policy changes, dependencies, licensing,
  financial execution, production deployments, credentials, migrations, etc.
  Requires an explicit authorization matching the exact PR HEAD SHA.
- None of these statuses grants trading LIVE, PayPal LIVE, production deployment,
  real-money execution, spending, external effects or permission to use data.
  Those operations still require their separate manual approval.

## Trusted enforcement, NOT self-approval
`.github/workflows/automated-risk.yml` tests the PR version and provides
advisory feedback. PR code can edit that version, so **DO NOT** use its
`automated-risk` status as the sole required branch-protection check.

`.github/workflows/automated-risk-trusted.yml` runs on
`pull_request_target`, checks out **only the protected default branch**, and
examines untrusted PR file metadata through GitHub's read-only API. It never
checks out or executes PR code. It fails closed on API errors, HEAD races, huge
change sets and missing exact-SHA approval.

The **required** ruleset status context must be `automated-risk-trusted`
(after this code is merged to the protected branch). Keep all existing build,
security and regression checks required as well.

## High-risk authorization (repository owner)
Inspect a HIGH-risk PR's actual changes, tests and full 40-character HEAD SHA.
In Settings → Secrets and variables → Actions → Variables, set the repository
variable `RISK_APPROVED_SHA` to that SHA. Rerun the failed risk check (or
synchronize the PR); **any subsequent commit invalidates that authorization**.
Clear or replace the variable after the decision. The variable is not a
license for the workflow to deploy or spend.

## Two-stage safe rollout
1. This initial draft PR can only use existing branch checks and the untrusted
   PR-version self-test. The trusted workflow does **not** run on this PR until
   the code is installed on the default branch. Retain current protection and
   human approval for this bootstrap.
2. After verifying and merging the bootstrap under the existing rules, confirm
   `automated-risk-trusted` runs and add it to **required status checks**.
3. Only after that, where plan and policy permit, set generic required PR
   approval count to 0 and disable mandatory CODEOWNERS reviews. Retain
   PR-only merges, security checks, no force pushes, deletion restrictions and
   required conversation resolution. Keep specific human gates for HIGH risk
   and all critical production/financial actions.

This uses read-only permissions and a GitHub-hosted runner with a short timeout,
avoiding self-hosted infrastructure or paid services. Available GitHub Actions
minutes and private-repository ruleset features still depend on the plan.
