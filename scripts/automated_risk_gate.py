#!/usr/bin/env python3
"""Fail-closed, dependency-free risk gate for GitHub pull requests.

Only source changes in the PR are inspected. This gate never deploys or trades.
HIGH risk requires the repository owner's exact-HEAD approval via an Actions
repository variable; this is not an independent reviewer requirement.
"""
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

SHA = re.compile(r"^[0-9a-fA-F]{40}$")
HIGH_TERMS = re.compile(
    r"(?:^|[/_.-])(?:paypal|payment|payments|capture|checkout|transfer|"
    r"withdraw|withdrawal|order|orders|execution|execute|binance|broker|"
    r"live|production|prod|deploy|release|webhook|secret|secrets|"
    r"credential|credentials|auth|authentication|authorization|iam|"
    r"permission|permissions|security|policy|risk|gate|approval|"
    r"billing|financial|capital|migration|migrations|license|"
    r"licensing|compliance|dataset)(?:$|[/_.-])",
    re.IGNORECASE,
)
HIGH_PREFIXES = (
    ".github/", "security/", "policy/", "infra/", "migrations/",
)
HIGH_EXACT = {
    "codeowners", ".env", ".gitignore", "dockerfile", "docker-compose.yml",
    "docker-compose.yaml", "pyproject.toml", "package.json",
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock",
}
CODE_SUFFIXES = (
    ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".go",
    ".rs", ".java", ".sh", ".bash", ".sql", ".html", ".css",
    ".vue", ".svelte", ".json", ".toml", ".yaml", ".yml",
)
ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def classify_path(path):
    value = path.replace("\\", "/").lower()
    while value.startswith("./"):
        value = value[2:]
    filename = value.rsplit("/", 1)[-1]
    if not value or value.startswith("../"):
        raise ValueError("Invalid changed path")
    if (
        value.startswith(HIGH_PREFIXES)
        or filename in HIGH_EXACT
        or filename.startswith("dockerfile")
        or filename.endswith((".lock", ".pem", ".key", ".env"))
        or HIGH_TERMS.search(value)
    ):
        return "HIGH"
    if value.startswith(("src/", "scripts/", "api/", "app/", "runtime/", "gateway/")):
        return "MEDIUM"
    if filename.endswith(CODE_SUFFIXES):
        return "MEDIUM"
    return "LOW"


def evaluate(paths, head_sha, approved_sha=""):
    if not SHA.fullmatch(head_sha):
        raise ValueError("Invalid PR head SHA")
    if not paths:
        raise ValueError("No changed paths; cannot certify PR")
    classes = {path: classify_path(path) for path in sorted(set(paths))}
    level = max(classes.values(), key=lambda item: ORDER[item])
    approved = bool(SHA.fullmatch(approved_sha)) and approved_sha.lower() == head_sha.lower()
    allowed = level != "HIGH" or approved
    return {
        "policy_version": "1.0",
        "head_sha": head_sha.lower(),
        "risk": level,
        "decision": "PASS" if allowed else "BLOCKED_HUMAN_AUTHORIZATION_REQUIRED",
        "exact_head_approval": bool(approved) if level == "HIGH" else None,
        "changed_files": classes,
        "financial_execution_authorized": False,
        "production_deploy_authorized": False,
    }


def git_changed_paths(base_sha, head_sha):
    if not SHA.fullmatch(base_sha) or not SHA.fullmatch(head_sha):
        raise ValueError("Invalid base/head SHA")
    completed = subprocess.run(
        ["git", "diff", "--no-renames", "--name-only", "-z",
         "--diff-filter=ACDMRTUXB", base_sha, head_sha],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return [part.decode("utf-8") for part in completed.stdout.split(b"\0") if part]



def github_api_json(url, token):
    """Read GitHub metadata only; never execute code from an untrusted PR."""
    if not token or not url.startswith("https://api.github.com/repos/"):
        raise ValueError("GitHub read token or API URL missing")
    request = urllib.request.Request(
        url, headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer " + token,
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "jadel-trusted-risk-gate",
        }
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def trusted_pr_paths(repository, number, expected_head, token):
    """Inspect PR filenames with a trusted default-branch runner and read-only API."""
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+", repository):
        raise ValueError("Invalid repository")
    if not str(number).isdigit() or int(number) < 1:
        raise ValueError("Invalid PR number")
    if not SHA.fullmatch(expected_head):
        raise ValueError("Invalid event PR HEAD")
    endpoint = f"https://api.github.com/repos/{repository}/pulls/{number}"
    before = github_api_json(endpoint, token)
    if before["head"]["sha"].lower() != expected_head.lower():
        raise ValueError("PR HEAD changed since the event")
    paths = []
    for page in range(1, 32):
        batch = github_api_json(f"{endpoint}/files?per_page=100&page={page}", token)
        if not isinstance(batch, list):
            raise ValueError("GitHub file API did not return a list")
        for item in batch:
            paths.append(item["filename"])
            if item.get("previous_filename"):
                paths.append(item["previous_filename"])
        if len(batch) < 100:
            break
    if len(paths) >= 3000:
        raise ValueError("PR has too many changed paths to certify")
    after = github_api_json(endpoint, token)
    if after["head"]["sha"].lower() != expected_head.lower():
        raise ValueError("PR HEAD changed during risk verification")
    return paths


def main():
    try:
        head = os.environ["PR_HEAD_SHA"]
        if os.environ.get("GITHUB_EVENT_NAME") == "pull_request_target":
            files = trusted_pr_paths(
                os.environ["GH_REPOSITORY"], os.environ["PR_NUMBER"],
                head, os.environ["GH_READ_TOKEN"],
            )
        else:
            base = os.environ["PR_BASE_SHA"]
            files = git_changed_paths(base, head)
        result = evaluate(files, head, os.environ.get("RISK_APPROVED_SHA", ""))
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError, UnicodeError, urllib.error.URLError) as exc:
        print(f"AUTOMATED_RISK_GATE=FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as output:
            output.write(
                "## Automated PR risk gate\n"
                f"- Risk: **{result['risk']}**\n"
                f"- Decision: **{result['decision']}**\n"
                f"- PR HEAD: `{result['head_sha']}`\n"
                "- Production and financial execution: **not authorized**\n"
            )
    return 0 if result["decision"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
