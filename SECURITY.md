# Security Policy

Do not report credentials, API keys, private keys, recovery codes, payment credentials, or personal identity documents in GitHub issues or pull requests.

For suspected credential exposure:
1. Revoke or rotate the credential at its provider first.
2. Preserve non-secret evidence such as timestamps, commit SHAs and affected component names.
3. Remove exposed material from active configuration and deployment paths.
4. Review logs and downstream systems without copying the secret into tickets or chat.

The production baseline intentionally exposes no autonomous external tools. Any future tool integration must use least privilege, an explicit allowlist, structured inputs/outputs, audit logs and human approval for sensitive actions.
