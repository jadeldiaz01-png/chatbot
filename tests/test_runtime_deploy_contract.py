import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "ops" / "chatbot-runtime-deploy-contract.json"
WORKFLOW = ROOT / ".github" / "workflows" / "deploy-chatbot-runtime-vps.yml"
REMOTE = ROOT / "ops" / "deploy_chatbot_runtime.sh"


def test_deploy_contract_is_fail_closed() -> None:
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0"
    assert data["service"] == "jadel-chatbot"
    assert data["runtime_provider"] == "openai"
    assert data["runtime_model"] == "gpt-5.6-luna"
    assert data["host_bind_address"] == "127.0.0.1"
    assert data["host_bind_port"] == 18501
    assert data["public_exposure_mutation"] is False
    assert data["financial_execution"] is False
    assert data["external_action_authority"] is False
    assert data["rollback_required"] is True
    assert data["tailscale_only_deployer"] is True


def test_remote_installer_requires_local_bind_and_rollback() -> None:
    text = REMOTE.read_text(encoding="utf-8")
    assert '-p "127.0.0.1:$HOST_BIND_PORT:8501"' in text
    assert "CHATBOT_ROLLBACK=PASS" in text
    assert "CHATBOT_ROLLBACK=FAIL" in text
    assert "PUBLIC_EXPOSURE_MUTATION=NO" in text
    assert "--restart unless-stopped" in text
    assert "--env-file" in text
    assert "docker info" in text
    assert "0.0.0.0" not in text
    assert "--network host" not in text


def test_workflow_requires_exact_sha_tailscale_and_confirmation() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "DEPLOY_CHATBOT_RUNTIME" in text
    assert 'test "$GITHUB_SHA" = "$EXPECTED_SHA"' in text
    assert "PROTECTED_MAIN_SHA_MATCH=PASS" in text
    assert "tailscale/github-action@780049a30b6ff5c378a9e7b389d15ece7a204888" in text
    assert "VPS_GATEWAY_TAILSCALE_IP" in text
    assert "OPENAI_API_KEY" in text
    assert "Refusing privileged deploy user" in text
    assert "PUBLIC_EXPOSURE_MUTATION=NO" in text
    assert "FINANCIAL_EXECUTION=NO" in text
    assert "0.0.0.0:18501" not in text
    assert "sudo " not in text
