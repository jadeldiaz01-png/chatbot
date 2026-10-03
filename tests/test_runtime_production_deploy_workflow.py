import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / ".github" / "workflows" / "runtime-release-candidate.yml"
DEPLOY = ROOT / ".github" / "workflows" / "runtime-production-deploy.yml"
AUTH = ROOT / "ops" / "chatbot-production-deploy-authorization.json"
REMOTE = ROOT / "ops" / "deploy_chatbot_runtime.sh"


def test_release_candidate_exports_deployable_image() -> None:
    text = RELEASE.read_text(encoding="utf-8")
    assert 'docker save -o release-evidence/chatbot-runtime.tar' in text
    assert '"image_tar_sha256":os.environ["IMAGE_TAR_SHA256"]' in text
    assert "sha256sum chatbot-runtime.tar requirements.lock release.json sbom.cdx.json" in text
    assert "chatbot-runtime-release-${{ env.EXPECTED_SHA }}" in text


def test_one_shot_authorization_is_exact_and_nonfinancial() -> None:
    data = json.loads(AUTH.read_text(encoding="utf-8"))
    assert data == {
        "schema_version": "1.0",
        "authorized": True,
        "authorized_base_sha": "06ce2c7797b656e24601c81272933d727ef9d907",
        "release_workflow": "Chatbot runtime release candidate",
        "deploy_workflow": "Chatbot production deployment",
        "run_once": True,
        "runtime_provider": "openai",
        "runtime_model": "gpt-5.6-luna",
        "host_bind_address": "127.0.0.1",
        "host_bind_port": 18501,
        "public_exposure_mutation": False,
        "financial_execution": False,
        "external_action_authority": False,
        "rollback_required": True,
        "tailscale_only_deployer": True,
    }


def test_deployment_consumes_only_certified_release() -> None:
    text = DEPLOY.read_text(encoding="utf-8")
    assert "Chatbot runtime release candidate" in text
    assert "github.event.workflow_run.conclusion == 'success'" in text
    assert "github.event.workflow_run.event == 'push'" in text
    assert "github.event.workflow_run.head_branch == 'main'" in text
    assert "RELEASE_RUN_ATTEMPT" in text
    assert 'test "$RELEASE_RUN_ATTEMPT" = "1"' in text
    assert "chatbot-runtime-release-$EXPECTED_SHA" in text
    assert "sha256sum -c SHA256SUMS" in text
    assert "image_tar_sha256" in text
    assert "artifact_digest" in text
    assert 'test "$(git rev-parse refs/remotes/origin/main)" = "$EXPECTED_SHA"' in text
    assert 'first_parent="$(git rev-parse "$EXPECTED_SHA^1")"' in text
    assert "06ce2c7797b656e24601c81272933d727ef9d907" in text


def test_deployment_is_tailscale_only_and_unprivileged() -> None:
    text = DEPLOY.read_text(encoding="utf-8")
    assert "tailscale/github-action@780049a30b6ff5c378a9e7b389d15ece7a204888" in text
    assert '[[ "$DEPLOY_IP" =~ ^100\\. ]]' in text
    assert 'case "$DEPLOY_USER" in root|ubuntu)' in text
    assert "StrictHostKeyChecking=yes" in text
    assert "EXPECTED_DEPLOY_KEY_FINGERPRINT" in text
    assert "ssh-keygen -lf" in text
    assert "sudo " not in text
    assert "0.0.0.0" not in text
    assert "--network host" not in text


def test_remote_deploy_preserves_local_bind_and_rollback() -> None:
    text = REMOTE.read_text(encoding="utf-8")
    assert '-p "127.0.0.1:$HOST_BIND_PORT:8501"' in text
    assert "CHATBOT_ROLLBACK=PASS" in text
    assert "CHATBOT_ROLLBACK=FAIL" in text
    assert "PUBLIC_EXPOSURE_MUTATION=NO" in text
    assert "--env-file" in text


def test_remote_preflight_requires_production_api_key_without_reading_value() -> None:
    text = DEPLOY.read_text(encoding="utf-8")
    assert 'grep -Eq "^OPENAI_API_KEY=.+$"' in text
    assert "cat $HOME/.config/jadel-chatbot/runtime.env" not in text
    assert "printenv OPENAI_API_KEY" not in text


def test_post_deploy_verifies_image_sha_label_bind_and_health() -> None:
    text = DEPLOY.read_text(encoding="utf-8")
    assert "jadel.source_sha" in text
    assert "docker port" in text
    assert "127.0.0.1:$HOST_BIND_PORT" in text
    assert "/_stcore/health" in text
    assert "CHATBOT_PRODUCTION_DEPLOYMENT=PASS" in text
    assert "PUBLIC_EXPOSURE_MUTATION=NO" in text
    assert "FINANCIAL_EXECUTION=NO" in text


if __name__ == "__main__":
    test_release_candidate_exports_deployable_image()
    test_one_shot_authorization_is_exact_and_nonfinancial()
    test_deployment_consumes_only_certified_release()
    test_deployment_is_tailscale_only_and_unprivileged()
    test_remote_deploy_preserves_local_bind_and_rollback()
    test_remote_preflight_requires_production_api_key_without_reading_value()
    test_post_deploy_verifies_image_sha_label_bind_and_health()
    print("RUNTIME_PRODUCTION_DEPLOY_WORKFLOW_TESTS=PASS")
