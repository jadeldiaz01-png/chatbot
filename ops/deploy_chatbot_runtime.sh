#!/usr/bin/env bash
set -euo pipefail

: "$EXPECTED_SHA"
: "$EXPECTED_TAR_SHA"
: "$EXPECTED_IMAGE_ID"
: "$CONTAINER_NAME"
: "$HOST_BIND_PORT"

incoming="$HOME/chatbot-incoming/$EXPECTED_SHA"
env_file="$HOME/.config/jadel-chatbot/runtime.env"

cd "$incoming"
actual_tar_sha="$(sha256sum chatbot-runtime.tar | awk '{print $1}')"
test "$actual_tar_sha" = "$EXPECTED_TAR_SHA"

command -v docker >/dev/null
docker info >/dev/null
test -f "$env_file"
test "$(stat -c '%a' "$env_file")" = "600"

docker load -i chatbot-runtime.tar >/dev/null
imported_id="$(docker image inspect "jadel-chatbot-runtime:$EXPECTED_SHA" --format '{{.Id}}')"
test "$imported_id" = "$EXPECTED_IMAGE_ID"

previous_image=""
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  previous_image="$(docker container inspect "$CONTAINER_NAME" --format '{{.Image}}')"
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

start_container() {
  image="$1"
  docker run -d     --name "$CONTAINER_NAME"     --restart unless-stopped     --env-file "$env_file"     -p "127.0.0.1:$HOST_BIND_PORT:8501"     --label "jadel.runtime=chatbot"     --label "jadel.source_sha=$EXPECTED_SHA"     "$image" >/dev/null
}

healthcheck() {
  attempts="$1"
  for _ in $(seq 1 "$attempts"); do
    if curl -fsS --max-time 3 "http://127.0.0.1:$HOST_BIND_PORT/_stcore/health" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  return 1
}

start_container "$EXPECTED_IMAGE_ID"

if ! healthcheck 30; then
  echo "CHATBOT_DEPLOY_HEALTH=FAIL" >&2
  docker logs --tail 80 "$CONTAINER_NAME" 2>&1     | sed -E 's/(sk-[A-Za-z0-9_-]{8,})/[REDACTED_KEY]/g' >&2 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

  if [ -n "$previous_image" ]; then
    start_container "$previous_image"
    if healthcheck 20; then
      echo "CHATBOT_ROLLBACK=PASS"
    else
      echo "CHATBOT_ROLLBACK=FAIL" >&2
      exit 51
    fi
  fi
  exit 50
fi

echo "CHATBOT_DEPLOY_HEALTH=PASS"
echo "CHATBOT_DEPLOY_IMAGE_ID=$EXPECTED_IMAGE_ID"
if [ -n "$previous_image" ]; then
  echo "CHATBOT_PREVIOUS_IMAGE_ID=$previous_image"
else
  echo "CHATBOT_PREVIOUS_IMAGE_ID=NONE"
fi
echo "CHATBOT_BIND=127.0.0.1:$HOST_BIND_PORT"
echo "PUBLIC_EXPOSURE_MUTATION=NO"
