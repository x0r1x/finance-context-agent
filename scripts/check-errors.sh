#!/usr/bin/env bash
set -euo pipefail

# Probe 400 responses that return before the thread lock and the model.
# The agent process must be up. Redis and the parser are not called.
# Env: BASE_URL, OUT_DIR, RUN_DIR, CHAT_TIMEOUT_SEC

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
need_cmds
ensure_run_dir

expect_error() {
  local name="$1"
  local body="$2"
  local want="$3"
  shift 3
  local request="$RUN_DIR/${name}.request.json"
  local response="$RUN_DIR/${name}.json"
  local code got
  printf '%s\n' "$body" >"$request"
  code="$(http_post_json /v1/chat/completions "$response" "$request" "$@")"
  if [[ "$code" != "400" ]]; then
    echo "${name} expected HTTP 400, got ${code}" >&2
    exit 1
  fi
  got="$(json_field "$response" error)"
  if [[ "$got" != "$want" ]]; then
    echo "${name} expected error ${want}, got ${got:-<empty>}" >&2
    exit 1
  fi
}

expect_error error-stream \
  '{"stream":true,"messages":[{"role":"user","content":"x"}]}' \
  stream_unsupported
expect_error error-messages '{}' messages_required
expect_error error-role \
  '{"messages":[{"role":"assistant","content":"x"}]}' \
  last_message_not_user
expect_error error-empty \
  '{"messages":[{"role":"user","content":" "}]}' \
  user_message_required
expect_error error-thread \
  '{"thread_id":"bad id","messages":[{"role":"user","content":"x"}]}' \
  bad_thread_id
expect_error error-job \
  '{"job_id":"bad id","messages":[{"role":"user","content":"x"}]}' \
  bad_job_id
expect_error error-header-job \
  '{"messages":[{"role":"user","content":"x"}]}' \
  bad_job_id \
  -H "X-Job-Id: bad id"

log_summary "error_contract=ok"
echo "OK  wrote ${RUN_DIR}"
