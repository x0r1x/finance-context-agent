#!/usr/bin/env bash
set -euo pipefail

# Send one user message to POST /v1/chat/completions.
# A pause (awaiting_user true) is a successful HTTP check.
# Usage: ask.sh [question]
# Env: BASE_URL, OUT_DIR, RUN_DIR, JOB_ID (required), THREAD_ID, QUESTION, CHAT_TIMEOUT_SEC

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
need_cmds
ensure_run_dir

if [[ $# -gt 1 ]]; then
  echo "pass the question as one argument" >&2
  exit 1
fi

question="${1:-${QUESTION:-}}"
if [[ -z "$question" ]]; then
  echo "question missing: pass it as an argument or set QUESTION" >&2
  exit 1
fi
if [[ -z "${JOB_ID:-}" ]]; then
  echo "JOB_ID is required" >&2
  exit 1
fi

python3 -c '
import json, os, sys
body = {
    "job_id": os.environ["JOB_ID"],
    "messages": [{"role": "user", "content": sys.argv[1]}],
}
thread_id = os.environ.get("THREAD_ID", "").strip()
if thread_id:
    body["thread_id"] = thread_id
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump(body, handle, ensure_ascii=False)
    handle.write("\n")
' "$question" "$RUN_DIR/ask.json"

code="$(http_post_json /v1/chat/completions "$RUN_DIR/completion.json" "$RUN_DIR/ask.json")"
if [[ "$code" != "200" ]]; then
  echo "chat completions expected HTTP 200, got ${code}" >&2
  exit 1
fi

object="$(json_field "$RUN_DIR/completion.json" object)"
model="$(json_field "$RUN_DIR/completion.json" model)"
finish="$(json_field "$RUN_DIR/completion.json" choices.0.finish_reason)"
thread_id="$(json_field "$RUN_DIR/completion.json" thread_id)"
awaiting="$(json_field "$RUN_DIR/completion.json" awaiting_user)"
satisfactory="$(json_field "$RUN_DIR/completion.json" satisfactory)"
content="$(json_field "$RUN_DIR/completion.json" choices.0.message.content)"

if [[ "$object" != "chat.completion" ]]; then
  echo "completion object is ${object:-<empty>}" >&2
  exit 1
fi
if [[ "$model" != "finance-context-agent" ]]; then
  echo "completion model is ${model:-<empty>}" >&2
  exit 1
fi
if [[ "$finish" != "stop" ]]; then
  echo "finish_reason is ${finish:-<empty>}" >&2
  exit 1
fi
if [[ -z "$thread_id" ]]; then
  echo "completion is missing thread_id" >&2
  exit 1
fi

log_summary "thread_id=${thread_id}"
log_summary "awaiting_user=${awaiting}"
log_summary "satisfactory=${satisfactory}"
log_summary "assistant=${content}"
echo "OK  wrote ${RUN_DIR}"
