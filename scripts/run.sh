#!/usr/bin/env bash
set -euo pipefail

# Calls the other API scripts. The agent must already be running.
# Usage: run.sh [question]
# Env: BASE_URL, OUT_DIR, RUN_DIR, JOB_ID, THREAD_ID, QUESTION, CHAT_TIMEOUT_SEC

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPTS/common.sh"
need_cmds

export RUN_DIR="${RUN_DIR:-$OUT_DIR/$(date +%Y%m%d-%H%M%S)}"
ensure_run_dir

"$SCRIPTS/check-service.sh"
"$SCRIPTS/check-errors.sh"

if [[ -n "${JOB_ID:-}" ]]; then
  "$SCRIPTS/ask.sh" "$@"
elif [[ $# -gt 0 ]]; then
  echo "JOB_ID is required to ask: $1" >&2
  exit 1
fi

log_summary "ok"
echo "OK  wrote ${RUN_DIR}"
