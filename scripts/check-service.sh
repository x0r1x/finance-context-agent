#!/usr/bin/env bash
set -euo pipefail

# Probe GET /healthz and GET /readyz. Writes JSON into OUT_DIR (default ./out).
# readyz is success only when Redis and the parser both answer.
# Env: BASE_URL, OUT_DIR, RUN_DIR (reuse an existing run folder).

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
need_cmds
ensure_run_dir

fail=0
code="$(http_get /healthz "$RUN_DIR/healthz.json")"
if [[ "$code" != "200" ]]; then
  echo "healthz expected HTTP 200, got ${code}" >&2
  fail=1
elif [[ "$(json_field "$RUN_DIR/healthz.json" status)" != "ok" ]]; then
  echo "healthz body status is not ok" >&2
  fail=1
fi

code="$(http_get /readyz "$RUN_DIR/readyz.json")"
redis_ok="$(json_field "$RUN_DIR/readyz.json" redis)"
parser_ok="$(json_field "$RUN_DIR/readyz.json" parser)"
if [[ "$code" != "200" ]]; then
  echo "readyz expected HTTP 200, got ${code} (redis=${redis_ok:-<empty>} parser=${parser_ok:-<empty>})" >&2
  fail=1
elif [[ "$(json_field "$RUN_DIR/readyz.json" status)" != "ready" \
  || "$redis_ok" != "true" || "$parser_ok" != "true" ]]; then
  echo "readyz is not ready (redis=${redis_ok:-<empty>} parser=${parser_ok:-<empty>})" >&2
  fail=1
fi

log_summary "run_dir=${RUN_DIR}"
if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
echo "OK  wrote ${RUN_DIR}"
