# Shared helpers for API scripts. Source from the other scripts; do not run directly.
# CHAT_TIMEOUT_SEC is how long this client waits for one POST. It does not
# release the server's thread lock. A dead run keeps the thread until
# LOCK_TTL_SEC on the server (default 900).

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$_COMMON_DIR/.." && pwd)"

BASE_URL="${BASE_URL:-http://127.0.0.1:8090}"
OUT_DIR="${OUT_DIR:-$ROOT/out}"
CHAT_TIMEOUT_SEC="${CHAT_TIMEOUT_SEC:-180}"

need_cmds() {
  local missing=0
  local cmd
  for cmd in curl python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      echo "missing required command: $cmd" >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    exit 1
  fi
}

ensure_run_dir() {
  if [[ -z "${RUN_DIR:-}" ]]; then
    RUN_DIR="$OUT_DIR/$(date +%Y%m%d-%H%M%S)"
  fi
  mkdir -p "$RUN_DIR"
  SUMMARY="${SUMMARY:-$RUN_DIR/summary.txt}"
  touch "$SUMMARY"
}

log_summary() {
  printf '%s\n' "$*" | tee -a "$SUMMARY" >&2
}

# Usage: json_field FILE dotted.key
# Dots walk objects. A numeric part indexes a list. Booleans print true/false.
json_field() {
  python3 -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
cur = data
for part in sys.argv[2].split("."):
    if isinstance(cur, list):
        cur = cur[int(part)]
    elif isinstance(cur, dict):
        cur = cur.get(part, "")
    else:
        cur = ""
        break
if isinstance(cur, bool):
    print("true" if cur else "false")
elif cur is None:
    print("")
else:
    print(cur)
' "$1" "$2"
}

# Usage: http_get PATH OUTFILE [quiet]
# Prints the HTTP status. The body is written to OUTFILE. Timeout is 10s.
http_get() {
  local path="$1"
  local outfile="$2"
  local quiet="${3:-}"
  local code
  mkdir -p "$(dirname "$outfile")"
  code="$(curl -sS --max-time 10 -o "$outfile" -w "%{http_code}" "${BASE_URL}${path}")"
  if [[ "$quiet" != "quiet" ]]; then
    log_summary "GET ${path} -> ${code} (${outfile#"$ROOT/"})"
  fi
  printf '%s\n' "$code"
}

# Usage: http_post_json PATH OUTFILE BODYFILE [curl args...]
# Prints the HTTP status. Does not use --fail: callers compare 400 bodies.
http_post_json() {
  local path="$1"
  local outfile="$2"
  local bodyfile="$3"
  shift 3
  local code
  mkdir -p "$(dirname "$outfile")"
  code="$(
    curl -sS --max-time "$CHAT_TIMEOUT_SEC" \
      -o "$outfile" \
      -w "%{http_code}" \
      -H "content-type: application/json" \
      --data-binary "@${bodyfile}" \
      "$@" \
      "${BASE_URL}${path}"
  )"
  log_summary "POST ${path} -> ${code} (${outfile#"$ROOT/"})"
  printf '%s\n' "$code"
}
