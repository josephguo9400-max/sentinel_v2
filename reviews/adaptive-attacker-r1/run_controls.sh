set -euo pipefail
export NO_PROXY='*' no_proxy='*'
cd /home/claude/wt-verify-att
SENTINEL_REGISTRY_PATH="$PWD/artifacts/registry.json" python3 -m uvicorn serving.app:app --host 127.0.0.1 --port 8000 --log-level warning &
APID=$!; trap 'kill $APID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do curl -s --noproxy '*' http://127.0.0.1:8000/health >/dev/null && break; sleep 1; done
curl -s --noproxy '*' http://127.0.0.1:8000/health | head -c 600; echo
python3 "$1" "$2"
