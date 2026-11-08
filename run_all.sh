#!/usr/bin/env bash
# Runs every part of the assignment end to end and collects everything into takeout.zip.
#
#   conda activate a5
#   bash run_all.sh
#
# Writes the graded outputs into partA/ partB/ partC/ partD/ and a full copy of all outputs, logs
# and machine details into takeout/ -> takeout.zip (git-ignored, never committed).
# Takes about 10-15 minutes. Do not use the laptop for heavy work while it runs (it affects latency).

cd "$(dirname "$0")"
ROOT="$(pwd)"
TAKE="$ROOT/takeout"
LOGS="$TAKE/logs"
export MLFLOW_TRACKING_URI="sqlite:///$ROOT/mlflow.db"
export MLFLOW_DISABLE_AGENT_HINT=1
export PYTHONWARNINGS="ignore"
PORTS="8000 8001 5101 5102 8100"
PIDS=()

rm -rf "$TAKE" takeout.zip
mkdir -p "$LOGS"
STATUS="$TAKE/STATUS.txt"
START_ALL=$(date +%s)
echo "run_all.sh started $(date '+%Y-%m-%d %H:%M:%S %Z')" | tee "$STATUS"

# ---------------------------------------------------------------- helpers
stop_servers() {
  for p in "${PIDS[@]}"; do
    pkill -TERM -P "$p" 2>/dev/null
    kill -TERM "$p" 2>/dev/null
  done
  sleep 2
  for p in "${PIDS[@]}"; do
    pkill -KILL -P "$p" 2>/dev/null
    kill -KILL "$p" 2>/dev/null
  done
  PIDS=()
}
trap 'stop_servers' EXIT
trap 'echo "interrupted"; exit 130' INT TERM

wait_http() {   # wait_http URL [timeout_s]
  local t=0
  until curl -sf "$1" >/dev/null 2>&1; do
    sleep 1; t=$((t + 1))
    if [ "$t" -ge "${2:-90}" ]; then return 1; fi
  done
}

step() {        # step NAME command...   -> logs/NAME.log, line in STATUS.txt
  local name="$1"; shift
  local t0=$(date +%s)
  echo; echo ">>> $name"
  "$@" > "$LOGS/$name.log" 2>&1
  local rc=$?
  stop_servers
  local dt=$(( $(date +%s) - t0 ))
  printf "%-28s %-4s exit=%-3s %4ss\n" "$name" "$([ $rc -eq 0 ] && echo OK || echo FAIL)" "$rc" "$dt" | tee -a "$STATUS"
  tail -n 4 "$LOGS/$name.log" | sed 's/^/    /'
  return $rc
}

start_api() {   # start_api MODEL PORT WORKERS
  ETA_MODEL="$1" uvicorn eta_app:app --app-dir partA --host 127.0.0.1 --port "$2" --workers "$3" \
      --log-level warning > "$LOGS/uvicorn_$2_w$3.log" 2>&1 &
  PIDS+=($!)
  wait_http "http://127.0.0.1:$2/healthz" 60
}

# ---------------------------------------------------------------- 0. pre-flight
echo ">>> pre-flight"
python - <<'EOF' || { echo "Activate the a5 environment and install requirements first (see README)."; exit 1; }
import sys
assert sys.version_info >= (3, 10), "Python 3.10+ needed"
import fastapi, uvicorn, sklearn, pandas, mlflow, ray, prometheus_client, pytest, requests  # noqa
print("python", sys.version.split()[0], "| mlflow", mlflow.__version__, "| ray", ray.__version__)
EOF
command -v curl >/dev/null || { echo "curl is missing: sudo apt install -y curl"; exit 1; }
for p in $PORTS; do
  if curl -s -o /dev/null "http://127.0.0.1:$p" 2>/dev/null; then
    echo "Port $p is already in use. Stop whatever runs there (see README, 'Reset'), then re-run."; exit 1
  fi
done

# ---------------------------------------------------------------- 1. machine description
{
  echo "collected: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "hostname: $(hostname)"
  echo "kernel: $(uname -srm)"
  grep -E '^(PRETTY_NAME)=' /etc/os-release 2>/dev/null
  echo "python: $(python --version 2>&1) at $(command -v python)"
  echo "conda env: ${CONDA_DEFAULT_ENV:-none}"
  echo "logical CPUs (nproc): $(nproc)"
  echo "physical cores: $(lscpu -p=CORE,SOCKET 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
  echo "RAM total: $(free -h | awk '/^Mem:/{print $2}')"
  echo "RAM available at start: $(free -h | awk '/^Mem:/{print $7}')"
  echo "load average at start: $(cut -d' ' -f1-3 /proc/loadavg)"
  echo "cpu governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
  echo "power profile: $(powerprofilesctl get 2>/dev/null || echo n/a)"
  echo "on AC power: $(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1 || echo n/a)"
  echo; echo "---- lscpu"; lscpu
  echo; echo "---- free -h"; free -h
  echo; echo "---- git"; git rev-parse HEAD 2>/dev/null; git status --short 2>/dev/null
} > "$TAKE/machine.txt" 2>&1
pip freeze > "$TAKE/pip_freeze.txt" 2>/dev/null
echo "machine description -> takeout/machine.txt"

# ---------------------------------------------------------------- 2. data and models
rm -rf data mlflow.db mlruns mlartifacts
step 01_eta_data python eta_data.py
python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"

# ---------------------------------------------------------------- 3. Part A
step 02_partA_pytest python -m pytest partA -v -p no:cacheprovider

loadtest_1w() {
  start_api data/eta_v1.joblib 8000 1 || { echo "API did not start"; return 1; }
  python load_test.py --url http://127.0.0.1:8000/v1/models/eta:predict --data data/eta_sample.csv \
         --concurrency 1 2 4 8 16 --duration 4 --out partA/loadtest_results.json || return 1
  echo "--- batch size 1 vs 100 at concurrency 1"
  python load_test.py --url http://127.0.0.1:8000/v1/models/eta:predict --data data/eta_sample.csv \
         --concurrency 1 --duration 4 --batch 1 --out "$TAKE/lt_batch1.json" || return 1
  python load_test.py --url http://127.0.0.1:8000/v1/models/eta:predict --data data/eta_sample.csv \
         --concurrency 1 --duration 4 --batch 100 --out "$TAKE/lt_batch100.json" || return 1
  curl -s http://127.0.0.1:8000/metrics > "$TAKE/partA_metrics_after_load.txt"
  stop_servers
}
step 03_partA_loadtest_1worker loadtest_1w

loadtest_2w() {
  start_api data/eta_v1.joblib 8000 2 || { echo "API did not start"; return 1; }
  python load_test.py --url http://127.0.0.1:8000/v1/models/eta:predict --data data/eta_sample.csv \
         --concurrency 1 2 4 8 16 --duration 4 --out partA/loadtest_results_2workers.json || return 1
  stop_servers
}
step 04_partA_loadtest_2workers loadtest_2w

python - <<'EOF' > "$LOGS/05_partA_batch_summary.log" 2>&1
import json
b1 = json.load(open("takeout/lt_batch1.json"))[0]; b100 = json.load(open("takeout/lt_batch100.json"))[0]
out = {}
for name, r, rows in (("batch_1", b1, 1), ("batch_100", b100, 100)):
    ms_per_req = 1000 / r["rps"] if r["rps"] else None
    out[name] = {**r, "rows_per_request": rows, "ms_per_request_mean": round(ms_per_req, 3),
                 "ms_per_row": round(ms_per_req / rows, 4), "rows_per_s": round(r["rps"] * rows, 1)}
json.dump(out, open("partA/loadtest_batch.json", "w"), indent=2)
print(json.dumps(out, indent=2))
EOF
echo "batch summary -> partA/loadtest_batch.json"

# ---------------------------------------------------------------- 4. Part B
step 06_partB_pytest python -m pytest partB -v -p no:cacheprovider
step 07_partB_register python partB/eta_mlflow.py register

gate() {
  echo "### Registry (MLflow model eta-model)"
  python partB/eta_mlflow.py table || return 1
  echo; echo "### promotion_gate() with default thresholds"
  python partB/eta_mlflow.py gate || return 1
  echo; echo "### promotion_gate(min_rel_gain=0.60)"
  python partB/eta_mlflow.py gate --min-rel-gain 0.60 || return 1
}
step 08_partB_gate gate && grep -v -E "WARNING|warn\(|UserWarning|color_warning" "$LOGS/08_partB_gate.log" > partB/gate_output.txt

shadow_run() {
  mlflow models serve -m "models:/eta-model@prod" -h 127.0.0.1 -p 5101 --env-manager local > "$LOGS/mlflow_serve_prod.log" 2>&1 &
  PIDS+=($!)
  mlflow models serve -m "models:/eta-model@candidate" -h 127.0.0.1 -p 5102 --env-manager local > "$LOGS/mlflow_serve_cand.log" 2>&1 &
  PIDS+=($!)
  wait_http http://127.0.0.1:5101/ping 120 && wait_http http://127.0.0.1:5102/ping 120 || { echo "mlflow servers did not start"; return 1; }
  echo "prod (alias prod -> v1) on :5101, candidate (alias candidate -> v2) on :5102; same 200 rows of data/eta_fresh.csv, one row per request"
  python partB/eta_mlflow.py shadow --prod-url http://127.0.0.1:5101/invocations \
         --cand-url http://127.0.0.1:5102/invocations --n 200 || return 1
  stop_servers
}
step 09_partB_shadow shadow_run && grep -v -E "WARNING|warn\(" "$LOGS/09_partB_shadow.log" > partB/shadow_output.txt

# ---------------------------------------------------------------- 5. Part C
step 10_partC_verify_ray python partC/verify_ray.py --port 8100
grep -v -E "^\(|WARNING|INFO|raylet" "$LOGS/10_partC_verify_ray.log" > partC/verify_output.txt

# ---------------------------------------------------------------- 6. Part D
smoke() {
  start_api data/eta_v1.joblib 8000 1 || return 1
  start_api data/eta_v2.joblib 8001 1 || return 1
  echo "##### 1) Part A serving v1, expecting v1 (should PASS)"
  python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1 --max-p95-ms 80; rc1=$?
  echo "exit code: $rc1"
  echo; echo "##### 2) Part A serving v2, expecting v2 (should PASS)"
  python partD/smoke_eta.py --url http://127.0.0.1:8001 --expect-version v2 --max-p95-ms 80; rc2=$?
  echo "exit code: $rc2"
  echo; echo "##### 3) wrong version deployed: serving v2 but release expects v1 (should FAIL -> exit 1)"
  python partD/smoke_eta.py --url http://127.0.0.1:8001 --expect-version v1 --max-p95-ms 80; rc3=$?
  echo "exit code: $rc3"
  stop_servers
  [ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] && [ $rc3 -eq 1 ]
}
step 11_partD_smoke smoke
cp "$LOGS/11_partD_smoke.log" partD/smoke_output.txt

# ---------------------------------------------------------------- 7. pack takeout.zip
mkdir -p "$TAKE/deliverables"
for f in partA/loadtest_results.json partA/loadtest_results_2workers.json partA/loadtest_batch.json \
         partB/gate_output.txt partB/shadow_output.txt partC/verify_output.txt partD/smoke_output.txt; do
  [ -f "$f" ] && cp "$f" "$TAKE/deliverables/$(echo "$f" | tr '/' '_')"
done
echo "total time: $(( $(date +%s) - START_ALL ))s" | tee -a "$STATUS"
python -m zipfile -c takeout.zip takeout/ && echo && echo "DONE -> $ROOT/takeout.zip  (send this file)"
grep -q FAIL "$STATUS" && echo "Some steps FAILED (see takeout/STATUS.txt). Send the zip anyway, the logs show why."
exit 0
