# AIOps Assignment 5: SwiftParcel ETA model served four ways

Mohammed Khaja Ghouse Mohiuddin (DA24B015)

SwiftParcel predicts the delivery time in minutes from seven features of a delivery. Two trained models
exist: **v1** (Ridge regression) and **v2** (gradient boosting). This repository serves them as:

| Part | What | Main files |
|---|---|---|
| A | FastAPI REST service with validation, request ids and Prometheus metrics, plus a load test | `partA/eta_app.py`, `partA/test_eta_app.py` |
| B | MLflow registry (aliases `prod` and `candidate`), a multi-criteria promotion gate, a shadow comparison | `partB/eta_mlflow.py`, `partB/test_gate.py` |
| C | Ray Serve app with a sticky per-customer canary, header override and autoscaling, fitting on 2 CPUs | `partC/eta_ray.py`, `partC/verify_ray.py` |
| D | Release-gate smoke test (exit 0 = release allowed, exit 1 = blocked) | `partD/smoke_eta.py` |

Written answers, tables, the machine description and the list of what was not tested are in **`REPORT.md`**.

## Repository layout

```
eta_data.py            given: generates data/ (train + fresh CSV) and the two model bundles
load_test.py           given: closed-loop load generator
requirements.txt       exact versions used
run_all.sh             runs every part end to end (about 5-15 min)
partA/  eta_app.py  test_eta_app.py  loadtest_results.json  loadtest_results_2workers.json  loadtest_batch.json
partB/  eta_mlflow.py  test_gate.py  gate_output.txt  shadow_output.txt
partC/  eta_ray.py  verify_ray.py  verify_output.txt
partD/  smoke_eta.py  smoke_output.txt
REPORT.md
```

`data/`, `mlflow.db` and `mlruns/` are not committed; `eta_data.py` and `run_all.sh` regenerate them.

## Setup (Ubuntu, from a clean checkout)

Needs Python 3.10+, a CPU with 2 or more cores, and `curl`.

```bash
git clone https://github.com/raisserv2/aiops-assignment5.git
cd aiops-assignment5
conda create -n a5 python=3.10 -y
conda activate a5
pip install -r requirements.txt
python eta_data.py
```

`eta_data.py` prints the fresh-data MAE of both models and writes `data/eta_v1.joblib` and `data/eta_v2.joblib`.

## Run everything with one command

```bash
bash run_all.sh
```

It runs all four parts in order and writes the graded outputs into the part folders. It only uses
ports 8000, 8001, 5101, 5102 and 8100, and stops every server it started when it finishes.

## Run each part by hand

Run all commands from the repository root with the `a5` environment active.

### Part A: REST service

```bash
ETA_MODEL=data/eta_v1.joblib uvicorn eta_app:app --app-dir partA --port 8000
```

```bash
curl -s -X POST localhost:8000/v1/models/eta:predict -H 'Content-Type: application/json' \
  -d '{"instances":[{"distance_km":12.5,"n_stops":4,"hour":17,"weekday":2,"traffic_index":0.6,"rain":1,"weight_kg":8}]}'
curl -s localhost:8000/healthz
curl -s localhost:8000/metrics | grep eta_
pytest partA -v
```

Load test (with the server above running):

```bash
python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"
python load_test.py --url http://localhost:8000/v1/models/eta:predict --data data/eta_sample.csv \
       --concurrency 1 2 4 8 16 --duration 4 --out partA/loadtest_results.json
```

The capacity change measured in the report is running two worker processes:
`uvicorn eta_app:app --app-dir partA --port 8000 --workers 2`.

Behaviour summary:

- `POST /v1/models/eta:predict` takes `{"instances": [...]}` (1 to 500 rows) and returns predictions rounded to one decimal, in input order, with `model_version` and `request_id`.
- Field ranges are enforced. A delivery with `distance_km > 150` and `n_stops > 20` is rejected as **infeasible** (422).
- Validation errors return `{"error": "validation_failed", "details": [{"field", "msg"}], "request_id"}`.
- Every response has an `X-Request-ID` header: the client's value is echoed, otherwise a UUID is generated.
- If the model raises, the client gets a generic 500 with no internals.
- `/metrics` exposes `eta_requests_total{path,status}` and the histogram `eta_request_latency_seconds`. Requests to `/metrics` itself are not counted.

### Part B: MLflow

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python partB/eta_mlflow.py register            # versions 1 (v1) and 2 (v2), aliases prod and candidate
python partB/eta_mlflow.py table
python partB/eta_mlflow.py gate                # default thresholds
python partB/eta_mlflow.py gate --min-rel-gain 0.60
pytest partB -v
```

Shadow comparison, with each server in its own terminal (and the same `MLFLOW_TRACKING_URI`):

```bash
mlflow models serve -m "models:/eta-model@prod"      -p 5101 --env-manager local
mlflow models serve -m "models:/eta-model@candidate" -p 5102 --env-manager local
python partB/eta_mlflow.py shadow --prod-url http://127.0.0.1:5101/invocations \
                                  --cand-url http://127.0.0.1:5102/invocations --n 200
```

`decide()` is pure. The gate promotes only if all of these hold:

- The relative MAE gain is at least `min_rel_gain`.
- The candidate's single-row p95 is at most `max_p95_ms`.
- Neither slice (rush hours 7–9 and 16–19, or `rain == 1`) gets worse than prod by more than `max_slice_regression` (relative).

### Part C: Ray Serve sticky canary

```bash
python partC/verify_ray.py            # starts Ray with 2 CPUs, deploys, runs checks (a)-(e), shuts down
```

To keep the app running and call it yourself:

```bash
python -c "import os, sys, time, ray; from ray import serve; sys.path.insert(0, 'partC'); import eta_ray
ray.init(num_cpus=2, runtime_env={'env_vars': {'PYTHONPATH': os.path.abspath('partC')}})
serve.run(eta_ray.build(weight=0.2), name='eta', route_prefix='/')
time.sleep(10**9)"
curl -s -X POST localhost:8000/v1/models/eta:predict -H 'Content-Type: application/json' \
  -d '{"customer_id":"cust-0042","instances":[{"distance_km":12.5,"n_stops":4,"hour":17,"weekday":2,"traffic_index":0.6,"rain":1,"weight_kg":8}]}'
```

Add `-H 'X-Beta: 1'` to the `curl` to force the candidate.

How the canary routes requests:

- **Sticky:** a customer goes to the candidate if `sha256(customer_id)`, mapped to [0, 1), is below `weight`. The same customer always gets the same arm, and raising the weight only moves customers from prod to the candidate.
- **Replicas:** prod has 2 fixed replicas. The candidate autoscales between 1 and 3 replicas.
- **CPUs:** every replica, including the router, reserves 0.25 CPU, so the app needs at most 1.5 CPUs.

### Part D: release gate

```bash
python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1 --max-p95-ms 80
echo $?    # 0 = release allowed, 1 = blocked
```

The script runs 11 checks:

- health endpoint
- model version and response schema
- plausible range
- deterministic output and order
- three model-sanity checks: longer distance, heavier traffic or more stops must not give a shorter ETA
- invalid input returns 422
- an infeasible delivery returns 422
- `X-Request-ID` echo
- p95 latency

It also works against the Part C router (add `--beta` to test the candidate arm).

## Credits

The structure of the service, registry workflow and Ray Serve canary follows the ideas of the Module 4
lecture notebooks (churn example). The code here is written for the ETA problem; see `AI_DISCLOSURE.md`.

## Demo video

https://drive.google.com/file/d/1pr9XJtAyQJNelbfCuKN66DPytJrkc5bR/view?usp=sharing
