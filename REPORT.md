# AIOps Assignment 5: SwiftParcel ETA served four ways

Mohammed Khaja Ghouse Mohiuddin (DA24B015)

## Machine and how the numbers were produced

| | |
|---|---|
| CPU | Intel Core i7-14650HX, 16 physical cores, 24 threads, max 5.2 GHz |
| RAM | 14 GiB (11 GiB free at start) |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-38 |
| Python | 3.10.22 (conda env `a5`), package versions in `requirements.txt` |
| Power | on AC, CPU governor `powersave`, power profile `balanced` |
| Ray | started with `num_cpus=2` for Part C |

Every number below comes from a single run of `bash run_all.sh` on 8 Nov 2026 at 19:00 IST (186 s in total; all 10 steps exit 0).
The load generator ran on the same laptop as the server. The raw files are `partA/*.json`, `partB/gate_output.txt`,
`partB/shadow_output.txt`, `partC/verify_output.txt` and `partD/smoke_output.txt`.

Citation: the overall structure (FastAPI service with request-id middleware and Prometheus metrics, MLflow pyfunc and aliases,
Ray Serve router with weighted canary) follows the ideas of the Module 4 lecture notebooks (churn example). `eta_data.py` and
`load_test.py` are the given starter files. AI use is declared in `AI_DISCLOSURE.md`.

## Part A: REST service

**A1.** `partA/eta_app.py` loads the bundle once at import (`model`, `FEATURES`, `MODEL_VERSION`; path from `ETA_MODEL`).
The seven requirements are implemented as follows:

1. **Request schema:** Pydantic models enforce the field ranges and a batch size of 1–500. Unknown fields inside an instance are rejected.
2. **Cross-field rule:** a model validator raises `infeasible delivery: distance_km > 150 and n_stops > 20`.
3. **Request id:** an HTTP middleware echoes or generates `X-Request-ID` on every response. That includes 422s and 500s, and the same id goes into the body.
4. **Validation errors:** a `RequestValidationError` handler rewrites them into `{"error": "validation_failed", "details": [{field, msg}], "request_id"}`.
5. **Inference errors:** if inference raises, the client gets `{"error": "internal_error", "request_id"}`. The middleware catches anything else the same way.
6. **Metrics:** `eta_requests_total{path,status}` and the histogram `eta_request_latency_seconds` (12 buckets) live in their own registry. `/metrics` itself is skipped.

**A2.** `partA/test_eta_app.py` has 17 tests, all passing. They cover:

- a valid request and the response schema
- batch order compared against the model directly
- `/healthz`
- invalid input: out of range, missing field, wrong type, bad category, `weight_kg = 0`, and malformed JSON
- the cross-field rule and its boundary (150 km / 20 stops is allowed)
- both batch limits (500 is accepted, 0 and 501 are rejected)
- request-id echo, generation, and presence on errors
- the metrics text and a counter increment
- a generic 500 using `monkeypatch`

I checked that the tests fail when the behaviour breaks. I removed the infeasible rule, the request-id echo, the batch cap and the "don't count /metrics" rule, and I reversed the output order. Each change made at least one test fail.

**A3.1 Load test, v1, one uvicorn worker, batch 1, 4 s per level** (`partA/loadtest_results.json`). The last column applies Little's Law: W = N / X, where N is concurrency and X is throughput.

| concurrency N | requests | throughput X (req/s) | p50 ms | p95 ms | p99 ms | errors | W = N/X (ms) |
|---|---|---|---|---|---|---|---|
| 1 | 1511 | 377.8 | 2.5 | 4.1 | 5.1 | 0 | 2.65 |
| 2 | 2398 | 599.5 | 3.1 | 5.0 | 6.2 | 0 | 3.34 |
| 4 | 2477 | 619.2 | 5.9 | 10.7 | 13.7 | 0 | 6.46 |
| 8 | 1412 | 353.0 | 23.0 | 32.2 | 39.1 | 0 | 22.66 |
| 16 | 1292 | 323.0 | 48.6 | 72.1 | 83.5 | 0 | 49.54 |

The `replicas` column of `load_test.py` shows 1 even with two workers, because it counts distinct `served_by` values and the Part A responses have no such field.

**A3.2 Saturation and capacity.** The load generator is closed-loop, so Little's Law gives the average time in the system: W = N / X.

- **Up to N = 2:** throughput still rises a lot (378 → 600 req/s) and W stays close to the single-request service time (about 2.6 to 3.3 ms).
- **From N = 2 to N = 4:** throughput only goes from 599.5 to 619.2 req/s, while W doubles from 3.3 to 6.5 ms. Extra concurrency is now turning into queueing instead of work.
- **From N = 8:** throughput actually drops, to 353 and then 323 req/s, and W grows roughly in proportion to N (22.7 and 49.5 ms). That is what a saturated server looks like. W also matches the measured p50 at every level (5.9, 23.0 and 48.6 ms), which confirms the arithmetic.

So the service **saturates at about N = 4, at roughly 620 req/s**.

Under the SLO of p95 ≤ 80 ms, every tested level passes (the worst is p95 = 72.1 ms at N = 16, although p99 is already 83.5 ms there). The binding limit is therefore throughput, not latency. The capacity under the SLO is the highest throughput that meets it, about **619 single-row requests per second, reached at N = 4**. I would plan for about 500 req/s (80%) to keep headroom.

I did not profile why throughput falls after N = 4. My hypothesis is that the event loop and the threads running the sync endpoint compete for the GIL in one process.

**A3.3 Cost per row, concurrency 1** (`partA/loadtest_batch.json`):

| batch | throughput (req/s) | mean ms per request | **ms per row** | rows per second |
|---|---|---|---|---|
| 1 | 404.8 | 2.470 | **2.470** | 405 |
| 100 | 320.8 | 3.117 | **0.031** | 32,080 |

A row costs about 79 times less in a batch of 100. Almost all of the 2.47 ms per request is fixed overhead that is paid once per request, whatever the batch size:

- the HTTP round trip
- JSON parsing and serialisation
- the middleware and metrics
- building one DataFrame
- one call into scikit-learn

The marginal cost of an extra row, (3.117 − 2.470) / 99 ≈ 0.0065 ms, is just Pydantic validation of one row and one more row in a vectorised matrix product. So clients that can batch should batch.

**A3.4 Change: two uvicorn worker processes** (`--workers 2`, `partA/loadtest_results_2workers.json`). One process is limited by one GIL, and the laptop has 16 cores, so a second process should roughly double capacity.

| concurrency | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| throughput (req/s) | 386.8 | 586.0 | 662.0 | 706.8 | 1053.5 |
| p95 ms | 4.0 | 5.6 | 11.5 | 23.2 | 24.0 |

The best throughput within the SLO went from 619.2 to **1053.5 req/s, which is 1.70 times, not a full doubling**. At N = 16 the gain is 3.3 times (323 → 1054), because a single worker had already collapsed there.

Throughput was still rising at N = 16, so the two-worker server was not saturated in the tested range. Its true capacity may be higher, but I did not test above 16. Two likely reasons it is below 2 times, neither of which I verified: at low N there is not enough load to keep two workers busy, and the Python load generator shares the same CPU.

One trade-off: with several workers, each process has its own metrics registry. Prometheus would need the client's multiprocess mode to add them up.

**A3.5** The mean hides the tail. A few very slow requests barely move it, but they are exactly what the unlucky users experience, and with many requests per user session almost everyone eventually hits them. SLOs are therefore written on p95/p99, which bound the experience of all but 5% (or 1%) of requests. Our own data shows this: at N = 16 the mean (49.5 ms) looks comfortable while p99 (83.5 ms) already breaks an 80 ms target.

## Part B: MLflow registry, gate and shadow

**B1.** `EtaPyfunc` is set up as follows:

- It loads the joblib bundle from the model artifacts in `load_context`.
- It returns a DataFrame with `eta_minutes` (rounded to one decimal) and `model_version`.
- Both bundles are logged with an inferred signature, a 5-row input example and pinned `pip_requirements` (mlflow, scikit-learn, pandas, numpy, joblib, cloudpickle).
- The tracking store is SQLite (`MLFLOW_TRACKING_URI=sqlite:///mlflow.db`).
- Both are registered as `eta-model`.

| version | alias | bundle |
|---|---|---|
| 1 | prod | v1 (Ridge) |
| 2 | candidate | v2 (GradientBoostingRegressor) |

**B2.** `decide()` is pure: inputs are not modified, and the same inputs give the same output. It returns four criteria, each with its value, threshold and pass flag, plus `promote` (true only if all four pass):

- `mae_gain = (prod.mae − cand.mae) / prod.mae`, which must be at least `min_rel_gain`
- `latency_p95 = cand.p95_ms`, which must be at most `max_p95_ms`
- the relative regression on the rush-hour slice (hours 7–9 and 16–19), which must be at most `max_slice_regression`
- the relative regression on the rain slice, with the same limit

Thresholds are inclusive, with a 1e-9 tolerance for floating point.

`promotion_gate()` works like this:

- It loads `models:/eta-model@prod` and `@candidate`.
- It evaluates them on `make_data(2000, seed=99)`: 571 rush-hour rows and 994 rain rows.
- It times 200 single-row `predict` calls after a warm-up.

`partB/test_gate.py` has 12 passing tests, with boundary cases just inside and just outside each threshold. For example, a MAE gain of 3.1% passes and 2.9% fails, a p95 of 10.0 ms passes and 10.01 ms fails, and slice regressions of +4.9% pass while +5.1% fail.

| | prod (v1) | candidate (v2) | criterion | default thresholds | `min_rel_gain=0.60` |
|---|---|---|---|---|---|
| MAE (min) | 11.955 | 6.105 | gain 0.489 | PASS (≥ 0.03) | **FAIL** (≥ 0.60) |
| single-row p95 (ms) | 1.48 | 1.43 | | PASS (≤ 10) | PASS |
| MAE rush hours | 19.680 | 8.580 | −56.4% | PASS (≤ +5%) | PASS |
| MAE rain | 11.623 | 6.285 | −45.9% | PASS (≤ +5%) | PASS |
| **promote** | | | | **True** | **False** |

The p95 values come from the default-threshold run. The second run measured 1.44 ms for prod and 1.40 ms for the candidate.

**B3. Shadow** (`mlflow models serve` with `@prod` on port 5101 and `@candidate` on port 5102). The same 200 rows of `eta_fresh.csv` were sent one per request, alternating which server went first:

| metric | value |
|---|---|
| (i) mean absolute difference between predictions | **9.48 min** (max 51.7 min) |
| (ii) requests differing by more than 10 min | **35.0%** |
| (iii) p95 latency, prod / candidate | **5.27 ms / 5.92 ms** (p50 4.01 / 3.97 ms) |
| MAE against the true ETA on the same 200 rows | prod 11.43, candidate 6.11 min |

I would promote v2:

- It nearly halves the error.
- It improves both risky slices instead of regressing them.
- Its HTTP p95 is within a millisecond of prod's.

But I would not flip the alias for everyone at once. A third of all customers would see their ETA move by more than 10 minutes, which is a visible change. So I would roll it out through the sticky canary in Part C with the abort criteria in Part D.

**Why moving an alias does not change a running server:** `mlflow models serve -m models:/eta-model@prod` resolves the alias to a version number only once, at startup. It then downloads that version's artifacts and keeps the loaded model in memory, and the process never asks the registry again. Moving the alias only changes a pointer in the database, which takes effect the next time something loads the model.

**Two ways to pick up the new version without dropping requests:**

1. **Blue-green or rolling replacement:** start a new server that loads the new alias and smoke-test it. Then switch the load balancer to it and drain the old server once its in-flight requests finish. A Ray Serve `serve.run` with the new version does this replica by replica.
2. **In-process hot reload:** a background thread polls the registry for the alias's current version. When it changes, the thread loads the new model next to the old one and atomically swaps the reference. Requests already running finish on the old model.

## Part C: Ray Serve sticky canary

**C1.** `partC/eta_ray.py` is built as follows:

- **Model deployment:** `EtaModel` loads one bundle per replica in `__init__` and predicts a batch.
- **Replicas:** prod runs 2 replicas. The candidate uses `autoscaling_config` with min 1, max 3, `target_ongoing_requests=2`, `upscale_delay_s=5` and `downscale_delay_s=60`.
- **Router:** a FastAPI ingress validates the same fields as Part A plus `customer_id`. It returns the model output plus `arm` and `customer_id`.
- **Routing rule:** the arm is the candidate if `X-Beta: 1` is set, or if `bucket(customer_id) < weight`. The bucket is the first 8 bytes of SHA-256 of the id, divided by 2^64. Because the test is "bucket < weight", raising the weight can only move customers from prod to the candidate.
- **CPU budget:** every replica, including the router, reserves 0.25 CPU. That is 1.0 CPU at minimum scale and 1.5 CPU at maximum, which fits in 2.
- **Entry point:** `build(weight=0.1, extra_work_ms=0.0)` returns the bound application.

**C2.** `partC/verify_ray.py` ran on Ray with `num_cpus=2`, 1000 ids and weight 0.2. All assertions passed:

| check | evidence from `verify_output.txt` |
|---|---|
| (a) share within 0.2 ± 0.04 | 203 of 1000 on the candidate, share 0.203 |
| (b) stable on a second call | 200 customers, 0 changed arm |
| (c) stable after redeploy | application deleted and redeployed, 0 of 1000 changed arm |
| (d) weight 0.2 → 0.5 | share 0.513; 0 of the 203 earlier candidate customers moved back; 310 moved v1 → v2 |
| (e) `X-Beta: 1` | 50 prod customers all got v2 with the header and v1 without it |
| setup | 1.00 CPU reserved at start; worst case 1.50 ≤ 2 |

**C3.**

1. **Why not `hash()`:** Python salts `hash()` of strings with a random seed in every process, unless `PYTHONHASHSEED` is fixed. Each replica, restart or redeploy would therefore put the same customer in a different bucket, which breaks stickiness. My run shows this: `hash("cust-0001") % 1000` was 226 with seed 1 and 142 with seed 2, while the SHA-256 bucket stayed 0.02308 in both. It is also not guaranteed to be uniform or the same across Python versions.
2. **Why stickiness matters:** for the product, one customer checking the same parcel should not see the ETA jump between two models. In our shadow test the two models disagree by more than 10 minutes on 35% of requests, so flip-flopping would look like a bug and erode trust. For the analysis, the customer is the unit of randomisation. If each customer sees only one arm, their outcomes (complaints, on-time rate, actual error) can be attributed to that arm without contamination, and the arms are independent groups for a statistical comparison. Ramping the weight up then keeps the already-exposed cohort intact.
3. **Autoscaling parameters:**
   - `target_ongoing_requests=2`: single-row inference is a few milliseconds of CPU work. About two requests in flight per replica keeps a replica busy without building a queue, so latency stays near the service time.
   - `upscale_delay_s=5`: this reacts quickly when a burst of beta traffic arrives.
   - `downscale_delay_s=60`: canary traffic is bursty, so waiting a minute before removing a replica avoids flapping and repeated cold starts.
   - `max_replicas=3` and `min_replicas=1`: max 3 is the most that fits the 2-CPU budget (0.25 × (1 + 2 + 3) = 1.5 CPU), and it is plenty for 10–20% of traffic. Min 1 keeps the candidate warm.

   In the 25 s autoscaling demo on my laptop, the candidate stayed at 1 replica (see "not tested").

## Part D: release gate

`partD/smoke_eta.py --url … --expect-version … [--max-p95-ms 80]` runs 11 checks and exits 1 if any fails:

- health
- version and response schema
- plausible range (1–1000 min)
- deterministic output and preserved order
- **three model-sanity checks:** a longer distance, heavier traffic or more stops must not give a shorter ETA
- **invalid input returns 422** (hour 24, rain 2, negative weight, empty batch)
- an infeasible delivery returns 422
- request-id echo
- latency p95 over 100 single-row requests

I chose those three sanity rules because I checked that they hold for both models. I did not use "rain must not shorten the ETA", because the GBR does not reliably satisfy it.

Results against Part A (`partD/smoke_output.txt`):

| scenario | result |
|---|---|
| v1 served, expect v1 | 11/11 pass, exit 0 (p95 4.4 ms) |
| v2 served, expect v2 | 11/11 pass, exit 0 (p95 4.9 ms) |
| v2 served, expect v1 | version check fails, **exit 1** (gate closed) |

**Abort criteria for a 10% canary of v2.** Each criterion compares v2 with v1, using the `arm` label. Any one of them triggers rollback:

1. **Errors:** the 5xx rate on the v2 arm is above 1% over a rolling 5-minute window (or more than twice v1's rate over the same window).
2. **Latency:** v2's p95 request latency is above 80 ms (the SLO) for two consecutive 5-minute windows, or more than 1.5 times v1's p95 over 10 minutes.
3. **Prediction sanity:** over 15 minutes, more than 0.1% of v2's predictions fall outside 1–1000 minutes, or v2's mean predicted ETA differs from v1's on the same hours by more than 15%.
4. **Real accuracy:** on deliveries completed in the last 24 hours, v2's MAE against the actual delivery time is more than 5% worse than v1's, overall or on the rush-hour or rain slices (the same rule as the B2 gate, applied to live data).

## Not tested

- I did not profile why one worker's throughput drops above N = 4. The GIL explanation is a hypothesis.
- I did not test two workers above N = 16. The two-worker server was not saturated in the tested range.
- The load generator and the server shared one laptop, so client CPU use was not isolated.
- Autoscaling scale-up was not observed on my laptop. In the 25 s demo with `extra_work_ms=30`, the candidate stayed at 1 replica. The autoscaling configuration is verified (`/config` and the replica states), but an actual scale-out is not. The demo window is probably shorter than Ray's metric look-back period.
- The smoke test was run only against Part A (v1 and v2). The Ray router supports the same request shape, but I did not run the smoke test against it in the recorded run. Not against the MLflow servers either, which use a different request format.
- `mlflow models serve` ran with `--env-manager local`. The pinned requirements were logged but never used to build an isolated environment.
- The `/metrics` endpoint was only read directly. It was not scraped by a real Prometheus server, and metrics are per process when running two workers.
