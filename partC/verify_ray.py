"""Part C2: deploy the sticky-canary app on a 2-CPU Ray and verify it with assertions.

Run from the repository root (after `python eta_data.py`):
    python partC/verify_ray.py | tee partC/verify_output.txt

Checks (1000 distinct customer ids, weight 0.2):
  (a) share routed to the candidate within +-4 percentage points of 0.2
  (b) 200 customers get the same arm on a second call
  (c) the same holds after redeploying the identical app (all 1000 customers)
  (d) after raising the weight 0.2 -> 0.5, no customer who was on the candidate moves back to prod
  (e) the X-Beta: 1 header forces the candidate for 50 requests
plus evidence for the deployment config (replicas, autoscaling, fractional CPUs) and hash stability.
"""
import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

import ray
import requests
from ray import serve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eta_ray  # noqa: E402

ROW = {"distance_km": 14.2, "n_stops": 5, "hour": 17, "weekday": 3, "traffic_index": 0.6, "rain": 1, "weight_kg": 9.5}
CUSTOMERS = [f"cust-{i:04d}" for i in range(1000)]


def section(t):
    print(f"\n=== {t} ===", flush=True)


def call(url, cid, beta=False, session=None):
    s = session or requests
    headers = {"X-Beta": "1"} if beta else {}
    r = s.post(url, json={"customer_id": cid, "instances": [ROW]}, headers=headers, timeout=30)
    r.raise_for_status()
    j = r.json()
    assert j["arm"] == j["model_version"], f"arm/model mismatch for {cid}: {j}"
    return j["arm"]


def arms_for(url, ids, beta=False, workers=8):
    def task(chunk):
        s = requests.Session()
        return [(cid, call(url, cid, beta, s)) for cid in chunk]
    chunks = [ids[i::workers] for i in range(workers)]
    out = {}
    with cf.ThreadPoolExecutor(workers) as ex:
        for res in ex.map(task, chunks):
            out.update(res)
    return out


def deploy(weight, extra_work_ms=0.0):
    serve.run(eta_ray.build(weight=weight, extra_work_ms=extra_work_ms), name="eta", route_prefix="/")


def wait_healthy(base, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(base + "/healthz", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("router did not become healthy")


def replica_counts():
    st = serve.status().applications["eta"].deployments
    return {name: dict(d.replica_states) for name, d in st.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--skip-autoscale-demo", action="store_true")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    url = base + "/v1/models/eta:predict"

    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False,
             runtime_env={"env_vars": {"PYTHONPATH": HERE}})
    serve.start(http_options={"host": "127.0.0.1", "port": a.port})
    try:
        section("setup")
        print(f"Ray cluster resources: CPU={ray.cluster_resources().get('CPU')}")
        deploy(0.2)
        wait_healthy(base)
        print("deployed with weight=0.2;", json.dumps(requests.get(base + "/config").json()))
        print("replica states:", replica_counts())
        used = ray.cluster_resources()["CPU"] - ray.available_resources().get("CPU", 0)
        print(f"CPU reserved by the app: {used:.2f} of 2 (router 0.25 + prod 2x0.25 + candidate 1x0.25 at min)")
        assert used <= 2.0 and abs(used - 1.0) < 1e-6, f"unexpected CPU reservation {used}"
        assert eta_ray.CANDIDATE_AUTOSCALING["min_replicas"] == 1 and eta_ray.CANDIDATE_AUTOSCALING["max_replicas"] == 3
        worst = (1 + eta_ray.PROD_REPLICAS + eta_ray.CANDIDATE_AUTOSCALING["max_replicas"]) * eta_ray.CPU_PER_REPLICA
        print(f"worst case at max autoscaling: {worst:.2f} CPU <= 2 -> fits")
        assert worst <= 2.0

        section("(a) share of customers on the candidate, weight=0.2, 1000 ids")
        t0 = time.time()
        first = arms_for(url, CUSTOMERS)
        share = sum(v == "v2" for v in first.values()) / len(first)
        print(f"{len(first)} customers in {time.time() - t0:.1f}s -> candidate share = {share:.3f} "
              f"(v2={sum(v == 'v2' for v in first.values())}, v1={sum(v == 'v1' for v in first.values())})")
        assert abs(share - 0.2) <= 0.04, "share outside 0.2 +- 0.04"
        print("PASS (a)")

        section("(b) 200 customers, second call gets the same arm")
        sample = CUSTOMERS[:200]
        second = arms_for(url, sample)
        changed = [c for c in sample if second[c] != first[c]]
        print(f"compared 200 customers: {len(changed)} changed arm; e.g. "
              + ", ".join(f"{c}:{first[c]}->{second[c]}" for c in sample[:5]))
        assert not changed
        print("PASS (b)")

        section("(c) redeploy the identical app, all 1000 customers keep their arm")
        serve.delete("eta")
        print("application deleted; replicas:", [d for d in serve.status().applications])
        deploy(0.2)
        wait_healthy(base)
        print("redeployed; replica states:", replica_counts())
        after = arms_for(url, CUSTOMERS)
        changed = [c for c in CUSTOMERS if after[c] != first[c]]
        print(f"after redeploy: {len(changed)} of 1000 customers changed arm")
        assert not changed
        print("PASS (c)")

        section("(d) raise weight 0.2 -> 0.5: nobody on the candidate moves back to prod")
        deploy(0.5)
        wait_healthy(base)
        print("config now:", requests.get(base + "/config").json()["weight"])
        raised = arms_for(url, CUSTOMERS)
        was_v2 = [c for c in CUSTOMERS if first[c] == "v2"]
        moved_back = [c for c in was_v2 if raised[c] == "v1"]
        newly = [c for c in CUSTOMERS if first[c] == "v1" and raised[c] == "v2"]
        share5 = sum(v == "v2" for v in raised.values()) / len(raised)
        print(f"candidate share now {share5:.3f}; previously-v2 customers: {len(was_v2)}, "
              f"moved back to v1: {len(moved_back)}, newly moved v1->v2: {len(newly)}")
        assert not moved_back
        assert abs(share5 - 0.5) <= 0.05
        print("PASS (d)")

        section("(e) header X-Beta: 1 forces the candidate for 50 requests")
        prod_customers = [c for c in CUSTOMERS if raised[c] == "v1"][:50]
        assert len(prod_customers) == 50
        forced = arms_for(url, prod_customers, beta=True)
        normal = arms_for(url, prod_customers[:10])
        print(f"50 prod customers with X-Beta: 1 -> arms {sorted(set(forced.values()))}; "
              f"same customers without header -> {sorted(set(normal.values()))}")
        assert all(v == "v2" for v in forced.values()) and all(v == "v1" for v in normal.values())
        print("PASS (e)")

        section("extra: bucketing is stable across processes, Python hash() is not")
        code = ("import sys; sys.path.insert(0, %r); import eta_ray; "
                "print([round(eta_ray.bucket(c), 6) for c in ['cust-0001','cust-0042']], "
                "[hash(c) %% 1000 for c in ['cust-0001','cust-0042']])" % HERE)
        outs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               env={**os.environ, "PYTHONHASHSEED": str(seed)}).stdout.strip() for seed in (1, 2)]
        for seed, o in zip((1, 2), outs):
            print(f"PYTHONHASHSEED={seed}: sha256 buckets / hash()%1000 -> {o}")
        assert outs[0].split("]")[0] == outs[1].split("]")[0], "sha256 buckets should not depend on the process"

        if not a.skip_autoscale_demo:
            section("extra: autoscaling demo (not an assertion): candidate under load, extra_work_ms=30")
            deploy(0.5, extra_work_ms=30)
            wait_healthy(base)
            stop = time.time() + 25

            def hammer(i):
                s, n = requests.Session(), 0
                while time.time() < stop:
                    call(url, f"load-{i}", beta=True, session=s)
                    n += 1
                return n
            with cf.ThreadPoolExecutor(8) as ex:
                futs = [ex.submit(hammer, i) for i in range(8)]
                while time.time() < stop:
                    time.sleep(5)
                    st = replica_counts().get("EtaModel_candidate", {})
                    print(f"  t+{25 - int(stop - time.time()):2d}s candidate replicas: {st}", flush=True)
                total = sum(f.result() for f in futs)
            print(f"  sent {total} requests to the candidate in 25 s with 8 concurrent clients")

        section("RESULT")
        print("ALL CHECKS PASSED (a) (b) (c) (d) (e)")
        return 0
    except AssertionError as e:
        print(f"\nCHECK FAILED: {e}")
        return 1
    finally:
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    sys.exit(main())
