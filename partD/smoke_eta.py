"""Part D: release-gate smoke test for an ETA endpoint. Exit 0 = all checks pass, 1 = any check fails.

    python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1 [--max-p95-ms 80]

--url may be the base URL or the full .../v1/models/eta:predict URL. Works against the Part A service
and the Part C router (a customer_id is always sent; use --beta to force the Ray candidate arm).
"""
import argparse
import math
import sys
import time
import uuid

import requests

PRED_PATH = "/v1/models/eta:predict"
BASE_ROW = {"distance_km": 20.0, "n_stops": 5, "hour": 11, "weekday": 2,
            "traffic_index": 0.3, "rain": 0, "weight_kg": 8.0}


def row(**over):
    r = dict(BASE_ROW)
    r.update(over)
    return r


class Smoke:
    def __init__(self, base, customer_id, beta):
        self.base, self.customer_id = base, customer_id
        self.s = requests.Session()
        self.headers = {"X-Beta": "1"} if beta else {}
        self.results = []

    def post(self, instances, headers=None, **body):
        payload = {"customer_id": self.customer_id, "instances": instances}
        payload.update(body)
        return self.s.post(self.base + PRED_PATH, json=payload,
                           headers={**self.headers, **(headers or {})}, timeout=10)

    def preds(self, instances):
        r = self.post(instances)
        r.raise_for_status()
        return r.json()["predictions"]

    def check(self, name, fn):
        try:
            ok, detail = fn()
        except Exception as e:                       # a crash inside a check is a failure, not a pass
            ok, detail = False, f"{type(e).__name__}: {e}"
        status = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
        self.results.append((name, status, detail))
        print(f"[{status}] {name:38s} {detail}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--expect-version", required=True, help="e.g. v1 or v2")
    ap.add_argument("--max-p95-ms", type=float, default=80.0)
    ap.add_argument("--latency-requests", type=int, default=100)
    ap.add_argument("--customer-id", default="smoke-test")
    ap.add_argument("--beta", action="store_true", help="send X-Beta: 1 (Ray router: force candidate)")
    a = ap.parse_args()

    base = a.url.split(PRED_PATH)[0].rstrip("/")
    sm = Smoke(base, a.customer_id, a.beta)
    print(f"smoke test against {base}  expect-version={a.expect_version}  max-p95={a.max_p95_ms} ms\n")

    def health():
        r = sm.s.get(base + "/healthz", timeout=5)
        return r.status_code == 200 and r.json().get("status") == "ok", f"HTTP {r.status_code} {r.text[:80]}"
    sm.check("1 health endpoint", health)

    def version_and_schema():
        rows = [row(), row(distance_km=60), row(distance_km=3, n_stops=1)]
        r = sm.post(rows)
        j = r.json()
        p = j.get("predictions", [])
        ok = (r.status_code == 200 and j.get("model_version") == a.expect_version and len(p) == len(rows)
              and all(isinstance(v, (int, float)) and math.isfinite(v) and round(v, 1) == v for v in p))
        return ok, f"HTTP {r.status_code}, model_version={j.get('model_version')!r}, predictions={p}"
    sm.check("2 version + response schema", version_and_schema)

    def plausible():
        p = sm.preds([row(distance_km=d) for d in (1, 20, 80, 200)])
        return all(1 <= v <= 1000 for v in p), f"predictions for 1/20/80/200 km = {p} (expected 1..1000 min)"
    sm.check("3 predictions in plausible range", plausible)

    def deterministic_and_ordered():
        rows = [row(distance_km=d, n_stops=s) for d, s in [(90, 3), (4, 1), (35, 12)]]
        batch = sm.preds(rows)
        singles = [sm.preds([r])[0] for r in rows]
        again = sm.preds(rows)
        return batch == singles == again, f"batch={batch} singles={singles} repeat={again}"
    sm.check("4 deterministic, order preserved", deterministic_and_ordered)

    def sanity_distance():
        dists = [5, 20, 50, 100, 150]
        p = sm.preds([row(distance_km=d) for d in dists])
        ok = all(b >= a_ for a_, b in zip(p, p[1:])) and p[-1] > p[0]
        return ok, "km " + ", ".join(f"{d}->{v}" for d, v in zip(dists, p))
    sm.check("5 sanity: longer distance not faster", sanity_distance)

    def sanity_traffic():
        lo, hi = sm.preds([row(distance_km=40, traffic_index=0.05), row(distance_km=40, traffic_index=0.95)])
        return hi >= lo, f"40 km: traffic 0.05 -> {lo} min, traffic 0.95 -> {hi} min"
    sm.check("6 sanity: heavy traffic not faster", sanity_traffic)

    def sanity_stops():
        lo, hi = sm.preds([row(distance_km=30, n_stops=2), row(distance_km=30, n_stops=18)])
        return hi >= lo, f"30 km: 2 stops -> {lo} min, 18 stops -> {hi} min"
    sm.check("7 sanity: more stops not faster", sanity_stops)

    def invalid_422():
        bad = [row(hour=24), row(rain=2), row(weight_kg=-1)]
        codes = [sm.post([b]).status_code for b in bad] + [sm.post([]).status_code]
        return all(c == 422 for c in codes), f"hour=24 / rain=2 / weight<0 / empty batch -> {codes}"
    sm.check("8 invalid input -> 422", invalid_422)

    def infeasible_422():
        r = sm.post([row(distance_km=200, n_stops=25)])
        return r.status_code == 422 and "infeasible" in r.text, f"HTTP {r.status_code} {r.text[:90]}"
    sm.check("9 infeasible delivery -> 422", infeasible_422)

    def request_id():
        rid = "smoke-" + uuid.uuid4().hex[:8]
        r = sm.post([row()], headers={"X-Request-ID": rid})
        if "X-Request-ID" not in r.headers:
            return None, "server does not send X-Request-ID (e.g. Ray router); skipped"
        body = r.json()
        ok = r.headers["X-Request-ID"] == rid and body.get("request_id", rid) == rid
        return ok, f"sent {rid}, header {r.headers['X-Request-ID']}, body {body.get('request_id', '(not in body)')}"
    sm.check("10 X-Request-ID echoed", request_id)

    def latency():
        for _ in range(5):
            sm.preds([row()])                         # warm-up
        lat = []
        for i in range(a.latency_requests):
            t0 = time.perf_counter()
            sm.preds([row(distance_km=5 + i % 50)])
            lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        p50, p95 = lat[len(lat) // 2], lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        return p95 <= a.max_p95_ms, f"{len(lat)} single-row requests: p50={p50:.1f} ms p95={p95:.1f} ms"
    sm.check(f"11 latency p95 <= {a.max_p95_ms:g} ms", latency)

    failed = [n for n, s, _ in sm.results if s == "FAIL"]
    passed = sum(s == "PASS" for _, s, _ in sm.results)
    skipped = sum(s == "SKIP" for _, s, _ in sm.results)
    print(f"\n{passed} passed, {len(failed)} failed, {skipped} skipped")
    print("RELEASE GATE: " + ("OPEN (exit 0)" if not failed else "CLOSED (exit 1): " + "; ".join(failed)))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
