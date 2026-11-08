"""Part B: MLflow registry, multi-criteria promotion gate and shadow comparison.

Run from the repository root (tracking URI from MLFLOW_TRACKING_URI, default sqlite:///mlflow.db):
    python partB/eta_mlflow.py register                 # log v1 + v2, register eta-model, set aliases
    python partB/eta_mlflow.py table                    # registry table (version, alias)
    python partB/eta_mlflow.py gate [--min-rel-gain 0.6]
    python partB/eta_mlflow.py shadow --prod-url http://127.0.0.1:5101/invocations \
                                      --cand-url http://127.0.0.1:5102/invocations --n 200

The pyfunc wrapper and alias workflow follow the ideas of the Module 4 lecture notebook (MLflow
registry section); decide() and the slice criteria are specific to this assignment.
"""
import argparse
import json
import os
import sys
import time

import mlflow
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_NAME = "eta-model"
RUSH_HOURS = (7, 8, 9, 16, 17, 18, 19)
FEATURES = ["distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg"]
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


# ------------------------------------------------------------------ custom pyfunc
class EtaPyfunc(mlflow.pyfunc.PythonModel):
    """Wraps a joblib bundle {model, feature_columns, version}.
    Output: DataFrame with columns eta_minutes (1 decimal) and model_version."""

    def load_context(self, context):
        import joblib
        b = joblib.load(context.artifacts["bundle"])
        self.model, self.features, self.version = b["model"], list(b["feature_columns"]), str(b["version"])

    def predict(self, context, model_input, params=None):
        X = pd.DataFrame(model_input)[self.features]
        y = np.round(self.model.predict(X).astype(float), 1)
        return pd.DataFrame({"eta_minutes": y, "model_version": [self.version] * len(y)})


def _setup():
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.MlflowClient()


def _pins():
    import joblib
    import sklearn
    import cloudpickle
    return [f"mlflow=={mlflow.__version__}", f"scikit-learn=={sklearn.__version__}",
            f"pandas=={pd.__version__}", f"numpy=={np.__version__}",
            f"joblib=={joblib.__version__}", f"cloudpickle=={cloudpickle.__version__}"]


# ------------------------------------------------------------------ B1: register
def register(data_dir=os.path.join(ROOT, "data")):
    """Log v1 and v2 as pyfunc models with signature, input example and pinned requirements,
    register them as eta-model (versions 1 and 2) and set aliases prod -> v1, candidate -> v2."""
    from mlflow.models import infer_signature
    sys.path.insert(0, ROOT)
    import eta_data

    client = _setup()
    try:                                   # start from a clean registry so versions are 1 and 2
        client.delete_registered_model(MODEL_NAME)
    except Exception:
        pass
    mlflow.set_experiment("swiftparcel-eta")

    example = eta_data.make_data(5, seed=7)[eta_data.FEATURES]
    versions = {}
    for tag in ("v1", "v2"):
        path = os.path.join(data_dir, f"eta_{tag}.joblib")
        wrapper = EtaPyfunc()

        class _Ctx:                        # compute the example output with the real wrapper
            artifacts = {"bundle": path}
        wrapper.load_context(_Ctx)
        signature = infer_signature(example, wrapper.predict(None, example))

        with mlflow.start_run(run_name=f"eta-{tag}"):
            mlflow.log_params({"bundle_version": tag, "estimator": type(wrapper.model).__name__})
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=EtaPyfunc(),
                artifacts={"bundle": path},
                code_paths=[os.path.abspath(__file__)],
                signature=signature,
                input_example=example,
                pip_requirements=_pins(),
                registered_model_name=MODEL_NAME,
            )
        versions[tag] = str(info.registered_model_version)
        client.set_model_version_tag(MODEL_NAME, versions[tag], "bundle_version", tag)

    client.set_registered_model_alias(MODEL_NAME, "prod", versions["v1"])
    client.set_registered_model_alias(MODEL_NAME, "candidate", versions["v2"])
    return registry_table()


def registry_table():
    client = _setup()
    rows = []
    for mv in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda m: int(m.version)):
        full = client.get_model_version(MODEL_NAME, mv.version)
        rows.append({"version": int(mv.version), "aliases": ",".join(full.aliases) or "-",
                     "bundle": full.tags.get("bundle_version", "?"), "run_id": mv.run_id[:8]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B2: gate
def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05):
    """Pure promotion decision. prod / cand: dicts with keys mae, mae_rush, mae_rain, p95_ms.

    Promote only if ALL hold:
      - relative MAE gain (prod.mae - cand.mae) / prod.mae >= min_rel_gain
      - cand.p95_ms <= max_p95_ms
      - each slice (rush, rain): cand <= prod * (1 + max_slice_regression)   (relative regression)
    Thresholds are inclusive; a tiny tolerance (1e-9) absorbs floating-point noise at the boundary.
    """
    eps = 1e-9
    rel_gain = (prod["mae"] - cand["mae"]) / prod["mae"]
    criteria = {
        "mae_gain": {"value": round(rel_gain, 4), "threshold": f">= {min_rel_gain}",
                     "pass": rel_gain >= min_rel_gain - eps},
        "latency_p95": {"value": round(cand["p95_ms"], 3), "threshold": f"<= {max_p95_ms} ms",
                        "pass": cand["p95_ms"] <= max_p95_ms + eps},
    }
    for sl in ("mae_rush", "mae_rain"):
        reg = (cand[sl] - prod[sl]) / prod[sl]
        criteria[f"slice_{sl[4:]}"] = {"value": round(reg, 4), "threshold": f"<= +{max_slice_regression}",
                                      "pass": reg <= max_slice_regression + eps}
    return {"criteria": criteria, "promote": all(c["pass"] for c in criteria.values())}


def _evaluate(m, data, n_latency=200):
    pred = m.predict(data[FEATURES])["eta_minutes"].to_numpy()
    err = np.abs(pred - data["eta_minutes"].to_numpy())
    rush = data["hour"].isin(RUSH_HOURS).to_numpy()
    rain = (data["rain"] == 1).to_numpy()
    X = data[FEATURES]
    for i in range(10):                    # warm-up, not timed
        m.predict(X.iloc[[i]])
    lat = []
    for i in range(n_latency):
        t0 = time.perf_counter()
        m.predict(X.iloc[[i]])
        lat.append((time.perf_counter() - t0) * 1000)
    return {"mae": float(err.mean()), "mae_rush": float(err[rush].mean()),
            "mae_rain": float(err[rain].mean()), "p95_ms": float(np.percentile(lat, 95)),
            "n_rows": int(len(data)), "n_rush": int(rush.sum()), "n_rain": int(rain.sum())}


def promotion_gate(**thresholds):
    """Load prod and candidate from the registry, evaluate on fresh data
    (eta_data.make_data(2000, seed=99)), measure single-row p95 over 200 calls, call decide()."""
    sys.path.insert(0, ROOT)
    import eta_data
    _setup()
    data = eta_data.make_data(2000, seed=99)
    prod_m = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@prod")
    cand_m = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@candidate")
    prod, cand = _evaluate(prod_m, data), _evaluate(cand_m, data)
    result = decide(prod, cand, **thresholds)
    result.update({"thresholds": thresholds or "defaults", "prod_metrics": prod, "cand_metrics": cand})
    return result


def _print_gate(result):
    print(f"thresholds: {result['thresholds']}")
    for side in ("prod_metrics", "cand_metrics"):
        m = result[side]
        print(f"  {side:13s} mae={m['mae']:.3f}  mae_rush={m['mae_rush']:.3f}  mae_rain={m['mae_rain']:.3f}"
              f"  p95_ms={m['p95_ms']:.3f}  (rows={m['n_rows']}, rush={m['n_rush']}, rain={m['n_rain']})")
    for name, c in result["criteria"].items():
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {name:13s} value={c['value']}  threshold {c['threshold']}")
    print(f"  PROMOTE = {result['promote']}")


# ------------------------------------------------------------------ B3: shadow
def shadow(prod_url, cand_url, n=200, data_path=os.path.join(ROOT, "data", "eta_fresh.csv")):
    """Send the same n single-row requests to both servers and compare predictions and latency."""
    import requests
    df = pd.read_csv(data_path).head(n)
    rows = df[FEATURES].to_dict(orient="records")
    s = requests.Session()
    out = {"prod": {"pred": [], "lat": [], "ver": set()}, "cand": {"pred": [], "lat": [], "ver": set()}}
    for i, r in enumerate(rows):
        order = [("prod", prod_url), ("cand", cand_url)]
        if i % 2:                            # alternate who goes first, so neither side is favoured
            order.reverse()
        for side, url in order:
            t0 = time.perf_counter()
            resp = s.post(url, json={"dataframe_records": [r]}, timeout=30)
            out[side]["lat"].append((time.perf_counter() - t0) * 1000)
            resp.raise_for_status()
            p = resp.json()["predictions"][0]
            out[side]["pred"].append(float(p["eta_minutes"]))
            out[side]["ver"].add(p["model_version"])
    pp, cp = np.array(out["prod"]["pred"]), np.array(out["cand"]["pred"])
    diff = np.abs(pp - cp)
    truth = df["eta_minutes"].to_numpy()
    return {
        "n_requests_each": len(rows),
        "prod_model_version": sorted(out["prod"]["ver"]), "cand_model_version": sorted(out["cand"]["ver"]),
        "mean_abs_diff_min": round(float(diff.mean()), 3),
        "pct_diff_gt_10_min": round(float((diff > 10).mean() * 100), 2),
        "max_abs_diff_min": round(float(diff.max()), 3),
        "prod_p50_ms": round(float(np.percentile(out["prod"]["lat"], 50)), 2),
        "prod_p95_ms": round(float(np.percentile(out["prod"]["lat"], 95)), 2),
        "cand_p50_ms": round(float(np.percentile(out["cand"]["lat"], 50)), 2),
        "cand_p95_ms": round(float(np.percentile(out["cand"]["lat"], 95)), 2),
        "prod_mae_vs_truth_min": round(float(np.abs(pp - truth).mean()), 3),
        "cand_mae_vs_truth_min": round(float(np.abs(cp - truth).mean()), 3),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("register")
    sub.add_parser("table")
    g = sub.add_parser("gate")
    g.add_argument("--min-rel-gain", type=float)
    g.add_argument("--max-p95-ms", type=float)
    g.add_argument("--max-slice-regression", type=float)
    sh = sub.add_parser("shadow")
    sh.add_argument("--prod-url", default="http://127.0.0.1:5101/invocations")
    sh.add_argument("--cand-url", default="http://127.0.0.1:5102/invocations")
    sh.add_argument("--n", type=int, default=200)
    a = ap.parse_args()

    if a.cmd == "register":
        print(register().to_string(index=False))
    elif a.cmd == "table":
        print(registry_table().to_string(index=False))
    elif a.cmd == "gate":
        th = {k: v for k, v in {"min_rel_gain": a.min_rel_gain, "max_p95_ms": a.max_p95_ms,
                                "max_slice_regression": a.max_slice_regression}.items() if v is not None}
        _print_gate(promotion_gate(**th))
    elif a.cmd == "shadow":
        res = shadow(a.prod_url, a.cand_url, a.n)
        for k, v in res.items():
            print(f"{k:24s} {v}")
        print(json.dumps(res))
