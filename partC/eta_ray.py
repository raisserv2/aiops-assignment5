"""Part C: Ray Serve application with a sticky (per-customer) canary.

    POST /v1/models/eta:predict   {"customer_id": "...", "instances": [ {7 features}, ... ]}
      -> {"predictions": [...], "model_version": "v1"|"v2", "arm": "v1"|"v2", "served_by": ..., "customer_id": ...}
    header  X-Beta: 1   forces the candidate (v2) for that request
    GET  /healthz

Deploy (from the repository root):
    python -c "import ray; from ray import serve; import sys; sys.path.insert(0,'partC'); import eta_ray; \
               ray.init(num_cpus=2); serve.run(eta_ray.build(weight=0.1), name='eta', route_prefix='/')"

CPU budget (fits on 2 CPUs): router 0.25 + prod 2 x 0.25 + candidate up to 3 x 0.25 = 1.5 CPU at most.
The deployment/router structure follows the ideas of the Module 4 lecture notebook (Ray Serve canary);
the sticky hashing, header override and autoscaling choices are specific to this assignment.
"""
import hashlib
import os
import time
from typing import List, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ray import serve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

CPU_PER_REPLICA = 0.25
PROD_REPLICAS = 2
CANDIDATE_AUTOSCALING = {
    "min_replicas": 1,
    "max_replicas": 3,
    "target_ongoing_requests": 2,     # scale out when a replica holds more than ~2 requests at once
    "upscale_delay_s": 5,             # react within seconds to a burst
    "downscale_delay_s": 60,          # but wait a minute before giving capacity back (avoids flapping)
}


def bucket(customer_id: str) -> float:
    """Stable bucket in [0, 1): first 8 bytes of SHA-256 of the id.
    Same id -> same bucket in every process, on every machine, after every restart
    (unlike Python's salted built-in hash())."""
    h = hashlib.sha256(customer_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def choose_arm(customer_id: str, weight: float, beta_header: str = None) -> str:
    if beta_header is not None and beta_header.strip() == "1":
        return "v2"
    return "v2" if bucket(customer_id) < weight else "v1"


class Instance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: Literal[0, 1]
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance_km > 150 and n_stops > 20")
        return self


class RoutedRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=200)
    instances: List[Instance] = Field(min_length=1, max_length=500)


@serve.deployment(ray_actor_options={"num_cpus": CPU_PER_REPLICA})
class EtaModel:
    """Loads one bundle ONCE per replica (in __init__) and predicts a batch of rows."""

    def __init__(self, bundle_path: str, extra_work_ms: float = 0.0):
        import joblib
        b = joblib.load(bundle_path)
        self.model, self.features, self.version = b["model"], list(b["feature_columns"]), str(b["version"])
        self.extra_work_ms = float(extra_work_ms)
        self.replica = serve.get_replica_context().replica_id.unique_id

    def predict(self, rows: List[dict]) -> dict:
        import pandas as pd
        if self.extra_work_ms > 0:            # simulated heavier model (busy CPU), used to exercise autoscaling
            end = time.perf_counter() + self.extra_work_ms / 1000
            while time.perf_counter() < end:
                pass
        y = self.model.predict(pd.DataFrame(rows, columns=self.features))
        return {"predictions": [round(float(v), 1) for v in y], "model_version": self.version,
                "served_by": self.replica}


api = FastAPI(title="SwiftParcel ETA - sticky canary")


@api.exception_handler(RequestValidationError)
async def _validation_failed(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in e.get("loc", ()) if p != "body") or "body",
                "msg": e.get("msg", "invalid value")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "validation_failed", "details": details})


@serve.deployment(ray_actor_options={"num_cpus": CPU_PER_REPLICA})
@serve.ingress(api)
class Router:
    def __init__(self, prod, candidate, weight: float = 0.1):
        self.arms = {"v1": prod, "v2": candidate}
        self.weight = float(weight)

    @api.post("/v1/models/eta:predict")
    async def predict(self, req: RoutedRequest, request: Request):
        arm = choose_arm(req.customer_id, self.weight, request.headers.get("x-beta"))
        out = await self.arms[arm].predict.remote([i.model_dump() for i in req.instances])
        out.update({"arm": arm, "customer_id": req.customer_id})
        return out

    @api.get("/healthz")
    async def healthz(self):
        return {"status": "ok", "weight": self.weight, "model_version": "v1 (prod) / v2 (candidate)"}

    @api.get("/config")
    async def config(self):
        return {"weight": self.weight, "prod_replicas": PROD_REPLICAS,
                "candidate_autoscaling": CANDIDATE_AUTOSCALING, "cpu_per_replica": CPU_PER_REPLICA}


def build(weight: float = 0.1, extra_work_ms: float = 0.0):
    """Return the Serve application (bind it with serve.run(build(...), route_prefix='/'))."""
    prod = EtaModel.options(name="EtaModel_prod", num_replicas=PROD_REPLICAS).bind(
        os.path.join(DATA, "eta_v1.joblib"), extra_work_ms)
    candidate = EtaModel.options(name="EtaModel_candidate", autoscaling_config=CANDIDATE_AUTOSCALING).bind(
        os.path.join(DATA, "eta_v2.joblib"), extra_work_ms)
    return Router.bind(prod, candidate, weight)
