"""Part A: FastAPI REST service for the SwiftParcel delivery-time (ETA) model.

Run from the repository root:
    ETA_MODEL=data/eta_v1.joblib uvicorn eta_app:app --app-dir partA --port 8000

Endpoints
    POST /v1/models/eta:predict   {"instances": [ {7 features}, ... ]}
    GET  /healthz
    GET  /metrics                 Prometheus text format

Request validation, the request-id middleware and the metrics follow the ideas of the
Module 4 lecture notebook (FastAPI churn service); the code is written for this problem.
"""
import logging
import os
import time
import uuid
from typing import List, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, model_validator

log = logging.getLogger("eta_app")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(path: str) -> str:
    """Use the path as given; if it is relative and missing, try it relative to the repo root."""
    if os.path.exists(path) or os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


# ---------------------------------------------------------------- model (loaded once, at import)
MODEL_PATH = _resolve(os.environ.get("ETA_MODEL", "data/eta_v1.joblib"))
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]                    # module-level variable required by the interface contract
FEATURES: List[str] = list(bundle["feature_columns"])
MODEL_VERSION: str = str(bundle["version"])

# ---------------------------------------------------------------- metrics (own registry)
REGISTRY = CollectorRegistry()
REQUESTS = Counter("eta_requests_total", "HTTP requests by route and status code",
                   ["path", "status"], registry=REGISTRY)
LATENCY = Histogram("eta_request_latency_seconds", "End-to-end request latency in seconds",
                    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.08, 0.1, 0.25, 0.5, 1.0, 2.5),
                    registry=REGISTRY)


# ---------------------------------------------------------------- request schema
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


class PredictRequest(BaseModel):
    # extra top-level keys (e.g. customer_id) are ignored so the same client works for Part C
    instances: List[Instance] = Field(min_length=1, max_length=500)


app = FastAPI(title="SwiftParcel ETA service", version=MODEL_VERSION)


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())


def _internal_error(rid: str) -> JSONResponse:
    # generic body on purpose: no exception text or stack trace leaves the server
    return JSONResponse(status_code=500, content={"error": "internal_error", "request_id": rid})


@app.middleware("http")
async def request_id_and_metrics(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:                      # anything not handled below still gets a clean 500
        log.exception("unhandled error, request_id=%s", rid)
        response = _internal_error(rid)
    response.headers["X-Request-ID"] = rid
    if request.url.path != "/metrics":      # scrapes of /metrics are not counted
        route = request.scope.get("route")
        path = route.path if route is not None else "unmatched"
        REQUESTS.labels(path=path, status=str(response.status_code)).inc()
        LATENCY.observe(time.perf_counter() - t0)
    return response


@app.exception_handler(RequestValidationError)
async def validation_failed(request: Request, exc: RequestValidationError):
    details = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", ()) if p != "body"]
        details.append({"field": ".".join(loc) or "body", "msg": err.get("msg", "invalid value")})
    rid = _rid(request)
    return JSONResponse(status_code=422, headers={"X-Request-ID": rid},
                        content={"error": "validation_failed", "details": details, "request_id": rid})


@app.post("/v1/models/eta:predict")
def predict(req: PredictRequest, request: Request):
    rid = _rid(request)
    X = pd.DataFrame([inst.model_dump() for inst in req.instances], columns=FEATURES)
    try:
        y = model.predict(X)
        preds = [round(float(v), 1) for v in y]
    except Exception:
        log.exception("inference failed, request_id=%s", rid)
        return _internal_error(rid)
    return {"predictions": preds, "model_version": MODEL_VERSION, "request_id": rid}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
