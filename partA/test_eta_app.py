"""Part A2: tests for eta_app.py.   Run from the repo root:  pytest partA -q"""
import os
import sys
import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eta_app  # noqa: E402

URL = "/v1/models/eta:predict"


@pytest.fixture
def client():
    return TestClient(eta_app.app, raise_server_exceptions=False)


def row(**over):
    r = {"distance_km": 12.5, "n_stops": 4, "hour": 10, "weekday": 2,
         "traffic_index": 0.4, "rain": 0, "weight_kg": 8.0}
    r.update(over)
    return r


def assert_validation_error(resp):
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "validation_failed"
    assert isinstance(body["details"], list) and body["details"]
    assert {"field", "msg"} <= set(body["details"][0])
    assert body["request_id"] == resp.headers["X-Request-ID"]
    return body


# ---------- valid requests
def test_valid_request(client):
    r = client.post(URL, json={"instances": [row()]})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"predictions", "model_version", "request_id"}
    assert len(body["predictions"]) == 1
    p = body["predictions"][0]
    assert isinstance(p, float) and p > 0 and round(p, 1) == p     # one decimal
    assert body["model_version"] == eta_app.MODEL_VERSION


def test_batch_order_preserved(client):
    rows = [row(distance_km=d, n_stops=s) for d, s in [(80, 9), (2, 1), (40, 15), (5, 3), (120, 2)]]
    r = client.post(URL, json={"instances": rows})
    assert r.status_code == 200
    expected = [round(float(v), 1) for v in eta_app.model.predict(pd.DataFrame(rows)[eta_app.FEATURES])]
    assert r.json()["predictions"] == expected


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": eta_app.MODEL_VERSION}


# ---------- three different kinds of invalid input
def test_out_of_range_value(client):
    body = assert_validation_error(client.post(URL, json={"instances": [row(hour=24)]}))
    assert any(d["field"].endswith("hour") for d in body["details"])


def test_missing_field(client):
    bad = row()
    del bad["weight_kg"]
    body = assert_validation_error(client.post(URL, json={"instances": [bad]}))
    assert any(d["field"].endswith("weight_kg") for d in body["details"])


def test_wrong_type_and_bad_category(client):
    assert_validation_error(client.post(URL, json={"instances": [row(n_stops="many")]}))
    assert_validation_error(client.post(URL, json={"instances": [row(rain=2)]}))
    assert_validation_error(client.post(URL, json={"instances": [row(weight_kg=0)]}))   # must be > 0


def test_malformed_json(client):
    r = client.post(URL, content=b"{not json", headers={"Content-Type": "application/json"})
    assert_validation_error(r)


# ---------- cross-field rule
def test_cross_field_rule_rejects_infeasible(client):
    r = client.post(URL, json={"instances": [row(), row(distance_km=151, n_stops=21)]})
    body = assert_validation_error(r)
    assert "infeasible" in r.text
    assert any("infeasible" in d["msg"] for d in body["details"])


def test_cross_field_rule_boundary_is_allowed(client):
    # exactly 150 km / 20 stops is not "> 150 and > 20", and each condition alone is fine
    rows = [row(distance_km=150, n_stops=20), row(distance_km=300, n_stops=5), row(distance_km=20, n_stops=50)]
    assert client.post(URL, json={"instances": rows}).status_code == 200


# ---------- batch-size limits
def test_batch_size_lower_limit(client):
    assert_validation_error(client.post(URL, json={"instances": []}))


def test_batch_size_upper_limit(client):
    assert client.post(URL, json={"instances": [row()] * 500}).status_code == 200
    assert_validation_error(client.post(URL, json={"instances": [row()] * 501}))


# ---------- request id
def test_request_id_echoed(client):
    r = client.post(URL, json={"instances": [row()]}, headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    assert r.json()["request_id"] == "abc-123"


def test_request_id_generated(client):
    r = client.post(URL, json={"instances": [row()]})
    rid = r.headers["X-Request-ID"]
    uuid.UUID(rid)                                    # a valid UUID
    assert r.json()["request_id"] == rid
    assert client.post(URL, json={"instances": [row()]}).headers["X-Request-ID"] != rid


def test_request_id_on_errors(client):
    r = client.post(URL, json={"instances": [row(hour=99)]}, headers={"X-Request-ID": "err-1"})
    assert r.status_code == 422 and r.headers["X-Request-ID"] == "err-1" and r.json()["request_id"] == "err-1"
    r = client.post(URL, json={"instances": []})
    uuid.UUID(r.headers["X-Request-ID"])
    assert r.json()["request_id"] == r.headers["X-Request-ID"]


# ---------- metrics
def test_metrics_output(client):
    client.post(URL, json={"instances": [row()]})
    client.post(URL, json={"instances": [row(hour=30)]})
    client.get("/metrics")
    text = client.get("/metrics").text
    assert 'eta_requests_total{path="/v1/models/eta:predict",status="200"}' in text
    assert 'eta_requests_total{path="/v1/models/eta:predict",status="422"}' in text
    assert 'eta_request_latency_seconds_bucket{le="+Inf"}' in text
    assert text.count("eta_request_latency_seconds_bucket{") >= 5
    assert "eta_request_latency_seconds_sum" in text and "eta_request_latency_seconds_count" in text
    assert 'path="/metrics"' not in text              # /metrics itself is not counted


def test_metrics_counter_increments(client):
    def count(text):
        key = 'eta_requests_total{path="/healthz",status="200"} '
        line = [ln for ln in text.splitlines() if ln.startswith(key)]
        return float(line[0].split()[-1]) if line else 0.0
    before = count(client.get("/metrics").text)
    client.get("/healthz")
    client.get("/healthz")
    assert count(client.get("/metrics").text) == before + 2


# ---------- generic 500
def test_inference_error_returns_generic_500(client, monkeypatch):
    class Broken:
        def predict(self, X):
            raise RuntimeError("secret internal detail: /opt/models/xyz")
    monkeypatch.setattr(eta_app, "model", Broken())
    r = client.post(URL, json={"instances": [row()]}, headers={"X-Request-ID": "boom-1"})
    assert r.status_code == 500
    assert r.json() == {"error": "internal_error", "request_id": "boom-1"}
    assert "secret" not in r.text and "Traceback" not in r.text and "RuntimeError" not in r.text
    assert r.headers["X-Request-ID"] == "boom-1"
