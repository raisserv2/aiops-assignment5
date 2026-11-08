"""Part B2: tests for the pure decide() function (synthetic numbers, no MLflow needed).
Run from the repo root:  pytest partB -q"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eta_mlflow import decide  # noqa: E402

PROD = {"mae": 10.0, "mae_rush": 12.0, "mae_rain": 11.0, "p95_ms": 5.0}


def cand(**over):
    c = {"mae": 9.0, "mae_rush": 11.0, "mae_rain": 10.0, "p95_ms": 5.0}   # clearly better everywhere
    c.update(over)
    return c


def test_clear_win_is_promoted():
    out = decide(PROD, cand())
    assert out["promote"] is True
    assert set(out["criteria"]) == {"mae_gain", "latency_p95", "slice_rush", "slice_rain"}
    assert all(c["pass"] for c in out["criteria"].values())


def test_worse_candidate_is_rejected():
    out = decide(PROD, cand(mae=10.5))
    assert out["promote"] is False and out["criteria"]["mae_gain"]["pass"] is False


# --- MAE gain boundary (default 3 % of 10.0 = 0.3 minutes)
def test_mae_gain_just_inside():
    assert decide(PROD, cand(mae=9.69))["promote"] is True          # 3.1 % gain


def test_mae_gain_exactly_at_threshold_passes():
    assert decide(PROD, cand(mae=9.7))["criteria"]["mae_gain"]["pass"] is True


def test_mae_gain_just_outside():
    out = decide(PROD, cand(mae=9.71))                               # 2.9 % gain
    assert out["promote"] is False and out["criteria"]["mae_gain"]["pass"] is False


# --- latency boundary
def test_latency_just_inside_and_at_limit():
    assert decide(PROD, cand(p95_ms=9.99))["promote"] is True
    assert decide(PROD, cand(p95_ms=10.0))["promote"] is True


def test_latency_just_outside():
    out = decide(PROD, cand(p95_ms=10.01))
    assert out["promote"] is False and out["criteria"]["latency_p95"]["pass"] is False


# --- slice boundaries (5 % relative regression allowed)
def test_rush_slice_small_regression_allowed():
    assert decide(PROD, cand(mae_rush=12.59))["promote"] is True     # +4.9 %


def test_rush_slice_regression_blocks_even_with_big_overall_gain():
    out = decide(PROD, cand(mae=5.0, mae_rush=12.61))                # +5.1 % on rush hours
    assert out["promote"] is False
    assert out["criteria"]["slice_rush"]["pass"] is False and out["criteria"]["mae_gain"]["pass"] is True


def test_rain_slice_just_inside_and_just_outside():
    assert decide(PROD, cand(mae_rain=11.54))["promote"] is True     # +4.9 %
    out = decide(PROD, cand(mae_rain=11.56))                         # +5.1 %
    assert out["promote"] is False and out["criteria"]["slice_rain"]["pass"] is False


# --- custom thresholds and purity
def test_custom_thresholds():
    assert decide(PROD, cand(mae=6.0), min_rel_gain=0.60)["promote"] is False   # 40 % < 60 %
    assert decide(PROD, cand(mae=3.9), min_rel_gain=0.60)["promote"] is True    # 61 %
    assert decide(PROD, cand(p95_ms=15), max_p95_ms=20.0)["promote"] is True
    assert decide(PROD, cand(mae_rush=13.0), max_slice_regression=0.10)["promote"] is True


def test_decide_is_pure():
    p, c = copy.deepcopy(PROD), cand()
    first = decide(p, c)
    assert p == PROD and c == cand()                                  # inputs not modified
    assert decide(p, c) == first                                     # same inputs, same output
