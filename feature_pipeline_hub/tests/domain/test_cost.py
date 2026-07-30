import pytest

from feature_pipeline.domain.cost import GPU_HOURLY_RATE_ENV, estimate_cost, gpu_hourly_rate


def test_hourly_rate_is_none_when_unset(monkeypatch):
    monkeypatch.delenv(GPU_HOURLY_RATE_ENV, raising=False)

    assert gpu_hourly_rate() is None


def test_hourly_rate_is_none_for_garbage(monkeypatch):
    monkeypatch.setenv(GPU_HOURLY_RATE_ENV, "not-a-number")

    assert gpu_hourly_rate() is None


def test_hourly_rate_is_none_for_zero_or_negative(monkeypatch):
    monkeypatch.setenv(GPU_HOURLY_RATE_ENV, "0")
    assert gpu_hourly_rate() is None

    monkeypatch.setenv(GPU_HOURLY_RATE_ENV, "-1.5")
    assert gpu_hourly_rate() is None


def test_hourly_rate_parses_a_valid_value(monkeypatch):
    monkeypatch.setenv(GPU_HOURLY_RATE_ENV, "1.20")

    assert gpu_hourly_rate() == 1.20


def test_estimate_cost_is_none_without_a_rate():
    assert estimate_cost(3600, hourly_rate=None) is None


def test_estimate_cost_is_none_for_zero_gpu_seconds():
    assert estimate_cost(0, hourly_rate=1.0) is None


def test_estimate_cost_scales_with_gpu_seconds_and_rate():
    # 1 hour of GPU time at $1.20/hour
    assert estimate_cost(3600, hourly_rate=1.20) == 1.20
    # 30 minutes at $2.00/hour
    assert estimate_cost(1800, hourly_rate=2.00) == 1.00
