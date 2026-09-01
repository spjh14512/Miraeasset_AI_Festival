from __future__ import annotations

from types import SimpleNamespace

import pytest

from vector_db import gpu_thermal_guard


def test_reads_cuda_zero_temperature(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return SimpleNamespace(stdout="43\n")

    monkeypatch.setattr(gpu_thermal_guard.subprocess, "run", fake_run)

    temperature = gpu_thermal_guard.read_cuda_zero_temperature()

    assert temperature == 43
    assert captured["command"] == [
        "nvidia-smi",
        "--id=0",
        "--query-gpu=temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    assert captured["kwargs"]["timeout"] == 10


def test_does_not_pause_below_threshold():
    sleeps = []
    messages = []
    guard = gpu_thermal_guard.GpuThermalGuard(
        temperature_reader=lambda: 81,
        sleeper=sleeps.append,
        reporter=messages.append,
    )

    guard.wait_until_ready()

    assert sleeps == []
    assert messages == []


def test_pauses_at_threshold_until_resume_temperature():
    temperatures = iter([82, 79, 72])
    sleeps = []
    messages = []
    guard = gpu_thermal_guard.GpuThermalGuard(
        pause_temperature=82,
        resume_temperature=72,
        poll_seconds=10,
        temperature_reader=lambda: next(temperatures),
        sleeper=sleeps.append,
        reporter=messages.append,
    )

    guard.wait_until_ready()

    assert sleeps == [10.0, 10.0]
    assert messages[0].startswith("GPU thermal pause: 82°C")
    assert messages[1] == "GPU thermal resume: 72°C."


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pause_temperature": 0}, "pause_temperature"),
        ({"resume_temperature": 82}, "lower than"),
        ({"poll_seconds": 0}, "poll_seconds"),
    ],
)
def test_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        gpu_thermal_guard.GpuThermalGuard(**kwargs)


def test_rejects_invalid_temperature_reader_result():
    guard = gpu_thermal_guard.GpuThermalGuard(
        temperature_reader=lambda: True,
    )

    with pytest.raises(RuntimeError, match="invalid value"):
        guard.validate()
