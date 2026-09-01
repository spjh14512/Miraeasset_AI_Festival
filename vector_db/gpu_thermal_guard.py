"""Pause local GPU embedding until the configured device cools down."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass


DEFAULT_GPU_PAUSE_TEMPERATURE = 82
DEFAULT_GPU_RESUME_TEMPERATURE = 72
DEFAULT_GPU_TEMPERATURE_POLL_SECONDS = 10.0

TemperatureReader = Callable[[], int]
Sleeper = Callable[[float], None]
Reporter = Callable[[str], None]


def read_cuda_zero_temperature() -> int:
    """Read the current GPU core temperature for CUDA device zero."""
    command = [
        "nvidia-smi",
        "--id=0",
        "--query-gpu=temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Failed to read GPU temperature with nvidia-smi") from error

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("nvidia-smi returned an unexpected GPU temperature result")
    try:
        temperature = int(lines[0])
    except ValueError as error:
        raise RuntimeError("nvidia-smi returned a non-integer GPU temperature") from error
    if not 0 <= temperature <= 150:
        raise RuntimeError("nvidia-smi returned an invalid GPU temperature")
    return temperature


@dataclass(slots=True)
class GpuThermalGuard:
    """Apply temperature hysteresis before each embedding window."""

    pause_temperature: int = DEFAULT_GPU_PAUSE_TEMPERATURE
    resume_temperature: int = DEFAULT_GPU_RESUME_TEMPERATURE
    poll_seconds: float = DEFAULT_GPU_TEMPERATURE_POLL_SECONDS
    temperature_reader: TemperatureReader = read_cuda_zero_temperature
    sleeper: Sleeper = time.sleep
    reporter: Reporter = print

    def __post_init__(self) -> None:
        for name, value in (
            ("pause_temperature", self.pause_temperature),
            ("resume_temperature", self.resume_temperature),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 1 <= value <= 150
            ):
                raise ValueError(f"{name} must be an integer between 1 and 150")
        if self.resume_temperature >= self.pause_temperature:
            raise ValueError("resume_temperature must be lower than pause_temperature")
        if (
            not isinstance(self.poll_seconds, (int, float))
            or isinstance(self.poll_seconds, bool)
            or self.poll_seconds <= 0
        ):
            raise ValueError("poll_seconds must be positive")
        if not callable(self.temperature_reader):
            raise ValueError("temperature_reader must be callable")
        if not callable(self.sleeper):
            raise ValueError("sleeper must be callable")
        if not callable(self.reporter):
            raise ValueError("reporter must be callable")

    def current_temperature(self) -> int:
        temperature = self.temperature_reader()
        if (
            not isinstance(temperature, int)
            or isinstance(temperature, bool)
            or not 0 <= temperature <= 150
        ):
            raise RuntimeError("GPU temperature reader returned an invalid value")
        return temperature

    def validate(self) -> int:
        """Fail before ingestion if the configured temperature reader is unavailable."""
        return self.current_temperature()

    def wait_until_ready(self) -> None:
        temperature = self.current_temperature()
        if temperature < self.pause_temperature:
            return

        self.reporter(
            "GPU thermal pause: "
            f"{temperature}°C >= {self.pause_temperature}°C; "
            f"waiting for {self.resume_temperature}°C or lower."
        )
        while temperature > self.resume_temperature:
            self.sleeper(float(self.poll_seconds))
            temperature = self.current_temperature()
        self.reporter(f"GPU thermal resume: {temperature}°C.")


__all__ = [
    "DEFAULT_GPU_PAUSE_TEMPERATURE",
    "DEFAULT_GPU_RESUME_TEMPERATURE",
    "DEFAULT_GPU_TEMPERATURE_POLL_SECONDS",
    "GpuThermalGuard",
    "read_cuda_zero_temperature",
]
