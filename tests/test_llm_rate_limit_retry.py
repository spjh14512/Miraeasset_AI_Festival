from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_graph import llm as llm_module


class _RateLimitError(Exception):
    def __init__(self, retry_after: str | None = None):
        headers = {} if retry_after is None else {"retry-after": retry_after}
        self.response = SimpleNamespace(headers=headers)
        super().__init__("rate limited")


class _SequenceRunnable:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def invoke(self, input_value):
        self.calls += 1
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output


def test_uses_exponential_backoff_when_retry_after_is_missing(monkeypatch):
    monkeypatch.setattr(llm_module, "RateLimitError", _RateLimitError)
    sleeps = []
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)
    runnable = _SequenceRunnable([
        _RateLimitError(),
        _RateLimitError(),
        "success",
    ])

    result = llm_module.invoke_with_rate_limit_retry(runnable, "input")

    assert result == "success"
    assert runnable.calls == 3
    assert sleeps == [1, 2]


def test_prefers_retry_after_header(monkeypatch):
    monkeypatch.setattr(llm_module, "RateLimitError", _RateLimitError)
    sleeps = []
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)
    runnable = _SequenceRunnable([_RateLimitError("3.5"), "success"])

    result = llm_module.invoke_with_rate_limit_retry(runnable, "input")

    assert result == "success"
    assert sleeps == [3.5]


def test_reraises_rate_limit_error_after_retries_are_exhausted(monkeypatch):
    monkeypatch.setattr(llm_module, "RateLimitError", _RateLimitError)
    monkeypatch.setattr(llm_module.time, "sleep", lambda _: None)
    final_error = _RateLimitError()
    runnable = _SequenceRunnable([
        _RateLimitError(),
        _RateLimitError(),
        final_error,
    ])

    with pytest.raises(_RateLimitError) as captured:
        llm_module.invoke_with_rate_limit_retry(runnable, "input")

    assert captured.value is final_error
    assert runnable.calls == 3
