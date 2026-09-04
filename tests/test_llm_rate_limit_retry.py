from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_graph import llm as llm_module


class _RateLimitError(Exception):
    def __init__(self, retry_after: str | None = None):
        headers = {} if retry_after is None else {"retry-after": retry_after}
        self.response = SimpleNamespace(headers=headers)
        super().__init__("rate limited")


class _BadRequestError(Exception):
    def __init__(self, code: str):
        self.code = code
        self.body = {"code": code}
        super().__init__(f"bad request: {code}")


class _APIStatusError(Exception):
    def __init__(self, status_code: int, retry_after: str | None = None):
        headers = {} if retry_after is None else {"retry-after": retry_after}
        self.status_code = status_code
        self.response = SimpleNamespace(headers=headers)
        super().__init__(f"status error: {status_code}")


class _APIConnectionError(Exception):
    pass


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


def test_retries_clova_unsupported_function_error(monkeypatch):
    monkeypatch.setattr(llm_module, "BadRequestError", _BadRequestError)
    sleeps = []
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)
    runnable = _SequenceRunnable([_BadRequestError("40009"), "success"])

    result = llm_module.invoke_with_rate_limit_retry(runnable, "input")

    assert result == "success"
    assert runnable.calls == 2
    assert sleeps == [1]


def test_reraises_other_bad_request_without_retry(monkeypatch):
    monkeypatch.setattr(llm_module, "BadRequestError", _BadRequestError)
    sleeps = []
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)
    error = _BadRequestError("40001")
    runnable = _SequenceRunnable([error])

    with pytest.raises(_BadRequestError) as captured:
        llm_module.invoke_with_rate_limit_retry(runnable, "input")

    assert captured.value is error
    assert runnable.calls == 1
    assert sleeps == []


def test_retries_server_error_and_prefers_retry_after(monkeypatch):
    monkeypatch.setattr(llm_module, "APIStatusError", _APIStatusError)
    sleeps = []
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)
    runnable = _SequenceRunnable([_APIStatusError(500, "2.5"), "success"])

    result = llm_module.invoke_with_rate_limit_retry(runnable, "input")

    assert result == "success"
    assert runnable.calls == 2
    assert sleeps == [2.5]


def test_reraises_non_retryable_status_error(monkeypatch):
    monkeypatch.setattr(llm_module, "APIStatusError", _APIStatusError)
    sleeps = []
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)
    error = _APIStatusError(403)
    runnable = _SequenceRunnable([error])

    with pytest.raises(_APIStatusError) as captured:
        llm_module.invoke_with_rate_limit_retry(runnable, "input")

    assert captured.value is error
    assert runnable.calls == 1
    assert sleeps == []


def test_retries_connection_error(monkeypatch):
    monkeypatch.setattr(llm_module, "APIConnectionError", _APIConnectionError)
    sleeps = []
    monkeypatch.setattr(llm_module.time, "sleep", sleeps.append)
    runnable = _SequenceRunnable([_APIConnectionError(), "success"])

    result = llm_module.invoke_with_rate_limit_retry(runnable, "input")

    assert result == "success"
    assert runnable.calls == 2
    assert sleeps == [1]
