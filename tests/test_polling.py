"""Tests for arch_sparring_agent.infra.polling."""

from unittest.mock import patch

import pytest

from arch_sparring_agent.infra.polling import ensure_resource, poll_until
from tests.conftest import FakeClientError


@patch("arch_sparring_agent.infra.polling.time")
def test_poll_until_succeeds_immediately(mock_time):
    mock_time.monotonic.side_effect = [0.0, 0.0, 0.1]
    result = poll_until(lambda: "done", interval=1.0, timeout=10.0, desc="test")
    assert result == "done"


@patch("arch_sparring_agent.infra.polling.time")
def test_poll_until_succeeds_after_retries(mock_time):
    mock_time.monotonic.side_effect = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    attempts = [0]

    def _check():
        attempts[0] += 1
        return "ok" if attempts[0] >= 3 else None

    result = poll_until(_check, interval=1.0, timeout=10.0, desc="test")
    assert result == "ok"
    assert attempts[0] == 3


@patch("arch_sparring_agent.infra.polling.time")
def test_poll_until_times_out(mock_time):
    call_count = [0]

    def advancing_monotonic():
        call_count[0] += 1
        return float(call_count[0] * 20)

    mock_time.monotonic.side_effect = advancing_monotonic

    result = poll_until(lambda: None, interval=1.0, timeout=10.0, desc="test")
    assert result is None


@patch("arch_sparring_agent.infra.polling.time")
def test_poll_until_aborts_on_stop_iteration(mock_time):
    mock_time.monotonic.side_effect = [0.0, 0.0, 1.0]

    def _check():
        raise StopIteration

    result = poll_until(_check, interval=1.0, timeout=10.0, desc="test")
    assert result is None


@patch("arch_sparring_agent.infra.polling.time")
def test_poll_until_applies_backoff(mock_time):
    mock_time.monotonic.side_effect = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    sleep_calls = []
    mock_time.sleep.side_effect = lambda d: sleep_calls.append(d)

    attempts = [0]

    def _check():
        attempts[0] += 1
        return "ok" if attempts[0] >= 4 else None

    poll_until(_check, interval=1.0, timeout=30.0, backoff=2.0, max_interval=5.0, desc="test")

    assert sleep_calls[0] == 1.0
    assert sleep_calls[1] == 2.0
    assert sleep_calls[2] == 4.0


def test_ensure_resource_creates_new():
    result = ensure_resource(lambda: "new-id", lambda: None, desc="test")
    assert result == "new-id"


def test_ensure_resource_finds_existing_on_conflict():
    def _create():
        raise FakeClientError("ConflictException")

    result = ensure_resource(_create, lambda: "existing-id", desc="test")
    assert result == "existing-id"


def test_ensure_resource_raises_on_non_conflict():
    def _create():
        raise FakeClientError("InternalServerError")

    with pytest.raises(FakeClientError):
        ensure_resource(_create, lambda: "existing-id", desc="test")


def test_ensure_resource_raises_when_find_returns_none():
    def _create():
        raise FakeClientError("ConflictException")

    with pytest.raises(FakeClientError):
        ensure_resource(_create, lambda: None, desc="test")
