"""Tests for arch_sparring_agent.infra.memory."""

from unittest.mock import MagicMock, patch

from arch_sparring_agent.infra.memory import (
    _find_memory_by_name,
    create_session_manager,
    setup_agentcore_memory,
)
from tests.conftest import FakeClientError

_MEM_ARN = "arn:aws:bedrock:us-east-1:111:memory/ArchReviewMemory"
_OTHER_ARN = "arn:aws:bedrock:us-east-1:111:memory/OtherMemory"


def test_find_memory_by_name_found():
    memories = [
        {"arn": _MEM_ARN, "id": "mem-1", "status": "ACTIVE"},
    ]
    mem_id, status = _find_memory_by_name(memories, "ArchReviewMemory")

    assert mem_id == "mem-1"
    assert status == "ACTIVE"


def test_find_memory_by_name_not_found():
    memories = [
        {"arn": _OTHER_ARN, "id": "mem-2", "status": "ACTIVE"},
    ]
    mem_id, status = _find_memory_by_name(memories, "ArchReviewMemory")

    assert mem_id is None
    assert status is None


@patch("arch_sparring_agent.infra.memory.MemoryClient")
def test_setup_finds_existing_active_memory(mock_memory_cls):
    mock_client = MagicMock()
    mock_memory_cls.return_value = mock_client
    mock_client.list_memories.return_value = [
        {"arn": _MEM_ARN, "id": "mem-1", "status": "ACTIVE"},
    ]

    config, mem_id = setup_agentcore_memory(region="us-east-1")

    assert mem_id == "mem-1"
    assert config is not None
    mock_client.create_memory.assert_not_called()


@patch("arch_sparring_agent.infra.memory.time")
@patch("arch_sparring_agent.infra.memory.MemoryClient")
def test_setup_creates_new_memory_and_waits(mock_memory_cls, mock_time):
    mock_client = MagicMock()
    mock_memory_cls.return_value = mock_client

    creating = [{"arn": _MEM_ARN, "id": "mem-new", "status": "CREATING"}]
    active = [{"arn": _MEM_ARN, "id": "mem-new", "status": "ACTIVE"}]
    mock_client.list_memories.side_effect = [[], creating, active]
    mock_client.create_memory.return_value = {"id": "mem-new"}

    config, mem_id = setup_agentcore_memory(region="us-east-1")

    assert mem_id == "mem-new"
    assert config is not None
    mock_client.create_memory.assert_called_once()


@patch("arch_sparring_agent.infra.memory.MemoryClient")
def test_setup_returns_none_on_failure(mock_memory_cls):
    mock_client = MagicMock()
    mock_memory_cls.return_value = mock_client
    mock_client.list_memories.side_effect = FakeClientError("AccessDenied")

    config, mem_id = setup_agentcore_memory(region="us-east-1")

    assert config is None
    assert mem_id is None


@patch("arch_sparring_agent.infra.memory.time")
@patch("arch_sparring_agent.infra.memory.MemoryClient")
def test_setup_returns_none_when_memory_stays_non_active(mock_memory_cls, mock_time):
    mock_client = MagicMock()
    mock_memory_cls.return_value = mock_client

    stuck = [{"arn": _MEM_ARN, "id": "mem-stuck", "status": "CREATING"}]
    mock_client.list_memories.side_effect = [[]] + [stuck] * 37
    mock_client.create_memory.return_value = {"id": "mem-stuck"}

    config, mem_id = setup_agentcore_memory(region="us-east-1")

    assert config is None
    assert mem_id is None


def test_create_session_manager_returns_none_for_none():
    result = create_session_manager(None)
    assert result is None


def test_create_session_manager_returns_manager_for_config():
    mock_config = MagicMock()
    mock_config.actor_id = "test-actor"

    result = create_session_manager(mock_config)

    assert result is not None
