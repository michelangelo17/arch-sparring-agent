"""AgentCore memory setup for session persistence."""

from __future__ import annotations

import logging
import time
from datetime import datetime

from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)

from ..config import DEFAULT_REGION
from ..exceptions import AWS_ERRORS

logger = logging.getLogger(__name__)


def _find_memory_by_name(memories: list, memory_name: str) -> tuple[str | None, str | None]:
    """Find memory ID and status by name from list of memories.

    MemorySummary from ListMemories has: arn, id, status, createdAt, updatedAt.
    There is no 'name' field — match against the ARN which embeds the name.
    The MemoryClient wrapper normalises id/memoryId on each item.
    """
    for m in memories:
        arn = m.get("arn", "")
        if memory_name.lower() in arn.lower():
            return m.get("id"), m.get("status", "")
    return None, None


def setup_agentcore_memory(
    region: str = DEFAULT_REGION,
    memory_name: str = "ArchReviewMemory",
    actor_id: str | None = None,
    session_id: str | None = None,
) -> tuple[AgentCoreMemoryConfig | None, str | None]:
    """Setup AgentCore memory for agents.

    Returns:
        Tuple of (memory_config, memory_id), or (None, None) on failure.
    """
    client = MemoryClient(region_name=region)

    try:
        memories = client.list_memories()
        memory_id, memory_status = _find_memory_by_name(memories, memory_name)

        if memory_id:
            if memory_status and memory_status.upper() != "ACTIVE":
                logger.info("Memory '%s' exists but status is %s", memory_name, memory_status)
                for attempt in range(12):
                    time.sleep(5)
                    elapsed = (attempt + 1) * 5
                    memories = client.list_memories()
                    memory_id, memory_status = _find_memory_by_name(memories, memory_name)
                    if memory_status and memory_status.upper() == "ACTIVE":
                        break
                    logger.info(
                        "Waiting for memory to become active... (%ds elapsed, status: %s)",
                        elapsed,
                        memory_status,
                    )

                if memory_status and memory_status.upper() != "ACTIVE":
                    logger.debug("Memory not active after waiting. Continuing without memory.")
                    return None, None

            logger.info("Using existing memory: %s", memory_name)
        else:
            memory = client.create_memory(
                name=memory_name, description="Memory for arch review agents"
            )
            memory_id = memory.get("id")
            logger.info("Created memory: %s", memory_name)

            # Wait for memory to become active (up to 3 minutes)
            logger.info("Waiting for memory to initialize...")
            memory_status = "CREATING"
            for attempt in range(36):  # 36 * 5s = 3 minutes max
                time.sleep(5)
                elapsed = (attempt + 1) * 5
                memories = client.list_memories()
                _, memory_status = _find_memory_by_name(memories, memory_name)
                status_upper = (memory_status or "").upper()

                if status_upper == "ACTIVE":
                    logger.info("Memory initialization done.")
                    break
                if status_upper == "FAILED":
                    logger.error("Memory creation failed.")
                    return None, None

                logger.info(
                    "Waiting for memory to initialize... (%ds elapsed, status: %s)",
                    elapsed,
                    memory_status,
                )
            else:
                logger.warning("Timeout waiting for memory. Status: %s", memory_status)
                return None, None

        if not memory_id:
            logger.warning("Could not resolve memory ID for '%s'. Skipping.", memory_name)
            return None, None

        actor_id = actor_id or f"actor_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        memory_config = AgentCoreMemoryConfig(
            memory_id=memory_id,
            session_id=session_id,
            actor_id=actor_id,
        )
        return memory_config, memory_id

    except AWS_ERRORS as e:
        logger.warning("Could not set up AgentCore Memory: %s", e)
        logger.warning("Continuing without memory.")
        return None, None


def create_session_manager(
    memory_config: AgentCoreMemoryConfig | None,
) -> AgentCoreMemorySessionManager | None:
    """Create a session manager from a memory config."""
    if not memory_config:
        return None

    actor_id = getattr(
        memory_config, "actor_id", f"actor_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    return AgentCoreMemorySessionManager(agentcore_memory_config=memory_config, actor_id=actor_id)
