"""Shared test fixtures and mock setup for arch-sparring-agent tests.

This module handles mocking of external dependencies (strands, boto3, bedrock_agentcore, etc.)
so individual test files don't need to repeat the boilerplate.
"""

import sys
from unittest.mock import MagicMock

import pytest

# List of all external modules that need mocking
_MOCKED_MODULES = [
    "strands",
    "strands.types",
    "strands.types.exceptions",
    "botocore",
    "botocore.exceptions",
    "boto3",
    "PIL",
    "frontmatter",
    "bedrock_agentcore",
    "bedrock_agentcore.memory",
    "bedrock_agentcore.memory.integrations",
    "bedrock_agentcore.memory.integrations.strands",
    "bedrock_agentcore.memory.integrations.strands.config",
    "bedrock_agentcore.memory.integrations.strands.session_manager",
    "bedrock_agentcore_starter_toolkit",
    "bedrock_agentcore_starter_toolkit.operations",
    "bedrock_agentcore_starter_toolkit.operations.gateway",
    "bedrock_agentcore_starter_toolkit.operations.gateway.client",
]


# --- Fake exception classes for isinstance checks ---


class FakeContextWindowOverflow(Exception):
    """Fake ContextWindowOverflowException for tests."""


class FakeMaxTokensReached(Exception):
    """Fake MaxTokensReachedException for tests."""


class FakeClientError(Exception):
    """Fake botocore ClientError for tests."""

    def __init__(self, error_code: str = "ValidationException"):
        self.response = {"Error": {"Code": error_code}}
        super().__init__(f"ClientError: {error_code}")


def _setup_mocks() -> None:
    """Install mocked modules into sys.modules."""
    for mod in _MOCKED_MODULES:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    # Wire fake exceptions into mocked modules so isinstance checks work
    sys.modules[
        "strands.types.exceptions"
    ].ContextWindowOverflowException = FakeContextWindowOverflow
    sys.modules["strands.types.exceptions"].MaxTokensReachedException = FakeMaxTokensReached
    sys.modules["botocore.exceptions"].ClientError = FakeClientError


# Install mocks at import time so all test modules can import application code
_setup_mocks()


@pytest.fixture(autouse=True)
def _reset_mocks():
    """Ensure mocks are fresh for each test."""
    _setup_mocks()
    yield
