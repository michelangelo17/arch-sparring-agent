"""Shared polling and idempotent-create utilities for infrastructure modules."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

T = TypeVar("T")

CONFLICT_CODES = ("ConflictException", "ResourceAlreadyExistsException")


def poll_until(
    check_fn: Callable[[], T | None],
    *,
    interval: float = 1.0,
    timeout: float = 60.0,
    backoff: float = 1.0,
    max_interval: float = 60.0,
    desc: str = "resource",
) -> T | None:
    """Poll *check_fn* until it returns a truthy value or *timeout* elapses.

    Args:
        check_fn: Returns a value on success, ``None`` to keep polling.
            Raise ``StopIteration`` to abort early (unrecoverable failure).
        interval: Initial delay between polls in seconds.
        timeout: Maximum wall-clock time in seconds.
        backoff: Multiplier applied to *interval* after each poll (1.0 = fixed).
        max_interval: Upper bound on the polling interval.
        desc: Human-readable label for log messages.

    Returns:
        The value from *check_fn*, or ``None`` on timeout / early abort.
    """
    start = time.monotonic()
    delay = interval

    while (time.monotonic() - start) < timeout:
        try:
            result = check_fn()
            if result is not None:
                logger.debug("%s ready after %.1fs", desc, time.monotonic() - start)
                return result
        except StopIteration:
            logger.debug("%s aborted early", desc)
            return None

        time.sleep(delay)
        delay = min(delay * backoff, max_interval)

    logger.warning("%s timed out after %.0fs", desc, timeout)
    return None


def ensure_resource(
    create_fn: Callable[[], T],
    find_fn: Callable[[], T | None],
    *,
    conflict_codes: tuple[str, ...] = CONFLICT_CODES,
    desc: str = "resource",
) -> T:
    """Try to create a resource; on conflict, find the existing one.

    Args:
        create_fn: Callable that creates and returns the resource.
        find_fn: Callable that locates an existing resource (or returns ``None``).
        conflict_codes: ``ClientError`` codes treated as "already exists".
        desc: Human-readable label for log messages.

    Returns:
        Result from *create_fn* or *find_fn*.

    Raises:
        The original ``ClientError`` if it is not a conflict or *find_fn*
        returns ``None``.
    """
    try:
        return create_fn()
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in conflict_codes:
            existing = find_fn()
            if existing is not None:
                logger.info("Using existing %s", desc)
                return existing
        raise
