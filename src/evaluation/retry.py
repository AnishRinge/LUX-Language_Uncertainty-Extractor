"""Configurable retry policy for the CP4.4 evaluator.

Keeps retry policy in configuration rather than hard-coding scientific
assumptions.

Classification of failures:
  * Transient / API failures (timeout, rate-limit, 5xx, connection errors) and
    parsing/schema failures are RETRYABLE.
  * Auth / invalid-argument errors are persistent and surface immediately.

The default ``sleep`` callable is ``time.sleep`` but is injectable so unit
tests never actually wait. Backoff is deterministic (no random jitter by
default) to keep evaluations reproducible.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Type


class RetryableError(Exception):
    """Marker base: exceptions that should be retried."""


class RetryExhausted(Exception):
    """Raised when a retryable operation exhausts the retry budget."""

    def __init__(self, last_error: BaseException, attempts: int):
        self.last_error = last_error
        self.attempts = attempts
        super().__init__(
            "retry exhausted after {0} attempt(s): {1}".format(attempts, last_error)
        )


def is_retryable_exception(exc: BaseException) -> bool:
    """Classify known transient error types by exception shape / name.

    Uses duck-typing on common attributes so it does not require importing the
    Google SDK (which may not be installed in the test environment).
    """
    if isinstance(exc, RetryableError):
        return True
    if isinstance(
        exc,
        (ConnectionError, TimeoutError, ConnectionResetError, BrokenPipeError),
    ):
        return True
    code = getattr(exc, "code", None) or getattr(exc, "grpc_status_code", None)
    if code is not None:
        if str(code) in ("429", "500", "502", "503", "504"):
            return True
    name = type(exc).__name__
    if name in (
        "RetryError",
        "Timeout",
        "ConnectionError",
        "RateLimitError",
        "ServiceUnavailable",
        "InternalServerError",
        "BadGateway",
        "GatewayTimeout",
    ):
        return True
    message = (str(exc) or "").lower()
    if any(
        token in message
        for token in ("timeout", "rate limit", "rate-limit", "throttl", "temporarily", "unavailable")
    ):
        return True
    return False


@dataclass
class RetryPolicy:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: float = 0.0
    sleep: Callable[[float], None] = field(default=lambda s: time.sleep(s))
    retryable_exceptions: Tuple[Type[BaseException], ...] = field(
        default_factory=tuple
    )

    def is_retryable(self, exc: BaseException) -> bool:
        if self.retryable_exceptions and isinstance(exc, self.retryable_exceptions):
            return True
        return is_retryable_exception(exc)

    def delay_for(self, attempt: int) -> float:
        """Pure exponential backoff (seconds) for the given 1-based attempt.

        No jitter is applied here; jitter is applied only at sleep time and
        defaults to 0 so the function is deterministic and reproducible.
        """
        delay = self.base_delay * (self.backoff_factor ** (attempt - 1))
        return min(delay, self.max_delay)

    def sleep_delay(self, attempt: int) -> float:
        delay = self.delay_for(attempt)
        if self.jitter and delay > 0:
            delay = delay * (1.0 - self.jitter)
        return max(0.0, delay)


def retry_call(
    func: Callable,
    policy: Optional[RetryPolicy] = None,
    on_attempt: Optional[Callable[[int, Optional[Exception]], None]] = None,
    description: str = "",
) -> Any:
    """Execute ``func()`` with a retry loop and return its result.

    ``on_attempt(attempt_number, exc)`` is invoked after every attempt:
    ``exc`` is None on success.

    Raises ``RetryExhausted`` if a retryable failure exhausts the budget, or
    re-raises the original exception if it is non-retryable (persistent).
    """
    if policy is None:
        result = func()
        if on_attempt:
            on_attempt(1, None)
        return result

    attempts = 0
    while True:
        attempts += 1
        try:
            result = func()
            if on_attempt:
                on_attempt(attempts, None)
            return result
        except Exception as exc:  # noqa: BLE001 - retry semantics are intentional
            if on_attempt:
                on_attempt(attempts, exc)
            if not policy.is_retryable(exc):
                raise
            if attempts > policy.max_retries:
                raise RetryExhausted(exc, attempts) from exc
            delay = policy.sleep_delay(attempts)
            if delay > 0:
                policy.sleep(delay)
