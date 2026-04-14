from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock


@dataclass
class BucketState:
    tokens: float
    last_refill_at: float


class TokenBucketRateLimiter:
    def __init__(self, capacity: int, refill_period_seconds: int) -> None:
        self.capacity = float(capacity)
        self.refill_period_seconds = float(refill_period_seconds)
        self.refill_rate = self.capacity / self.refill_period_seconds
        self._lock = Lock()
        self._buckets: dict[str, BucketState] = {}

    def consume(self, key: str, tokens: float = 1.0) -> tuple[bool, float]:
        now = time.monotonic()

        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = BucketState(tokens=self.capacity, last_refill_at=now)
                self._buckets[key] = bucket

            elapsed = now - bucket.last_refill_at
            bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_rate)
            bucket.last_refill_at = now

            if bucket.tokens >= tokens:
                bucket.tokens -= tokens
                return True, 0.0

            missing_tokens = tokens - bucket.tokens
            retry_after = missing_tokens / self.refill_rate
            return False, retry_after
