from __future__ import annotations

from collections import deque
from threading import Lock


class MetricsService:
    def __init__(self, history_size: int = 500) -> None:
        self._latencies_ms: deque[float] = deque(maxlen=history_size)
        self._lock = Lock()

    def record_query_latency(self, latency_ms: float) -> None:
        with self._lock:
            self._latencies_ms.append(latency_ms)

    def average_query_latency_ms(self) -> float | None:
        with self._lock:
            if not self._latencies_ms:
                return None
            return sum(self._latencies_ms) / len(self._latencies_ms)
