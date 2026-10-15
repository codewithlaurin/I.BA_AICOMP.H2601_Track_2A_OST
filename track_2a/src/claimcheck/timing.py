"""Per-case wall time minus the union of external API call intervals."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time


class CaseTimer:
    def __init__(self):
        self.start = time.perf_counter()
        self.api_intervals: list[tuple[float, float]] = []
        self.preparation_seconds = 0.0

    def local_ms(self) -> int:
        end = time.perf_counter()
        api_seconds = 0.0
        previous_end = self.start
        for start, stop in sorted(self.api_intervals):
            start, stop = max(start, self.start), min(stop, end)
            api_seconds += max(0.0, stop - max(start, previous_end))
            previous_end = max(previous_end, stop)
        total_seconds = end - self.start + self.preparation_seconds
        return round(max(0.0, total_seconds - api_seconds) * 1000)


_current_timer: ContextVar[CaseTimer | None] = ContextVar("case_timer", default=None)


def measure_local_time(function):
    """Share a timer across nested prediction helpers; finalize every returned metric."""
    @wraps(function)
    def measured(*args, **kwargs):
        timer = _current_timer.get()
        token = None
        if timer is None:
            timer = CaseTimer()
            token = _current_timer.set(timer)
        try:
            result = function(*args, **kwargs)
            result.get("metrics", result)["inference_time_ms"] = timer.local_ms()
            return result
        finally:
            if token is not None:
                _current_timer.reset(token)
    return measured


@contextmanager
def external_api_call():
    """Exclude the entire SDK call, including retries and failed requests."""
    timer = _current_timer.get()
    if timer is None:
        raise RuntimeError("external_api_call requires a measured prediction")
    start = time.perf_counter()
    try:
        yield
    finally:
        timer.api_intervals.append((start, time.perf_counter()))


def include_preparation(seconds: float) -> None:
    """Charge eager, shared document preparation to its first consuming case only."""
    timer = _current_timer.get()
    if timer is None:
        raise RuntimeError("include_preparation requires a measured prediction")
    timer.preparation_seconds += seconds
