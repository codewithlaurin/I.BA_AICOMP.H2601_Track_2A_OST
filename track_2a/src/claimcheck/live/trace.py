"""What the claim checker sent to Apertus and got back, for the trace view of a claim in the page.

    with collect() as steps:      # one thread: extraction of a segment, or the check of a claim
        ...                       # every call(...) and note(...) in this thread lands in `steps`

Outside collect() call() is plain nli.complete() and note() does nothing, so the verdict procedure
behaves the same with and without tracing. The steps are written next to the claims cache
(<video>.trace.jsonl) and only read when somebody opens a claim's trace.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path

_local = threading.local()


@contextmanager
def collect():
    previous = getattr(_local, "steps", None)
    steps: list[dict] = []
    _local.steps = steps
    try:
        yield steps
    finally:
        _local.steps = previous


def note(kind: str, **data) -> None:
    steps = getattr(_local, "steps", None)
    if steps is not None:
        steps.append({"kind": kind, **data})


def call(name: str, system: str, user: str, max_tokens: int) -> dict:
    """nli.complete(), with prompt, request and raw answer recorded under `name`."""
    from ..nli import complete   # looked up at call time: tests patch claimcheck.nli.complete

    try:
        result = complete(system, user, max_tokens)
    except Exception as exc:
        note("call", name=name, system=system, user=user, max_tokens=max_tokens,
             error=f"{exc.__class__.__name__}: {exc}")
        raise
    note("call", name=name, system=system, user=user, max_tokens=max_tokens, response=result.get("text", ""),
         model=result.get("model"), input_tokens=result.get("input_tokens"),
         output_tokens=result.get("output_tokens"), ms=result.get("api_time_ms"))
    return result


def trace_path(claims_cache: Path) -> Path:
    """x.claims.jsonl -> x.trace.jsonl"""
    name = Path(claims_cache).name
    stem = name[:-len(".claims.jsonl")] if name.endswith(".claims.jsonl") else Path(name).stem
    return Path(claims_cache).with_name(stem + ".trace.jsonl")


def find(path: Path, kind: str, key: str, field: str, match: dict) -> dict | None:
    """The last record of `kind` whose `field` is `key` and that has the values in `match`."""
    if not path.exists():
        return None
    needle = json.dumps(key, ensure_ascii=False)
    found = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if needle not in line:   # cheap pre-filter: the file holds whole booklet pages per call
                continue
            try:
                record = json.loads(line)
            except ValueError:   # a line a worker is still writing
                continue
            if (record.get("type") == kind and record.get(field) == key
                    and all(record.get(k) == v for k, v in match.items())):
                found = record
    return found
