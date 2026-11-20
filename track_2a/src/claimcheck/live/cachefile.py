"""JSONL cache files that belong to one specific video file.

The first line records a fingerprint of the video (size + hash of its first megabyte). A cache
written for a different file with the same name is set aside instead of being reused.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def video_fingerprint(video: Path) -> str:
    with video.open("rb") as stream:
        head = stream.read(1 << 20)
    return f"{video.stat().st_size}-{hashlib.sha1(head).hexdigest()[:16]}"


def load_cache(path: Path, fingerprint: str | None) -> list[dict]:
    """Records of the cache, without the fingerprint line. Starts a fresh file when the
    fingerprint differs (the old file is kept as <name>.old-<timestamp>)."""
    records = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    if fingerprint is None:
        return [r for r in records if r.get("type") != "video"]
    stored = next((r["fingerprint"] for r in records if r.get("type") == "video"), None)
    if stored is not None and stored != fingerprint:
        path.rename(path.with_name(f"{path.name}.old-{int(time.time())}"))
        records, stored = [], None
    if stored is None:  # new file, or a cache from before fingerprints existed: adopt it
        append_cache(path, {"type": "video", "fingerprint": fingerprint})
    return [r for r in records if r.get("type") != "video"]


def append_cache(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
