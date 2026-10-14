"""Content-addressed local cache shared by PDF extraction and page retrieval."""

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import tempfile


def cache_dir() -> Path:
    default = Path(__file__).resolve().parents[2] / ".cache" / "claimcheck"
    return Path(os.environ.get("CLAIMCHECK_CACHE_DIR", str(default))).expanduser().resolve()


def content_key(value) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@lru_cache(maxsize=128)
def _file_digest(path: Path, size: int, mtime_ns: int, ctime_ns: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_key(path: Path) -> str:
    path = path.resolve()
    stat = path.stat()
    return _file_digest(path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@contextmanager
def cache_lock(destination: Path):
    """Only one process builds a given artifact, including on a cold cache."""
    import fcntl

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(destination.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def atomic_file(destination: Path):
    """Publish complete artifacts; an interrupted write leaves the old file intact."""
    fd, name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
