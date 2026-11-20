"""The arenas of the live demo: one debate video each, with its booklet and its web sources.

Configured in data/demo/arenas.json (next to the videos, not in git), for example:

    {"arenas": [{
        "id": "kmg",
        "title": "Kriegsmaterialgesetz",
        "vote": "Änderung des Kriegsmaterialgesetzes",
        "arena_date": "2026-11-06", "vote_date": "2026-11-29",
        "video": "data/demo/arena_kriegsmaterialgesetz_2026-11-06.mp4",
        "booklet": "data/booklets/pdfs/BAR_Erlaeuterungen_Nov26_DE_v10.pdf",
        "pages": "10-11,50-61",
        "keywords": ["Kriegsmaterial"],
        "sources": ["https://www.admin.ch/de/kriegsmaterialgesetz", "..."]
    }]}

Paths are relative to track_2a/. See arenas.example.json.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # track_2a/
DEFAULT_CONFIG = ROOT / "data" / "demo" / "arenas.json"


@dataclass
class Arena:
    id: str
    title: str
    vote: str
    video: Path
    booklet: Path | None = None
    pages: str | None = None
    arena_date: str = ""
    vote_date: str = ""
    keywords: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @property
    def subtitles_cache(self) -> Path:
        return self.video.with_suffix(".subtitles.jsonl")

    @property
    def claims_cache(self) -> Path:
        return self.video.with_suffix(".claims.jsonl")

    @property
    def web_cache(self) -> Path:
        return self.video.with_suffix(".web.jsonl")

    @property
    def poster(self) -> Path:
        return self.video.with_suffix(".poster.jpg")


def resolve(path: str | None) -> Path | None:
    if not path:
        return None
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_arenas(config: Path = DEFAULT_CONFIG) -> list[Arena]:
    data = json.loads(Path(config).read_text(encoding="utf-8"))
    arenas = []
    for item in data["arenas"]:
        arenas.append(Arena(id=item["id"], title=item["title"], vote=item.get("vote", item["title"]),
                            video=resolve(item["video"]), booklet=resolve(item.get("booklet")),
                            pages=item.get("pages"), arena_date=item.get("arena_date", ""),
                            vote_date=item.get("vote_date", ""), keywords=item.get("keywords", []),
                            sources=item.get("sources", [])))
    ids = [a.id for a in arenas]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{config}: arena ids must be unique")
    return arenas


_durations: dict[Path, float] = {}
_lock = threading.Lock()


def video_duration(video: Path) -> float:
    """Seconds, read from the file header with the bundled ffmpeg (cached)."""
    with _lock:
        if video in _durations:
            return _durations[video]
    import imageio_ffmpeg

    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(video)], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stderr
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if not match:
        raise RuntimeError(f"cannot read the duration of {video}")
    hours, minutes, seconds = match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    with _lock:
        _durations[video] = duration
    return duration


def make_poster(arena: Arena, at: float | None = None) -> Path:
    """A still frame of the video for the start screen (made once, stored next to the video)."""
    if not arena.poster.exists():
        import imageio_ffmpeg

        if at is None:
            at = min(600.0, video_duration(arena.video) / 4)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-ss", str(at), "-i", str(arena.video),
                        "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "4", str(arena.poster)], check=True)
    return arena.poster
