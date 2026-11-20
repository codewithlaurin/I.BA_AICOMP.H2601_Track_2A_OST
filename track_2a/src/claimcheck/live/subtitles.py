"""Subtitle state for the live demo.

Keeps the transcribed segments, remembers which parts of the video are already transcribed
(both are cached in a JSONL file next to the video) and runs at most one live transcription
session that follows playback. A second run of the same video needs no speech service.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from .cachefile import append_cache, load_cache

END_OF_VIDEO = 1e9
LOOKAHEAD = 8.0   # open a live session when fewer seconds than this are transcribed ahead of playback
IDLE_STOP = 2.0   # close the live session this long after the video paused (it would run ahead of the picture)


class Subtitles:
    """start_transcriber(start_seconds, emit) -> object with .stop(); None disables live transcription.

    The transcriber calls emit(event) from its own threads with
      {"type": "interim", "start", "end", "text"}            text may still change
      {"type": "final", "start", "end", "text", "words"}     words: [{"t": seconds, "w": word}]
      {"type": "error", "message"} / {"type": "end"}         end: the video's audio is finished
    All times are seconds from the start of the video.
    """

    def __init__(self, cache: Path, start_transcriber: Callable | None = None,
                 clock: Callable[[], float] = time.monotonic, fingerprint: str | None = None):
        self.cache = Path(cache)
        self._fingerprint = fingerprint
        self._start = start_transcriber
        self._clock = clock
        self._lock = threading.RLock()
        self.finals: list[dict] = []
        self.ranges: list[list[float]] = []   # transcribed stretches of the video, merged, sorted
        self.interim: dict | None = None
        self.error: str | None = None
        self._session: dict | None = None
        self._token = 0
        self._last_playing = clock()
        self._load()

    # ---------------------------------------------------------------- cache

    def _load(self) -> None:
        for record in load_cache(self.cache, self._fingerprint):
            if record.get("type") == "range":
                self._add_range(record["start"], record["end"])
            elif record.get("type") == "final":
                self.finals.append({**record, "id": len(self.finals)})

    def _save(self, record: dict) -> None:
        append_cache(self.cache, record)

    def snapshot(self) -> list[dict]:
        """All final segments so far, in the order of the video."""
        with self._lock:
            return sorted(self.finals, key=lambda segment: segment["start"])

    def _add_range(self, start: float, end: float) -> None:
        merged: list[list[float]] = []
        for a, b in sorted(self.ranges + [[start, end]]):
            if merged and a <= merged[-1][1] + 0.5:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        self.ranges = merged

    def gaps(self, duration: float, min_length: float = 3.0) -> list[tuple[float, float]]:
        """Stretches of [0, duration] that are not transcribed yet."""
        gaps, position = [], 0.0
        for a, b in self.ranges:
            if a - position >= min_length:
                gaps.append((position, min(a, duration)))
            position = max(position, b)
        if duration - position >= min_length:
            gaps.append((position, duration))
        return [(a, b) for a, b in gaps if b - a >= min_length]

    def add_transcribed(self, start: float, end: float, finals: list[dict]) -> None:
        """Store a stretch transcribed outside a live session (batch mode)."""
        with self._lock:
            for event in sorted(finals, key=lambda e: e["start"]):
                record = {"type": "final", "id": len(self.finals),
                          **{k: event[k] for k in ("start", "end", "text", "words")}}
                self.finals.append(record)
                self._save(record)
            self._add_range(start, end)
            self._save({"type": "range", "start": round(start, 2), "end": round(end, 2)})

    def _covered_until(self, t: float) -> float:
        """End of the transcribed stretch that contains t; t itself when it is not transcribed."""
        for a, b in self.ranges:
            if a - 0.5 <= t < b:
                return b
        return t

    # ---------------------------------------------------------------- live session

    def _open(self, start: float) -> None:
        self._token += 1
        token = self._token
        session = {"start": start, "t0": self._clock(), "last_end": start, "transcriber": None}
        self._session = session
        try:
            session["transcriber"] = self._start(start, lambda event: self._on_event(token, event))
        except Exception as exc:  # missing key, missing package, no network: show it, stop trying
            self._session, self._start = None, None
            self.error = f"{exc.__class__.__name__}: {exc}"
            return
        session["t0"] = self._clock()

    def _close(self, end: float | None = None) -> None:
        session = self._session
        if session is None:
            return
        self._session, self.interim = None, None
        self._token += 1  # events of this session that are still on their way are ignored
        try:
            if session["transcriber"] is not None:
                session["transcriber"].stop()
        except Exception:
            pass
        end = session["last_end"] if end is None else end
        if end > session["start"]:
            self._add_range(session["start"], end)
            self._save({"type": "range", "start": round(session["start"], 2), "end": round(end, 2)})

    def _on_event(self, token: int, event: dict) -> None:
        with self._lock:
            session = self._session
            if session is None or token != self._token:
                return
            kind = event.get("type")
            if kind == "interim":
                self.interim = {k: event[k] for k in ("start", "end", "text")}
            elif kind == "final":
                if any(a + 0.3 <= event["start"] < b - 0.3 for a, b in self.ranges):
                    return  # this stretch was transcribed in an earlier session
                record = {"type": "final", "id": len(self.finals),
                          **{k: event[k] for k in ("start", "end", "text", "words")}}
                self.finals.append(record)
                self._save(record)
                session["last_end"] = max(session["last_end"], event["end"])
                self.interim = None
            elif kind == "error":
                self.error = str(event.get("message"))
                self._close()
            elif kind == "end":
                self._close(event.get("until", END_OF_VIDEO))

    def stop_if_idle(self) -> None:
        """Called periodically: a paused or closed player must not keep the speech service running."""
        with self._lock:
            if self._session is not None and self._clock() - self._last_playing > IDLE_STOP:
                self._close()

    def shutdown(self) -> None:
        with self._lock:
            self._close()

    # ---------------------------------------------------------------- what the page asks for

    def sync(self, t: float, playing: bool, since: int) -> dict:
        """The page calls this a few times per second with its playback position.

        Returns the final segments it does not have yet (ids >= since) and the current interim text;
        makes sure a live session is transcribing where playback is about to need it.
        """
        with self._lock:
            now = self._clock()
            if playing and self._start is not None:
                self._last_playing = now
                edge = self._covered_until(t)
                session = self._session
                if session is not None:
                    head = session["start"] + now - session["t0"]  # the session runs in real time
                    if any(a + 1 < head < b for a, b in self.ranges):
                        self._close(head)        # it has reached a stretch that is already done
                    elif session["start"] > edge + 0.5 or head < t - 1.5:
                        self._close()            # the viewer jumped somewhere else
                if self._session is None and edge - t < LOOKAHEAD:
                    self._open(edge)
            return {"segments": self.finals[max(since, 0):],
                    "interim": self.interim if self._session is not None else None,
                    "live": self._session is not None,
                    "can_transcribe": self._start is not None,
                    "error": self.error}
