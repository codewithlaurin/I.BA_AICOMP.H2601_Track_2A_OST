"""Web server of the live demo: the page, the video file and the sync endpoint (subtitles + claims)."""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

from .claims import ClaimChecker
from .subtitles import Subtitles

STATIC = Path(__file__).parent / "static"


def create_app(video: Path, subtitles: Subtitles, checker: ClaimChecker | None = None,
               booklet: dict | None = None) -> FastAPI:
    """booklet: {"path", "vote", "pages"} enables /api/dev/claims (prompt development)."""
    app = FastAPI(title="claimcheck live")

    @app.get("/api/dev/claims")
    def dev_claims(start: float = 0.0, end: float = 1e9, verify: int = 1, model: str = "", debug: int = 0) -> dict:
        """Run claim extraction (and the check) over the cached transcript between start and end
        seconds with the prompts as they are on disk right now. Nothing is cached or shown in the
        page: this is for trying out prompt changes without restarting the server."""
        import importlib
        from concurrent.futures import ThreadPoolExecutor

        from . import claims as module

        if booklet is None:
            return {"error": "started without --booklet"}
        if model:   # e.g. swiss-ai/Apertus-v1.5-70B; applies to the whole server from now on
            import os
            os.environ["LLM_NAME"] = model
        import time
        began = time.perf_counter()
        module = importlib.reload(module)
        verifier = module.BookletVerifier(booklet["path"], booklet["vote"], booklet["pages"])
        segments = subtitles.snapshot()

        def run(index: int) -> dict:
            segment = segments[index]
            out = {"start": segment["start"], "end": segment["end"], "text": segment["text"], "claims": []}
            if len(segment["text"]) < module.MIN_SEGMENT_CHARS:
                return out
            before = " ".join(s["text"] for s in segments[max(0, index - 2):index])
            after = segments[index + 1]["text"] if index + 1 < len(segments) else ""
            try:
                if debug:   # every statement with its type and why it was not checked
                    out["statements"] = module.extract_statements(booklet["vote"], before, segment["text"], after)
                    return out
                for claim in module.extract_claims(booklet["vote"], before, segment["text"], after):
                    if verify:
                        try:
                            claim.update(verifier.verify(claim["text"]))
                        except Exception as exc:
                            claim["error"] = f"{exc.__class__.__name__}: {exc}"
                    out["claims"].append(claim)
            except Exception as exc:
                out["error"] = f"{exc.__class__.__name__}: {exc}"
            return out

        wanted = [i for i, s in enumerate(segments) if start <= s["start"] < end]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, wanted))
        from ..nli import remote_model
        return {"version": module.CLAIMS_VERSION, "model": remote_model(),
                "seconds": round(time.perf_counter() - began, 1), "segments": results}

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8").replace("{{VIDEO_NAME}}", video.name)

    @app.get("/video")
    def video_file() -> FileResponse:  # FileResponse answers Range requests, so seeking works
        return FileResponse(video, media_type="video/mp4")

    @app.get("/api/sync")
    def sync(t: float = 0.0, playing: int = 0, since: int = 0, rev: int = 0) -> dict:
        data = subtitles.sync(t, bool(playing), since)
        data["claims_enabled"] = checker is not None
        if checker is not None:
            if playing:
                checker.follow(t, subtitles.snapshot(), data["interim"])
            data.update(checker.changes(rev))
        return data

    stop = threading.Event()

    def watchdog() -> None:
        while not stop.wait(2.0):
            subtitles.stop_if_idle()

    threading.Thread(target=watchdog, daemon=True).start()

    @app.on_event("shutdown")
    def shutdown() -> None:
        stop.set()
        subtitles.shutdown()

    return app
