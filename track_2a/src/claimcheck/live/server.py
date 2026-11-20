"""Web server of the live demo: the start screen with all arenas and, per arena, the player page,
the video file and the sync endpoint (subtitles + claims)."""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from .arenas import make_poster
from .runtime import Runtime

STATIC = Path(__file__).parent / "static"


def create_app(runtimes: list[Runtime]) -> FastAPI:
    app = FastAPI(title="claimcheck live")
    by_id = {r.arena.id: r for r in runtimes}

    def get(arena_id: str) -> Runtime:
        if arena_id not in by_id:
            raise HTTPException(404, f"unknown arena {arena_id!r}")
        return by_id[arena_id]

    # ---------------------------------------------------------------- start screen

    @app.get("/", response_class=HTMLResponse)
    def start() -> str:
        return (STATIC / "start.html").read_text(encoding="utf-8")

    @app.get("/api/arenas")
    def arenas() -> list[dict]:
        return [r.summary() for r in runtimes]

    @app.get("/arena/{arena_id}/poster.jpg")
    def poster(arena_id: str) -> FileResponse:
        try:
            return FileResponse(make_poster(get(arena_id).arena), media_type="image/jpeg")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(404, f"no poster: {exc}")

    # ---------------------------------------------------------------- player

    @app.get("/arena/{arena_id}")
    def player_redirect(arena_id: str) -> RedirectResponse:
        return RedirectResponse(f"/arena/{arena_id}/")   # the page uses relative URLs

    @app.get("/arena/{arena_id}/", response_class=HTMLResponse)
    def player(arena_id: str) -> str:
        arena = get(arena_id).arena
        return ((STATIC / "index.html").read_text(encoding="utf-8")
                .replace("{{TITLE}}", arena.title).replace("{{VIDEO_NAME}}", arena.video.name))

    @app.get("/arena/{arena_id}/video")
    def video_file(arena_id: str) -> FileResponse:  # FileResponse answers Range requests, so seeking works
        return FileResponse(get(arena_id).arena.video, media_type="video/mp4")

    @app.get("/arena/{arena_id}/api/sync")
    def sync(arena_id: str, t: float = 0.0, playing: int = 0, since: int = 0, rev: int = 0, web: int = 0) -> dict:
        runtime = get(arena_id)
        data = runtime.subtitles.sync(t, bool(playing), since)
        checker = runtime.checker
        data["claims_enabled"] = checker is not None
        data["web_pages"] = runtime.web_pages
        if checker is not None:
            checker.web_on = bool(web) and runtime.web_pages > 0
            if playing:
                checker.follow(t, runtime.subtitles.snapshot(), data["interim"])
            elif checker.web_on:
                checker.follow(-1e9, [], None)   # research claims already shown, also while paused
            data.update(checker.changes(rev))
        return data

    @app.get("/arena/{arena_id}/api/dev/claims")
    def dev_claims(arena_id: str, start: float = 0.0, end: float = 1e9, verify: int = 1, debug: int = 0) -> dict:
        """Run claim extraction (and the check) over the cached transcript between start and end
        seconds with the prompts as they are on disk right now. Nothing is cached or shown in the
        page: this is for trying out prompt changes without restarting the server."""
        import importlib
        import time
        from concurrent.futures import ThreadPoolExecutor

        from . import claims as module

        runtime = get(arena_id)
        if runtime.dev is None:
            return {"error": "no booklet, or offline"}
        began = time.perf_counter()
        module = importlib.reload(module)
        dev = runtime.dev
        verifier = module.BookletVerifier(dev["booklet"], dev["vote"], dev["pages"])
        segments = runtime.subtitles.snapshot()

        def run(index: int) -> dict:
            segment = segments[index]
            out = {"start": segment["start"], "end": segment["end"], "text": segment["text"], "claims": []}
            if len(segment["text"]) < module.MIN_SEGMENT_CHARS:
                return out
            before = " ".join(s["text"] for s in segments[max(0, index - 2):index])
            after = segments[index + 1]["text"] if index + 1 < len(segments) else ""
            try:
                if debug:   # every statement with its type and why it was not checked
                    out["statements"] = module.extract_statements(dev["vote"], before, segment["text"], after)
                    return out
                for claim in module.extract_claims(dev["vote"], before, segment["text"], after):
                    if verify:
                        try:
                            claim.update(verifier.verify(claim["text"]))
                            if claim["label"] == "neutral" and dev.get("web") is not None:
                                claim["web"] = dev["web"].verify(claim["text"])
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

    # ---------------------------------------------------------------- housekeeping

    stop = threading.Event()

    def watchdog() -> None:
        while not stop.wait(2.0):
            for runtime in runtimes:
                runtime.subtitles.stop_if_idle()

    threading.Thread(target=watchdog, daemon=True).start()

    @app.on_event("shutdown")
    def shutdown() -> None:
        stop.set()
        for runtime in runtimes:
            runtime.subtitles.shutdown()

    return app
