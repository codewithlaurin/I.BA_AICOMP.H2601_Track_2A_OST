"""Prepare whole arenas without watching them: collect the web sources, transcribe what is not
transcribed yet (faster than real time, several stretches in parallel), then extract and check all
claims, against the booklet and, where the booklet is silent, against the web sources.

    uv run --with azure-cognitiveservices-speech --with imageio-ffmpeg --with trafilatura \
        python -m claimcheck.live.batch kmg ahv          # arena ids from data/demo/arenas.json
    ... python -m claimcheck.live.batch --all

Writes the same caches as the live demo, so the demo afterwards plays everything from the cache, and
a readable list of all claims per arena (<video>.claims.txt). Running it again only does what is
missing; --refresh-web fetches the web sources again, --no-web skips them.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .__main__ import ROOT, load_env
from .arenas import DEFAULT_CONFIG, load_arenas, video_duration
from .runtime import build_runtime

CHUNK = 600.0   # seconds of video per transcription job
RETRIES = 3     # attempts per stretch after the first one
RETRY_WAIT = 10 # seconds, growing with each attempt
LABELS = {"entailment": "BELEGT", "contradiction": "WIDERSPRUCH", "neutral": "nicht belegt"}


def transcribe_stretch(video: Path, start: float, duration: float, language: str) -> list[dict]:
    from .transcribe import AzureTranscriber

    finals, problems, done = [], [], threading.Event()

    def emit(event: dict) -> None:
        if event["type"] == "final":
            finals.append(event)
        elif event["type"] == "error":
            problems.append(event["message"])
            done.set()
        elif event["type"] == "end":
            done.set()

    transcriber = AzureTranscriber(video, start, emit, language, duration=duration, realtime=False)
    try:
        done.wait()
    finally:
        transcriber.stop()
    if problems:
        raise RuntimeError(problems[0])
    return finals


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def report(arena, checker) -> Path:
    claims = sorted(checker.claims.values(), key=lambda c: c["start"])
    lines, counts = [], {}
    for claim in claims:
        label = LABELS.get(claim.get("label"), claim["status"])
        web = claim.get("web")
        if claim.get("label") == "neutral" and web and web["label"] != "neutral":
            label = f"{LABELS[web['label']]} (Web)"
        counts[label] = counts.get(label, 0) + 1
        lines.append(f"[{clock(claim['start'])}] {label}: {claim['text']}")
        if claim["status"] == "error" and claim.get("note"):
            lines.append(f"      Fehler: {claim['note']}")
        for item in claim.get("evidence") or []:
            lines.append(f"      Büchlein S. {item['page']}: {' '.join(item['text'].split())}")
        for item in (web or {}).get("evidence") or []:
            lines.append(f"      {item['source']} ({item['url']}): {' '.join(item['text'].split())}")
    summary = ", ".join(f"{n} {label}" for label, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    path = arena.video.with_suffix(".claims.txt")
    path.write_text(f"{arena.title}: {len(claims)} Claims ({summary})\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {len(claims)} claims: {summary}\n  -> {path}", flush=True)
    return path


def prepare(arena, args) -> None:
    print(f"\n=== {arena.id}: {arena.title}", flush=True)

    # ---- 1. web sources
    if not args.no_web and arena.sources and (args.refresh_web or not arena.web_cache.exists()):
        from .research import collect

        print(f"collecting web sources ({len(arena.sources)} seeds) ...", flush=True)
        pages = collect(arena.sources, arena.keywords, arena.web_cache)
        print(f"  {pages} pages -> {arena.web_cache.name}", flush=True)

    runtime = build_runtime(arena, args.language, offline=False)
    subtitles, checker = runtime.subtitles, runtime.checker

    # ---- 2. transcription
    duration = video_duration(arena.video)
    jobs = []
    for a, b in subtitles.gaps(duration):
        while a < b:
            jobs.append((a, min(a + CHUNK, b)))
            a += CHUNK
    print(f"{clock(duration)} video, {clock(sum(b - a for a, b in jobs))} to transcribe in {len(jobs)} parts",
          flush=True)
    began, finished = time.perf_counter(), 0

    failed = []

    def transcribe(job):
        nonlocal finished
        a, b = job
        for attempt in range(RETRIES + 1):
            try:
                finals = transcribe_stretch(arena.video, a, b - a, args.language)
                break
            except Exception as exc:   # network hiccups (DNS, Wi-Fi): wait and try the stretch again
                if attempt == RETRIES:
                    failed.append(job)
                    print(f"  FAILED {clock(a)}-{clock(b)}: {str(exc).splitlines()[0]}", flush=True)
                    return
                print(f"  retry {clock(a)}-{clock(b)} in {RETRY_WAIT * (attempt + 1)} s: {str(exc).splitlines()[0][:120]}",
                      flush=True)
                time.sleep(RETRY_WAIT * (attempt + 1))
        subtitles.add_transcribed(a, b, finals)
        finished += 1
        print(f"  transcribed {clock(a)}-{clock(b)} ({len(finals)} segments) "
              f"[{finished}/{len(jobs)}, {time.perf_counter() - began:.0f} s]", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        list(pool.map(transcribe, jobs))
    if failed:   # what is transcribed is kept; the claims below are checked for it
        print(f"  {len(failed)} of {len(jobs)} parts not transcribed: run the batch again later", flush=True)

    # ---- 3. claims
    if checker is None:
        print("no booklet configured: claims skipped", flush=True)
        return
    checker._submit = lambda function, *a: function(*a)   # each segment's checks in the same worker
    checker.web_on = not args.no_web and runtime.web_pages > 0
    segments = subtitles.snapshot()
    print(f"checking claims in {len(segments)} segments"
          f"{' (web research on)' if checker.web_on else ''} ...", flush=True)
    began = time.perf_counter()

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total} [{time.perf_counter() - began:.0f} s]", flush=True)

    checker.process_all(segments, progress=progress)
    report(arena, checker)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arenas", nargs="*", help="arena ids from the config")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--language", default="de-CH")
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-70B")
    parser.add_argument("--parallel", type=int, default=3, help="transcription jobs at the same time")
    parser.add_argument("--no-web", action="store_true", help="no web research")
    parser.add_argument("--refresh-web", action="store_true", help="fetch the web sources again")
    args = parser.parse_args()
    load_env(ROOT / ".env")
    os.environ["LLM_NAME"] = args.model
    arenas = load_arenas(args.config)
    chosen = arenas if args.all else [a for a in arenas if a.id in args.arenas]
    unknown = set(args.arenas) - {a.id for a in arenas}
    if unknown or not chosen:
        parser.error(f"choose arenas by id ({', '.join(a.id for a in arenas)}) or --all")
    for arena in chosen:
        if not arena.video.exists():
            print(f"\n=== {arena.id}: video not found: {arena.video}", flush=True)
            continue
        prepare(arena, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
