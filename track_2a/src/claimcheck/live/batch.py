"""Process a whole debate video in one go, without watching it: transcribe what is not transcribed
yet (faster than real time, several stretches in parallel), then extract and check all claims.

    uv run --with azure-cognitiveservices-speech --with imageio-ffmpeg --with rapidfuzz \
        python -m claimcheck.live.batch data/demo/arena_nachhaltigkeitsinitiative_2026-05-13.mp4 \
        --booklet data/booklets/pdfs/2026-06-14_....pdf --pages 4-5,8-22 \
        --vote "Volksinitiative «Keine 10-Millionen-Schweiz! (Nachhaltigkeitsinitiative)»"

Writes the same caches as the live demo (<video>.subtitles.jsonl, <video>.claims.jsonl), so the demo
afterwards plays everything from the cache, and a readable list of all claims (<video>.claims.txt).
Running it again only does what is missing.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .__main__ import ROOT, load_env

CHUNK = 600.0   # seconds of video per transcription job
LABELS = {"entailment": "BELEGT", "contradiction": "WIDERSPRUCH", "neutral": "nicht im Büchlein"}


def video_duration(video: Path) -> float:
    import imageio_ffmpeg

    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(video)], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stderr
    match = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if not match:
        raise RuntimeError(f"cannot read the duration of {video}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--booklet", type=Path, required=True)
    parser.add_argument("--pages", help="pages of the proposal in the booklet, e.g. 4-5,8-22")
    parser.add_argument("--vote", default="")
    parser.add_argument("--language", default="de-CH")
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-70B")
    parser.add_argument("--parallel", type=int, default=3, help="transcription jobs at the same time")
    args = parser.parse_args()
    video = args.video.resolve()
    if not video.exists() or not args.booklet.exists():
        parser.error("video or booklet not found")
    load_env(ROOT / ".env")
    os.environ["LLM_NAME"] = args.model

    from .cachefile import video_fingerprint
    from .claims import CLAIMS_VERSION, BookletVerifier, ClaimChecker, extract_claims, parse_pages
    from .subtitles import Subtitles

    fingerprint = video_fingerprint(video)
    subtitles = Subtitles(video.with_suffix(".subtitles.jsonl"), None, fingerprint=fingerprint)

    # ---- 1. transcription
    duration = video_duration(video)
    jobs = []
    for a, b in subtitles.gaps(duration):
        while a < b:
            jobs.append((a, min(a + CHUNK, b)))
            a += CHUNK
    todo = sum(b - a for a, b in jobs)
    print(f"{video.name}: {clock(duration)}, {len(subtitles.finals)} segments cached, "
          f"{clock(todo)} to transcribe in {len(jobs)} parts", flush=True)
    began, finished = time.perf_counter(), 0

    def transcribe(job):
        nonlocal finished
        a, b = job
        finals = transcribe_stretch(video, a, b - a, args.language)
        subtitles.add_transcribed(a, b, finals)
        finished += 1
        print(f"  transcribed {clock(a)}-{clock(b)} ({len(finals)} segments) "
              f"[{finished}/{len(jobs)}, {time.perf_counter() - began:.0f} s]", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        for future in [pool.submit(transcribe, job) for job in jobs]:
            future.result()

    # ---- 2. claims
    verifier = BookletVerifier(args.booklet, args.vote, parse_pages(args.pages))
    checker = ClaimChecker(video.with_suffix(".claims.jsonl"),
                           lambda before, segment, after: extract_claims(args.vote, before, segment, after),
                           verifier.verify, fingerprint=f"{fingerprint}-{CLAIMS_VERSION}",
                           submit=lambda function, *a: function(*a))
    segments = subtitles.snapshot()
    print(f"checking claims in {len(segments)} segments with {args.model} ...", flush=True)
    began = time.perf_counter()

    def progress(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total} [{time.perf_counter() - began:.0f} s]", flush=True)

    checker.process_all(segments, progress=progress)

    # ---- 3. report
    claims = sorted(checker.claims.values(), key=lambda c: c["start"])
    lines, counts = [], {}
    for claim in claims:
        label = LABELS.get(claim.get("label"), claim["status"])
        counts[label] = counts.get(label, 0) + 1
        lines.append(f"[{clock(claim['start'])}] {label}: {claim['text']}")
        for item in claim.get("evidence") or []:
            lines.append(f"      S. {item['page']}: {' '.join(item['text'].split())}")
    summary = ", ".join(f"{n} {label}" for label, n in sorted(counts.items(), key=lambda kv: -kv[1]))
    report = video.with_suffix(".claims.txt")
    report.write_text(f"{video.name}: {len(claims)} Claims ({summary})\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{len(claims)} claims: {summary}\n-> {report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
