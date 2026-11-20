"""Live demo: pick a debate on the start screen, play it with live subtitles, and see its claims
checked against the voting booklet (and, with the toggle, against collected web sources).

    uv run --with fastapi --with uvicorn --with azure-cognitiveservices-speech --with imageio-ffmpeg \
        python -m claimcheck.live

Reads the arenas from data/demo/arenas.json (see arenas.py) and opens http://localhost:8766.
While a video plays, its audio goes to Azure Speech (de-CH) and the text appears under the picture;
Apertus picks checkable claims out of it and checks each one against the booklet pages. Transcript,
claims and web pages are cached next to each video, so a stretch that was played once costs nothing
the second time. Prepare a whole arena without watching it: python -m claimcheck.live.batch <id>.

--offline: caches only, no speech service and no model calls.
A single video without a config file still works:
    python -m claimcheck.live VIDEO --booklet PDF --pages 10-11,50-61 --vote "..."

Prompt development: GET /arena/<id>/api/dev/claims?start=0&end=600 runs extraction and check over
the cached transcript with the prompts as they are on disk (no restart, nothing cached).
"""

import argparse
import os
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # track_2a/


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def build_runtime(arena, language: str, offline: bool, log=print):
    """Subtitles, claim checker and verifiers for one arena (see server.Runtime)."""
    from .cachefile import video_fingerprint
    from .claims import CLAIMS_VERSION, BookletVerifier, ClaimChecker, extract_claims, parse_pages
    from .research import WebIndex, WebVerifier, load_passages
    from .server import Runtime
    from .subtitles import Subtitles

    video = arena.video
    fingerprint = video_fingerprint(video)
    start_transcriber = None
    if not offline:
        from .transcribe import AzureTranscriber

        def start_transcriber(start, emit):
            return AzureTranscriber(video, start, emit, language)

    subtitles = Subtitles(arena.subtitles_cache, start_transcriber, fingerprint=fingerprint)
    passages = load_passages(arena.web_cache)
    web_pages = len({p["url"] for p in passages})
    checker, dev = None, None
    if arena.booklet:
        if not arena.booklet.exists():
            raise FileNotFoundError(f"{arena.id}: booklet not found: {arena.booklet}")
        extract = verify = web_verify = None
        if not offline:
            verify = BookletVerifier(arena.booklet, arena.vote, parse_pages(arena.pages)).verify
            web = WebVerifier(WebIndex(passages), arena.vote) if passages else None
            web_verify = web.verify if web else None

            def extract(before, segment, after):
                return extract_claims(arena.vote, before, segment, after)

            dev = {"booklet": arena.booklet, "vote": arena.vote, "pages": parse_pages(arena.pages), "web": web}
        checker = ClaimChecker(arena.claims_cache, extract, verify, fingerprint=f"{fingerprint}-{CLAIMS_VERSION}",
                               web_verify=web_verify)
    log(f"  {arena.id:8} {arena.title}: {len(subtitles.finals)} segments, "
        f"{len(checker.claims) if checker else 'no'} claims, {web_pages} web pages cached")
    return Runtime(arena, subtitles, checker, web_pages, dev)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, nargs="?", help="a single video instead of the arenas config")
    parser.add_argument("--config", type=Path, default=None, help="arenas config (default data/demo/arenas.json)")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--language", default="de-CH")
    parser.add_argument("--booklet", type=Path, help="single video: voting booklet PDF")
    parser.add_argument("--pages", help="single video: pages of the proposal in the booklet, e.g. 10-11,50-61")
    parser.add_argument("--vote", default="", help="single video: title of the proposal")
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-70B",
                        help="Apertus model for claim extraction and checking (the 8B misjudges spoken claims)")
    parser.add_argument("--offline", action="store_true", help="caches only: no speech service, no model calls")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    load_env(ROOT / ".env")
    os.environ["LLM_NAME"] = args.model   # over .env: the demo was tuned with the 70B model

    import uvicorn

    from .arenas import DEFAULT_CONFIG, Arena, load_arenas
    from .runtime import build_runtime
    from .server import create_app

    if args.video:
        video = args.video.resolve()
        arenas = [Arena(id="video", title=args.vote or video.stem, vote=args.vote, video=video,
                        booklet=args.booklet.resolve() if args.booklet else None, pages=args.pages)]
    else:
        config = args.config or DEFAULT_CONFIG
        if not config.exists():
            parser.error(f"no arenas config at {config}: create it (see arenas.example.json) or pass a VIDEO")
        arenas = load_arenas(config)
    missing = [a.id for a in arenas if not a.video.exists()]
    if missing:
        print(f"videos not found, left out: {', '.join(missing)}", flush=True)
    print(f"{'cache only' if args.offline else 'live: Azure ' + args.language + ', ' + args.model}", flush=True)
    runtimes = [build_runtime(a, args.language, args.offline) for a in arenas if a.video.exists()]
    url = f"http://localhost:{args.port}"
    if len(runtimes) == 1 and args.video:
        url += f"/arena/{runtimes[0].arena.id}/"
    print(url, flush=True)
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(create_app(runtimes), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
