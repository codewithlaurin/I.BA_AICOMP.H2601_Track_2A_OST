"""Live demo: play a debate video, show live subtitles under it and checked claims beside it.

    uv run --with fastapi --with uvicorn --with azure-cognitiveservices-speech --with imageio-ffmpeg \
        python -m claimcheck.live data/demo/arena_kriegsmaterialgesetz_2026-11-06.mp4 \
        --booklet data/booklets/pdfs/BAR_Erlaeuterungen_Nov26_DE_v10.pdf --pages 10-11,50-61 \
        --vote "Änderung des Kriegsmaterialgesetzes"

Opens http://localhost:8766. While the video plays, its audio goes to Azure Speech (de-CH) and the
text appears under the picture; Apertus picks checkable claims out of it and checks each one against
the booklet pages. Transcript and claims are cached next to the video (<video>.subtitles.jsonl,
<video>.claims.jsonl), so a stretch that was played once costs nothing the second time.
--offline: caches only, no speech service and no model calls. Without --booklet: subtitles only.

Prompt development: with the server running, GET /api/dev/claims?start=0&end=600 runs extraction and
check over the cached transcript with the prompts as they are on disk (no restart, nothing cached).
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--language", default="de-CH")
    parser.add_argument("--booklet", type=Path, help="voting booklet PDF; enables the claim check")
    parser.add_argument("--pages", help="pages of the proposal in the booklet, e.g. 10-11,50-61")
    parser.add_argument("--vote", default="", help="title of the proposal")
    parser.add_argument("--model", default="swiss-ai/Apertus-v1.5-70B",
                        help="Apertus model for claim extraction and checking (the 8B misjudges spoken claims)")
    parser.add_argument("--offline", action="store_true", help="caches only: no speech service, no model calls")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    video = args.video.resolve()
    if not video.exists():
        parser.error(f"video not found: {video}")
    load_env(ROOT / ".env")
    os.environ["LLM_NAME"] = args.model   # over .env: the demo was tuned with the 70B model

    import uvicorn

    from .cachefile import video_fingerprint
    from .claims import CLAIMS_VERSION, BookletVerifier, ClaimChecker, extract_claims, parse_pages
    from .server import create_app
    from .subtitles import Subtitles

    fingerprint = video_fingerprint(video)

    start_transcriber = None
    if not args.offline:
        from .transcribe import AzureTranscriber

        def start_transcriber(start, emit):
            return AzureTranscriber(video, start, emit, args.language)

    subtitles = Subtitles(video.with_suffix(".subtitles.jsonl"), start_transcriber, fingerprint=fingerprint)

    checker = None
    if args.booklet:
        if not args.booklet.exists():
            parser.error(f"booklet not found: {args.booklet}")
        extract = verify = None
        if not args.offline:
            verifier = BookletVerifier(args.booklet, args.vote, parse_pages(args.pages))
            print(f"booklet: {args.booklet.name}, {len(verifier.chunks(''))} pages per check"
                  if args.pages else f"booklet: {args.booklet.name}, retrieval over all pages", flush=True)
            verify = verifier.verify

            def extract(before, segment, after):
                return extract_claims(args.vote, before, segment, after)

        checker = ClaimChecker(video.with_suffix(".claims.jsonl"), extract, verify,
                               fingerprint=f"{fingerprint}-{CLAIMS_VERSION}")

    url = f"http://localhost:{args.port}"
    print(f"{url}   ({video.name}, {len(subtitles.finals)} cached segments, "
          f"{len(checker.claims) if checker else 'no'} cached claims, "
          f"{'cache only' if args.offline else 'live: Azure ' + args.language})", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    booklet = ({"path": args.booklet, "vote": args.vote, "pages": parse_pages(args.pages)}
               if args.booklet and not args.offline else None)
    uvicorn.run(create_app(video, subtitles, checker, booklet), host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
