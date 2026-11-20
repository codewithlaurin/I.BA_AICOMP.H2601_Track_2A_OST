"""What the demo keeps per arena: subtitles, claim checker, verifiers. Used by the web server and
by the batch mode (which runs without FastAPI)."""

from __future__ import annotations

from .arenas import Arena, video_duration
from .claims import ClaimChecker
from .subtitles import Subtitles


class Runtime:
    """Everything the server keeps for one arena. checker is None without a booklet; dev holds what
    /api/dev/claims needs (None in offline mode)."""

    def __init__(self, arena: Arena, subtitles: Subtitles, checker: ClaimChecker | None,
                 web_pages: int = 0, dev: dict | None = None):
        self.arena, self.subtitles, self.checker, self.web_pages, self.dev = arena, subtitles, checker, web_pages, dev

    def summary(self) -> dict:
        arena = self.arena
        try:
            duration = video_duration(arena.video)
        except Exception:
            duration = None
        finals = self.subtitles.snapshot()
        last = max((s["end"] for s in finals), default=0.0)
        covered = sum(max(0.0, min(b, duration or last) - a) for a, b in self.subtitles.ranges)
        return {"id": arena.id, "title": arena.title, "vote": arena.vote, "arena_date": arena.arena_date,
                "vote_date": arena.vote_date, "duration": duration, "transcribed": round(covered),
                "claims": self.checker.counts() if self.checker else None,
                "claims_web": self.checker.counts(web=True) if self.checker else None,
                "web_pages": self.web_pages, "video_found": arena.video.exists()}



def build_runtime(arena, language: str, offline: bool, log=print):
    """Subtitles, claim checker and verifiers for one arena (see server.Runtime)."""
    from .cachefile import video_fingerprint
    from .claims import CLAIMS_VERSION, BookletVerifier, ClaimChecker, extract_claims, parse_pages
    from .research import WebIndex, WebVerifier, load_passages
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
