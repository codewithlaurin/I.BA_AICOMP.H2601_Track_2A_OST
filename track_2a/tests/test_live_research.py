"""Web research of the live demo: whitelist, collection, passages, and the checker's web step (no network)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimcheck.live import research  # noqa: E402
from claimcheck.live.claims import ClaimChecker  # noqa: E402

LONG = "Jedes Jahr hat das SECO ungefähr 2500 Gesuche um Export, und einmal kam das vor."


class WhitelistTest(unittest.TestCase):
    def test_allowed_and_source_names(self):
        self.assertTrue(research.allowed("https://www.bfs.admin.ch/bfs/de/home.html"))
        self.assertTrue(research.allowed("https://www.srf.ch/news/x"))
        self.assertFalse(research.allowed("https://www.admin.ch.evil.com/x"))
        self.assertFalse(research.allowed("https://www.nzz.ch/x"))
        self.assertFalse(research.allowed("ftp://admin.ch/x"))
        self.assertEqual(research.source_name("https://www.bfs.admin.ch/x"), "BFS")
        self.assertEqual(research.source_name("https://www.bafu.admin.ch/x"), "BAFU")
        self.assertEqual(research.source_name("https://www.admin.ch/de/x"), "Bundesrat")
        self.assertEqual(research.source_name("https://www.srf.ch/news/x"), "SRF")

    def test_speeches_and_interviews_are_left_out(self):
        page = {"url": "https://www.ejpd.admin.ch/de/newnsb/x", "title": "Medienkonferenz", "text": "Fakten. " * 100}
        self.assertFalse(research.is_campaign(page))
        self.assertTrue(research.is_campaign({**page, "text": page["text"] + " Es gilt das gesprochene Wort."}))
        self.assertTrue(research.is_campaign({**page, "title": "«Wir erleben den Brexit-Moment der Schweiz»"}))
        self.assertFalse(research.is_campaign({**page, "url": "https://www.srf.ch/news/abstimmungs-arena-x",
                                               "title": "«Abstimmungs-Arena» - Soll das Gesetz gelockert werden?"}))
        self.assertTrue(research.is_campaign({**page, "title": "«Abstimmungs-Arena» zur Initiative «Keine 10-Millionen-Schweiz»"}))
        self.assertTrue(research.is_campaign({**page, "title": "Bundesrat Beat Jans im Abstimmungs-Interview"}))
        self.assertTrue(research.is_campaign({**page, "text": "Ich finde, ich meine. " + page["text"].replace("Fakten.", "Fakten, ich weiss.")}))

    def test_web_sources_only_confirm(self):
        from unittest.mock import patch

        index = research.WebIndex([{"url": "https://www.srf.ch/news/x", "source": "SRF", "title": "t", "text": "B"}])
        index.search = lambda query, k: index.passages
        verdict = {"label": "contradiction", "evidence": [{"text": "B"}], "note": None, "why": {"how": "second look"}}
        with patch("claimcheck.live.claims.judge", return_value=verdict):
            result = research.WebVerifier(index, "vote").verify("A")
        self.assertEqual((result["label"], result["evidence"]), ("neutral", []))
        with patch("claimcheck.live.claims.judge", return_value={**verdict, "label": "entailment"}):
            self.assertEqual(research.WebVerifier(index, "vote").verify("A")["label"], "entailment")

    def test_web_prompt_talks_about_sources(self):
        from claimcheck.live.claims import VERIFY_PROMPT

        prompt = research.web_prompt(VERIFY_PROMPT)
        self.assertIn("Steps:", prompt)
        self.assertIn("SRF", prompt)
        self.assertNotIn("booklet sentence", prompt)


class CollectTest(unittest.TestCase):
    def test_seeds_and_linked_pages_about_the_topic_are_kept(self):
        topic = "Das Kriegsmaterialgesetz regelt den Export. " * 10
        pages = {
            "https://www.admin.ch/de/seed": {"text": topic, "links": [
                "https://www.srf.ch/news/a", "https://www.srf.ch/news/b", "https://www.nzz.ch/news/c",
                "https://www.srf.ch/play/tv/video", "https://www.admin.ch/de", "https://swissvotes.ch/attachments/f00",
                "https://www.srf.ch/news/copy", "https://www.srf.ch/news/menu"]},
            "https://www.srf.ch/news/a": {"text": topic + " Neu.", "links": ["https://www.srf.ch/news/d"]},
            "https://www.srf.ch/news/b": {"text": "Etwas ganz anderes. " * 30, "links": []},
            "https://www.srf.ch/news/menu": {"text": "Kriegsmaterialgesetz. " + "Sport und Wetter. " * 30, "links": []},
            "https://www.srf.ch/news/copy": {"text": topic + " Neu.", "links": []},
            "https://www.srf.ch/news/d": {"text": topic, "links": []},
            "https://www.admin.ch/de": {"text": topic, "links": []},
            "https://swissvotes.ch/attachments/f00": {"text": topic, "links": [], "pdf": True},
        }
        fetched = []

        def fake_read(url):
            fetched.append(url)
            page = pages[url]
            return {"url": url, "source": research.source_name(url), "title": url.rsplit("/", 1)[-1], "text": page["text"],
                    "links": list(page["links"]), "pdf": page.get("pdf", False)}

        original, research.read_page = research.read_page, fake_read
        sleep, research.time.sleep = research.time.sleep, lambda s: None
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "x.web.jsonl"
                count = research.collect(["https://www.admin.ch/de/seed"], ["kriegsmaterial"], out, log=lambda m: None)
                saved = [json.loads(line)["url"] for line in out.read_text(encoding="utf-8").splitlines()]
                passages = research.load_passages(out)
        finally:
            research.read_page, research.time.sleep = original, sleep
        # b: other topic; menu: mentions it once; copy: same text as a; d: two levels deep
        self.assertEqual(saved, ["https://www.admin.ch/de/seed", "https://www.srf.ch/news/a"])
        self.assertEqual(count, 2)
        # never fetched: other domain, media, start page, two levels deep (the PDF is fetched, then left out)
        for skipped in ("https://www.nzz.ch/news/c", "https://www.srf.ch/play/tv/video", "https://www.admin.ch/de",
                        "https://www.srf.ch/news/d"):
            self.assertNotIn(skipped, fetched)
        self.assertTrue(all(len(p["text"]) <= research.PASSAGE_CHARS + 200 for p in passages))
        self.assertEqual({p["source"] for p in passages}, {"Bundesrat", "SRF"})


class CheckerWebTest(unittest.TestCase):
    def test_web_research_only_for_neutral_claims_and_only_when_switched_on(self):
        texts = iter(["Die Nettozuwanderung betrug 2025 rund 70 000 Personen.", "Das Gesetz gilt für 36 Partnerstaaten."])
        labels = {"Die Nettozuwanderung betrug 2025 rund 70 000 Personen.": "neutral",
                  "Das Gesetz gilt für 36 Partnerstaaten.": "entailment"}
        web_calls = []

        def web_verify(claim):
            web_calls.append(claim)
            return {"label": "entailment", "evidence": [{"url": "https://www.bfs.admin.ch/x", "source": "BFS",
                                                         "title": "Migration", "text": "70 432"}], "note": None}

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "v.claims.jsonl"

            def make():
                return ClaimChecker(cache, lambda b, s, a: [{"text": next(texts), "quote": ""}],
                                    lambda claim: {"label": labels[claim], "evidence": [], "note": None},
                                    submit=lambda function, *args: function(*args), web_verify=web_verify)

            checker = make()
            checker.follow(30, [{"start": 6, "end": 20, "text": LONG}])
            self.assertEqual(web_calls, [])                             # toggle off: no research
            checker.web_on = True
            checker.follow(60, [{"start": 6, "end": 20, "text": LONG}, {"start": 40, "end": 50, "text": LONG + " Zwei."}])
            self.assertEqual(web_calls, ["Die Nettozuwanderung betrug 2025 rund 70 000 Personen."])
            self.assertEqual(checker.counts(), {"entailment": 1, "contradiction": 0, "neutral": 1})
            self.assertEqual(checker.counts(web=True), {"entailment": 2, "contradiction": 0, "neutral": 0})

            again = make()                                              # cached: no new calls
            again.web_on = True
            again.follow(60, [])
            self.assertEqual(len(web_calls), 1)
            self.assertEqual(again.counts(web=True)["entailment"], 2)


if __name__ == "__main__":
    unittest.main()
