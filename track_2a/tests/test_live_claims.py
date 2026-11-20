"""Claim extraction parsing and the claim checker of the live demo (fake model, no network)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimcheck.live.cachefile import load_cache, video_fingerprint  # noqa: E402
from claimcheck.live.claims import ClaimChecker, figures_differ, grounded_in, numbers, scale_slip, page_sections, parse_claims, parse_pages, quoted_from, settle, usable  # noqa: E402

LONG = "Jedes Jahr hat das SECO ungefähr 2500 Gesuche um Export, und einmal kam das vor."


def segment(start, end, text=LONG):
    return {"start": start, "end": end, "text": text}


class ParseTest(unittest.TestCase):
    def test_parse_claims_keeps_only_checkable_statements(self):
        text = ('Sure:\n```json\n{"statements": [{"type": "number", "claim": "Das  SECO erhält\\n2500 Gesuche.", "quote": "2500 fragen"},'
                ' {"type": "OPINION", "claim": "Das Gesetz ist gefährlich.", "quote": "gefährlich"},'
                ' {"type": "LAW", "claim": "Das Abstimmungsbüchlein nennt drei Bestimmungen.", "quote": "3 Bestimmungen"}, {"x": 1}]}\n```')
        self.assertEqual(parse_claims(text), [{"text": "Das SECO erhält 2500 Gesuche.", "quote": "2500 fragen"}])
        self.assertEqual(parse_claims("no json here"), [])
        self.assertEqual(parse_claims('{"statements": []}'), [])

    def test_quote_must_come_from_the_segment(self):
        said = "Nein, stimmt nicht, es gibt 3 Bestimmungen, eine sieht vor, dass diese 36 Länder beliefert werden."
        self.assertTrue(quoted_from("eine sieht vor, dass diese 36 Länder beliefert werden", said))
        self.assertFalse(quoted_from("Die Ukraine ist nicht eingeschlossen in diese Wiederausfuhr", said))
        self.assertFalse(quoted_from("es gibt", said))

    def test_claim_must_be_what_was_said(self):
        said = "Fragen kann oder vorschreiben kann, dass es eine Nichtwiederausfuhrerklärung braucht."
        self.assertFalse(grounded_in("Die Lockerungen ermöglichen, dass ein Land trotz Verbot beliefert werden kann.", said))
        self.assertTrue(grounded_in("Man kann eine Nichtwiederausfuhrerklärung vorschreiben.", said))
        question = {"text": "Die Lockerungen will man im neuen Gesetz machen.", "quote": "Was kritisieren Sie? Ja genau solche Lockerungen will man machen."}
        self.assertFalse(usable(question, question["quote"]))
        said = "es geht hier um 36 Länder und die Schweiz soll klare gesetzliche Kriterien für die Waffenexporte haben"
        self.assertFalse(usable({"text": "Es geht um 36 Länder.", "quote": "es geht hier um 36 Länder"}, said))
        self.assertFalse(usable({"text": "Die Schweiz soll klare gesetzliche Kriterien für Waffenexporte haben.",
                                 "quote": "die Schweiz soll klare gesetzliche Kriterien für die Waffenexporte haben"}, said))

    def test_numbers_decide_whether_a_figure_was_said(self):
        said = "mit der vorgeschlagenen Erhöhung werden 1,7 Mia. eingenommen, der Bedarf ist 4 bis 5 Milliarden"
        self.assertTrue(grounded_in("Die Zusatzfinanzierung bringt der AHV 1,7 Milliarden Franken ein.", said))
        self.assertFalse(grounded_in("Die Zusatzfinanzierung bringt der AHV 1,4 Milliarden Franken ein.", said))
        law = {"text": "Die Mehrwertsteuer soll um 0,4 Prozentpunkte erhöht werden.", "quote": "die Mehrwertsteuer soll um 0,4 Prozent erhöht werden"}
        self.assertTrue(usable(law, "Ja, die Mehrwertsteuer soll um 0,4 Prozent erhöht werden, das ist moderat."))

    def test_numbers_and_thousands(self):
        self.assertEqual(numbers("rund 40'000 oder 37’500, 2,6 Mio."), {"40000", "37500", "2,6"})
        self.assertEqual(numbers("im Jahr 2025 100 Personen"), {"2025", "100"})
        self.assertEqual(numbers("netto 40 000 Personen"), {"40000"})

    def test_scale_slip_is_speech_recognition_not_contradiction(self):
        source = ["Rein rechnerisch könnten bis 2050 jährlich noch gegen 40’000 Personen dazukommen (genau 37’500)."]
        self.assertTrue(scale_slip("Es könnten jährlich netto 40 Personen zuwandern.", source))
        self.assertFalse(scale_slip("Es könnten jährlich netto 60'000 Personen zuwandern.", source))
        self.assertFalse(scale_slip("Die Zuwanderung ist gesunken.", source))

    def test_figures_differ(self):
        quote = ["Die Erhöhung verschafft der AHV zusätzliche Einnahmen von schätzungsweise 1,4 Milliarden Franken pro Jahr."]
        self.assertTrue(figures_differ("Mit der Vorlage kommen 1,7 Milliarden Franken in die AHV.", quote))
        self.assertFalse(figures_differ("Mit der Vorlage kommen 1,4 Milliarden Franken in die AHV.", quote))
        self.assertFalse(figures_differ("Die Vorlage verschafft der AHV zusätzliche Einnahmen.", quote))
        gdp = ["Die Wirtschaftsleistung der Schweiz nahm seit 2002 pro Kopf um 24 Prozent zu."]
        self.assertFalse(figures_differ("Das BIP pro Kopf ist seit 1999 um 25 bis 26,5 % gestiegen.", gdp))   # another period
        self.assertTrue(figures_differ("Das BIP pro Kopf ist seit 2002 um 30 % gestiegen.", gdp))
        self.assertFalse(figures_differ("Das BIP pro Kopf ist seit 2002 gestiegen.", gdp))   # the year is no figure

    def test_final_label_is_one_of_the_three(self):
        facts, committee, council = ["Erläuterungen"], ["Argument des Komitees"], ["Argument von Bundesrat und Parlament"]
        self.assertEqual(settle("SUPPORT", facts), "entailment")
        self.assertEqual(settle("SUPPORT", committee), "entailment")
        self.assertEqual(settle("CONFLICT", facts), "contradiction")
        self.assertEqual(settle("CONFLICT", committee), "neutral")   # one side's view refutes nothing
        self.assertEqual(settle("CONFLICT", council, "Bundesrat und Parlament empfehlen ein Nein."), "contradiction")
        self.assertEqual(settle("UNRELATED", facts), "neutral")
        self.assertEqual(settle(None, []), "neutral")

    def test_page_sections_follow_the_headings(self):
        pages = [{"page": 50, "text": "50 Vierte Vorlage Im Detail Vorlage im Detail Argumente Referendumskomitee Argumente Bundesrat und Parlament"},
                 {"page": 51, "text": "51 Ausgangslage Rolle der Rüstungsindustrie"},
                 {"page": 56, "text": "56 Vierte Vorlage Argumente Referendumskomitee Argumente Mit der Schwächung"},
                 {"page": 57, "text": "57 Wie Schweizer Handgranaten beim IS landeten"},
                 {"page": 58, "text": "58 Vierte Vorlage Argumente Bundesrat und Parlament Weltweit"},
                 {"page": 60, "text": "60 Vierte Vorlage Abstimmungstext Bundesgesetz"}]
        self.assertEqual(page_sections(pages), {50: "Erläuterungen", 51: "Erläuterungen", 56: "Argument des Komitees",
                                                57: "Argument des Komitees", 58: "Argument von Bundesrat und Parlament",
                                                60: "Gesetzestext"})

    def test_parse_pages(self):
        self.assertEqual(parse_pages("10-11, 50-52"), {10, 11, 50, 51, 52})
        self.assertIsNone(parse_pages(None))


class ClaimCheckerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "video.claims.jsonl"
        self.extracted, self.verified = [], []

    def tearDown(self):
        self.tmp.cleanup()

    def checker(self, extract=True, **kwargs) -> ClaimChecker:
        def do_extract(before, text, after):
            self.extracted.append((before, text, after))
            return [{"text": "Das SECO erhält jährlich rund 2500 Exportgesuche.", "quote": "2500"}]

        def do_verify(claim):
            self.verified.append(claim)
            return {"label": "entailment", "evidence": [{"page": 52, "text": "rund 2500 Gesuche"}], "note": None}

        return ClaimChecker(self.cache, do_extract if extract else None, do_verify if extract else None,
                            submit=lambda function, *args: function(*args), **kwargs)

    def test_segment_near_playback_is_extracted_and_checked_once(self):
        checker = self.checker()
        segments = [segment(0, 5, "Guten Abend und willkommen in der Arena, schön sind Sie da."), segment(6, 20)]
        checker.follow(28, segments[1:])
        checker.follow(29, segments[1:])
        self.assertEqual(len(self.extracted), 1)
        changes = checker.changes(0)
        self.assertEqual([(c["status"], c["label"], c["t"]) for c in changes["claims"]], [("done", "entailment", 20)])
        self.assertEqual(checker.changes(changes["rev"])["claims"], [])   # the page is up to date

    def test_context_is_the_two_segments_before(self):
        checker = self.checker()
        segments = [segment(0, 5, "A" * 50), segment(6, 10, "B" * 50), segment(11, 15, "C" * 50), segment(16, 20)]
        checker.follow(100, segments)          # far ahead: only the last minute is worked on
        self.assertEqual(self.extracted, [])
        checker.follow(30, segments)
        self.assertIn(("B" * 50 + " " + "C" * 50, LONG, ""), self.extracted)

    def test_waits_for_what_follows_the_segment(self):
        checker = self.checker()
        checker.follow(21, [segment(6, 20)])                      # just ended, nothing after it yet
        self.assertEqual(self.extracted, [])
        checker.follow(22, [segment(6, 20)], {"start": 20.5, "end": 22, "text": "Aber"})
        self.assertEqual(self.extracted, [])                      # too little to complete a sentence
        following = "Aber das gilt natürlich nur, wenn der Bundesrat dem zustimmt, sagen die Gegner."
        checker.follow(24, [segment(6, 20)], {"start": 20.5, "end": 24, "text": following})
        self.assertEqual(self.extracted, [("", LONG, following)])

    def test_next_segment_is_the_text_after(self):
        checker = self.checker()
        checker.follow(10, [segment(6, 20), segment(21, 30, "N" * 70)])   # cached transcript, ahead of playback
        self.assertEqual(self.extracted[0], ("", LONG, "N" * 70))

    def test_far_ahead_and_short_segments_cost_nothing(self):
        checker = self.checker()
        checker.follow(0, [segment(1, 3, "In die Runde."), segment(100, 120)])
        self.assertEqual(self.extracted, [])

    def test_same_claim_again_is_not_checked_twice(self):
        checker = self.checker()
        checker.follow(30, [segment(6, 20)])
        checker.follow(50, [segment(6, 20), segment(21, 40, LONG + " Wirklich.")])
        self.assertEqual(len(self.extracted), 2)
        self.assertEqual(len(self.verified), 1)

    def test_reworded_duplicate_is_recognised(self):
        texts = iter(["36 Partnerstaaten dürfen künftig auch beliefert werden, wenn sie in einen bewaffneten Konflikt verwickelt sind.",
                      "Künftig dürfen 36 Partnerstaaten beliefert werden, auch wenn sie in einen bewaffneten Konflikt verwickelt sind."])
        checker = ClaimChecker(self.cache, lambda b, s, a: [{"text": next(texts), "quote": ""}],
                               lambda claim: self.verified.append(claim) or {"label": "entailment", "evidence": [], "note": None},
                               submit=lambda function, *args: function(*args))
        checker.follow(30, [segment(6, 20)])
        checker.follow(50, [segment(6, 20), segment(21, 40, LONG + " Wirklich.")])
        self.assertEqual(len(self.verified), 1)

    def test_process_all_does_every_segment_once(self):
        checker = self.checker()
        segments = [segment(0, 5, "In die Runde."), segment(6, 20), segment(100, 120, LONG + " Und nochmals etwas anderes gesagt.")]
        checker.process_all(segments, workers=2)
        checker.process_all(segments, workers=2)
        self.assertEqual(len(self.extracted), 2)
        self.assertEqual(self.extracted[0][2], segments[2]["text"])   # the next segment is the text after
        self.assertEqual(len(self.verified), 1)                      # the second claim is a duplicate

    def test_cache_makes_the_second_run_free(self):
        self.checker().follow(30, [segment(6, 20)])
        again = self.checker()
        again.follow(30, [segment(6, 20)])
        self.assertEqual((len(self.extracted), len(self.verified)), (1, 1))
        self.assertEqual([c["label"] for c in again.changes(0)["claims"]], ["entailment"])

    def test_offline_shows_the_cache_and_calls_nothing(self):
        self.checker().follow(30, [segment(6, 20)])
        offline = self.checker(extract=False)
        offline.follow(60, [segment(6, 20), segment(30, 50)])
        self.assertEqual((len(self.extracted), len(self.verified)), (1, 1))
        self.assertEqual(len(offline.changes(0)["claims"]), 1)

    def test_failed_check_is_shown_and_not_cached(self):
        def verify(claim):
            raise RuntimeError("endpoint down")
        checker = ClaimChecker(self.cache, lambda b, s, a: [{"text": "Ein Claim über das Gesetz.", "quote": ""}], verify,
                               submit=lambda function, *args: function(*args))
        checker.follow(30, [segment(6, 20)])
        self.assertEqual(checker.changes(0)["claims"][0]["status"], "error")
        self.assertFalse(any(r["type"] == "result" for r in load_cache(self.cache, None)))

    def test_cache_of_another_video_is_set_aside(self):
        video = Path(self.tmp.name) / "video.mp4"
        video.write_bytes(b"first video")
        self.checker(fingerprint=video_fingerprint(video)).follow(30, [segment(6, 20)])
        video.write_bytes(b"a different video")
        fresh = self.checker(fingerprint=video_fingerprint(video))
        self.assertEqual(fresh.changes(0)["claims"], [])
        self.assertEqual(len(list(Path(self.tmp.name).glob("video.claims.jsonl.old-*"))), 1)


class BookletPagesTest(unittest.TestCase):
    def test_parallel_workers_read_the_booklet_one_at_a_time(self):
        """pdfium is not thread-safe: the batch's workers failed with 'Failed to load page'."""
        import threading
        import time
        from unittest.mock import patch

        from claimcheck import predict
        from claimcheck.live.claims import booklet_pages

        inside, overlaps = [0], []

        def pdf_pages(path):
            inside[0] += 1
            overlaps.append(inside[0])
            time.sleep(0.02)
            inside[0] -= 1
            return ({"page": 1, "text": "x"},)

        with patch.object(predict, "pdf_pages", pdf_pages):
            threads = [threading.Thread(target=booklet_pages, args=(Path("b.pdf"),)) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(max(overlaps), 1)


if __name__ == "__main__":
    unittest.main()
