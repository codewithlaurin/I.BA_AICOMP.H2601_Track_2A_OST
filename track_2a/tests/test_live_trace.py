"""Trace view of the live demo: every Apertus call behind a card is recorded and served (no network)."""

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimcheck.live import trace  # noqa: E402
from claimcheck.live.claims import ClaimChecker, judge  # noqa: E402

PAGE = {"page": 53, "text": "Davon sind neu aber 36 Partnerstaaten ausgenommen."}
FIRST = json.dumps({"claim_says": "36 countries are exempt.", "subject_addressed": True,
                    "passage_says": "36 partner states are exempt.", "same_meaning": True, "label": "ENTAIL",
                    "citations": [{"chunk": 0, "quote": "Davon sind neu aber 36 Partnerstaaten ausgenommen"}]})


def fake_complete(answers):
    answers = iter(answers)

    def complete(system, user, max_tokens=1024):
        return {"text": next(answers), "model": "apertus-test", "input_tokens": 10, "output_tokens": 5, "api_time_ms": 7}
    return complete


class CollectTest(unittest.TestCase):
    def test_calls_and_rules_are_recorded_in_order(self):
        with patch("claimcheck.nli.complete", fake_complete([FIRST, '{"relation": "SUPPORT"}'])):
            with trace.collect() as steps:
                result = judge("36 Länder sind ausgenommen.", [PAGE], "KMG", "SYSTEM", lambda i, t: {"page": 53, "text": t})
        self.assertEqual(result["label"], "entailment")
        self.assertEqual([s.get("name", s["kind"]) for s in steps], ["Erste Prüfung", "Zweiter Blick", "rules"])
        first = steps[0]
        self.assertEqual((first["system"], first["response"], first["model"]), ("SYSTEM", FIRST, "apertus-test"))
        self.assertEqual(json.loads(first["user"])["claim"], "36 Länder sind ausgenommen.")
        self.assertEqual((steps[2]["second_look"], steps[2]["label"]), ("SUPPORT", "entailment"))

    def test_without_collect_nothing_is_kept(self):
        with patch("claimcheck.nli.complete", fake_complete(["x"])):
            self.assertEqual(trace.call("Zweiter Blick", "S", "U", 10)["text"], "x")
        trace.note("rules", label="neutral")   # no error, no effect

    def test_failed_calls_are_recorded_too(self):
        def broken(system, user, max_tokens=1024):
            raise TimeoutError("Request timed out.")
        with patch("claimcheck.nli.complete", broken), trace.collect() as steps:
            with self.assertRaises(TimeoutError):
                trace.call("Erste Prüfung", "S", "U", 10)
        self.assertEqual(steps[0]["error"], "TimeoutError: Request timed out.")

    def test_threads_keep_their_own_steps(self):
        barrier, seen = threading.Barrier(2), {}

        def work(name):
            with trace.collect() as steps:
                trace.note("mark", who=name)
                barrier.wait()
                trace.note("mark", who=name)
            seen[name] = [s["who"] for s in steps]
        threads = [threading.Thread(target=work, args=(n,)) for n in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(seen, {"a": ["a", "a"], "b": ["b", "b"]})

    def test_trace_file_sits_next_to_the_claims_cache(self):
        self.assertEqual(trace.trace_path(Path("d/x.claims.jsonl")), Path("d/x.trace.jsonl"))


class CheckerTraceTest(unittest.TestCase):
    SEGMENTS = [{"start": 0.0, "end": 5.0, "text": "Vorher wurde etwas gesagt, das den Zusammenhang gibt."},
                {"start": 6.0, "end": 20.0, "text": "Jedes Jahr hat das SECO ungefähr 2500 Gesuche um Export, und einmal kam das vor."},
                {"start": 21.0, "end": 30.0, "text": "Danach ging die Diskussion weiter mit einem anderen Punkt der Vorlage."}]

    def test_recorded_calls_are_served_per_claim(self):
        def extract(before, segment, after):
            trace.note("call", name="Claim-Erkennung", system="E", user=json.dumps({"segment": segment}), response="[]")
            trace.note("statements", items=[{"text": "Das SECO erhält 2500 Gesuche.", "type": "NUMBER", "rejected": None}])
            return [{"text": "Das SECO erhält 2500 Gesuche.", "quote": "2500 Gesuche"}]

        def verify(claim):
            trace.note("call", name="Erste Prüfung", system="V", user="{}", response="{}")
            return {"label": "neutral", "evidence": [], "note": None, "why": {"how": "not addressed"}}

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "v.claims.jsonl"
            checker = ClaimChecker(cache, extract, verify, submit=lambda function, *args: function(*args))
            checker.process_all(self.SEGMENTS)
            claim_id = next(iter(checker.claims))
            view = checker.trace_for(claim_id, self.SEGMENTS)
            self.assertTrue((Path(tmp) / "v.trace.jsonl").exists())
        self.assertFalse(view["transcript"]["reconstructed"])
        self.assertEqual(view["transcript"]["segment"], self.SEGMENTS[1]["text"])
        self.assertEqual(view["transcript"]["after"], self.SEGMENTS[2]["text"])
        self.assertEqual([s["kind"] for s in view["extraction"]["steps"]], ["call", "statements"])
        self.assertEqual(view["booklet"]["steps"][0]["name"], "Erste Prüfung")
        self.assertEqual(view["booklet"]["why"], {"how": "not addressed"})
        self.assertIsNone(view["web"])

    def test_claims_from_before_tracing_are_rebuilt_not_invented(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "v.claims.jsonl"
            key = ClaimChecker.segment_key(self.SEGMENTS[1])
            lines = [{"type": "segment", "key": key, "t": 20.0, "start": 6.0,
                      "claims": [{"id": key + "#0", "text": "Das SECO erhält 2500 Gesuche.", "quote": "2500 Gesuche"}]},
                     {"type": "result", "id": key + "#0", "label": "neutral", "evidence": [], "v": "verify-3"}]
            cache.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
            checker = ClaimChecker(cache, None, None)
            view = checker.trace_for(key + "#0", self.SEGMENTS)
            self.assertIsNone(checker.trace_for("nothing#0", self.SEGMENTS))
        self.assertTrue(view["transcript"]["reconstructed"])
        self.assertEqual(view["transcript"]["before"], self.SEGMENTS[0]["text"])
        self.assertEqual(view["transcript"]["segment"], self.SEGMENTS[1]["text"])
        self.assertIsNone(view["extraction"])
        self.assertIsNone(view["booklet"])


if __name__ == "__main__":
    unittest.main()
