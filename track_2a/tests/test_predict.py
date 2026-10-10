import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from claimcheck import nli, predict
from claimcheck.contracts import validate_prediction, validate_request

REFERENCE = "OUI. Le Conseil fédéral et le Parlement recommandent d'accepter l'initiative."


def request(task="B", vote="Initiative"):
    obj = {"id": "case-1", "vote": vote, "claim": {"text": "Der Bundesrat empfiehlt ein Nein.", "language": "de"}}
    if task == "A":
        obj["booklet"] = {"path": "booklets/x.pdf", "language": "fr"}
    else:
        obj["reference"] = {"text": REFERENCE, "language": "fr"}
    return validate_request(obj, Path("/data"))


def answer(label="CONTRADICT", chunk=0):
    return json.dumps({"claim_says": "x", "subject_addressed": True, "passage_says": "y", "same_meaning": False,
                       "label": label, "citations": [{"chunk": chunk, "quote": "recommandent d'accepter l'initiative"}]})


def fake_complete(text):
    calls = []

    def complete(system, user, max_tokens=600):
        calls.append(json.loads(user))
        return {"text": text, "model": "swiss-ai/Apertus-v1.5-8B", "input_tokens": 100, "output_tokens": 20,
                "inference_time_ms": 5}
    return patch.object(nli, "complete", side_effect=complete), calls


class ApertusPredictorTests(unittest.TestCase):
    def test_task_b_reference_is_the_only_chunk(self):
        patcher, calls = fake_complete(answer())
        with patcher:
            prediction = predict.ApertusPredictor().predict(request())
        self.assertEqual(validate_prediction(prediction, request()), prediction)
        self.assertEqual((prediction["label"], prediction["label_name"]), (2, "contradiction"))
        self.assertEqual(prediction["evidence"], [{"page": None, "text": "recommandent d'accepter l'initiative"}])
        self.assertEqual(prediction["metrics"], {"input_tokens": 100, "output_tokens": 20, "inference_time_ms": 5})
        self.assertEqual(calls[0]["vote"], "Initiative")
        self.assertEqual(calls[0]["chunks"], [{"chunk": 0, "page": None, "paragraph": None, "text": REFERENCE}])

    def test_task_a_pages_are_chunks_with_page_numbers(self):
        pages = ({"text": "Seite eins.", "page": 1}, {"text": REFERENCE, "page": 7})
        patcher, calls = fake_complete(answer(chunk=1))
        with patcher, patch.object(predict, "pdf_pages", return_value=pages):
            prediction = predict.ApertusPredictor().predict(request("A", vote={"title": "Initiative", "date": "2026"}))
        self.assertEqual(validate_prediction(prediction, request("A")), prediction)
        self.assertEqual(prediction["evidence"], [{"page": 7, "text": "recommandent d'accepter l'initiative"}])
        self.assertEqual(calls[0]["vote"], "Initiative")
        self.assertEqual(calls[0]["chunks"][1]["page"], 7)

    def test_failures_degrade_to_neutral(self):
        stderr = io.StringIO()
        with patch.object(nli, "complete", side_effect=RuntimeError("offline")), contextlib.redirect_stderr(stderr):
            prediction = predict.ApertusPredictor().predict(request())
        self.assertEqual(validate_prediction(prediction, request()), prediction)
        self.assertEqual(prediction["label"], 1)
        self.assertIn("offline", stderr.getvalue())
        with patch.object(predict, "pdf_pages", side_effect=ValueError("no text")), contextlib.redirect_stderr(stderr):
            self.assertEqual(predict.ApertusPredictor().predict(request("A"))["label"], 1)

    def test_vote_title(self):
        self.assertEqual(predict.vote_title("x"), "x")
        self.assertEqual(predict.vote_title({"title": "t", "date": "d"}), "t")
        self.assertIsNone(predict.vote_title(None))


if __name__ == "__main__":
    unittest.main()
