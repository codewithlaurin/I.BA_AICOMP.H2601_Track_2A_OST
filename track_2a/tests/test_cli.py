import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from claimcheck.cli import main, run
from claimcheck.contracts import ContractError, LABEL_NAMES, validate_prediction, validate_request
from claimcheck.predict import NeutralPredictor


def case(identifier="case-001", task="B"):
    result = {
        "id": identifier,
        "vote": {"date": "2026-09-27", "title": "Example vote"},
        "claim": {"text": "Une affirmation.", "language": "fr"},
    }
    if task == "A":
        result["booklet"] = {"path": "booklets/example.pdf", "language": "de"}
    else:
        result["reference"] = {"text": "Un testo di riferimento.", "language": "it"}
    return result


class ContractTests(unittest.TestCase):
    def test_languages_can_differ_for_both_tasks(self):
        for task in ("A", "B"):
            for claim_language in ("de", "fr", "it"):
                for source_language in ("de", "fr", "it"):
                    obj = case(task=task)
                    obj["claim"]["language"] = claim_language
                    obj["booklet" if task == "A" else "reference"]["language"] = source_language
                    request = validate_request(obj, Path("/data"))
                    self.assertEqual(request.task, task)
                    self.assertEqual(request.claim.language, claim_language)
                    self.assertEqual(request.source.language, source_language)

    def test_relative_and_absolute_booklet_paths(self):
        obj = case(task="A")
        self.assertEqual(validate_request(obj, Path("/data")).source.path, Path("/data/booklets/example.pdf"))
        obj["booklet"]["path"] = "/other/example.pdf"
        self.assertEqual(validate_request(obj, Path("/data")).source.path, Path("/other/example.pdf"))

    def test_invalid_requests(self):
        invalid = [([], "object")]
        for field in ("id", "vote", "claim", "reference"):
            obj = case()
            del obj[field]
            invalid.append((obj, "required|exactly one"))
        obj = case()
        obj["booklet"] = {"path": "x.pdf", "language": "de"}
        invalid.append((obj, "exactly one"))
        for identifier in (None, True, 1.5, [], {}, " "):
            invalid.append((case(identifier), "id must"))
        for field, value in (("claim", None), ("reference", "text")):
            obj = case()
            obj[field] = value
            invalid.append((obj, "object"))
        for field in ("claim", "reference"):
            for value in ("en", None, []):
                obj = case()
                obj[field]["language"] = value
                invalid.append((obj, "language"))
            for value in ("", " ", 3):
                obj = case()
                obj[field]["text"] = value
                invalid.append((obj, "text"))
        for value in ("", None, "bad\x00path"):
            obj = case(task="A")
            obj["booklet"]["path"] = value
            invalid.append((obj, "path"))
        for obj, message in invalid:
            with self.subTest(obj=obj), self.assertRaisesRegex(ContractError, message):
                validate_request(obj, Path("/data"))

    def test_label_mapping_and_valid_evidence(self):
        for task in ("A", "B"):
            request = validate_request(case(task=task), Path("/data"))
            for label, name in LABEL_NAMES.items():
                prediction = NeutralPredictor().predict(request)
                prediction.update(label=label, label_name=name)
                prediction["evidence"] = [{"page": 1 if task == "A" else None, "text": "ü" * 5000}] * 5
                self.assertEqual(validate_prediction(prediction, request), prediction)

    def test_invalid_predictions(self):
        for task in ("A", "B"):
            request = validate_request(case(task=task), Path("/data"))
            base = NeutralPredictor().predict(request)
            changes = [
                ({"id": "different"}, "id"),
                ({"label": True}, "label"),
                ({"label": 3}, "label"),
                ({"label_name": "NEUTRAL"}, "label_name"),
                ({"evidence": None}, "list"),
                ({"evidence": [{}] * 6}, "five"),
                ({"evidence": [{"text": "quote"}]}, "page"),
                ({"evidence": [{"page": 1 if task == "A" else None, "text": "x" * 5001}]}, "5,000"),
                ({"evidence": [{"page": 1 if task == "A" else None, "text": " "}]}, "text"),
            ]
            for page in ((None, 0, -1, True, 1.5) if task == "A" else (0, 1, "null")):
                changes.append(({"evidence": [{"page": page, "text": "quote"}]}, "page"))
            if task == "A":
                for label in (0, 2):
                    changes.append(({"label": label, "label_name": LABEL_NAMES[label]}, "require evidence"))
            for field, values in {
                "input_tokens": (-1, 0.5, True, None),
                "output_tokens": (-1, 0.5, True, None),
                "inference_time_ms": (-1, True, None, float("nan"), float("inf")),
            }.items():
                for value in values:
                    metrics = dict(base["metrics"], **{field: value})
                    changes.append(({"metrics": metrics}, field))
            for change, message in changes:
                with self.subTest(task=task, change=change), self.assertRaisesRegex(ContractError, message):
                    validate_prediction(dict(base, **change), request)

    def test_task_b_entailment_and_contradiction_need_no_evidence(self):
        request = validate_request(case(), Path("/data"))
        for label in (0, 2):
            prediction = NeutralPredictor().predict(request)
            prediction.update(label=label, label_name=LABEL_NAMES[label])
            validate_prediction(prediction, request)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "data" / "cases.jsonl"
        self.input.parent.mkdir()
        self.output = self.root / "output" / "predictions.jsonl"

    def write_cases(self, *cases):
        self.input.write_text("".join(json.dumps(obj, ensure_ascii=False) + "\n" for obj in cases), encoding="utf-8")

    def invoke(self, *args):
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        return subprocess.run(
            [sys.executable, "-m", "claimcheck", *map(str, args)],
            cwd=self.root, env=env, text=True, capture_output=True,
        )

    def test_real_cli_mixed_file_preserves_ids_and_order(self):
        self.write_cases(case("0001", "A"), case(7), case("7"))
        result = self.invoke("--input", self.input, "--output", self.output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("stub", result.stderr)
        predictions = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual([obj["id"] for obj in predictions], ["0001", 7, "7"])
        for obj in predictions:
            self.assertEqual(obj["label"], 1)
            self.assertEqual(obj["label_name"], "neutral")
            self.assertEqual(obj["evidence"], [])
            self.assertEqual(obj["metrics"], {"input_tokens": 0, "output_tokens": 0, "inference_time_ms": 0})

    def test_predictor_receives_resolved_path_and_vote(self):
        self.write_cases(case(task="A"))
        observed = []

        class RecordingPredictor(NeutralPredictor):
            def predict(self, request):
                observed.append(request)
                return super().predict(request)

        run(self.input, self.output, RecordingPredictor())
        self.assertEqual(observed[0].source.path, (self.input.parent / "booklets/example.pdf").resolve())
        self.assertEqual(observed[0].vote, case()["vote"])

    def test_malformed_and_duplicate_input_preserve_existing_output(self):
        self.output.parent.mkdir()
        self.output.write_text("existing output\n")
        valid = json.dumps(case())
        invalid_inputs = [
            (valid + "\n{bad json}\n", "line 2"),
            (valid + "\n" + valid + "\n", "duplicate id.*line 1"),
            (valid + "\n\n", "line 2.*blank line"),
            ("", "no requests"),
            ('{"id": 1, "id": 2}\n', "duplicate JSON field"),
            (valid.replace('"vote": {', '"vote": NaN, "unused": {') + "\n", "invalid JSON number"),
            ('[]\n', "object"),
        ]
        for data, message in invalid_inputs:
            with self.subTest(data=data):
                self.input.write_text(data)
                with self.assertRaisesRegex(ContractError, message):
                    run(self.input, self.output, NeutralPredictor())
                self.assertEqual(self.output.read_text(), "existing output\n")

    def test_predictor_failure_is_nonzero_without_partial_output(self):
        self.write_cases(case("first"), case("second"))

        class BrokenPredictor(NeutralPredictor):
            def predict(self, request):
                if request.id == "second":
                    raise RuntimeError("backend unavailable")
                return super().predict(request)

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = main(["--input", str(self.input), "--output", str(self.output)], predictor=BrokenPredictor())
        self.assertEqual(status, 1)
        self.assertIn("line 2, id 'second': backend unavailable", stderr.getvalue())
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.output.parent.iterdir()), [])

    def test_invalid_prediction_preserves_existing_output(self):
        self.write_cases(case())
        self.output.parent.mkdir()
        self.output.write_text("previous\n")

        class InvalidPredictor(NeutralPredictor):
            def predict(self, request):
                return dict(super().predict(request), label_name="contradiction")

        with self.assertRaisesRegex(ContractError, "label_name"):
            run(self.input, self.output, InvalidPredictor())
        self.assertEqual(self.output.read_text(), "previous\n")
        self.assertEqual(list(self.output.parent.iterdir()), [self.output])

    def test_missing_input_and_invalid_utf8_fail_on_stderr(self):
        for data in (None, b"\xff\n"):
            if data is not None:
                self.input.write_bytes(data)
            result = self.invoke("--input", self.input, "--output", self.output)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("error:", result.stderr)
            self.assertFalse(self.output.exists())

    def test_required_arguments(self):
        result = self.invoke("--input", self.input)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--output", result.stderr)

    def test_input_and_booklet_cannot_be_overwritten(self):
        obj = case(task="A")
        obj["booklet"]["path"] = str(self.output)
        self.write_cases(obj)
        for destination in (self.input, self.output):
            with self.assertRaisesRegex(ContractError, "--output"):
                run(self.input, destination, NeutralPredictor())
        self.assertEqual(json.loads(self.input.read_text()), obj)


if __name__ == "__main__":
    unittest.main()
