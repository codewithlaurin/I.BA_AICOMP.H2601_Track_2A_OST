from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import benchmark as b
from scripts.apertus_nli import ApiConfig, create_client
from scripts.evaluate import score
from scripts.prepare_cases import prepare


def case(identifier="sample", task="B"):
    result = {"id": identifier, "vote": "Renteninitiative",
              "claim": {"text": "La pensione aumenta.", "language": "it"}}
    result["reference" if task == "B" else "booklet"] = (
        {"text": "Die Rente steigt. Weitere Angaben fehlen.", "language": "de"}
        if task == "B" else {"path": "booklets/test.pdf", "language": "de"})
    return result


def api_result(label="ENTAIL", quote="Die Rente steigt.", usage=True):
    result = {"model": b.DEFAULT_MODEL, "choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps({"label": label, "explanation": "Evidenza.", "citations":
                               [] if label == "NEUTRAL" else [{"evidence_id": "E1", "quote": quote}]})}}]}
    if usage:
        result["usage"] = {"prompt_tokens": 120, "completion_tokens": 30,
                           "completion_tokens_details": {"reasoning_tokens": 10}}
    return result


class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.config = ApiConfig(model=b.DEFAULT_MODEL)

    def test_all_labels_and_actual_usage(self):
        for name, number in b.LABEL_IDS.items():
            with self.subTest(label=name), patch.object(b, "api_request", return_value=api_result(name)) as call:
                result = b.predict(case(), Path("."), self.config)
                self.assertEqual(result["label"], number)
                self.assertEqual(result["label_name"], b.LABELS[number])
                self.assertEqual(result["metrics"]["input_tokens"], 120)
                self.assertEqual(result["metrics"]["output_tokens"], 30)
                self.assertGreaterEqual(result["metrics"]["inference_time_ms"], 0)
                if number != 1:
                    self.assertIsNone(result["evidence"][0]["page"])
                call.assert_called_once()

    def test_vote_language_and_no_gold_in_prompt(self):
        request = {**case(), "entailment_label": "SECRET_GOLD", "reference_string": "SECRET_REFERENCE"}
        with patch.object(b, "api_request", return_value=api_result()) as call:
            b.predict(request, Path("."), self.config)
        payload = call.call_args.args[2]
        data = json.loads(payload["messages"][1]["content"])
        self.assertEqual(data["vote"], request["vote"])
        self.assertEqual(data["claim"]["language"], "it")
        self.assertNotIn("SECRET", json.dumps(payload))

    def test_page_number_preserved(self):
        with patch.object(b, "booklet_path", return_value=Path("test.pdf")), patch.object(
            b, "extract_pdf", return_value=((7, case()["reference"]["text"]),)
        ), patch.object(b, "api_request", return_value=api_result()):
            result = b.predict(case(task="A"), Path("."), self.config)
        self.assertEqual(result["evidence"], [{"page": 7, "text": "Die Rente steigt."}])

    def test_unverifiable_and_whole_passage_quotes_fail(self):
        for quote in ("Erfundene Aussage.", case()["reference"]["text"]):
            with patch.object(b, "api_request", return_value=api_result(quote=quote)):
                result = b.predict(case(), Path("."), self.config)
            self.assertNotIn("label", result)
            self.assertIn("error", result)
            self.assertEqual(result["metrics"]["input_tokens"], 120)

    def test_incomplete_response_retains_usage(self):
        raw = api_result()
        raw["choices"][0]["finish_reason"] = "length"
        with patch.object(b, "api_request", return_value=raw):
            result = b.predict(case(), Path("."), self.config)
        self.assertIn("error", result)
        self.assertEqual(result["metrics"]["output_tokens"], 30)

    def test_transport_and_missing_usage_are_not_zero(self):
        for kwargs in ({"side_effect": RuntimeError("offline")}, {"return_value": api_result(usage=False)}):
            with patch.object(b, "api_request", **kwargs):
                result = b.predict(case(), Path("."), self.config)
            self.assertIn("error", result)
            self.assertIsNone(result["metrics"]["input_tokens"])

    def test_budget_failure_does_not_call_api(self):
        with patch.object(b, "api_request") as call:
            result = b.predict(case(), Path("."), ApiConfig(model=b.DEFAULT_MODEL, max_prompt_chars=10))
        call.assert_not_called()
        self.assertIn("error", result)
        self.assertEqual(result["metrics"]["input_tokens"], 0)

    def test_injected_environment_wins(self):
        env = {"BASE_URL": "https://proxy.example/v1", "API_KEY": "judge-key",
               "CSCS_INFERENCE_BASE_URL": "https://old.example/v1", "CSCS_INFERENCE_API_KEY": "old-key",
               "LLM_NAME": b.DEFAULT_MODEL, "APERTUS_MODEL": "old-model"}
        with patch.dict(os.environ, env, clear=True), patch("openai.OpenAI") as client:
            config = b.benchmark_config()
            create_client(config)
        self.assertEqual(config.base_url, env["BASE_URL"])
        self.assertEqual(client.call_args.kwargs["api_key"], "judge-key")
        self.assertEqual(client.call_args.kwargs["max_retries"], 0)

    def test_empty_judge_key_does_not_fall_back(self):
        with patch.dict(os.environ, {"API_KEY": "", "CSCS_INFERENCE_API_KEY": "old-key"}, clear=True):
            with self.assertRaises(ValueError):
                b.benchmark_config()

    def test_older_apertus_rejected(self):
        with patch.dict(os.environ, {"API_KEY": "x", "APERTUS_MODEL": "swiss-ai/Apertus-8B-Instruct-2509"}, clear=True):
            with self.assertRaisesRegex(ValueError, "v1.5"):
                b.benchmark_config()


class BatchTests(unittest.TestCase):
    def test_citation_comparison_is_saved_in_predictions_for_both_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "booklets").mkdir()
            (root / "booklets/test.pdf").write_bytes(b"%PDF fake; parser mocked")
            inputs, output = root / "cases.jsonl", root / "predictions.jsonl"
            cases = [case("a", "A"), case("b")]
            inputs.write_text("".join(json.dumps(c) + "\n" for c in cases))
            source_text = "Die Rente steigt.\r\nWeitere Angaben fehlen."
            rejected_quote = "La pensione aumenta."
            with patch.object(b, "benchmark_config", return_value=ApiConfig(model=b.DEFAULT_MODEL)), patch.object(
                b, "extract_pdf", return_value=((7, source_text),)
            ), patch.object(b, "api_request", return_value=api_result(quote=rejected_quote)):
                self.assertEqual(b.run(inputs, output), 1)
            records = b.read_jsonl(output)
            for record, request, page, text in zip(
                records, cases, (7, None), (source_text, cases[1]["reference"]["text"]), strict=True
            ):
                with self.subTest(task=request["id"]):
                    self.assertNotIn("label", record)
                    self.assertEqual(record["metrics"]["input_tokens"], 120)
                    self.assertEqual(record["diagnostics"], {
                        "evidence_id": "E1", "model_quote": rejected_quote,
                        "source_text": text, "chunk_id": f"{request['id']}:1",
                        "page_number": page,
                        "source_pdf": "booklets/test.pdf" if page else None,
                    })
            report = score(cases, records, [{"id": c["id"], "label": 0} for c in cases])
            self.assertEqual(report["overall"]["errors"], 2)

    def test_real_pdf_extraction_preserves_physical_page_numbers(self):
        # Minimal two-page PDF: blank first page, text on the second. Exercises
        # the real PDFium API rather than mocking extraction/resource lifetimes.
        stream = b"BT /F1 12 Tf 40 700 Td (Die Rente steigt.) Tj ET"
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 6 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        ]
        pdf, offsets = b"%PDF-1.4\n", [0]
        for number, obj in enumerate(objects, 1):
            offsets.append(len(pdf))
            pdf += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
        xref = len(pdf)
        pdf += b"xref\n0 7\n0000000000 65535 f \n"
        pdf += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
        pdf += f"trailer\n<< /Size 7 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.pdf"
            path.write_bytes(pdf)
            pages = b.extract_pdf(path)
        self.assertEqual(pages, ((2, "Die Rente steigt."),))

    def test_mixed_batch_continues_after_failure_and_is_order_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "booklets").mkdir()
            (root / "booklets/test.pdf").write_bytes(b"%PDF fake; parser mocked")
            cases = [case("a", "A"), case("b"), case("c")]
            input_path, output = root / "cases.jsonl", root / "out/predictions.jsonl"
            def respond(path, config, request):
                data = json.loads(request["messages"][1]["content"])
                return api_result("NEUTRAL" if data["evidence"][0]["page_number"] else "ENTAIL")
            with patch.object(b, "benchmark_config", return_value=ApiConfig(model=b.DEFAULT_MODEL)), patch.object(
                b, "extract_pdf", return_value=((1, case()["reference"]["text"]),)
            ), patch.object(b, "api_request", side_effect=respond):
                input_path.write_text("".join(json.dumps(c) + "\n" for c in cases))
                self.assertEqual(b.run(input_path, output), 0)
                forward = b.read_jsonl(output)
                input_path.write_text("".join(json.dumps(c) + "\n" for c in reversed(cases)))
                self.assertEqual(b.run(input_path, output), 0)
                backward = b.read_jsonl(output)
            self.assertEqual({r["id"]: r["label"] for r in forward}, {r["id"]: r["label"] for r in backward})
            with patch.object(b, "benchmark_config", return_value=ApiConfig(model=b.DEFAULT_MODEL)), patch.object(
                b, "extract_pdf", return_value=((1, case()["reference"]["text"]),)
            ), patch.object(b, "api_request", side_effect=[RuntimeError("offline"), api_result(), api_result()]):
                self.assertEqual(b.run(input_path, output), 1)
                results = b.read_jsonl(output)
                self.assertEqual(len(results), 3)
                self.assertIn("error", results[0])
                self.assertEqual(results[1]["label"], 0)

    def test_invalid_inputs_preserve_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, output = root / "cases.jsonl", root / "predictions.jsonl"
            output.write_text("keep me")
            inputs.write_text(json.dumps(case()) + "\n" + json.dumps(case()) + "\n")
            with self.assertRaises(ValueError):
                b.run(inputs, output)
            self.assertEqual(output.read_text(), "keep me")
            with self.assertRaises(ValueError):
                b.run(inputs, inputs)

    def test_source_exclusivity_languages_and_traversal(self):
        for invalid in ({**case(), "booklet": case(task="A")["booklet"]},
                        {**case(), "claim": {"text": "test", "language": "en"}}):
            with self.assertRaises(ValueError):
                b.validate_case(invalid)
        for path in ("../secret.pdf", "/etc/secret.pdf"):
            request = case(task="A")
            request["booklet"]["path"] = path
            with self.assertRaises(ValueError):
                b.booklet_path(request, Path("/tmp/data"))


class ScorerTests(unittest.TestCase):
    def test_macro_f1_includes_missing_errors_and_all_classes(self):
        cases = [case(str(i)) for i in range(4)]
        predictions = []
        for i, label in enumerate(("ENTAIL", "CONTRADICT", "CONTRADICT")):
            with patch.object(b, "api_request", return_value=api_result(label)):
                predictions.append(b.predict(cases[i], Path("."), ApiConfig(model=b.DEFAULT_MODEL)))
        gold = [{"id": str(i), "label": label} for i, label in enumerate((0, 1, 2, 0))]
        report = score(cases, predictions, gold)
        metrics = report["by_task"]["B"]
        self.assertAlmostEqual(metrics["macro_f1"], 4 / 9)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["prediction_coverage"], 0.75)
        self.assertEqual(metrics["confusion_matrix"]["values"], [[1, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0]])
        self.assertEqual(metrics["usage"]["input_tokens"], {"reported_total": 360, "missing_cases": 1})
        self.assertIn("de->it", metrics["by_language_pair"])
        self.assertIsNone(report["by_task"]["A"]["macro_f1"])

    def test_invalid_citation_is_not_a_valid_prediction(self):
        with patch.object(b, "api_request", return_value=api_result()):
            prediction = b.predict(case(), Path("."), ApiConfig(model=b.DEFAULT_MODEL))
        prediction["evidence"][0]["text"] = "invented"
        report = score([case()], [prediction], [{"id": "sample", "label": 0}])
        self.assertEqual(report["overall"]["errors"], 1)

    def test_unknown_prediction_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            score([case()], [{"id": "other"}], [{"id": "sample", "label": 0}])


class PreparationTests(unittest.TestCase):
    def test_reproducible_sampling_original_ids_and_no_label_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "dataset.jsonl"
            rows = [{"row_index": i + 10, "claim": "test", "claim_language": "it",
                     "vote": "Vote", "reference_string": "Titre uniquement", "reference_language": "fr",
                     "entailment_label": i % 3, "booklet_url": "https://example.com/booklet.pdf"}
                    for i in range(5)]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows))
            for name in ("first", "second"):
                self.assertEqual(prepare(source, root / name, "both", 2, 42, False), 4)
            self.assertEqual((root / "first/cases.jsonl").read_bytes(), (root / "second/cases.jsonl").read_bytes())
            cases = b.read_jsonl(root / "first/cases.jsonl")
            self.assertEqual(set(cases[0]), {"id", "vote", "claim", "booklet"})
            self.assertEqual(cases[1]["reference"]["text"], "Titre uniquement")
            self.assertTrue(all(int(c["id"].split("-")[2]) >= 10 for c in cases))
            with self.assertRaises(FileExistsError):
                prepare(source, root / "first", "B", 2, 42, False)


if __name__ == "__main__":
    unittest.main()
