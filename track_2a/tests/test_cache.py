import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

# Initialize the native runtimes before patch.dict snapshots/restores os.environ;
# OpenMP itself registers runtime state in environment variables on macOS.
import sentence_transformers  # noqa: F401
import faiss
import numpy as np

faiss.omp_set_num_threads(1)

from claimcheck import predict, retrieval
from claimcheck.cli import main, run


PAGES = ({"page": 2, "text": "first passage"}, {"page": 7, "text": "second passage"})


class RecordingEmbedder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return np.array([[0.0, 1.0] if "second" in text else [1.0, 0.0] for text in texts], dtype="float32")


class DocumentCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.env = patch.dict(os.environ, {"CLAIMCHECK_CACHE_DIR": str(self.cache),
                                          "EMBEDDING_MODEL": "test-model", "EMBEDDING_MODEL_REVISION": "v1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.pdf = self.root / "booklet.pdf"
        self.pdf.write_bytes(b"PDF content v1")
        self.clear_memory()
        self.addCleanup(self.clear_memory)

    @staticmethod
    def clear_memory():
        predict._cached_pdf_pages.cache_clear()
        retrieval._INDEXES.clear()

    def test_extracted_pages_survive_memory_eviction_and_identical_pdf_copy(self):
        with patch.object(predict, "_extract_pdf_pages", return_value=PAGES) as extract:
            self.assertEqual(predict.pdf_pages(self.pdf), PAGES)
            self.clear_memory()
            self.assertEqual(predict.pdf_pages(self.pdf), PAGES)
            copy = self.root / "copy.pdf"
            copy.write_bytes(self.pdf.read_bytes())
            self.assertEqual(predict.pdf_pages(copy), PAGES)
        extract.assert_called_once_with(self.pdf.resolve())

    def test_changed_pdf_reextracts_and_corrupt_document_cache_recovers(self):
        with patch.object(predict, "_extract_pdf_pages", return_value=PAGES) as extract:
            predict.pdf_pages(self.pdf)
            self.pdf.write_bytes(b"PDF content v2")
            predict.pdf_pages(self.pdf)
            self.assertEqual(extract.call_count, 2)
            self.clear_memory()
            for file in (self.cache / "documents").glob("*.json"):
                file.write_text('{"pages": [], "checksum": "bad"}')
            self.assertEqual(predict.pdf_pages(self.pdf), PAGES)
            self.assertEqual(extract.call_count, 3)

    def test_index_survives_restart_and_different_claims_only_embed_queries(self):
        model = RecordingEmbedder()
        with patch.object(retrieval, "embedder", return_value=model):
            first = retrieval.top_pages(self.pdf, PAGES, "first", None, k=1)
            second = retrieval.top_pages(self.pdf, PAGES, "second", None, k=1)
        self.assertEqual([first[0]["page"], second[0]["page"]], [2, 7])
        self.assertEqual(model.calls, [["passage: first passage", "passage: second passage"],
                                       ["query: first"], ["query: second"]])

        # A genuinely fresh process must read the persisted FAISS artifact and
        # still retrieve the right page, with passage embedding forbidden.
        script = """
import numpy as np
from pathlib import Path
from unittest.mock import patch
from claimcheck import retrieval
class QueryOnly:
    def encode(self, texts, **kwargs):
        assert texts == ['query: second'], texts
        return np.array([[0.0, 1.0]], dtype='float32')
pages = ({'page': 2, 'text': 'first passage'}, {'page': 7, 'text': 'second passage'})
with patch.object(retrieval, 'embedder', return_value=QueryOnly()):
    assert retrieval.top_pages(Path('another.pdf'), pages, 'second', None, 1)[0]['page'] == 7
"""
        env = dict(os.environ, PYTHONPATH=str(Path(predict.__file__).resolve().parents[1]))
        result = subprocess.run([sys.executable, "-c", script], env=env, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_eviction_does_not_reembed_booklet(self):
        model = RecordingEmbedder()
        with patch.object(retrieval, "embedder", return_value=model):
            original = retrieval.booklet_index(self.pdf, PAGES)
            for number in range(16):
                pages = ({"page": 1, "text": f"other document {number}"},)
                retrieval.booklet_index(self.root / f"{number}.pdf", pages)
            self.assertEqual(len(retrieval._INDEXES), 16)
            recovered = retrieval.booklet_index(self.pdf, PAGES)
        self.assertEqual(len(model.calls), 17)
        np.testing.assert_array_equal(original.reconstruct_n(), recovered.reconstruct_n())

    def test_changed_pages_model_revision_and_encoding_settings_rebuild(self):
        model = RecordingEmbedder()
        with patch.object(retrieval, "embedder", return_value=model):
            retrieval.booklet_index(self.pdf, PAGES)
            retrieval.booklet_index(self.pdf, tuple(reversed(PAGES)))
            retrieval.booklet_index(self.pdf, ({"page": 2, "text": "changed"},) + PAGES[1:])
            with patch.dict(os.environ, {"EMBEDDING_MODEL": "another-model"}):
                retrieval.booklet_index(self.pdf, PAGES)
            with patch.dict(os.environ, {"EMBEDDING_MODEL_REVISION": "v2"}):
                retrieval.booklet_index(self.pdf, PAGES)
            with patch.object(retrieval, "PASSAGE_CHARS", 5):
                retrieval.booklet_index(self.pdf, PAGES)
            with patch.object(retrieval, "TOP_K", 1):
                retrieval.booklet_index(self.pdf, PAGES)
        self.assertEqual(len(model.calls), 6)

    def test_changed_local_model_files_rebuild(self):
        local_model = self.root / "model"
        local_model.mkdir()
        weights = local_model / "model.safetensors"
        weights.write_bytes(b"weights")
        model = RecordingEmbedder()
        with patch.dict(os.environ, {"EMBEDDING_MODEL": str(local_model)}), \
                patch.object(retrieval, "embedder", return_value=model):
            retrieval.booklet_index(self.pdf, PAGES)
            weights.write_bytes(b"updated weights")
            retrieval.booklet_index(self.pdf, PAGES)
        self.assertEqual(len(model.calls), 2)

    def test_corrupt_faiss_index_recovers(self):
        model = RecordingEmbedder()
        with patch.object(retrieval, "embedder", return_value=model):
            retrieval.booklet_index(self.pdf, PAGES)
            self.clear_memory()
            next((self.cache / "indexes").glob("*.faiss")).write_bytes(b"interrupted index")
            self.assertEqual(retrieval.top_pages(self.pdf, PAGES, "second", None, 1)[0]["page"], 7)
        self.assertEqual(len(model.calls), 3)  # Original build, recovery, query.
        self.assertFalse(list(self.cache.rglob("*.tmp")))

    def test_concurrent_processes_embed_document_once(self):
        marker = self.root / "builds.txt"
        script = """
import sys
import time
import numpy as np
from pathlib import Path
from unittest.mock import patch
from claimcheck import retrieval
class Embedder:
    def encode(self, texts, **kwargs):
        with Path(sys.argv[1]).open('a') as out:
            out.write('build\\n')
        time.sleep(0.2)
        return np.eye(2, dtype='float32')
pages = ({'page': 2, 'text': 'first passage'}, {'page': 7, 'text': 'second passage'})
with patch.object(retrieval, 'embedder', return_value=Embedder()):
    assert retrieval.booklet_index(Path('booklet.pdf'), pages).ntotal == 2
"""
        env = dict(os.environ, PYTHONPATH=str(Path(predict.__file__).resolve().parents[1]))
        processes = [subprocess.Popen([sys.executable, "-c", script, str(marker)], env=env,
                                      text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
        try:
            for process in processes:
                _, stderr = process.communicate(timeout=60)
                self.assertEqual(process.returncode, 0, stderr)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        self.assertEqual(marker.read_text().splitlines(), ["build"])

    def test_prepare_only_builds_cache_without_inference_or_output(self):
        source = self.root / "cases.jsonl"
        source.write_text(json.dumps({"id": "a", "vote": None,
                                     "claim": {"text": "second", "language": "de"},
                                     "booklet": {"path": str(self.pdf), "language": "de"}}) + "\n")
        model = RecordingEmbedder()
        with patch.object(predict, "_extract_pdf_pages", return_value=PAGES), \
                patch.object(retrieval, "embedder", return_value=model), \
                patch("claimcheck.nli.complete", side_effect=AssertionError("inference forbidden")) as complete, \
                contextlib.redirect_stderr(io.StringIO()):
            status = main(["--input", str(source), "--prepare-only"], predictor=predict.ApertusPredictor(top_k=1))
        self.assertEqual(status, 0)
        complete.assert_not_called()
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(list((self.cache / "indexes").glob("*.faiss"))), 1)

    def test_prepare_failure_is_nonzero_and_prediction_still_degrades_to_neutral(self):
        source = self.root / "cases.jsonl"
        source.write_text(json.dumps({"id": "a", "vote": None,
                                     "claim": {"text": "second", "language": "de"},
                                     "booklet": {"path": str(self.root / "missing.pdf"), "language": "de"}}) + "\n")
        predictor = predict.ApertusPredictor()
        output = self.root / "predictions.jsonl"
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--input", str(source), "--prepare-only"], predictor=predictor), 1)
            self.assertEqual(run(source, output, predictor), 1)
        self.assertEqual(json.loads(output.read_text())["label"], 1)

    def test_cli_prepares_all_unique_documents_before_classifying(self):
        second_pdf = self.root / "second.pdf"
        second_pdf.write_bytes(b"another PDF")
        paths = [self.pdf, second_pdf, self.pdf]
        cases = [{"id": number, "vote": None, "claim": {"text": "second", "language": "de"},
                  "booklet": {"path": str(path), "language": "de"}} for number, path in enumerate(paths)]
        source = self.root / "cases.jsonl"
        output = self.root / "predictions.jsonl"
        source.write_text("".join(json.dumps(case) + "\n" for case in cases))
        model = RecordingEmbedder()

        def complete(*args, **kwargs):
            self.assertEqual(extract.call_count, 2)  # Both PDFs prepared before the first claim.
            return {"text": '{"label":"NEUTRAL", "subject_addressed":false}',
                    "model": "test", "input_tokens": 10, "output_tokens": 5, "inference_time_ms": 1}

        with patch.object(predict, "_extract_pdf_pages", return_value=PAGES) as extract, \
                patch.object(retrieval, "embedder", return_value=model), \
                patch("claimcheck.nli.complete", side_effect=complete), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(run(source, output, predict.ApertusPredictor(top_k=1)), 3)
            self.clear_memory()
            self.assertEqual(run(source, output, predict.ApertusPredictor(top_k=1)), 3)
        self.assertEqual(extract.call_count, 2)
        # Identical extracted passages share an index, even across different PDFs.
        self.assertEqual(sum(call[0].startswith("passage:") for call in model.calls), 1)
        self.assertEqual(sum(call[0].startswith("query:") for call in model.calls), 6)
        self.assertEqual([json.loads(line)["id"] for line in output.read_text().splitlines()], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
