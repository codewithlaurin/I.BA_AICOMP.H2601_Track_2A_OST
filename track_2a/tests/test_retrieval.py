import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from claimcheck import retrieval, sparse


class BM25Tests(unittest.TestCase):
    def test_unicode_words_case_and_swiss_numbers(self):
        self.assertEqual(sparse.tokenize("Straße ÉCOLE pensione 13’000 4,5%"),
                         ["strasse", "école", "pensione", "13000", "4.5"])
        pages = ({"text": "Straße école pensione 13'000 4.5"}, {"text": "other content"})
        index = sparse.BM25Index.build(pages)
        for query in ("STRASSE", "ÉCOLE", "PENSIONE", "13’000", "4,5"):
            with self.subTest(query=query):
                self.assertEqual([page for page, _ in index.search(query, 5)], [0])

    def test_relevance_length_normalization_and_common_terms(self):
        pages = ({"text": "pension pension"}, {"text": "pension " + "unrelated " * 40},
                 {"text": "pension"}, {"text": "different subject"})
        hits = sparse.BM25Index.build(pages).search("pension", 4)
        self.assertEqual([index for index, _ in hits], [0, 2, 1])
        self.assertTrue(all(score > 0 for _, score in hits))  # Also when df exceeds half the corpus.

    def test_no_shared_terms_or_empty_pages_produce_no_candidates(self):
        self.assertEqual(sparse.BM25Index.build(({"text": "Bundesrat"},)).search("consiglio", 3), [])
        self.assertEqual(sparse.BM25Index.build(({"text": ""},)).search("topic", 3), [])
        self.assertEqual(sparse.BM25Index.build(()).search("topic", 3), [])


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        env = patch.dict(os.environ, {"CLAIMCHECK_CACHE_DIR": str(self.cache),
                                      "CLAIMCHECK_RETRIEVAL_CANDIDATES": "32", "CLAIMCHECK_RRF_CONSTANT": "60"})
        env.start()
        self.addCleanup(env.stop)
        sparse._INDEXES.clear()
        self.addCleanup(sparse._INDEXES.clear)

    def test_rrf_promotes_agreement_without_summing_raw_scores(self):
        fused = retrieval.reciprocal_rank_fusion([[0, 1, 2], [1, 2, 3]])
        self.assertEqual([page for page, _ in fused], [1, 2, 0, 3])
        self.assertAlmostEqual(fused[0][1], 1 / 62 + 1 / 61)
        self.assertEqual(retrieval.reciprocal_rank_fusion([[4, 2], []]), [(4, 1 / 61), (2, 1 / 62)])

    def test_rrf_deduplicates_per_retriever_and_keeps_dense_order_on_ties(self):
        fused = retrieval.reciprocal_rank_fusion([[2, 2, 0], [0, 2]])
        self.assertEqual([page for page, _ in fused], [2, 0])
        self.assertAlmostEqual(fused[0][1], 1 / 61 + 1 / 62)
        self.assertEqual(fused[0][1], fused[1][1])

    def test_sparse_exact_number_rescues_page_beyond_dense_truncation(self):
        pages = ({"page": 3, "text": "climate taxation"}, {"page": 7, "text": "economy jobs"},
                 {"page": 12, "text": "background " * retrieval.PASSAGE_CHARS + "13'000"})
        dense = Mock()
        dense.search.return_value = (np.array([[0.9, 0.8, -0.2]]), np.array([[0, 1, 2]]))
        model = Mock()
        model.encode.return_value = np.array([[1.0, 0.0]])
        with patch.object(retrieval, "booklet_index", return_value=dense), \
                patch.object(retrieval, "embedder", return_value=model):
            result = retrieval.top_pages(self.root / "booklet.pdf", pages, "13’000", None, k=1)
        self.assertEqual([page["page"] for page in result], [12])
        self.assertAlmostEqual(result[0]["score"], 1 / 63 + 1 / 61)
        self.assertEqual(dense.search.call_args.args[1], 3)  # More candidates than the final top-1.
        model.encode.assert_called_once_with(["query: 13’000"])

    def test_cross_language_no_lexical_match_preserves_dense_ranking(self):
        pages = tuple({"page": number, "text": "Bundesrat"} for number in (3, 7, 12))
        dense = Mock()
        dense.search.return_value = (np.array([[0.9, 0.8, 0.7]]), np.array([[1, 2, 0]]))
        model = Mock()
        model.encode.return_value = np.array([[1.0, 0.0]])
        with patch.object(retrieval, "booklet_index", return_value=dense), \
                patch.object(retrieval, "embedder", return_value=model):
            result = retrieval.top_pages(self.root / "booklet.pdf", pages, "consiglio federale", None, k=2)
        self.assertEqual([page["page"] for page in result], [7, 12])
        self.assertEqual([page["score"] for page in result], [1 / 61, 1 / 62])

    def test_vote_title_is_used_by_both_searches_and_window_covers_top_k(self):
        pages = tuple({"page": number, "text": "generic text"} for number in (3, 7, 12))
        dense = Mock()
        dense.search.return_value = (np.array([[0.9, 0.8]]), np.array([[0, 1]]))
        bm25 = Mock()
        bm25.search.return_value = [(2, 4.0)]
        model = Mock()
        model.encode.return_value = np.array([[1.0, 0.0]])
        with patch.object(retrieval, "booklet_index", return_value=dense), \
                patch.object(retrieval, "booklet_bm25", return_value=bm25), \
                patch.object(retrieval, "embedder", return_value=model), \
                patch.dict(os.environ, {"CLAIMCHECK_RETRIEVAL_CANDIDATES": "1"}):
            result = retrieval.top_pages(self.root / "booklet.pdf", pages, "claim", "vote title", k=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(len({page["page"] for page in result}), 2)
        self.assertEqual(dense.search.call_args.args[1], 2)
        bm25.search.assert_called_once_with("claim vote title", 2)
        model.encode.assert_called_once_with(["query: claim vote title"])

    def test_small_booklet_bypasses_both_indexes(self):
        pages = ({"page": 1, "text": "short"},)
        with patch.object(retrieval, "booklet_index", side_effect=AssertionError("dense unnecessary")), \
                patch.object(retrieval, "booklet_bm25", side_effect=AssertionError("sparse unnecessary")):
            self.assertEqual(retrieval.top_pages(self.root / "booklet.pdf", pages, "claim", None, 2),
                             [{"page": 1, "text": "short", "score": None}])

    def test_sparse_index_survives_restart_without_retokenizing_documents(self):
        pages = ({"page": 2, "text": "climate"}, {"page": 7, "text": "pension"})
        self.assertEqual(sparse.booklet_bm25(pages).search("pension", 1)[0][0], 1)
        script = """
from unittest.mock import patch
from claimcheck.sparse import BM25Index, booklet_bm25
pages = ({'page': 2, 'text': 'climate'}, {'page': 7, 'text': 'pension'})
with patch.object(BM25Index, 'build', side_effect=AssertionError('document tokenization forbidden')):
    assert booklet_bm25(pages).search('pension', 1)[0][0] == 1
"""
        env = dict(os.environ, PYTHONPATH=str(Path(retrieval.__file__).resolve().parents[1]))
        result = subprocess.run([sys.executable, "-c", script], env=env, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sparse_cache_changes_with_pages_and_recovers_from_corruption(self):
        pages = ({"page": 2, "text": "climate"}, {"page": 7, "text": "pension"})
        with patch.object(sparse.BM25Index, "build", wraps=sparse.BM25Index.build) as build:
            sparse.booklet_bm25(pages)
            sparse.booklet_bm25(pages)
            sparse._INDEXES.clear()
            sparse.booklet_bm25(pages)
            self.assertEqual(build.call_count, 1)
            for file in (self.cache / "sparse").glob("*.bm25.json"):
                file.write_text("{broken json")
            sparse._INDEXES.clear()
            self.assertEqual(sparse.booklet_bm25(pages).search("pension", 1)[0][0], 1)
            self.assertEqual(build.call_count, 2)
            changed = ({"page": 2, "text": "pension"}, {"page": 7, "text": "climate"})
            self.assertEqual(sparse.booklet_bm25(changed).search("pension", 1)[0][0], 0)
            self.assertEqual(build.call_count, 3)

    def test_deleted_sparse_file_restores_from_memory_without_tokenizing(self):
        pages = ({"page": 1, "text": "pension"},)
        index = sparse.booklet_bm25(pages)
        next((self.cache / "sparse").glob("*.bm25.json")).unlink()
        with patch.object(sparse.BM25Index, "build", side_effect=AssertionError("no retokenization")):
            self.assertIs(sparse.booklet_bm25(pages), index)
            self.assertEqual(len(list((self.cache / "sparse").glob("*.bm25.json"))), 1)


if __name__ == "__main__":
    unittest.main()
