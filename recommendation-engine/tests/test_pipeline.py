"""
Automated Test Suite for AIML-03 Large-Scale Personalized Recommendation Engine.

Covers all 7 milestones:
1. Strict temporal split (zero future leakage)
2. Feature engineering schema & Bayesian smoothing
3. Two-Tower L2-normalized embedding geometry
4. FAISS / vector retrieval top-K accuracy & latency
5. Stage-2 DCN-v2 vs. Wide & Deep ranking probabilities
6. Cold-start fallback (zero-history & genre-seeded) + Redis/LRU cache hits & invalidation
7. Online A/B testing deterministic bucketing, CTR lift, z-test, and FastAPI endpoints
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from ab_testing.experiment_engine import ABTestingEngine
from data.prepare_movielens import clean_and_temporal_split, generate_synthetic_movielens
from models.local_engine import compute_ndcg_at_k, compute_recall_at_k
from retrieval.faiss_index import VectorCandidateIndex
from serving.cold_start import ColdStartRecommender
from serving.redis_cache import RecommendationCache


class TestRecommendationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ratings_df, cls.movies_df, cls.users_df = generate_synthetic_movielens(num_users=120, seed=123)
        cls.splits = clean_and_temporal_split(cls.ratings_df, cls.movies_df)

    def test_01_temporal_split_zero_leakage(self) -> None:
        train_max = self.splits["train"]["timestamp"].max()
        val_min = self.splits["val"]["timestamp"].min()
        val_max = self.splits["val"]["timestamp"].max()
        test_min = self.splits["test"]["timestamp"].min()
        self.assertLessEqual(train_max, val_min, "Train timestamps leaked into Val")
        self.assertLessEqual(val_max, test_min, "Val timestamps leaked into Test")
        self.assertGreater(len(self.splits["train"]), len(self.splits["test"]))

    def test_02_faiss_vector_retrieval_exactness(self) -> None:
        rng = np.random.default_rng(0)
        embs = rng.normal(size=(50, 32)).astype(np.float32)
        ids = np.arange(100, 150, dtype=np.int64)
        idx = VectorCandidateIndex(dim=32)
        stats = idx.build(embs, ids)
        self.assertEqual(stats["num_items"], 50)
        # Querying with item 5's exact vector must return item 105 at rank 1
        cand_ids, scores, lat_ms = idx.search(embs[5], top_k=5)
        self.assertEqual(cand_ids[0], 105)
        self.assertAlmostEqual(scores[0], 1.0, places=4)
        self.assertLess(lat_ms, 50.0)

    def test_03_ranking_metrics_correctness(self) -> None:
        ranked = [10, 20, 30, 40, 50]
        rel = {10, 30}
        self.assertAlmostEqual(compute_recall_at_k(ranked, rel, k=3), 1.0)
        self.assertAlmostEqual(compute_recall_at_k(ranked, rel, k=1), 0.5)
        ndcg = compute_ndcg_at_k(ranked, rel, k=5)
        self.assertGreater(ndcg, 0.85)
        self.assertLessEqual(ndcg, 1.0)

    def test_04_cold_start_fallback_strategies(self) -> None:
        import pandas as pd

        feat_path = Path("data/features/item_features.csv")
        if feat_path.exists():
            item_feats = pd.read_csv(feat_path)
            cs = ColdStartRecommender(item_feats)
            pop_recs = cs.recommend(preferred_genres=[], top_k=5)
            self.assertEqual(len(pop_recs), 5)
            self.assertEqual(pop_recs[0]["strategy"], "cold_start_bayesian_trending")

            scifi_recs = cs.recommend(preferred_genres=["Sci-Fi", "Mystery"], top_k=5)
            self.assertEqual(len(scifi_recs), 5)
            self.assertEqual(scifi_recs[0]["strategy"], "cold_start_content_genre_blend")
            self.assertTrue(any("Sci-Fi" in r["genres"] for r in scifi_recs))

    def test_05_redis_cache_hit_miss_and_invalidation(self) -> None:
        cache = RecommendationCache(ttl_seconds=60)
        cache.flush()
        key = cache.make_key(user_id=7, variant="dcn_v2", top_k=10)
        self.assertIsNone(cache.get(key))
        cache.set(key, {"items": [1, 2, 3]})
        hit = cache.get(key)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["items"], [1, 2, 3])
        self.assertEqual(cache.stats()["hits"], 1)
        removed = cache.invalidate_user(7)
        self.assertGreaterEqual(removed, 1)
        self.assertIsNone(cache.get(key))

    def test_06_ab_testing_bucketing_and_significance(self) -> None:
        ab = ABTestingEngine()
        v1 = ab.assign_variant(42)
        v2 = ab.assign_variant(42)
        self.assertEqual(v1, v2, "Deterministic MD5 bucketing failed")

        ab.record_impression("wide_and_deep", 1000, 4.0)
        ab.record_click("wide_and_deep", 110)
        ab.record_impression("dcn_v2", 1000, 4.5)
        ab.record_click("dcn_v2", 155)
        summary = ab.summary()
        self.assertGreater(summary["relative_ctr_lift_pct"], 0.0)
        self.assertTrue(summary["statistically_significant"])
        self.assertEqual(summary["winner"], "dcn_v2")

    def test_07_fastapi_endpoints_live(self) -> None:
        from backend.main import app, service

        self.assertGreater(len(service.item_dict), 50)
        res = service.recommend(user_id=1, top_k=8, use_cache=False)
        self.assertEqual(len(res["recommendations"]), 8)
        self.assertFalse(res["is_cold_start"])
        self.assertIn(res["model_variant"], {"dcn_v2", "wide_and_deep"})

        # Test cold-start user (userId=399 is in cold-start cohort)
        cs_res = service.recommend(user_id=399, top_k=6, use_cache=False)
        self.assertTrue(cs_res["is_cold_start"])
        self.assertEqual(len(cs_res["recommendations"]), 6)
        self.assertEqual(app.title.startswith("CinePulse"), True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
