"""
Milestone 6: Cold-Start Recommendation Fallback Engine.

Handles two cold-start regimes:
1. Zero-history users (no onboarding genres): Bayesian-weighted popularity blended with
   recency-decayed trending score (prevents stale classics from monopolizing new users).
2. Onboarding-seeded users (selected 1-3 preferred genres): Content-based TF-IDF/cosine
   similarity over multi-hot genre vectors blended 70/30 with Bayesian quality.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class ColdStartRecommender:
    """Popularity + Content-Based fallback for new users and sparse histories."""

    def __init__(self, item_features: pd.DataFrame):
        self.item_features = item_features.copy()
        self.genre_cols = [c for c in item_features.columns if c.startswith("genre_")]
        self.genre_names = [c.replace("genre_", "") for c in self.genre_cols]

        # Precompute normalized genre matrix (TF-IDF weighted by inverse genre frequency)
        g_mat = self.item_features[self.genre_cols].to_numpy(dtype=np.float32)
        df_counts = np.maximum(g_mat.sum(axis=0), 1.0)
        idf = np.log((len(self.item_features) + 1.0) / df_counts) + 1.0
        tfidf = g_mat * idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        self.item_tfidf = tfidf / np.where(norms == 0, 1.0, norms)
        self.idf = idf

        # Normalize Bayesian rating & trending score to [0, 1]
        bayes = self.item_features["item_bayesian_rating"].to_numpy(dtype=np.float32)
        trend = self.item_features["item_trending_score"].to_numpy(dtype=np.float32)
        self.norm_bayes = (bayes - bayes.min()) / max(float(bayes.max() - bayes.min()), 1e-6)
        self.norm_trend = (trend - trend.min()) / max(float(trend.max() - trend.min()), 1e-6)
        self.popularity_score = 0.65 * self.norm_bayes + 0.35 * self.norm_trend

    def recommend(
        self,
        preferred_genres: Optional[List[str]] = None,
        top_k: int = 12,
        exclude_ids: Optional[set] = None,
    ) -> List[Dict[str, object]]:
        """Generate cold-start recommendations with explicit strategy attribution."""
        exclude_ids = exclude_ids or set()
        preferred = [g.strip() for g in (preferred_genres or []) if g.strip() in self.genre_names]

        if preferred:
            q_vec = np.zeros(len(self.genre_names), dtype=np.float32)
            for g in preferred:
                q_vec[self.genre_names.index(g)] = 1.0
            q_vec = q_vec * self.idf
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm
            content_sim = (self.item_tfidf @ q_vec).ravel()
            final_scores = 0.70 * content_sim + 0.30 * self.popularity_score
            strategy = "cold_start_content_genre_blend"
        else:
            content_sim = np.zeros(len(self.item_features), dtype=np.float32)
            final_scores = self.popularity_score
            strategy = "cold_start_bayesian_trending"

        order = np.argsort(-final_scores)
        results: List[Dict[str, object]] = []
        for idx in order:
            row = self.item_features.iloc[int(idx)]
            mid = int(row["movieId"])
            if mid in exclude_ids:
                continue
            results.append({
                "movieId": mid,
                "title": str(row["title"]),
                "clean_title": str(row.get("clean_title", row["title"])),
                "release_year": int(row.get("release_year", 2000)),
                "genres": str(row["genres"]).split("|"),
                "stage1_score": round(float(content_sim[idx]), 4),
                "final_score": round(float(final_scores[idx]), 4),
                "bayesian_rating": round(float(row["item_bayesian_rating"]), 2),
                "strategy": strategy,
                "reason": (
                    f"Matches onboarding genres ({', '.join(preferred)}) + top Bayesian quality"
                    if preferred
                    else "Top trending & Bayesian-weighted audience favorite (zero-history fallback)"
                ),
            })
            if len(results) >= top_k:
                break
        return results
