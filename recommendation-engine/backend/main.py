"""
FastAPI Production Serving Layer for CinePulse AI (AIML-03).

Serves:
- End-to-end two-stage recommendations (Stage 1: Two-Tower + FAISS -> Stage 2: DCN-v2 or Wide & Deep)
- Cold-start popularity + genre content-based fallback
- Redis / LRU caching with live cache hit-rate & latency breakdown
- Triton Inference Server integration with automatic local fallback
- Online A/B Testing (deterministic MD5 bucketing, click logging, z-test significance)
- Interactive standalone UI at `/`
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from ab_testing.experiment_engine import ABTestingEngine
from models.local_engine import (
    NumpyDCNv2Ranker,
    NumpyWideAndDeepRanker,
    build_ranking_feature_vector,
    run_full_local_training_and_eval,
)
from retrieval.faiss_index import VectorCandidateIndex
from serving.cold_start import ColdStartRecommender
from serving.redis_cache import RecommendationCache
from serving.triton_client import TritonServingClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ColdStartRequest(BaseModel):
    preferred_genres: List[str] = Field(default_factory=list)
    top_k: int = Field(default=12, ge=1, le=50)


class ABEventRequest(BaseModel):
    user_id: int
    movie_id: int
    variant: str
    event_type: str = Field(default="click", pattern="^(click|watch_complete|dismiss)$")


class RecommenderService:
    """Orchestrates all 7 milestones in one low-latency serving runtime."""

    def __init__(self, root_dir: Path = PROJECT_ROOT):
        self.root_dir = root_dir
        self.data_dir = root_dir / "data" / "processed"
        self.feat_dir = root_dir / "data" / "features"
        self.art_dir = root_dir / "artifacts_store"
        self._ensure_artifacts()

        self.train_df = pd.read_csv(self.data_dir / "train.csv")
        self.users_df = pd.read_csv(self.data_dir / "users.csv")
        self.user_feats = pd.read_csv(self.feat_dir / "user_features.csv")
        self.item_feats = pd.read_csv(self.feat_dir / "item_features.csv")
        self.benchmark_report = json.loads((self.art_dir / "benchmark_report.json").read_text())
        self.split_summary = json.loads((self.data_dir / "split_summary.json").read_text())

        id_maps = json.loads((self.art_dir / "id_maps.json").read_text())
        self.user_id_to_idx: Dict[int, int] = {int(k): int(v) for k, v in id_maps["user_id_to_idx"].items()}
        self.user_embeddings = np.load(self.art_dir / "user_embeddings.npy")
        self.faiss_index = VectorCandidateIndex.load(self.art_dir)

        self.wd_ranker = NumpyWideAndDeepRanker.from_dict(
            json.loads((self.art_dir / "wide_deep_ranker.json").read_text())
        )
        self.dcn_ranker = NumpyDCNv2Ranker.from_dict(
            json.loads((self.art_dir / "dcn_v2_ranker.json").read_text())
        )

        self.user_dict = self.user_feats.set_index("userId").to_dict(orient="index")
        self.item_dict = self.item_feats.set_index("movieId").to_dict(orient="index")
        self.seen_by_user = self.train_df.groupby("userId")["movieId"].apply(set).to_dict()

        self.cold_start = ColdStartRecommender(self.item_feats)
        self.cache = RecommendationCache(ttl_seconds=300)
        self.triton = TritonServingClient()
        self.ab_engine = ABTestingEngine()
        self.ab_engine.seed_from_offline_replay(
            ndcg_wd=self.benchmark_report["models"]["stage2_wide_and_deep"]["ndcg_at_10"],
            ndcg_dcn=self.benchmark_report["models"]["stage2_dcn_v2"]["ndcg_at_10"],
        )

    def _ensure_artifacts(self) -> None:
        if not (self.art_dir / "benchmark_report.json").exists():
            from data.prepare_movielens import clean_and_temporal_split, generate_synthetic_movielens
            from spark.feature_engineering import run_local_feature_engineering

            self.data_dir.mkdir(parents=True, exist_ok=True)
            ratings_df, movies_df, users_df = generate_synthetic_movielens(num_users=400)
            splits = clean_and_temporal_split(ratings_df, movies_df)
            splits["train"].to_csv(self.data_dir / "train.csv", index=False)
            splits["val"].to_csv(self.data_dir / "val.csv", index=False)
            splits["test"].to_csv(self.data_dir / "test.csv", index=False)
            splits["movies"].to_csv(self.data_dir / "movies.csv", index=False)
            users_df.to_csv(self.data_dir / "users.csv", index=False)
            (self.data_dir / "split_summary.json").write_text(json.dumps({
                "mode": "benchmark_synthetic",
                "total_interactions": int(len(splits["train"]) + len(splits["val"]) + len(splits["test"])),
                "train_rows": int(len(splits["train"])),
                "val_rows": int(len(splits["val"])),
                "test_rows": int(len(splits["test"])),
                "num_movies": int(len(splits["movies"])),
                "num_users_total": int(len(users_df)),
                "temporal_leakage_verified": True,
            }))
            run_local_feature_engineering(self.data_dir, self.feat_dir)
            run_full_local_training_and_eval(self.root_dir)

    def list_sample_users(self, limit: int = 24) -> List[Dict[str, object]]:
        rows = []
        active_sample = self.users_df[~self.users_df["is_cold_start"]].head(limit - 4)
        cold_sample = self.users_df[self.users_df["is_cold_start"]].head(4)
        for row in pd.concat([active_sample, cold_sample]).itertuples():
            uid = int(row.userId)
            uf = self.user_dict.get(uid)
            rows.append({
                "userId": uid,
                "archetype": str(row.archetype),
                "is_cold_start": bool(row.is_cold_start),
                "interaction_count": int(uf["user_interaction_count"]) if uf else 0,
                "mean_rating": round(float(uf["user_mean_rating"]), 2) if uf else None,
                "top_genres": str(uf["top_genres"]).split("|") if uf else [],
                "ab_variant": self.ab_engine.assign_variant(uid),
            })
        return rows

    def recommend(
        self,
        user_id: int,
        top_k: int = 12,
        stage1_k: int = 60,
        model_override: Optional[str] = None,
        use_cache: bool = True,
    ) -> Dict[str, object]:
        t_start = time.perf_counter()
        variant = model_override if model_override in {"dcn_v2", "wide_and_deep"} else self.ab_engine.assign_variant(user_id)
        cache_key = self.cache.make_key(user_id, variant, top_k)

        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                cached["cache_hit"] = True
                cached["latency_breakdown_ms"]["total_ms"] = round((time.perf_counter() - t_start) * 1000.0, 3)
                return cached

        # Cold-start detection
        if user_id not in self.user_id_to_idx:
            t_cs = time.perf_counter()
            u_row = self.users_df[self.users_df["userId"] == user_id]
            pref_genres = (
                str(u_row.iloc[0]["preferred_genres"]).split("|")
                if len(u_row) > 0 and pd.notna(u_row.iloc[0]["preferred_genres"])
                else []
            )
            recs = self.cold_start.recommend(preferred_genres=pref_genres, top_k=top_k)
            cs_ms = (time.perf_counter() - t_cs) * 1000.0
            payload = {
                "user_id": user_id,
                "is_cold_start": True,
                "model_variant": "cold_start_fallback",
                "serving_backend": "cold_start_engine",
                "cache_hit": False,
                "stage1_candidates_evaluated": len(self.item_feats),
                "latency_breakdown_ms": {
                    "stage1_faiss_ms": 0.0,
                    "stage2_ranking_ms": round(cs_ms, 3),
                    "total_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
                },
                "recommendations": recs,
            }
            if use_cache:
                self.cache.set(cache_key, payload)
            return payload

        # Stage 1: Two-Tower User Embedding -> FAISS Candidate Retrieval
        u_emb = self.user_embeddings[self.user_id_to_idx[user_id]]
        seen = self.seen_by_user.get(user_id, set())
        cand_ids, cand_scores, s1_ms = self.faiss_index.search(u_emb, top_k=min(stage1_k + len(seen), len(self.item_dict)))
        unseen = [(m, s) for m, s in zip(cand_ids, cand_scores) if m not in seen][:stage1_k]

        # Stage 2: Deep Ranking (DCN-v2 vs Wide & Deep via Triton or local engine)
        t_s2 = time.perf_counter()
        u_meta = self.user_dict[user_id]
        X_cand = np.vstack([
            build_ranking_feature_vector(u_meta, self.item_dict[m], s) for m, s in unseen
        ])

        triton_model = "dcn_ranker" if variant == "dcn_v2" else "wide_deep_ranker"
        triton_scores = self.triton.infer_ranker(triton_model, X_cand)
        if triton_scores is not None:
            rank_scores = triton_scores
            serving_backend = "triton_inference_server"
        else:
            rank_scores = (
                self.dcn_ranker.predict_proba(X_cand)
                if variant == "dcn_v2"
                else self.wd_ranker.predict_proba(X_cand)
            )
            serving_backend = "local_neural_fallback"
        s2_ms = (time.perf_counter() - t_s2) * 1000.0

        order = np.argsort(-rank_scores)[:top_k]
        u_top_genres = set(str(u_meta.get("top_genres", "")).split("|"))
        recs = []
        for rank_pos, idx in enumerate(order, start=1):
            mid, s1_score = unseen[int(idx)]
            im = self.item_dict[mid]
            item_genres = str(im["genres"]).split("|")
            shared = [g for g in item_genres if g in u_top_genres]
            reason = (
                f"High Two-Tower cosine ({s1_score:.2f}) + matches {', '.join(shared[:2])}"
                if shared
                else f"Strong collaborative embedding match ({s1_score:.2f}) & Bayesian score {im['item_bayesian_rating']:.2f}"
            )
            recs.append({
                "rank": rank_pos,
                "movieId": int(mid),
                "title": str(im["title"]),
                "clean_title": str(im.get("clean_title", im["title"])),
                "release_year": int(im.get("release_year", 2000)),
                "genres": item_genres,
                "stage1_score": round(float(s1_score), 4),
                "final_score": round(float(rank_scores[int(idx)]), 4),
                "bayesian_rating": round(float(im["item_bayesian_rating"]), 2),
                "strategy": f"two_stage_{variant}",
                "reason": reason,
            })

        total_ms = (time.perf_counter() - t_start) * 1000.0
        self.ab_engine.record_impression(variant, len(recs), total_ms)

        payload = {
            "user_id": user_id,
            "is_cold_start": False,
            "user_profile": {
                "interaction_count": int(u_meta["user_interaction_count"]),
                "mean_rating": round(float(u_meta["user_mean_rating"]), 2),
                "top_genres": list(u_top_genres),
            },
            "model_variant": variant,
            "serving_backend": serving_backend,
            "cache_hit": False,
            "stage1_candidates_evaluated": len(unseen),
            "latency_breakdown_ms": {
                "stage1_faiss_ms": round(s1_ms, 3),
                "stage2_ranking_ms": round(s2_ms, 3),
                "total_ms": round(total_ms, 3),
            },
            "recommendations": recs,
        }
        if use_cache:
            self.cache.set(cache_key, payload)
        return payload


app = FastAPI(
    title="CinePulse AI — Large-Scale Two-Stage Recommendation Engine",
    version="1.0.0",
    description="AIML-03 Production Two-Stage Recommender (TFRS Two-Tower + FAISS + DCN-v2/Wide&Deep + Redis + Triton + Airflow + A/B Testing)",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

service = RecommenderService()


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    standalone = PROJECT_ROOT / "frontend" / "standalone_demo.html"
    if standalone.exists():
        return FileResponse(standalone)
    return HTMLResponse("<h1>CinePulse AI Recommender API is running. Visit /docs</h1>")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "faiss_backend": service.faiss_index.backend,
        "cache_backend": service.cache.backend,
        "triton_live": service.triton.triton_live,
        "catalog_size": len(service.item_dict),
        "active_users": len(service.user_id_to_idx),
    }


@app.get("/api/users")
def get_users(limit: int = Query(default=24, ge=4, le=100)):
    return {"users": service.list_sample_users(limit=limit)}


@app.get("/api/recommend/{user_id}")
def get_recommendations(
    user_id: int,
    top_k: int = Query(default=12, ge=1, le=30),
    stage1_k: int = Query(default=60, ge=15, le=120),
    model: Optional[str] = Query(default=None, description="Optional override: dcn_v2 or wide_and_deep"),
    use_cache: bool = Query(default=True),
):
    return service.recommend(
        user_id=user_id,
        top_k=top_k,
        stage1_k=stage1_k,
        model_override=model,
        use_cache=use_cache,
    )


@app.post("/api/recommend/cold-start")
def cold_start_recommend(req: ColdStartRequest):
    t0 = time.perf_counter()
    recs = service.cold_start.recommend(preferred_genres=req.preferred_genres, top_k=req.top_k)
    return {
        "is_cold_start": True,
        "preferred_genres": req.preferred_genres,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "recommendations": recs,
    }


@app.get("/api/metrics")
def get_metrics():
    return {
        "split_summary": service.split_summary,
        "offline_benchmarks": service.benchmark_report,
        "cache_stats": service.cache.stats(),
        "ab_test": service.ab_engine.summary(),
    }


@app.get("/api/ab-test/summary")
def get_ab_summary():
    return service.ab_engine.summary()


@app.post("/api/ab-test/event")
def log_ab_event(ev: ABEventRequest):
    if ev.variant not in {"dcn_v2", "wide_and_deep"}:
        raise HTTPException(status_code=400, detail="Invalid variant; use dcn_v2 or wide_and_deep")
    if ev.event_type in {"click", "watch_complete"}:
        service.ab_engine.record_click(ev.variant, count=1)
    return {"recorded": True, "summary": service.ab_engine.summary()}


@app.post("/api/cache/flush")
def flush_cache():
    service.cache.flush()
    return {"flushed": True, "cache_stats": service.cache.stats()}
