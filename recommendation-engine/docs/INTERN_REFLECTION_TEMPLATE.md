# Individual Intern Reflection — AIML-03: Large-Scale Personalized Recommendation Engine

- **Intern Name**: Badal Singh
- **Specialization Track**: AI / Machine Learning
- **Project**: AIML-03 — Large-Scale Personalized Recommendation Engine (CinePulse AI)
- **Timeline**: 29 October 2026 – 30 November 2026

---

## 1. What I Owned
- Designed and implemented the end-to-end two-stage recommendation architecture spanning temporal data preparation, Apache Spark feature engineering, Two-Tower retrieval, FAISS indexing, Stage-2 deep ranking (DCN-v2 vs. Wide & Deep), cold-start fallbacks, Redis caching, Triton Inference Server configs, Airflow retraining orchestration, and online A/B testing.
- Built the FastAPI serving layer and interactive CinePulse AI web UI to visualize Stage-1 vs. Stage-2 scoring, cache latency reductions, cold-start genre onboarding, and live A/B z-test convergence.

## 2. Key Technical Learnings
- **Why Temporal Splitting Matters**: Random train/test splitting leaks future popularity and user drift into training, inflating offline NDCG unrealistically. Enforcing strict timestamp cutoffs (`max(train.ts) <= min(val.ts) <= min(test.ts)`) gave trustworthy generalization numbers.
- **Negative Sampling in Stage-2 Ranking**: Training Stage-2 rankers solely on observed ratings caused the ranker to tie Stage-1 retrieval; adding 2x negative sampling of uninteracted items enabled DCN-v2 to lift NDCG@10 from `0.1910` (Stage-1) to `0.2328` (+21.9%).
- **Latency vs. Accuracy Trade-offs**: Separating FAISS ANN retrieval (`0.048 ms` P50) from DCN-v2 re-ranking (`0.086 ms` P50 total) and adding Redis 300s TTL caching (`< 0.02 ms` cache hit) keeps end-to-end serving well within real-time SLAs.

## 3. Challenges Faced & How I Resolved Them
- *[Add 1–2 personal notes from running the full MovieLens 25M dataset / Docker Compose on your machine before final submission]*

## 4. Next Steps & Future Improvements
- Add session-based sequential modeling (SASRec / Transformers4Rec) to capture short-term in-session intent shifts.
- Implement multi-objective ranking (balancing CTR, watch completion, and genre diversity via MMR re-ranking).
