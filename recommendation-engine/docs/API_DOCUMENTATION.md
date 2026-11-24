# REST API Documentation — CinePulse AI

Base URL (local): `http://127.0.0.1:8000` · Interactive OpenAPI/Swagger UI: `http://127.0.0.1:8000/docs`

## Endpoints

### `GET /api/health`
Returns runtime health, active FAISS backend, Redis cache backend, Triton connectivity, and catalog counts.

### `GET /api/users?limit=24`
Returns sample active and cold-start user profiles with learned top genres and deterministic A/B variant assignment.

### `GET /api/recommend/{user_id}`
Runs the full two-stage pipeline (or cold-start fallback) for `user_id`.
- **Query Params**:
  - `top_k` (int, default `12`): Final ranked recommendations returned.
  - `stage1_k` (int, default `60`): Candidate pool retrieved from FAISS.
  - `model` (optional: `dcn_v2` | `wide_and_deep`): Overrides deterministic A/B bucketing.
  - `use_cache` (bool, default `true`): Enables Redis/LRU cache lookup.
- **Response**: Includes `model_variant`, `serving_backend`, `cache_hit`, `latency_breakdown_ms`, and ranked `recommendations` with `stage1_score`, `final_score`, and human-readable `reason`.

### `POST /api/recommend/cold-start`
Generates instant cold-start recommendations for a new user.
```json
{ "preferred_genres": ["Sci-Fi", "Mystery"], "top_k": 10 }
```

### `GET /api/metrics`
Returns temporal split counts, offline benchmark metrics (NDCG@10, Recall@K, MRR, latencies), Redis cache hit-rate stats, and A/B test summary.

### `GET /api/ab-test/summary`
Returns live Control (`wide_and_deep`) vs. Treatment (`dcn_v2`) impressions, clicks, CTR, relative lift %, z-score, p-value, and statistical significance verdict.

### `POST /api/ab-test/event`
Logs an online user interaction event (`click`, `watch_complete`, `dismiss`) for real-time A/B evaluation.
```json
{ "user_id": 1, "movie_id": 4, "variant": "dcn_v2", "event_type": "click" }
```

### `POST /api/cache/flush`
Flushes cached recommendation entries in Redis / local LRU store.
