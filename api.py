"""
api.py
Production FastAPI serving the two-stage recommendation pipeline.
Request flow:
1. User ID comes in via /recommend endpoint
2. Stage 1: FAISS retrieves top-100 candidate movies (< 5ms)
3. Stage 2: Neural ranker re-scores all 100 candidates (< 50ms)
4. Return top-N movies with titles, scores, and latency breakdown
Total latency target: < 100ms
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
import torch
import faiss
import joblib
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from train_ranker import NeuralRanker, HIDDEN_DIMS, DROPOUT
from train_embeddings import EMBEDDING_DIM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "models")
DATA_DIR = os.getenv("DATA_DIR", "data")

GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]

app = FastAPI(
    title="Movie Recommendation API",
    description="Two-stage recommendation: FAISS retrieval + Neural ranking",
    version="1.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

#Globals loaded at startup
faiss_index = None
user_embeddings = None
movie_embeddings = None
ranker_model = None
movies_df = None
movie_id_map = None
idx_to_movie_id = None
genre_lookup = None
stats = None
models_loaded = False
device = torch.device("cpu")  # CPU for serving — no GPU needed for inference


def load_all_models():
    global faiss_index, user_embeddings, movie_embeddings
    global ranker_model, movies_df, movie_id_map
    global idx_to_movie_id, genre_lookup, stats, models_loaded

    try:
        logger.info("Loading FAISS index...")
        faiss_index = faiss.read_index(os.path.join(MODEL_DIR, "faiss_index.bin"))

        logger.info("Loading embeddings...")
        user_embeddings = np.load(os.path.join(MODEL_DIR, "user_embeddings_norm.npy"))
        movie_embeddings = np.load(os.path.join(MODEL_DIR, "movie_embeddings.npy"))

        logger.info("Loading neural ranker...")
        with open(os.path.join(MODEL_DIR, "ranker_config.json")) as f:
            ranker_config = json.load(f)

        ranker_model = NeuralRanker(
            ranker_config["input_dim"],
            ranker_config["hidden_dims"],
            ranker_config["dropout"]
        ).to(device)
        ranker_model.load_state_dict(
            torch.load(os.path.join(MODEL_DIR, "ranker_model.pt"), map_location=device)
        )
        ranker_model.eval()

        logger.info("Loading metadata...")
        movies_df = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))

        with open(os.path.join(DATA_DIR, "movie_id_map.json")) as f:
            movie_id_map = json.load(f)

        with open(os.path.join(DATA_DIR, "stats.json")) as f:
            stats = json.load(f)

        # Precompute reverse map and genre lookup for fast access
        idx_to_movie_id = {v: int(k) for k, v in movie_id_map.items()}
        genre_lookup = {}
        for _, row in movies_df.iterrows():
            genre_lookup[int(row["movie_id"])] = row[GENRE_COLS].values.astype(np.float32)

        models_loaded = True
        logger.info("All models loaded successfully.")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        models_loaded = False


@app.on_event("startup")
def startup():
    load_all_models()


#Schemas
class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=0, description="User index (0-based)")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of recommendations")
    n_candidates: int = Field(default=100, ge=10, le=500, description="Stage 1 retrieval size")


class MovieRecommendation(BaseModel):
    rank: int
    movie_idx: int
    title: str
    genres: list[str]
    retrieval_score: float  # cosine similarity from FAISS
    ranking_score: float  # neural ranker probability


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[MovieRecommendation]
    latency_ms: dict  # breakdown: retrieval_ms, ranking_ms, total_ms
    n_candidates: int
    timestamp: str


#Stage 1: FAISS Retrieval
def retrieve_candidates(user_idx: int, n_candidates: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Stage 1: Fast retrieval using FAISS.

    Takes user embedding, finds top-n_candidates most similar movies.
    Returns (movie_indices, cosine_similarities).

    This is approximate — optimised for speed.
    Target: < 5ms for 1682 movies.
    """
    query = user_embeddings[user_idx:user_idx + 1]  # shape: (1, 64)
    distances, indices = faiss_index.search(query, k=n_candidates)
    return indices[0], distances[0]  # both shape: (n_candidates,)


#Stage 2: Neural Ranking
def rank_candidates(
        user_idx: int,
        movie_indices: np.ndarray,
        retrieval_scores: np.ndarray
) -> np.ndarray:
    """
    Stage 2: Re-score candidates with neural ranker.

    Builds feature vectors for all (user, movie) pairs,
    runs them through the ranker in one batch.
    Returns ranking scores for each candidate.

    Target: < 50ms for 100 candidates.
    """
    u_emb = movie_embeddings[user_idx]  # raw (unnormalised) user embedding

    feature_list = []
    for movie_idx in movie_indices:
        m_emb = movie_embeddings[int(movie_idx)]

        interaction = u_emb * m_emb
        cos_sim = float(
            np.dot(u_emb, m_emb) /
            (np.linalg.norm(u_emb) * np.linalg.norm(m_emb) + 1e-8)
        )
        original_id = idx_to_movie_id.get(int(movie_idx), -1)
        genre_vec = genre_lookup.get(original_id, np.zeros(19, dtype=np.float32))

        feature_list.append(np.concatenate([u_emb, m_emb, interaction, genre_vec, [cos_sim]]))

    X = torch.FloatTensor(np.array(feature_list)).to(device)  # (100, 212)

    with torch.no_grad():
        logits = ranker_model(X)
        scores = torch.sigmoid(logits).cpu().numpy()  # (100,)

    return scores


#Endpoints
@app.get("/health")
def health():
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "n_users": stats["n_users"] if stats else None,
        "n_movies": stats["n_movies"] if stats else None,
        "faiss_indexed": faiss_index.ntotal if faiss_index else 0,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if request.user_id >= stats["n_users"]:
        raise HTTPException(
            status_code=400,
            detail=f"user_id {request.user_id} out of range. Max: {stats['n_users'] - 1}"
        )

    t_total_start = time.time()

    # Stage 1: Retrieval
    t_ret_start = time.time()
    movie_indices, retrieval_scores = retrieve_candidates(request.user_id, request.n_candidates)
    retrieval_ms = (time.time() - t_ret_start) * 1000

    # Stage 2: Ranking
    t_rank_start = time.time()
    ranking_scores = rank_candidates(request.user_id, movie_indices, retrieval_scores)
    ranking_ms = (time.time() - t_rank_start) * 1000

    # Sort by ranking score and take top_k
    sorted_indices = np.argsort(ranking_scores)[::-1][:request.top_k]

    # Build response
    recommendations = []
    for rank, idx in enumerate(sorted_indices, start=1):
        movie_idx = int(movie_indices[idx])
        original_id = idx_to_movie_id.get(movie_idx, -1)
        movie_row = movies_df[movies_df["movie_id"] == original_id]

        title = movie_row["title"].values[0] if len(movie_row) > 0 else "Unknown"
        genres = [g for g in GENRE_COLS if len(movie_row) > 0 and movie_row[g].values[0] == 1]

        recommendations.append(MovieRecommendation(
            rank=rank,
            movie_idx=movie_idx,
            title=title,
            genres=genres,
            retrieval_score=round(float(retrieval_scores[idx]), 4),
            ranking_score=round(float(ranking_scores[idx]), 4),
        ))

    total_ms = (time.time() - t_total_start) * 1000

    return RecommendResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        latency_ms={
            "retrieval_ms": round(retrieval_ms, 2),
            "ranking_ms": round(ranking_ms, 2),
            "total_ms": round(total_ms, 2),
        },
        n_candidates=request.n_candidates,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/user/{user_id}/profile")
def user_profile(user_id: int):
    """Returns user embedding stats — useful for debugging."""
    if not models_loaded or user_id >= stats["n_users"]:
        raise HTTPException(status_code=400, detail="Invalid user_id")

    emb = user_embeddings[user_id]
    return {
        "user_id": user_id,
        "embedding_dim": len(emb),
        "embedding_norm": round(float(np.linalg.norm(emb)), 4),
        "top_dimensions": np.argsort(np.abs(emb))[::-1][:5].tolist(),
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
