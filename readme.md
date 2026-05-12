# Two-Stage Movie Recommendation System

Production recommendation system using FAISS retrieval + Neural ranking.

## Stack
`PyTorch` · `FAISS` · `FastAPI` · `MovieLens 100K` · `NumPy`

## Architecture

```
User ID
   ↓
[STAGE 1 — RETRIEVAL]
User Embedding (64-dim) → FAISS Index → Top-100 candidates
Latency: < 5ms

   ↓
[STAGE 2 — RANKING]
For each candidate:
[user_emb | movie_emb | interaction | genres | cosine_sim] → Neural Ranker
212-dim feature vector → 3 hidden layers → ranking score
Latency: < 50ms

   ↓
Top-10 recommendations with titles, genres, scores
Total latency: < 100ms
```

## Project Structure

```
movie_recommender/
├── src/
│   ├── prepare_data.py       # download + process MovieLens 100K
│   ├── train_embeddings.py   # learn user/movie embeddings (Matrix Factorization)
│   ├── build_index.py        # build FAISS index on movie embeddings
│   ├── train_ranker.py       # train neural ranker (Stage 2)
│   └── api.py                # FastAPI two-stage serving
├── models/                   # saved model artifacts
├── data/                     # processed dataset files
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt

# Run in order:
python src/prepare_data.py       # ~1 min
python src/train_embeddings.py   # ~5 min
python src/build_index.py        # ~10 sec
python src/train_ranker.py       # ~3 min
uvicorn src.api:app --port 8000  # serve

# Test
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 0, "top_k": 10, "n_candidates": 100}'
```

## Key Concepts

**Why two stages?**
Scoring all 1682 movies with a complex ranker takes too long.
Stage 1 uses fast embedding similarity to get 100 candidates (< 5ms).
Stage 2 uses a powerful neural network on only 100 items (< 50ms).
Total: < 100ms. Netflix/YouTube use this exact pattern at scale.

**Embeddings:** 64-dimensional vectors learned by matrix factorization.
Users who like similar movies get similar embeddings.
Movies liked by similar users get similar embeddings.

**FAISS:** Pre-built index on movie embeddings.
Finds top-100 nearest movie embeddings to a user embedding in milliseconds.
Uses cosine similarity (IndexFlatIP on normalised vectors).

**Neural Ranker:** 3-layer network with 212 input features:
user_emb (64) + movie_emb (64) + interaction (64) + genres (19) + cosine_sim (1)
Richer features than FAISS alone → better final ranking.

## Resume Bullet Points

- Built two-stage recommendation system: FAISS embedding retrieval (Stage 1) + neural ranker (Stage 2), serving top-10 recommendations in < 100ms
- Trained matrix factorization model on MovieLens 100K to learn 64-dimensional user and movie embeddings
- Built FAISS cosine similarity index on movie embeddings for sub-5ms candidate retrieval from 1682 movies
- Designed neural ranker with 212-dim feature vectors (embeddings + genre features + interaction terms) achieving AUC > 0.75
- Deployed full pipeline via FastAPI with latency breakdown per stage
