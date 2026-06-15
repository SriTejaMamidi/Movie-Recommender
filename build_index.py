"""
build_index.py
Builds a FAISS index on movie embeddings for fast nearest-neighbour retrieval.
This is Stage 1 of the two-stage recommendation pipeline.
What FAISS does:
- Takes all movie embeddings (1682 movies × 64 dimensions)
- Builds an index structure that allows finding the K most similar
  movies to any query embedding in milliseconds
- Without FAISS: scan all 1682 movies linearly every query
- With FAISS: pre-built index returns top-K in microseconds
For MovieLens 1682 movies, IndexFlatL2 (exact search) is fine.
For 10M+ items (production), you'd use IndexIVFFlat (approximate).
"""

import os
import json
import logging
import numpy as np
import faiss
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = "models"
DATA_DIR = "data"


#Normalising the  embeddings
def normalise_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2 normalise embeddings so each vector has unit length (magnitude = 1).

    Why normalise?
    After normalisation, dot product = cosine similarity.
    Cosine similarity measures the ANGLE between vectors — direction only.
    This is better than raw dot product which also depends on magnitude.

    Example:
    User A: [0.6, 0.8] magnitude=1.0 (already unit)
    Movie X: [0.6, 0.8] → cosine similarity = 1.0 (perfect match)
    Movie Y: [0.8, 0.6] → cosine similarity = 0.96 (close match)
    Movie Z: [-0.6, -0.8] → cosine similarity = -1.0 (opposite taste)

    FAISS IndexFlatIP (inner product) on normalised vectors
    = cosine similarity search.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)  # prevent division by zero
    return (embeddings / norms).astype(np.float32)


#Building the FAISS index
def build_faiss_index(movie_embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS index on movie embeddings.

    Index types:

    IndexFlatL2 — exact L2 distance search
    - Checks every single vector
    - 100% accurate — no approximation
    - Good for small datasets (< 100K items)
    - We use this for MovieLens (1682 movies)

    IndexFlatIP — exact inner product (dot product) search
    - On normalised vectors: equivalent to cosine similarity
    - We use this after normalising embeddings

    IndexIVFFlat — approximate search (for production scale)
    - Clusters vectors into groups (cells)
    - At query time, only searches nearby clusters
    - Much faster at 1M+ items, slight accuracy loss
    - Would use this for Netflix-scale (millions of movies)

    IndexHNSW — graph-based approximate search
    - Best accuracy/speed tradeoff for very large datasets
    - Used by companies like Spotify, Pinterest at scale
    """
    embedding_dim = movie_embeddings.shape[1]  # 64

    # IndexFlatIP = exact inner product search
    # On unit-normalised vectors this equals cosine similarity
    index = faiss.IndexFlatIP(embedding_dim)

    # Wrap with IDMap so we can use movie indices as IDs
    # Without IDMap, FAISS assigns sequential IDs 0,1,2...
    # With IDMap, we can assign our own movie_idx as the ID
    # This lets us directly get movie_idx back from search results
    index_with_ids = faiss.IndexIDMap(index)

    # Add all movie embeddings to the index
    # ids = array of movie indices [0, 1, 2, ..., 1681]
    ids = np.arange(len(movie_embeddings)).astype(np.int64)
    index_with_ids.add_with_ids(movie_embeddings, ids)

    logger.info(f"FAISS index built: {index_with_ids.ntotal} vectors indexed")
    return index_with_ids


# Verifying index works
def verify_index(
        index: faiss.Index,
        user_embeddings: np.ndarray,
        movie_embeddings: np.ndarray,
        movies_df: pd.DataFrame,
        movie_id_map: dict,
        n_test_users: int = 3
):
    """
    Quick sanity check — retrieves top-10 movies for a few users
    and prints results so we can visually verify they look reasonable.
    """
    # Reverse movie_id_map: new_idx → original_movie_id
    idx_to_movie_id = {v: int(k) for k, v in movie_id_map.items()}

    logger.info("\n--- Retrieval Verification ---")
    for user_idx in range(n_test_users):
        query = user_embeddings[user_idx:user_idx + 1]  # shape: (1, 64)

        # Search: returns distances and movie indices
        distances, movie_indices = index.search(query, k=10)

        logger.info(f"\nUser {user_idx} — Top 10 retrieved movies:")
        for rank, (movie_idx, dist) in enumerate(
                zip(movie_indices[0], distances[0]), start=1
        ):
            original_id = idx_to_movie_id.get(int(movie_idx), -1)
            movie_row = movies_df[movies_df["movie_id"] == original_id]
            title = movie_row["title"].values[0] if len(movie_row) > 0 else "Unknown"
            logger.info(f"  {rank:2}. [{dist:.4f}] {title}")


# Main
def main():
    # Load embeddings
    logger.info("Loading embeddings...")
    user_embeddings = np.load(os.path.join(MODEL_DIR, "user_embeddings.npy"))
    movie_embeddings = np.load(os.path.join(MODEL_DIR, "movie_embeddings.npy"))
    logger.info(f"User embeddings:  {user_embeddings.shape}")
    logger.info(f"Movie embeddings: {movie_embeddings.shape}")

    # Load movies metadata and ID map for verification
    movies_df = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    with open(os.path.join(DATA_DIR, "movie_id_map.json")) as f:
        movie_id_map = json.load(f)

    # Normalise
    logger.info("\nNormalising embeddings...")
    user_embeddings_norm = normalise_embeddings(user_embeddings)
    movie_embeddings_norm = normalise_embeddings(movie_embeddings)

    # Save normalised user embeddings — used at query time
    np.save(
        os.path.join(MODEL_DIR, "user_embeddings_norm.npy"),
        user_embeddings_norm
    )

    # Build FAISS index on normalised movie embeddings
    logger.info("\nBuilding FAISS index...")
    index = build_faiss_index(movie_embeddings_norm)

    # Save index to disk
    index_path = os.path.join(MODEL_DIR, "faiss_index.bin")
    faiss.write_index(index, index_path)
    logger.info(f"Index saved to {index_path}")

    # Verify
    verify_index(
        index,
        user_embeddings_norm,
        movie_embeddings_norm,
        movies_df,
        movie_id_map
    )

    logger.info("\nFAISS index built successfully!")
    logger.info("Next step: run train_ranker.py")


if __name__ == "__main__":
    main()
