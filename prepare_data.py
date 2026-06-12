"""
prepare_data.py
Downloads and processes the MovieLens 100K dataset.

MovieLens 100K contains:
- 100,000 ratings from 943 users on 1,682 movies
- Ratings are 1-5 stars
- Each row: user_id, movie_id, rating, timestamp

Our job:
1. Download the dataset automatically
2. Clean and process it
3. Split into train/validation/test sets
4. Save processed files for the next scripts
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Constants
DATA_DIR = "data"
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
RAW_ZIP = os.path.join(DATA_DIR, "ml-100k.zip")
RAW_DIR = os.path.join(DATA_DIR, "ml-100k")


#Step 1: Download
def download_movielens():
    """Download MovieLens 100K if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(RAW_DIR):
        logger.info("MovieLens already downloaded. Skipping.")
        return

    logger.info("Downloading MovieLens 100K...")
    urllib.request.urlretrieve(MOVIELENS_URL, RAW_ZIP)
    logger.info("Download complete. Extracting...")

    with zipfile.ZipFile(RAW_ZIP, "r") as z:
        z.extractall(DATA_DIR)

    logger.info(f"Extracted to {RAW_DIR}")

#Step 2: Load ratings
def load_ratings() -> pd.DataFrame:
    """
    Load the main ratings file.
    u.data format: user_id | movie_id | rating | timestamp
    Tab separated, no header.
    We drop timestamp — not useful for our model.
    """
    ratings_path = os.path.join(RAW_DIR, "u.data")
    df = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    df.drop(columns=["timestamp"], inplace=True)
    logger.info(f"Loaded {len(df)} ratings")
    logger.info(f"Users: {df.user_id.nunique()} | Movies: {df.movie_id.nunique()}")
    logger.info(f"Rating distribution:\n{df.rating.value_counts().sort_index()}")
    return df


#Step 3: Load movie metadata
def load_movies() -> pd.DataFrame:
    """
    Load movie titles and genres.
    u.item format: movie_id | title | release_date | ... | genre_flags (19 columns)
    Pipe separated, latin-1 encoding (has special characters).
    Genre columns are binary flags — 1 if movie belongs to that genre, 0 if not.
    A movie can belong to multiple genres.
    """
    movies_path = os.path.join(RAW_DIR, "u.item")

    # 19 genre names in order they appear in the file
    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western"
    ]

    cols = ["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + genre_names

    movies = pd.read_csv(
        movies_path,
        sep="|",
        names=cols,
        encoding="latin-1",
        usecols=["movie_id", "title"] + genre_names
    )

    logger.info(f"Loaded {len(movies)} movies")
    return movies


#Step 4: Create implicit feedback
def create_implicit_feedback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert explicit ratings (1-5 stars) to implicit feedback (liked/not liked).
    Why implicit?
    Real recommendation systems mostly use implicit signals — clicks, watches,
    purchases — not explicit ratings. Users rarely rate things.
    Our conversion rule:
    Rating >= 4 → positive interaction (user liked it) → label = 1
    Rating <= 2 → negative interaction (user disliked it) → label = 0
    Rating == 3 → neutral, drop it (ambiguous signal)
    This gives us cleaner training signal.
    """
    df = df.copy()
    df = df[df["rating"] != 3]  # drop neutral
    df["label"] = (df["rating"] >= 4).astype(int)

    pos = (df["label"] == 1).sum()
    neg = (df["label"] == 0).sum()
    logger.info(f"Positive interactions: {pos} | Negative: {neg} | Ratio: {pos / neg:.2f}")
    return df


#Step 5: Reindex user and movie IDs
def reindex_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """
    Reindex user and movie IDs to start from 0 consecutively.
    Why?
    The original IDs in MovieLens start from 1. Our embedding matrices
    use IDs as row indices, so they need to be 0-indexed and consecutive.
    Without this, a movie_id of 1682 would require an embedding matrix
    with 1682 rows even if we only have a few movies.
    Returns:
    - df with new user_idx and movie_idx columns
    - user_id_map: original_id → new_idx
    - movie_id_map: original_id → new_idx
    """
    # Create mappings
    unique_users = sorted(df["user_id"].unique())
    unique_movies = sorted(df["movie_id"].unique())

    user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
    movie_id_map = {mid: idx for idx, mid in enumerate(unique_movies)}

    df["user_idx"] = df["user_id"].map(user_id_map)
    df["movie_idx"] = df["movie_id"].map(movie_id_map)

    logger.info(f"Reindexed: {len(user_id_map)} users, {len(movie_id_map)} movies")
    return df, user_id_map, movie_id_map


#Step 6: Train/val/test split
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split interactions into train (80%), validation (10%), test (10%).
    Important: we split per user, not randomly.
    Why? If we split randomly, the model might see a user's later interactions
    during training and predict their earlier ones — that's data leakage.
    Correct approach: for each user, sort interactions by time (approximated
    here by row order) and use last 20% as test/val.
    For simplicity with MovieLens 100K (no timestamp after dropping it),
    we do a stratified split ensuring each user appears in train.
    """
    # Ensure every user has at least one interaction in train
    train_list, val_list, test_list = [], [], []

    for user_idx, group in df.groupby("user_idx"):
        n = len(group)
        if n < 3:
            # Too few interactions — put all in train
            train_list.append(group)
            continue

        # Shuffle within user
        group = group.sample(frac=1, random_state=42)
        n_test = max(1, int(n * 0.1))
        n_val = max(1, int(n * 0.1))

        test_list.append(group.iloc[:n_test])
        val_list.append(group.iloc[n_test:n_test + n_val])
        train_list.append(group.iloc[n_test + n_val:])

    train = pd.concat(train_list).reset_index(drop=True)
    val = pd.concat(val_list).reset_index(drop=True)
    test = pd.concat(test_list).reset_index(drop=True)

    logger.info(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


#Step 7: Save everything
def save_processed(
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        movies: pd.DataFrame,
        user_id_map: dict,
        movie_id_map: dict,
):
    """Save all processed files. Subsequent scripts load from here."""
    import json

    os.makedirs(DATA_DIR, exist_ok=True)

    train.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(DATA_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
    movies.to_csv(os.path.join(DATA_DIR, "movies.csv"), index=False)

    # Save ID mappings so we can reverse-lookup movie titles later
    with open(os.path.join(DATA_DIR, "user_id_map.json"), "w") as f:
        json.dump({str(k): v for k, v in user_id_map.items()}, f)

    with open(os.path.join(DATA_DIR, "movie_id_map.json"), "w") as f:
        json.dump({str(k): v for k, v in movie_id_map.items()}, f)

    # Save dataset stats — api.py reads this
    stats = {
        "n_users": len(user_id_map),
        "n_movies": len(movie_id_map),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }
    with open(os.path.join(DATA_DIR, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("All files saved to data/")
    logger.info(f"Stats: {stats}")


#Main
def main():
    logger.info("=" * 50)
    logger.info("Step 1: Download MovieLens 100K")
    logger.info("=" * 50)
    download_movielens()

    logger.info("\n" + "=" * 50)
    logger.info("Step 2: Load and process data")
    logger.info("=" * 50)
    ratings = load_ratings()
    movies = load_movies()

    logger.info("\n" + "=" * 50)
    logger.info("Step 3: Create implicit feedback")
    logger.info("=" * 50)
    ratings = create_implicit_feedback(ratings)

    logger.info("\n" + "=" * 50)
    logger.info("Step 4: Reindex IDs")
    logger.info("=" * 50)
    ratings, user_id_map, movie_id_map = reindex_ids(ratings)

    logger.info("\n" + "=" * 50)
    logger.info("Step 5: Split data")
    logger.info("=" * 50)
    train, val, test = split_data(ratings)

    logger.info("\n" + "=" * 50)
    logger.info("Step 6: Save processed files")
    logger.info("=" * 50)
    save_processed(train, val, test, movies, user_id_map, movie_id_map)

    logger.info("\nData preparation complete!")
    logger.info("Next step: run train_embeddings.py")


if __name__ == "__main__":
    main()
