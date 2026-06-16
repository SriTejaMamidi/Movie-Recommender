"""
train_embeddings.py
Learns user and movie embeddings using Neural Collaborative Filtering.
What this script does:
- Builds a PyTorch model with embedding tables for users and movies
- Trains it on interaction data (user liked/disliked movie)
- After training, every user and every movie has a dense vector (embedding)
- These embeddings are used by FAISS in the next step for fast retrieval
Architecture:
User ID → User Embedding (64-dim)  ┐
                                    ├→ Dot product → probability of interaction
Movie ID → Movie Embedding (64-dim) ┘
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Configs
EMBEDDING_DIM = 64  # size of each embedding vector
BATCH_SIZE = 1024  # interactions per training step
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5  # L2 regularisation — prevents overfitting
DATA_DIR = "data"
MODEL_DIR = "models"


#Dataset
class InteractionDataset(Dataset):
    """
    PyTorch Dataset for user-movie interactions.

    PyTorch needs a Dataset class that:
    - Knows how many examples there are (__len__)
    - Can return any single example by index (__getitem__)

    DataLoader then batches these automatically.
    """

    def __init__(self, df: pd.DataFrame):
        # Convert to tensors once at init — faster than converting per batch
        self.user_idx = torch.LongTensor(df["user_idx"].values)
        self.movie_idx = torch.LongTensor(df["movie_idx"].values)
        self.labels = torch.FloatTensor(df["label"].values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_idx[idx], self.movie_idx[idx], self.labels[idx]


#Model
class MatrixFactorization(nn.Module):
    """
    Neural Matrix Factorization model.

    Classic matrix factorization idea:
    - Each user has a latent vector (embedding) representing their taste
    - Each movie has a latent vector representing its characteristics
    - If user and movie embeddings point in the same direction → high match

    We extend classic MF with:
    - Bias terms (some users rate high, some movies are universally liked)
    - Sigmoid output (converts score to probability 0-1)
    - Dropout (prevents overfitting)

    Forward pass:
    user_embedding (64-dim) · movie_embedding (64-dim) = score
    score + user_bias + movie_bias = adjusted_score
    sigmoid(adjusted_score) = probability user likes movie
    """

    def __init__(self, n_users: int, n_movies: int, embedding_dim: int = 64):
        super().__init__()

        # Embedding tables — these are the learnable parameters
        # n_users × embedding_dim matrix — one row per user
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        # n_movies × embedding_dim matrix — one row per movie
        self.movie_embeddings = nn.Embedding(n_movies, embedding_dim)

        # Bias terms — scalar per user and per movie
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)

        # Dropout — randomly zeros 20% of embedding values during training
        # Forces model to not rely on any single dimension → better generalisation
        self.dropout = nn.Dropout(p=0.2)

        # Initialise embeddings with small random values
        # Without this, all embeddings start the same → symmetry breaking problem
        nn.init.normal_(self.user_embeddings.weight, mean=0, std=0.01)
        nn.init.normal_(self.movie_embeddings.weight, mean=0, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_idx: torch.Tensor, movie_idx: torch.Tensor) -> torch.Tensor:
        # Look up embeddings for the batch
        user_emb = self.dropout(self.user_embeddings(user_idx))  # (batch, 64)
        movie_emb = self.dropout(self.movie_embeddings(movie_idx))  # (batch, 64)

        # Dot product — element-wise multiply then sum across embedding dim
        # Measures alignment between user taste and movie characteristics
        dot = (user_emb * movie_emb).sum(dim=1)  # (batch,)

        # Add biases
        u_bias = self.user_bias(user_idx).squeeze()  # (batch,)
        m_bias = self.movie_bias(movie_idx).squeeze()  # (batch,)

        score = dot + u_bias + m_bias

        # Sigmoid → probability between 0 and 1
        return torch.sigmoid(score)

    def get_user_embeddings(self) -> np.ndarray:
        """Extract all user embeddings as numpy array for FAISS."""
        return self.user_embeddings.weight.detach().cpu().numpy()

    def get_movie_embeddings(self) -> np.ndarray:
        """Extract all movie embeddings as numpy array for FAISS."""
        return self.movie_embeddings.weight.detach().cpu().numpy()


#Training
def train_epoch(model, loader, optimizer, criterion, device) -> float:
    """One full pass through training data. Returns average loss."""
    model.train()
    total_loss = 0.0

    for user_idx, movie_idx, labels in loader:
        user_idx = user_idx.to(device)
        movie_idx = movie_idx.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # clear gradients from last step
        predictions = model(user_idx, movie_idx)
        loss = criterion(predictions, labels)
        loss.backward()  # compute gradients
        optimizer.step()  # update weights

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Evaluate on val/test set. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # no gradient computation needed for evaluation
        for user_idx, movie_idx, labels in loader:
            user_idx = user_idx.to(device)
            movie_idx = movie_idx.to(device)
            labels = labels.to(device)

            predictions = model(user_idx, movie_idx)
            loss = criterion(predictions, labels)
            total_loss += loss.item()

            # Accuracy: prediction > 0.5 → predicted positive
            predicted = (predictions > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


#Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))

    with open(os.path.join(DATA_DIR, "stats.json")) as f:
        stats = json.load(f)

    n_users = stats["n_users"]
    n_movies = stats["n_movies"]
    logger.info(f"Users: {n_users} | Movies: {n_movies}")

    # Create datasets and loaders
    train_dataset = InteractionDataset(train_df)
    val_dataset = InteractionDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model
    model = MatrixFactorization(n_users, n_movies, EMBEDDING_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Loss and optimiser
    # BCELoss = Binary Cross Entropy — standard loss for binary classification
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler — halves LR if val loss doesn't improve for 3 epochs
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Training loop
    best_val_loss = float("inf")
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("\nStarting training...")
    logger.info(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>10} | {'Val Acc':>8}")
    logger.info("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        logger.info(f"{epoch:>6} | {train_loss:>12.4f} | {val_loss:>10.4f} | {val_acc:>8.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "embeddings_model.pt"))
            logger.info(f"  → Saved best model (val_loss={val_loss:.4f})")

    # Load best model for embedding extraction
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "embeddings_model.pt")))
    model.eval()

    # Extract and save embeddings
    logger.info("\nExtracting embeddings...")
    user_embeddings = model.get_user_embeddings()  # shape: (n_users, 64)
    movie_embeddings = model.get_movie_embeddings()  # shape: (n_movies, 64)

    np.save(os.path.join(MODEL_DIR, "user_embeddings.npy"), user_embeddings)
    np.save(os.path.join(MODEL_DIR, "movie_embeddings.npy"), movie_embeddings)

    logger.info(f"User embeddings shape:  {user_embeddings.shape}")
    logger.info(f"Movie embeddings shape: {movie_embeddings.shape}")
    logger.info(f"Saved to {MODEL_DIR}/")
    logger.info("\nNext step: run build_index.py")


if __name__ == "__main__":
    main()
