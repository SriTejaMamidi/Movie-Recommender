"""
train_ranker.py
Stage 2: Neural Ranker
After Stage 1 retrieves 100 candidate movies for a user,
Stage 2 re-scores those candidates with a more powerful model
that uses richer features than just embedding similarity.
Why a separate ranker?
- Stage 1 (FAISS) uses only embedding dot product — fast but limited
- Stage 2 can use: user features + movie features + interaction features
- The ranker sees fewer items (100 not 1M) so can afford more computation
- This is the same pattern used by YouTube, Netflix, Airbnb
Ranker architecture:
[user_embedding | movie_embedding | genre_features | interaction_features]
→ 3 hidden layers with BatchNorm + Dropout
→ single score (probability user likes movie)
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
from sklearn.metrics import roc_auc_score
import joblib
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Config
EMBEDDING_DIM = 64
HIDDEN_DIMS = [256, 128, 64]  # three hidden layers, getting smaller
BATCH_SIZE = 512
EPOCHS = 15
LEARNING_RATE = 0.001
DROPOUT = 0.3
DATA_DIR = "data"
MODEL_DIR = "models"

GENRE_COLS = [
    "unknown", "Action", "Adventure", "Animation", "Children",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]


#Feature engineering
def build_ranker_features(
        df: pd.DataFrame,
        user_embeddings: np.ndarray,
        movie_embeddings: np.ndarray,
        movies_df: pd.DataFrame,
        movie_id_map: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build rich feature vectors for each (user, movie) pair.

    Feature vector = concatenation of:
    1. User embedding (64-dim) — who the user is
    2. Movie embedding (64-dim) — what the movie is
    3. Element-wise product of embeddings (64-dim) — interaction signal
       (captures which specific dimensions of taste align)
    4. Genre features (19-dim) — movie genres as binary flags
    5. Cosine similarity (1-dim) — scalar alignment score

    Total: 64 + 64 + 64 + 19 + 1 = 212 features per pair

    Why element-wise product?
    Dot product gives one number — loses information about WHICH
    dimensions matched. Element-wise product keeps all 64 dimensions
    showing exactly where user taste and movie characteristics align.
    """
    # Reverse movie_id_map for lookup
    idx_to_movie_id = {v: int(k) for k, v in movie_id_map.items()}

    # Build movie_idx → genre vector mapping
    genre_lookup = {}
    for _, row in movies_df.iterrows():
        genre_vec = row[GENRE_COLS].values.astype(np.float32)
        genre_lookup[int(row["movie_id"])] = genre_vec

    features_list = []
    labels_list = []

    for _, row in df.iterrows():
        u_idx = int(row["user_idx"])
        m_idx = int(row["movie_idx"])

        u_emb = user_embeddings[u_idx]  # (64,)
        m_emb = movie_embeddings[m_idx]  # (64,)

        # Interaction features
        interaction = u_emb * m_emb  # element-wise product (64,)

        # Cosine similarity
        cos_sim = float(
            np.dot(u_emb, m_emb) /
            (np.linalg.norm(u_emb) * np.linalg.norm(m_emb) + 1e-8)
        )

        # Genre features
        original_movie_id = idx_to_movie_id.get(m_idx, -1)
        genre_vec = genre_lookup.get(original_movie_id, np.zeros(19, dtype=np.float32))

        # Concatenate all features
        feature_vec = np.concatenate([
            u_emb,
            m_emb,
            interaction,
            genre_vec,
            [cos_sim]
        ]).astype(np.float32)

        features_list.append(feature_vec)
        labels_list.append(float(row["label"]))

    X = np.array(features_list)  # (n_samples, 212)
    y = np.array(labels_list)  # (n_samples,)

    logger.info(f"Feature matrix shape: {X.shape}")
    return X, y


#Dataset
class RankerDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#Neural Ranker Model
class NeuralRanker(nn.Module):
    """
    Multi-layer neural network for ranking candidates.

    Architecture: Input → [Linear → BatchNorm → ReLU → Dropout] × 3 → Output

    BatchNorm (Batch Normalisation):
    - Normalises activations within each batch
    - Prevents internal covariate shift — each layer's input stays stable
    - Allows higher learning rates → faster training
    - Acts as mild regularisation

    ReLU (Rectified Linear Unit):
    - Activation function: max(0, x)
    - Introduces non-linearity — without it, stacked linear layers
      collapse to a single linear layer (no benefit from depth)

    Dropout(0.3):
    - Randomly zeros 30% of neurons during training
    - Forces network to learn redundant representations
    - Reduces overfitting significantly

    Tower architecture (256 → 128 → 64 → 1):
    - Progressively compresses features
    - Forces model to learn increasingly abstract representations
    - Final output = single score for ranking
    """

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Final output layer — single score, no activation
        # Sigmoid applied in loss function (BCEWithLogitsLoss)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # Weight initialisation — Kaiming He init for ReLU networks
        # Better than default Xavier init when using ReLU
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)  # (batch, 1) → (batch,)


#Training
def train_ranker(model, train_loader, val_loader, device):
    """Train the ranker and return best model."""

    # BCEWithLogitsLoss = sigmoid + binary cross entropy in one step
    # More numerically stable than applying sigmoid then BCELoss separately
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_auc = 0.0
    best_model_state = None

    logger.info(f"\n{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>10} | {'Val AUC':>8}")
    logger.info("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            # Gradient clipping — prevents exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(all_labels, all_probs)
        scheduler.step(val_loss)

        logger.info(f"{epoch:>6} | {train_loss:>12.4f} | {val_loss:>10.4f} | {val_auc:>8.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(f"  → Best model (AUC={val_auc:.4f})")

    model.load_state_dict(best_model_state)
    return model, best_val_auc


#Main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    movies_df = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))

    with open(os.path.join(DATA_DIR, "movie_id_map.json")) as f:
        movie_id_map = json.load(f)

    # Load embeddings
    user_embeddings = np.load(os.path.join(MODEL_DIR, "user_embeddings.npy"))
    movie_embeddings = np.load(os.path.join(MODEL_DIR, "movie_embeddings.npy"))

    # Build features
    logger.info("Building ranker features...")
    X_train, y_train = build_ranker_features(train_df, user_embeddings, movie_embeddings, movies_df, movie_id_map)
    X_val, y_val = build_ranker_features(val_df, user_embeddings, movie_embeddings, movies_df, movie_id_map)

    # Datasets
    train_loader = DataLoader(RankerDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RankerDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    # Model
    input_dim = X_train.shape[1]  # 212
    model = NeuralRanker(input_dim, HIDDEN_DIMS, DROPOUT).to(device)
    logger.info(f"Ranker input dim: {input_dim}")
    logger.info(f"Ranker parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    model, best_auc = train_ranker(model, train_loader, val_loader, device)
    logger.info(f"\nBest validation AUC: {best_auc:.4f}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "ranker_model.pt"))

    # Save config for loading at inference time
    ranker_config = {"input_dim": input_dim, "hidden_dims": HIDDEN_DIMS, "dropout": DROPOUT}
    with open(os.path.join(MODEL_DIR, "ranker_config.json"), "w") as f:
        json.dump(ranker_config, f, indent=2)

    logger.info(f"Ranker saved to {MODEL_DIR}/ranker_model.pt")
    logger.info("Next step: run api.py")


if __name__ == "__main__":
    main()
