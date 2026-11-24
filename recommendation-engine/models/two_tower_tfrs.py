"""
Milestone 3: Two-Tower Retrieval Model in TensorFlow Recommenders (TFRS).

Architecture:
- User Tower: StringLookup/IntegerLookup -> Embedding(64) + Dense(128, relu) + Dense(64) -> L2 Normalize
- Item Tower: MovieId Embedding(64) + Genre TextVectorization -> Dense(128, relu) + Dense(64) -> L2 Normalize
- Loss: In-batch sampled softmax via tfrs.tasks.Retrieval with FactorizedTopK metrics.
- Export: Saves `user_tower` SavedModel for Triton Inference Server & extracts L2-normalized
  item embeddings `.npy` for Milestone 4 FAISS indexing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    import tensorflow_recommenders as tfrs
    TFRS_AVAILABLE = True
except ImportError:
    TFRS_AVAILABLE = False


if TFRS_AVAILABLE:

    class UserTower(tf.keras.Model):
        """Encodes user ID + continuous behavioral stats into a unit-norm 64-d vector."""

        def __init__(self, unique_user_ids: List[int], embedding_dim: int = 64):
            super().__init__()
            self.user_embedding = tf.keras.Sequential([
                tf.keras.layers.IntegerLookup(vocabulary=unique_user_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dim),
            ])
            self.mlp = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.15),
                tf.keras.layers.Dense(embedding_dim),
            ])

        def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
            uid_emb = self.user_embedding(inputs["userId"])
            stats = tf.stack([
                tf.cast(inputs["user_mean_rating"], tf.float32) / 5.0,
                tf.math.log1p(tf.cast(inputs["user_interaction_count"], tf.float32)) / 5.0,
            ], axis=-1)
            combined = tf.concat([uid_emb, stats], axis=-1)
            vec = self.mlp(combined)
            return tf.math.l2_normalize(vec, axis=-1)

    class ItemTower(tf.keras.Model):
        """Encodes movie ID + genres + popularity into a unit-norm 64-d vector."""

        def __init__(self, unique_movie_ids: List[int], unique_genres: List[str], embedding_dim: int = 64):
            super().__init__()
            self.movie_embedding = tf.keras.Sequential([
                tf.keras.layers.IntegerLookup(vocabulary=unique_movie_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dim),
            ])
            self.genre_vectorizer = tf.keras.layers.TextVectorization(
                max_tokens=len(unique_genres) + 5,
                split="whitespace",
                output_mode="multi_hot",
            )
            self.genre_vectorizer.adapt(unique_genres)
            self.mlp = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.15),
                tf.keras.layers.Dense(embedding_dim),
            ])

        def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
            mid_emb = self.movie_embedding(inputs["movieId"])
            genre_vec = self.genre_vectorizer(inputs["genres_clean"])
            pop = tf.expand_dims(tf.cast(inputs["item_log_popularity"], tf.float32) / 10.0, axis=-1)
            combined = tf.concat([mid_emb, genre_vec, pop], axis=-1)
            vec = self.mlp(combined)
            return tf.math.l2_normalize(vec, axis=-1)

    class TwoTowerRetrievalModel(tfrs.models.Model):
        """Full TFRS Two-Tower model with in-batch softmax and temperature scaling."""

        def __init__(
            self,
            unique_user_ids: List[int],
            unique_movie_ids: List[int],
            unique_genres: List[str],
            candidate_dataset: tf.data.Dataset,
            embedding_dim: int = 64,
            temperature: float = 0.05,
        ):
            super().__init__()
            self.user_tower = UserTower(unique_user_ids, embedding_dim)
            self.item_tower = ItemTower(unique_movie_ids, unique_genres, embedding_dim)
            self.task = tfrs.tasks.Retrieval(
                temperature=temperature,
                metrics=tfrs.metrics.FactorizedTopK(
                    candidates=candidate_dataset.batch(128).map(self.item_tower)
                ),
            )

        def compute_loss(self, features: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
            user_emb = self.user_tower(features)
            item_emb = self.item_tower(features)
            return self.task(user_emb, item_emb, compute_metrics=not training)


def train_tfrs_two_tower(
    data_dir: Path,
    feature_dir: Path,
    artifact_dir: Path,
    epochs: int = 5,
    batch_size: int = 2048,
    embedding_dim: int = 64,
) -> None:
    """Train TFRS Two-Tower retrieval model and export embeddings + Triton SavedModel."""
    if not TFRS_AVAILABLE:
        raise RuntimeError(
            "TensorFlow Recommenders is not installed in this environment. "
            "Install `tensorflow` and `tensorflow-recommenders`, or run `python models/local_engine.py` "
            "for the pure-NumPy/SciPy runnable pipeline."
        )

    train_df = pd.read_csv(data_dir / "train.csv")
    user_feats = pd.read_csv(feature_dir / "user_features.csv")
    item_feats = pd.read_csv(feature_dir / "item_features.csv")
    item_feats["genres_clean"] = item_feats["genres"].str.replace("|", " ", regex=False)

    # Positive interactions for retrieval training (rating >= 4.0)
    pos_df = train_df[train_df["rating"] >= 4.0].merge(
        user_feats[["userId", "user_mean_rating", "user_interaction_count"]], on="userId", how="left"
    ).merge(
        item_feats[["movieId", "genres_clean", "item_log_popularity"]], on="movieId", how="left"
    )

    unique_users = sorted(user_feats["userId"].astype(int).unique().tolist())
    unique_movies = sorted(item_feats["movieId"].astype(int).unique().tolist())
    unique_genres = sorted({g for s in item_feats["genres_clean"] for g in str(s).split()})

    train_ds = tf.data.Dataset.from_tensor_slices({
        "userId": pos_df["userId"].astype(np.int32).values,
        "user_mean_rating": pos_df["user_mean_rating"].astype(np.float32).values,
        "user_interaction_count": pos_df["user_interaction_count"].astype(np.float32).values,
        "movieId": pos_df["movieId"].astype(np.int32).values,
        "genres_clean": pos_df["genres_clean"].astype(str).values,
        "item_log_popularity": pos_df["item_log_popularity"].astype(np.float32).values,
    }).shuffle(100_000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    cand_ds = tf.data.Dataset.from_tensor_slices({
        "movieId": item_feats["movieId"].astype(np.int32).values,
        "genres_clean": item_feats["genres_clean"].astype(str).values,
        "item_log_popularity": item_feats["item_log_popularity"].astype(np.float32).values,
    })

    model = TwoTowerRetrievalModel(unique_users, unique_movies, unique_genres, cand_ds, embedding_dim)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.08))
    model.fit(train_ds, epochs=epochs)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Export User Tower SavedModel for Triton Inference Server
    tf.saved_model.save(model.user_tower, str(artifact_dir / "triton_user_tower"))

    # Extract L2-normalized item embeddings for FAISS indexing
    item_embs = np.vstack([model.item_tower(batch).numpy() for batch in cand_ds.batch(512)])
    np.save(artifact_dir / "item_embeddings.npy", item_embs.astype(np.float32))
    np.save(artifact_dir / "item_ids.npy", item_feats["movieId"].astype(np.int64).values)
    print(f"[Milestone 3] TFRS Two-Tower trained & exported: {item_embs.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--feature-dir", type=str, default="data/features")
    parser.add_argument("--artifact-dir", type=str, default="artifacts_store")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    train_tfrs_two_tower(Path(args.data_dir), Path(args.feature_dir), Path(args.artifact_dir), args.epochs)
