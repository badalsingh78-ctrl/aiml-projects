"""
Milestone 5: Stage-2 Deep Ranking Models (DCN-v2 & Wide & Deep) in TensorFlow Recommenders.

Both models take Stage-1 retrieved candidate pairs (user features + item features + two-tower
dot-product similarity + genre overlap) and predict click/high-rating probability for precise
re-ranking.

Models implemented:
1. DCNv2Ranker: Stacked explicit polynomial feature-crossing layers (`tfrs.layers.dcn.Cross`)
   in parallel with a Deep MLP network.
2. WideAndDeepRanker: Wide linear memorization branch over sparse/cross features combined
   with a Deep generalization MLP.
"""

from __future__ import annotations

from typing import Dict

try:
    import tensorflow as tf
    import tensorflow_recommenders as tfrs
    TFRS_AVAILABLE = True
except ImportError:
    TFRS_AVAILABLE = False


if TFRS_AVAILABLE:

    class DCNv2Ranker(tfrs.Model):
        """Deep & Cross Network v2 (parallel cross + deep structure) for Stage-2 re-ranking."""

        def __init__(self, num_features: int, cross_layers: int = 3, deep_units=(128, 64)):
            super().__init__()
            self.input_norm = tf.keras.layers.BatchNormalization()
            self.cross_net = [
                tfrs.layers.dcn.Cross(projection_dim=16, kernel_initializer="glorot_uniform")
                for _ in range(cross_layers)
            ]
            self.deep_net = tf.keras.Sequential([
                tf.keras.layers.Dense(u, activation="relu") for u in deep_units
            ])
            self.head = tf.keras.layers.Dense(1, activation="sigmoid")
            self.task = tfrs.tasks.Ranking(
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    tf.keras.metrics.AUC(name="auc"),
                    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                ],
            )

        def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
            x0 = self.input_norm(features["dense_features"])
            xl = x0
            for cross in self.cross_net:
                xl = cross(x0, xl)
            deep_out = self.deep_net(x0)
            combined = tf.concat([xl, deep_out], axis=-1)
            return self.head(combined)

        def compute_loss(self, features: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
            labels = features["clicked"]
            preds = self(features, training=training)
            return self.task(labels=labels, predictions=preds)

    class WideAndDeepRanker(tfrs.Model):
        """Wide & Deep baseline ranker for A/B comparison against DCN-v2."""

        def __init__(self, num_features: int, deep_units=(128, 64)):
            super().__init__()
            self.input_norm = tf.keras.layers.BatchNormalization()
            self.wide = tf.keras.layers.Dense(1, activation=None)
            self.deep = tf.keras.Sequential([
                tf.keras.layers.Dense(u, activation="relu") for u in deep_units
            ] + [tf.keras.layers.Dense(1, activation=None)])
            self.task = tfrs.tasks.Ranking(
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    tf.keras.metrics.AUC(name="auc"),
                    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                ],
            )

        def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
            x = self.input_norm(features["dense_features"])
            logits = self.wide(x) + self.deep(x)
            return tf.nn.sigmoid(logits)

        def compute_loss(self, features: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
            labels = features["clicked"]
            preds = self(features, training=training)
            return self.task(labels=labels, predictions=preds)
