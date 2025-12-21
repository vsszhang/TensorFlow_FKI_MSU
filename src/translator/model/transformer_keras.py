from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int

    max_len: int = 128
    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 512
    num_layers: int = 4
    dropout: float = 0.1


# 2) Token Embedding + Positional Embedding
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        dropout: float = 0.1,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=d_model, name="token_embedding"
        )

        self.pos_emb = tf.keras.layers.Embedding(
            input_dim=max_len,
            output_dim=d_model,
            name="pos_embedding",
        )

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, token_ids: tf.Tensor, training: bool = False) -> tf.Tensor:
        # dynamic seq_len
        seq_len = tf.shape(token_ids)[1]

        x = self.token_emb(token_ids)

        positions = tf.range(start=0, limit=seq_len, delta=1)

        pos = self.pos_emb(positions)

        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = x + pos

        return self.dropout(x, training=training)


# 3) Feed-Forward Network (FFN) used in Transformer
class FeedForwardNetwork(tf.keras.layers.Layer):
    """Position-wise Feed-Forward Network

       The second sub-block inside each Transformer layer.

       For each position (token) independently:
            x -> Dense(d_ff, relu) -> Dense(d_model)

       Shapes:
            input: (batch, seq_len, d_model)
            output: (batch, seq_len, d_model)

    Args:
        tf (keras Layer subject): keras Layer subject
    """

    def __init__(
        self, d_model: int, d_ff: int, dropout: float, name: str | None = None
    ):
        super().__init__(name=name)

        # keep FFN as a small Sequential for clarity
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(d_ff, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ],
            name="ffn",
        )
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.net(x)
        return self.dropout(x, training=training)


# 4) Encoder Layer (Self-Attention + FFN)
class EncoderLayer(tf.keras.layers.Layer):
    """A sigle Transformer Encoder layer

    Structure (classic):
        x -> Self-Attention -> Dropout -> Add & Norm
          -> FFN            -> Dropout -> Add & Norm

    Note on masks:
        We accept an optional `padding_mask` of shape (batch, seq_len)
        Here, padding_mask should be a boolean/int mask where:
            1 / True => keep token (Not padding)
            0 / False => padding position (should be masked)

        Keras MultiHeadAttention uses `attention_mask` where True/1 means
        "this. key position is visible"
        Expected shape: (batch, query_len, key_len)

    Args:
        tf (tf keras Layer): tf keras Layer
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        name: str | None = None,
    ):
        super().__init__(name=name)

        if d_model % num_heads != 0:
            raise ValueError(
                f"[INFO] d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        # multi-head self-attention
        # key_dim is per-head dimension
        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name="self_attention",
        )

        # dropout for attention output
        self.attn_dropout = tf.keras.layers.Dropout(dropout)

        # layer normalization after the first residual
        self.attn_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="attn_norm"
        )

        # FFN sub-block
        self.ffn = FeedForwardNetwork(
            d_model=d_model, d_ff=d_ff, dropout=dropout, name="ffn_block"
        )

        # layer normalization after the first residual
        self.ffn_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="ffn_norm"
        )

    @staticmethod
    def _make_attention_mask(padding_mask: tf.Tensor, seq_len: tf.Tensor) -> tf.Tensor:
        """Convert (batch, seq_len) padding mask -> (batch, seq_len, seq_len) attention mask.

        Args:
            padding_mask (tf.Tensor): padding mask
            seq_len (tf.Tensor): senquence length

        Returns:
            tf.Tensor: tf Tensor subject
        """
        # Ensure boolean mask
        mask = tf.cast(padding_mask, tf.bool)

        # expand to (batch, 1, seq_len) then broadcast to (batch, seq_len, seq_len)
        # so every query position uses the same key visibility mask
        mask = mask[:, tf.newaxis, :]
        return tf.tile(mask, [1, seq_len, 1])

    def call(
        self,
        x: tf.Tensor,
        padding_mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """Forward pass

        Args:
            x (tf.Tensor): (batch, seq_len, d_model)
            padding_mask (tf.Tensor | None, optional): (batch, seq_len) with 1/True for real tokens. Defaults to None.
            training (bool, optional): whether in training mode. Defaults to False.

        Returns:
            tf.Tensor: tf Tensor subject
        """
        seq_len = tf.shape(x)[1]

        # 1) self-attention block
        attn_mask = None
        if padding_mask is not None:
            attn_mask = self._make_attention_mask(padding_mask, seq_len)

        # (batch, seq_len, d_model)
        attn_out = self.self_attn(
            query=x, value=x, key=x, attention_mask=attn_mask, training=training
        )

        attn_out = self.attn_dropout(attn_out, training=training)

        # redidual + normalize
        x = self.attn_norm(x + attn_out)

        # 2) FFN block
        ffn_out = self.ffn(x, training=training)

        # residual + normalize
        x = self.ffn_norm(x + ffn_out)

        return x
