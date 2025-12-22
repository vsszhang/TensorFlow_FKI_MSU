from __future__ import annotations

from dataclasses import asdict, dataclass

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
            tf.Tensor: tf Tensor object
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
            tf.Tensor: tf Tensor object
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


# 5) Decoder Layer (Masker SA + Cross-Attn)
class DecoderLayer(tf.keras.layers.Layer):
    """A single Transformer Decoder layer.

    Structure (classic):
        x -> Masked Self-Attention -> Dropput -> Add & Norm
          -> Cross-Attention (enc <-> dec) -> Dropout -> Add & Norm
          -> FFN -> Dropout -> Add & Norm

    Masks:
    - tgt_padding_mask: (batch, tgt_len) 1/True = real token, 0/False = padding
    - src_padding_mask: (batch, src_len) 1/True = real token, 0/False = padding

    For masked self-attention we need TWO constraints:
    1) look-ahead (causal) mask: position i cannot see future tokens j>i
    2) padding mask: queries do not attend to padding keys

    Keras MultiHeadAttention expects `attention_mask` where True/1 means
    "this key position is visible"

    Shapes:
        self-attn mask: (batch, tgt_len, tgt_len)
        cross-attn mask: (batch, tgt_len, src_len)

    Args:
        tf (tf keras Layer): tf keras Layer object
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

        # step 1: masked self-attention (decoder attends to itself)
        self.self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name="masked_self_attention",
        )
        self.self_attn_dropout = tf.keras.layers.Dropout(dropout)
        self.self_attn_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="self_attn_norm"
        )

        # step 2: cross-attention (decoder attends to encoder output)
        self.cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name="cross_attention",
        )
        self.cross_attn_dropout = tf.keras.layers.Dropout(dropout)
        self.cross_attn_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="cross_attn_norm"
        )

        # step 3: FFN sub-block
        self.ffn = FeedForwardNetwork(
            d_model=d_model, d_ff=d_ff, dropout=dropout, name="ffn_block"
        )
        self.ffn_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="ffn_norm"
        )

    @staticmethod
    def _make_look_ahead_mask(tgt_len: tf.Tensor) -> tf.Tensor:
        """Create causal mask of shape (tgt_len, tgt_len)

        True means visible (allowed), False means masked.

        Example (tgt_len=4):
            [[1, 0, 0, 0],
             [1, 1, 0, 0],
             [1, 1, 1, 0],
             [1, 1, 1, 1]]

        Args:
            tgt_len (tf.Tensor): tgt length

        Returns:
            tf.Tensor: tf Tensor object
        """
        mask = tf.linalg.band_part(tf.ones((tgt_len, tgt_len), dtype=tf.bool), -1, 0)
        return mask

    @staticmethod
    def _make_self_attention_mask(
        tgt_padding_mask: tf.Tensor | None, tgt_len: tf.Tensor
    ) -> tf.Tensor:
        """Combine lood-ahead + target padding mask into (batch, tgt_len, tgt_len).

        Args:
            tgt_padding_mask (tf.Tensor | None): tgt padding mask
            tgt_len (tf.Tensor): tgt len

        Returns:
            tf.Tensor: tf Tensor object
        """
        # (tgt_len, tgt_len)
        causal = DecoderLayer._make_look_ahead_mask(tgt_len)

        # expand to (1, tgt_len, tgt_len) then brodcast to (batch, tgt_len, tgt_len)
        causal = causal[tf.newaxis, :, :]

        if tgt_padding_mask is None:
            return causal

        # tgt_padding_mask: (batch, tgt_len) where 1/True = real token
        key_visible = tf.cast(tgt_padding_mask, tf.bool)  # (batch, tgt_len)
        key_visible = key_visible[:, tf.newaxis, :]  # (batch, 1, tgt_len)
        key_visible = tf.tile(key_visible, [1, tgt_len, 1])  # (batch, tgt_len, tgt_len)

        # both constrains must be satisfied
        return tf.logical_and(key_visible, causal)

    @staticmethod
    def _make_cross_attention_mask(
        src_padding_mask: tf.Tensor | None, tgt_len: tf.Tensor
    ) -> tf.Tensor | None:
        if src_padding_mask is None:
            return None

        # 1/True means visible
        mask = tf.cast(src_padding_mask, tf.bool)  # (batch, src_len)
        mask = mask[:, tf.newaxis, :]  # (batch, 1, src_len)
        return tf.tile(mask, [1, tgt_len, 1])  # (batch, tgt_len, src_len)

    def call(
        self,
        x: tf.Tensor,
        enc_out: tf.Tensor,
        tgt_padding_mask: tf.Tensor | None = None,
        src_padding_mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        tgt_len = tf.shape(x)[1]

        # step 1: masked self-attention
        self_mask = self._make_self_attention_mask(tgt_padding_mask, tgt_len)

        attn1 = self.self_attn(
            query=x, value=x, key=x, attention_mask=self_mask, training=training
        )
        attn1 = self.self_attn_dropout(attn1, training=training)
        x = self.self_attn_norm(x + attn1)

        # step 2: cross-attention: query=decoder states, key/value=encoder states
        cross_mask = self._make_cross_attention_mask(src_padding_mask, tgt_len)
        attn2 = self.cross_attn(
            query=x,
            value=enc_out,
            key=enc_out,
            attention_mask=cross_mask,
            training=training,
        )
        attn2 = self.cross_attn_dropout(attn2, training=training)
        x = self.cross_attn_norm(x + attn2)

        # step 3: FFN
        ffn_out = self.ffn(x, training=training)
        x = self.ffn_norm(x + ffn_out)

        return x


# 6) Encoder / Decoder stacks (num_layers)
class Encoder(tf.keras.layers.Layer):
    def __init__(self, cfg: TransformerConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg

        self.embed = PositionalEmbedding(
            vocab_size=cfg.src_vocab_size,
            d_model=cfg.d_model,
            max_len=cfg.max_len,
            dropout=cfg.dropout,
            name="src_embedding",
        )

        self.layers = [
            EncoderLayer(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
                name=f"encoder_layer_{i}",
            )
            for i in range(cfg.num_layers)
        ]

    def call(
        self,
        src_ids: tf.Tensor,
        src_padding_mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """_summary_

        Args:
            src_ids (tf.Tensor): (batch, src_len)
            src_padding_mask (tf.Tensor | None, optional): (batch, src_len) with 1/True for real tokens. Defaults to None.
            training (bool, optional): whether in training mode. Defaults to False.

        Returns:
            tf.Tensor: tf Tensor object
        """
        x = self.embed(src_ids, training=training)  # (batch, src_len, d_model)
        for layer in self.layers:
            x = layer(x, padding_mask=src_padding_mask, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"cfg": asdict(self.cfg)})
        return config


class Decoder(tf.keras.layers.Layer):
    def __init__(self, cfg: TransformerConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg

        self.embed = PositionalEmbedding(
            vocab_size=cfg.tgt_vocab_size,
            d_model=cfg.d_model,
            max_len=cfg.max_len,
            dropout=cfg.dropout,
            name="tgt_embedding",
        )

        self.layers = [
            DecoderLayer(
                d_model=cfg.d_model,
                num_heads=cfg.num_heads,
                d_ff=cfg.d_ff,
                dropout=cfg.dropout,
                name=f"decoder_layer_{i}",
            )
            for i in range(cfg.num_layers)
        ]

    def call(
        self,
        tgt_ids: tf.Tensor,
        enc_out: tf.Tensor,
        tgt_padding_mask: tf.Tensor | None = None,
        src_padding_mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """_summary_

        Args:
            tgt_ids (tf.Tensor): (batch, tgt_len)
            enc_out (tf.Tensor): (batch, src_len, d_model)
            tgt_padding_mask (tf.Tensor | None, optional): (batch, tgt_len) with 1/True for real tokens. Defaults to None.
            src_padding_mask (tf.Tensor | None, optional): (batch, src_len) wiht 1/True for real tokens. Defaults to None.
            training (bool, optional): whether in training model. Defaults to False.

        Returns:
            tf.Tensor: tf Tensor object
        """
        x = self.embed(tgt_ids, training=training)  # (batch, tgt_len, d_model)
        for layer in self.layers:
            x = layer(
                x,
                enc_out,
                tgt_padding_mask=tgt_padding_mask,
                src_padding_mask=src_padding_mask,
                training=training,
            )
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"cfg": asdict(self.cfg)})
        return config


# 7) Full Transformer model (compile/fit)
class Transformer(tf.keras.Model):
    def __init__(self, cfg: TransformerConfig, name: str | None = None):
        super().__init__(name=name)
        self.cfg = cfg

        self.encoder = Encoder(cfg, name="encoder")
        self.decoder = Decoder(cfg, name="decoder")

        # final projection to vocabulary
        self.out_proj = tf.keras.layers.Dense(cfg.tgt_vocab_size, name="vocab_logits")

    @staticmethod
    def make_padding_mask(token_ids: tf.Tensor, pad_id: int = 3) -> tf.Tensor:
        """Create padding mask (batch, seq_len) where 1=real token, 0=pad

        Set pad_id=3 to match SentencePiece config (--pad_id=3)

        Args:
            token_ids (tf.Tensor): toekn ids
            pad_id (int, optional): _description_. Defaults to 3.

        Returns:
            tf.Tensor: tf Tensor object
        """
        return tf.cast(tf.not_equal(token_ids, pad_id), tf.int32)

    def call(
        self,
        inputs: tuple[tf.Tensor, tf.Tensor],
        training: bool = False,
    ) -> tf.Tensor:
        """Keras forward.

        Args:
            inputs (tuple[tf.Tensor, tf.Tensor]): (src_ids, tgt_in_i)
            training (bool, optional): whether in training model. Defaults to False.

        Returns:
            tf.Tensor: logits (batch, tgt_len,  tgt_vocab_size)
        """
        src_ids, tgt_in_ids = inputs

        src_padding_mask = self.make_padding_mask(src_ids)
        tgt_padding_mask = self.make_padding_mask(tgt_in_ids)

        enc_out = self.encoder(
            src_ids, src_padding_mask=src_padding_mask, training=training
        )
        dec_out = self.decoder(
            tgt_in_ids,
            enc_out,
            tgt_padding_mask=tgt_padding_mask,
            src_padding_mask=src_padding_mask,
            training=training,
        )

        return self.out_proj(dec_out)

    def get_config(self):
        # Required for saving to the native `.keras` format when __init__ has non-primitive args.
        config = super().get_config()
        config.update({"cfg": asdict(self.cfg)})
        return config

    @classmethod
    def from_config(cls, config):
        cfg_dict = config.pop("cfg")
        cfg = TransformerConfig(**cfg_dict)
        return cls(cfg=cfg, **config)
