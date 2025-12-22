from __future__ import annotations

import csv
from pathlib import Path

import sentencepiece as spm
import tensorflow as tf
from model.transformer_keras import Transformer, TransformerConfig

# 0) Contant value (match SentencePiece config)
UNK_ID = 0
BOS_ID = 1
EOS_ID = 2
PAD_ID = 3


# 1) Load SentencePiece models
def load_sp(model_path: Path) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp


# 2) Encode one line
def encode_line(sp: spm.SentencePieceProcessor, text, max_len: int) -> list[int]:
    """We add BOS/EOS and then pad/truncate.

    Args:
        sp (spm.SentencePieceProcessor): _description_
        text (bytes): bytes string from tf.data.TextLineDataset
        max_len (int): _description_

    Returns:
        list[int]: a list of token ids length <= max_len.
    """
    # When called via `tf.py_function`, `text` is a `tf.Tensor` (EagerTensor), not raw `bytes`.
    # So we must convert it to bytes first.
    if isinstance(text, (bytes, bytearray)):
        raw = bytes(text)
    else:
        # EagerTensor -> bytes
        raw = text.numpy()

    s = raw.decode("utf-8").strip()
    ids = sp.encode(s, out_type=int)

    # add special tokens for seq2seq
    ids = [BOS_ID] + ids + [EOS_ID]

    # truncate
    ids = ids[:max_len]

    # pad
    if len(ids) < max_len:
        ids = ids + [PAD_ID] * (max_len - len(ids))

    return ids


# 2.5) Build parallel text files from CSV
def export_parallel_from_csv(
    csv_path: Path,
    ru_out: Path,
    en_out: Path,
    *,
    ru_col: int = 0,
    en_col: int = 1,
    encoding: str = "utf-8",
    max_rows: int | None = None,
) -> tuple[int, int]:
    """Read `csv_path` and export two parallel files: ru_out / en_out.

    CSV format assumption (can be adjusted by ru_col/en_col):
    - column `ru_col` -> Russian text
    - column `en_col` -> English text

    Returns:
        (written_rows, skipped_rows)
    """
    ru_out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with (
        csv_path.open("r", encoding=encoding, newline="") as f_in,
        ru_out.open("w", encoding="utf-8", newline="\n") as f_ru,
        en_out.open("w", encoding="utf-8", newline="\n") as f_en,
    ):
        reader = csv.reader(f_in)

        for row in reader:
            if max_rows is not None and written >= max_rows:
                break

            # Basic validation
            if not row or len(row) <= max(ru_col, en_col):
                skipped += 1
                continue

            ru = (row[ru_col] or "").strip()
            en = (row[en_col] or "").strip()

            if not ru or not en:
                skipped += 1
                continue

            # One sample per line
            f_ru.write(ru + "\n")
            f_en.write(en + "\n")
            written += 1

    return written, skipped


# 3) Build tf.data pipeline (RU -> EN)
def make_dataset(
    ru_path: Path,
    en_path: Path,
    sp_ru: spm.SentencePieceProcessor,
    sp_en: spm.SentencePieceProcessor,
    max_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> tf.data.Dataset:
    ru_ds = tf.data.TextLineDataset(str(ru_path))
    en_ds = tf.data.TextLineDataset(str(en_path))

    ds = tf.data.Dataset.zip((ru_ds, en_ds))

    if shuffle:
        ds = ds.shuffle(10_000, reshuffle_each_iteration=True)

    def tf_encode(ru_bytes: tf.Tensor, en_bytes: tf.Tensor):
        src_ids = tf.py_function(
            func=lambda x: tf.constant(encode_line(sp_ru, x, max_len), dtype=tf.int32),
            inp=[ru_bytes],
            Tout=tf.int32,
        )
        tgt_full = tf.py_function(
            func=lambda x: tf.constant(encode_line(sp_en, x, max_len), dtype=tf.int32),
            inp=[en_bytes],
            Tout=tf.int32,
        )

        # set static shapes
        src_ids.set_shape((max_len,))
        tgt_full.set_shape((max_len,))

        tgt_in = tgt_full[:-1]  # decoder input
        tgt_out = tgt_full[1:]  # decoder label

        tgt_in.set_shape((max_len - 1,))
        tgt_out.set_shape((max_len - 1,))

        return (src_ids, tgt_in), tgt_out

    ds = ds.map(tf_encode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# 4) Loss (ignore PAD positions)
def masked_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """ignore PAD_ID in y_true

    Args:
        y_true (tf.Tensor): (batch, tgt_len-1)
        y_pred (tf.Tensor): (batch, tgt_len-1, vocab)

    Returns:
        tf.Tensor: _description_
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    per_token = loss_fn(y_true, y_pred)  # (batch, tgt_len-1)

    mask = tf.cast(tf.not_equal(y_true, PAD_ID), tf.float32)
    per_token = per_token * mask

    # average over non-pad tokens
    return tf.reduce_sum(per_token) / tf.reduce_sum(mask)


# 5) Main training entry
def main():
    base = Path(__file__).resolve().parent
    data_dir = base / "data" / "processed"
    tok_dir = base / "tokenizer" / "models"
    out_dir = base / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    ru_train = data_dir / "train.ru"
    en_train = data_dir / "train.en"

    # CSV is English,Russian, swap ru_col/en_col.
    csv_corpus = base / "data" / "raw" / "en2ru.csv"

    if csv_corpus.exists():
        written, skipped = export_parallel_from_csv(
            csv_corpus,
            ru_out=ru_train,
            en_out=en_train,
            ru_col=0,
            en_col=1,
            encoding="utf-8",
        )
        print(f"[CSV] Exported parallel files: written={written}, skipped={skipped}")
    else:
        print(
            f"[CSV] Not found: {csv_corpus} (fallback to existing train.ru/train.en if present)"
        )

    sp_ru = load_sp(tok_dir / "spm_ru_bpe_16k.model")
    sp_en = load_sp(tok_dir / "spm_en_bpe_16k.model")

    # keep it small first
    cfg = TransformerConfig(
        src_vocab_size=sp_ru.get_piece_size(),
        tgt_vocab_size=sp_en.get_piece_size(),
        max_len=64,
        d_model=256,
        num_heads=8,
        d_ff=512,
        num_layers=4,
        dropout=0.1,
    )

    batch_size = 64

    train_ds = make_dataset(
        ru_path=ru_train,
        en_path=en_train,
        sp_ru=sp_ru,
        sp_en=sp_en,
        max_len=cfg.max_len,
        batch_size=batch_size,
        shuffle=True,
    )

    model = Transformer(cfg)

    # build model
    model.build(input_shape=[(None, cfg.max_len), (None, cfg.max_len - 1)])
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=masked_loss
    )

    # quick sanity run: 1 epoch
    model.fit(train_ds, epochs=1, steps_per_epoch=200)

    # save model
    save_path = out_dir / "translator_ru_en.keras"
    model.save(str(save_path))
    print(f"[OK] Saved model to: {save_path}")


if __name__ == "__main__":
    main()
