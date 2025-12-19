from __future__ import annotations

from pathlib import Path

import sentencepiece as spm


def train_one(
    input_path: Path,
    model_prefix: Path,
    vocab_size: int = 16000,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
) -> None:
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    # SentencePiece uses a single command string
    args = " ".join(
        [
            f"--input={input_path}",
            f"--model_prefix={model_prefix}",
            f"--vocab_size={vocab_size}",
            f"--model_type={model_type}",
            f"--character_coverage={character_coverage}",
            "--unk_id=0",
            "--bos_id=1",
            "--eos_id=2",
            "--pad_id=3",
        ]
    )

    spm.SentencePieceTrainer.Train(args)


def main() -> None:
    # parse path
    base = Path(__file__).resolve().parents[1]
    data_dir = base / "data" / "processed"
    out_dir = base / "tokenizer" / "models"

    # input
    en_in = data_dir / "train.en"
    ru_in = data_dir / "train.ru"

    # outputs
    en_prefix = out_dir / "spm_en_bpe_16k"
    ru_prefix = out_dir / "spm_ru_bpe_16k"

    # file exist check
    if not en_in.exists():
        raise FileNotFoundError(f"[INFO] Missing: {en_in}")
    if not ru_in.exists():
        raise FileNotFoundError(f"[INFO] Missing: {ru_in}")

    train_one(
        en_in, en_prefix, vocab_size=16000, model_type="bpe", character_coverage=1.0
    )
    train_one(
        ru_in, ru_prefix, vocab_size=16000, model_type="bpe", character_coverage=1.0
    )

    print("[SUCCESS] Tonkenizer models created")
    print(f"    {en_prefix}.model / {en_prefix}.vocab")
    print(f"    {ru_prefix}.model / {ru_prefix}.vocab")


if __name__ == "__main__":
    main()
