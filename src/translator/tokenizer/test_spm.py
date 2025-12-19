from pathlib import Path

import sentencepiece as spm


def load(model_path: Path) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp


def main():
    base = Path(__file__).resolve().parents[1]
    model_dir = base / "tokenizer" / "models"

    sp_en = load(model_dir / "spm_en_bpe_16k.model")
    sp_ru = load(model_dir / "spm_ru_bpe_16k.model")

    en_text = "Thank you so much, Chris."
    ru_text = "Спасибо, Крис."

    print("=== EN ===")
    print("text :", en_text)
    print("pieces:", sp_en.encode(en_text, out_type=str))
    print("ids   :", sp_en.encode(en_text, out_type=int))
    print("back  :", sp_en.decode(sp_en.encode(en_text, out_type=int)))

    print("\n=== RU ===")
    print("text :", ru_text)
    print("pieces:", sp_ru.encode(ru_text, out_type=str))
    print("ids   :", sp_ru.encode(ru_text, out_type=int))
    print("back  :", sp_ru.decode(sp_ru.encode(ru_text, out_type=int)))


if __name__ == "__main__":
    main()
