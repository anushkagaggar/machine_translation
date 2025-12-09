import os
from pathlib import Path
import sentencepiece as spm


PROCESSED_DIR = Path("data/processed")
BPE_DIR = Path("models/bpe")

VOCAB_SIZE = 32000
MODEL_PREFIX = str(BPE_DIR / "spm")  # will create spm.model and spm.vocab


def build_joint_corpus():
    """
    Concatenate cleaned train.en + train.de into a single joint corpus file.
    """
    src_path = PROCESSED_DIR / "train.en"
    tgt_path = PROCESSED_DIR / "train.de"
    joint_path = PROCESSED_DIR / "joint_corpus.txt"

    assert src_path.exists(), f"Missing {src_path}"
    assert tgt_path.exists(), f"Missing {tgt_path}"

    with src_path.open("r", encoding="utf-8") as f_en, \
         tgt_path.open("r", encoding="utf-8") as f_de, \
         joint_path.open("w", encoding="utf-8") as f_out:

        for line in f_en:
            line = line.strip()
            if line:
                f_out.write(line + "\n")

        for line in f_de:
            line = line.strip()
            if line:
                f_out.write(line + "\n")

    print(f"Joint corpus written to: {joint_path}")
    return joint_path


def train_sentencepiece(input_path: Path):
    """
    Train a SentencePiece BPE model on the joint corpus.
    """
    BPE_DIR.mkdir(parents=True, exist_ok=True)

    # SentencePieceTrainer accepts a string of args
    spm.SentencePieceTrainer.Train(
        input=str(input_path),
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=0.9995,
        input_sentence_size=10000000,  # sample up to 10M sentences if needed
        shuffle_input_sentence=True
    )

    print(f"SentencePiece model trained:")
    print(f"  {MODEL_PREFIX}.model")
    print(f"  {MODEL_PREFIX}.vocab")


def quick_test():
    """
    Quick sanity check: encode/decode English + German sentences.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(f"{MODEL_PREFIX}.model")

    examples = [
        "This is a small test sentence in English.",
        "Dies ist ein kleiner Testsatz auf Deutsch."
    ]

    for text in examples:
        ids = sp.encode(text, out_type=int)
        pieces = sp.encode(text, out_type=str)
        decoded = sp.decode(ids)

        print("\nText:", text)
        print("Pieces:", pieces[:20])
        print("IDs:", ids[:20])
        print("Decoded:", decoded)


if __name__ == "__main__":
    joint_corpus = build_joint_corpus()
    train_sentencepiece(joint_corpus)
    quick_test()
