import os
import unicodedata
from pathlib import Path


RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")

MIN_TOKENS = 3
MAX_TOKENS = 120
MAX_LEN_RATIO = 3.0


def normalize_text(text: str) -> str:
    # Unicode normalize + strip
    text = unicodedata.normalize("NFKC", text)
    text = text.strip()
    return text


def count_tokens(text: str) -> int:
    # simple whitespace tokenization (pre-BPE)
    return len(text.split())


def keep_pair(en: str, de: str) -> bool:
    len_en = count_tokens(en)
    len_de = count_tokens(de)

    # too short
    if len_en < MIN_TOKENS or len_de < MIN_TOKENS:
        return False

    # too long
    if len_en > MAX_TOKENS or len_de > MAX_TOKENS:
        return False

    # extreme length mismatch
    shorter = min(len_en, len_de)
    longer = max(len_en, len_de)

    if shorter == 0:
        return False

    if longer / shorter > MAX_LEN_RATIO:
        return False

    return True


def process_split(raw_name: str, out_name: str):
    raw_en = RAW_DIR / f"{raw_name}.en"
    raw_de = RAW_DIR / f"{raw_name}.de"

    out_en = PROC_DIR / f"{out_name}.en"
    out_de = PROC_DIR / f"{out_name}.de"

    assert raw_en.exists(), f"Missing file: {raw_en}"
    assert raw_de.exists(), f"Missing file: {raw_de}"

    total = 0
    kept = 0
    dropped_too_short = 0
    dropped_too_long = 0
    dropped_ratio = 0

    with raw_en.open("r", encoding="utf-8") as f_en_in, \
         raw_de.open("r", encoding="utf-8") as f_de_in, \
         out_en.open("w", encoding="utf-8") as f_en_out, \
         out_de.open("w", encoding="utf-8") as f_de_out:

        for en_line, de_line in zip(f_en_in, f_de_in):
            total += 1

            en = normalize_text(en_line)
            de = normalize_text(de_line)

            len_en = count_tokens(en)
            len_de = count_tokens(de)

            # too short
            if len_en < MIN_TOKENS or len_de < MIN_TOKENS:
                dropped_too_short += 1
                continue

            # too long
            if len_en > MAX_TOKENS or len_de > MAX_TOKENS:
                dropped_too_long += 1
                continue

            shorter = min(len_en, len_de)
            longer = max(len_en, len_de)

            if shorter == 0 or longer / shorter > MAX_LEN_RATIO:
                dropped_ratio += 1
                continue

            # keep
            f_en_out.write(en + "\n")
            f_de_out.write(de + "\n")
            kept += 1

    print(f"=== Split: {raw_name} -> {out_name} ===")
    print(f"Total pairs:       {total}")
    print(f"Kept:              {kept}")
    print(f"Dropped (short):   {dropped_too_short}")
    print(f"Dropped (long):    {dropped_too_long}")
    print(f"Dropped (ratio):   {dropped_ratio}")
    print()


if __name__ == "__main__":
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # raw_name -> output_name
    splits = [
        ("train", "train"),
        ("validation", "valid"),
        ("test", "test"),
    ]

    for raw_name, out_name in splits:
        process_split(raw_name, out_name)
