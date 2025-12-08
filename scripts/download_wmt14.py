import os
from datasets import load_dataset

def save_split(split, out_dir):
    en_path = os.path.join(out_dir, f"{split}.en")
    de_path = os.path.join(out_dir, f"{split}.de")

    with open(en_path, "w", encoding="utf-8") as f_en, \
         open(de_path, "w", encoding="utf-8") as f_de:

        for sample in data[split]:
            en = sample["translation"]["en"]
            de = sample["translation"]["de"]

            # avoid blank lines
            if en.strip() == "" or de.strip() == "":
                continue

            f_en.write(en.strip() + "\n")
            f_de.write(de.strip() + "\n")

    print(f"Saved {split}: {en_path}, {de_path}")


if __name__ == "__main__":
    print("Downloading WMT14 EN→DE...")
    data = load_dataset("wmt14", "de-en")

    out_dir = "data/raw"
    os.makedirs(out_dir, exist_ok=True)

    for split in ["train", "validation", "test"]:
        save_split(split, out_dir)

    print("Done.")
