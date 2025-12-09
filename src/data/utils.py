from pathlib import Path
from typing import List, Tuple


def compute_length_stats(src_path: str, tgt_path: str) -> Tuple[List[int], List[int]]:
    src_file = Path(src_path)
    tgt_file = Path(tgt_path)

    assert src_file.exists(), f"Missing: {src_file}"
    assert tgt_file.exists(), f"Missing: {tgt_file}"

    src_lengths = []
    tgt_lengths = []

    with src_file.open("r", encoding="utf-8") as f_src, \
         tgt_file.open("r", encoding="utf-8") as f_tgt:

        for s_line, t_line in zip(f_src, f_tgt):
            s_line = s_line.strip()
            t_line = t_line.strip()
            if not s_line or not t_line:
                continue

            src_lengths.append(len(s_line.split()))
            tgt_lengths.append(len(t_line.split()))

    assert len(src_lengths) == len(tgt_lengths)
    return src_lengths, tgt_lengths
