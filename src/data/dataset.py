import os
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset
import sentencepiece as spm


class MTTextDataset(Dataset):
    """
    Machine Translation dataset for EN->DE using SentencePiece tokenizer.
    Returns raw token ID sequences (no padding).
    """

    def __init__(
        self,
        src_path: str,
        tgt_path: str,
        spm_model_path: str,
        max_source_len: int = 128,
        max_target_len: int = 128,
    ):
        self.src_path = Path(src_path)
        self.tgt_path = Path(tgt_path)

        assert self.src_path.exists(), f"Source file not found: {self.src_path}"
        assert self.tgt_path.exists(), f"Target file not found: {self.tgt_path}"

        # Load raw lines into memory
        with self.src_path.open("r", encoding="utf-8") as f:
            self.src_lines = [line.strip() for line in f if line.strip()]

        with self.tgt_path.open("r", encoding="utf-8") as f:
            self.tgt_lines = [line.strip() for line in f if line.strip()]

        assert len(self.src_lines) == len(self.tgt_lines), (
            f"Source and target must have same number of lines, "
            f"got {len(self.src_lines)} vs {len(self.tgt_lines)}"
        )

        # Load SentencePiece model
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)

        self.vocab_size = self.sp.get_piece_size()
        self.pad_id = 0  # SentencePiece default
        self.bos_id = self.sp.bos_id()  # usually 1
        self.eos_id = self.sp.eos_id()  # usually 2

        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.src_lines)

    def _encode(self, text: str, max_len: int, add_bos: bool, add_eos: bool) -> List[int]:
        ids = self.sp.encode(text, out_type=int)

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        # truncate if too long
        if len(ids) > max_len:
            ids = ids[:max_len]

        return ids

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src_text = self.src_lines[idx]
        tgt_text = self.tgt_lines[idx]

        # Encoder input: BOS not strictly required, but EOS is useful
        src_ids = self._encode(
            src_text,
            max_len=self.max_source_len,
            add_bos=False,
            add_eos=True,
        )

        # Decoder input: BOS + tokens (for teacher forcing)
        tgt_input_ids = self._encode(
            tgt_text,
            max_len=self.max_target_len,
            add_bos=True,
            add_eos=False,
        )

        # Decoder labels: tokens + EOS (shifted target)
        tgt_label_ids = self._encode(
            tgt_text,
            max_len=self.max_target_len,
            add_bos=False,
            add_eos=True,
        )

        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_input_ids": torch.tensor(tgt_input_ids, dtype=torch.long),
            "tgt_labels": torch.tensor(tgt_label_ids, dtype=torch.long),
        }
