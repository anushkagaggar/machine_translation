from typing import List, Dict

import torch


def collate_mt_batch(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Pads a list of examples into batch tensors and builds padding masks.
    """

    # Extract fields
    src_seqs = [item["src_ids"] for item in batch]
    tgt_in_seqs = [item["tgt_input_ids"] for item in batch]
    tgt_label_seqs = [item["tgt_labels"] for item in batch]

    batch_size = len(batch)
    max_src_len = max(seq.size(0) for seq in src_seqs)
    max_tgt_len = max(seq.size(0) for seq in tgt_in_seqs)

    # Allocate padded tensors
    src_ids = torch.full((batch_size, max_src_len), pad_id, dtype=torch.long)
    tgt_input_ids = torch.full((batch_size, max_tgt_len), pad_id, dtype=torch.long)
    tgt_labels = torch.full((batch_size, max_tgt_len), -100, dtype=torch.long)  # -100 ignored by CrossEntropyLoss

    # Padding masks: True where PAD, False where real token
    src_padding_mask = torch.ones((batch_size, max_src_len), dtype=torch.bool)
    tgt_padding_mask = torch.ones((batch_size, max_tgt_len), dtype=torch.bool)

    for i, (src, tgt_in, tgt_lab) in enumerate(zip(src_seqs, tgt_in_seqs, tgt_label_seqs)):
        src_len = src.size(0)
        tgt_len = tgt_in.size(0)

        src_ids[i, :src_len] = src
        tgt_input_ids[i, :tgt_len] = tgt_in
        tgt_labels[i, :tgt_len] = tgt_lab

        src_padding_mask[i, :src_len] = False
        tgt_padding_mask[i, :tgt_len] = False

    batch_dict = {
        "src_ids": src_ids,                       # [B, S_src]
        "tgt_input_ids": tgt_input_ids,           # [B, S_tgt]
        "tgt_labels": tgt_labels,                 # [B, S_tgt]
        "src_padding_mask": src_padding_mask,     # [B, S_src]  True = PAD
        "tgt_padding_mask": tgt_padding_mask,     # [B, S_tgt]  True = PAD
    }

    return batch_dict
