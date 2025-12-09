from torch.utils.data import DataLoader

from src.data.dataset import MTTextDataset
from src.data.utils import compute_length_stats
from src.data.batch_sampler import TokenBatchSampler
from src.data.collate import collate_mt_batch


def main():
    train_en = "data/processed/train.en"
    train_de = "data/processed/train.de"
    spm_model = "models/bpe/spm.model"

    max_source_len = 128
    max_target_len = 128
    max_tokens_per_batch = 4000  # you can tune this later

    dataset = MTTextDataset(
        src_path=train_en,
        tgt_path=train_de,
        spm_model_path=spm_model,
        max_source_len=max_source_len,
        max_target_len=max_target_len,
    )

    # approximate lengths based on whitespace tokens
    src_lengths, tgt_lengths = compute_length_stats(train_en, train_de)

    batch_sampler = TokenBatchSampler(
        src_lengths=src_lengths,
        tgt_lengths=tgt_lengths,
        max_tokens_per_batch=max_tokens_per_batch,
        shuffle=True,
    )

    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_mt_batch,
        num_workers=2,   # you can increase if your CPU is strong
        pin_memory=True,
    )

    # Take one batch and inspect shapes
    batch = next(iter(train_loader))
    print("src_ids:", batch["src_ids"].shape)
    print("tgt_input_ids:", batch["tgt_input_ids"].shape)
    print("tgt_labels:", batch["tgt_labels"].shape)
    print("src_padding_mask:", batch["src_padding_mask"].shape)
    print("tgt_padding_mask:", batch["tgt_padding_mask"].shape)


if __name__ == "__main__":
    main()
