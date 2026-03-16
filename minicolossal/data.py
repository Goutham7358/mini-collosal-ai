"""
Data loading for GPT-2 training using WikiText-2 dataset.

WikiText-2 is a standard language modeling benchmark (~2M tokens).
We use tiktoken's GPT-2 tokenizer (byte-pair encoding, 50257 vocab).

The dataset is loaded via HuggingFace 'datasets' library, tokenized,
and chunked into fixed-length sequences for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class WikiTextDataset(Dataset):
    """
    Loads WikiText-2, tokenizes it, and serves fixed-length chunks.

    Each sample is (input_ids, target_ids) where target = input shifted by 1.
    This is the standard next-token-prediction setup for language models.
    """
    def __init__(self, split="train", seq_len=256, max_tokens=None):
        """
        Args:
            split: "train", "validation", or "test"
            seq_len: length of each token sequence
            max_tokens: if set, truncate dataset to this many tokens (for quick testing)
        """
        import tiktoken
        from datasets import load_dataset

        # Load WikiText-2 from HuggingFace
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

        # Tokenize using GPT-2's BPE tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")
        all_tokens = []
        for example in dataset:
            text = example["text"]
            if text.strip():  # skip empty lines
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)

        # Optionally truncate for fast testing
        if max_tokens is not None:
            all_tokens = all_tokens[:max_tokens]

        # Store as a single flat tensor
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.seq_len = seq_len

        # Number of complete sequences we can extract
        # We need seq_len + 1 tokens per sample (input + 1 shifted target)
        self.n_samples = (len(self.tokens) - 1) // seq_len

        print(f"  [{split}] Loaded {len(self.tokens):,} tokens -> {self.n_samples:,} samples "
              f"(seq_len={seq_len})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        input_ids = self.tokens[start:end]
        target_ids = self.tokens[start + 1:end + 1]
        return input_ids, target_ids


def get_dataloader(split="train", seq_len=256, batch_size=8, num_workers=2,
                   distributed=False, rank=0, world_size=1, max_tokens=None):
    """
    Creates a DataLoader for WikiText-2.

    Args:
        split: dataset split
        seq_len: sequence length
        batch_size: per-GPU batch size
        num_workers: dataloader workers
        distributed: if True, use DistributedSampler to shard data across GPUs
        rank: current process rank (for distributed)
        world_size: total number of processes (for distributed)
        max_tokens: truncate dataset for quick testing
    Returns:
        (dataloader, dataset)
    """
    dataset = WikiTextDataset(split=split, seq_len=seq_len, max_tokens=max_tokens)

    sampler = None
    shuffle = True
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        shuffle = False  # sampler handles shuffling

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # drop incomplete batches for consistent batch sizes
    )

    return dataloader, dataset
