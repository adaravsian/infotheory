import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import os


def build_transcript(
    out_dir: str,
    hf_dataset_name: str = "wikitext",
    hf_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    block_size: int = 256,
    max_sequences: int = 50000,
):
    """
    Build an ordered transcript from a HuggingFace text dataset.

    The transcript is a HF dataset saved to disk with columns:
      - index: int (0, 1, 2, ...)
      - tokens: list[int] (length block_size)

    Args:
      out_dir: where to save the dataset (for load_from_disk).
      hf_dataset_name: HF dataset name (e.g. 'wikitext', 'roneneldan/TinyStories').
      hf_config: dataset config / subset (e.g. 'wikitext-2-raw-v1').
      split: split to use (e.g. 'train').
      block_size: sequence length in tokens.
      max_sequences: cap on number of sequences (e.g. 50000).
    """
    print(f"Loading dataset {hf_dataset_name} / {hf_config} split={split} ...")
    raw = load_dataset(hf_dataset_name, hf_config, split=split)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        # returns a dict with "input_ids", "attention_mask", ...
        return tokenizer(examples["text"])

    print("Tokenizing...")
    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw.column_names,
        desc="Tokenizing",
    )

    def group_texts(examples):
        # Flatten then cut into fixed blocks
        concatenated = np.concatenate(examples["input_ids"])
        total_length = (len(concatenated) // block_size) * block_size
        concatenated = concatenated[:total_length]

        # force int64 so HF doesn't try to store as floats
        input_ids = np.array(concatenated, dtype=np.int64).reshape(-1, block_size)
        return {"tokens": input_ids.tolist()}

    print("Grouping into fixed-length sequences...")
    chunked = tokenized.map(
        group_texts,
        batched=True,
        remove_columns=tokenized.column_names,
        desc="Grouping",
    )

    # Flatten any batching so we have a simple 1D index over sequences
    # (map with group_texts can create multiple rows per original row)
    chunked = chunked.flatten_indices()

    # Cap to max_sequences
    n_total = len(chunked)
    n_used = min(max_sequences, n_total)
    if n_used < max_sequences:
        print(f"Warning: dataset only produced {n_total} sequences; using {n_used}.")
    else:
        print(f"Using first {n_used} sequences out of {n_total}.")
    chunked = chunked.select(range(n_used))

    print("Adding index column (this defines training order)...")
    chunked = chunked.map(
        lambda ex, idx: {"index": int(idx)},
        with_indices=True,
        desc="Indexing",
    )

    # Ensure tokens are pure ints (avoid float issues later)
    def fix_tokens(example):
        return {"tokens": [int(t) for t in example["tokens"]]}

    chunked = chunked.map(fix_tokens, desc="Casting tokens to int")

    # Reorder columns: index, tokens
    chunked = chunked.select_columns(["index", "tokens"])

    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving transcript to {out_dir} ...")
    chunked.save_to_disk(out_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/wiki_transcript",
        help="Where to save the HF dataset (for load_from_disk).",
    )
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name, e.g. 'wikitext' or 'roneneldan/TinyStories'.",
    )
    parser.add_argument(
        "--hf_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset config / subset, e.g. 'wikitext-2-raw-v1'.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Data split to use, e.g. 'train', 'validation', 'test'.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=256,
        help="Sequence length in tokens.",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=50000,
        help="Maximum number of sequences to keep (prefix in order).",
    )
    args = parser.parse_args()

    build_transcript(
        out_dir=args.out_dir,
        hf_dataset_name=args.hf_dataset_name,
        hf_config=args.hf_config,
        split=args.split,
        block_size=args.block_size,
        max_sequences=args.max_sequences,
    )
