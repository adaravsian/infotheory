import argparse
import os
import random

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    AutoModelForCausalLM,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_batch(batch, device):
    """
    batch: list of {"index": int, "tokens": [int]}
    Returns input_ids and labels tensors on the given device.
    """
    input_ids = torch.tensor(
        [b["tokens"] for b in batch],
        dtype=torch.long,
        device=device,
    )
    # Standard LM: labels = inputs (model handles shifting internally)
    return {"input_ids": input_ids, "labels": input_ids}


def build_or_load_model(base_model: str, tokenizer, block_size: int,
                        n_embd: int, n_layer: int, n_head: int):
    """
    base_model:
      - "scratch" => build small GPT-2 from scratch
      - path / HF ID => load with AutoModelForCausalLM.from_pretrained
    """
    if base_model == "scratch":
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=block_size,
            n_ctx=block_size,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
        )
        model = GPT2LMHeadModel(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model)
    return model


def train_model(
    model,
    dataset,
    tokenizer,
    out_dir: str,
    num_epochs: int = 1,
    batch_size: int = 32,
    lr: float = 3e-4,
    shuffle: bool = False,
):
    """
    Generic training loop.
    If shuffle=False, the DataLoader respects dataset order (important for tracing).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # optional; main order is controlled at dataset level
        collate_fn=lambda b: collate_batch(b, device),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    global_step = 0
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % 100 == 0:
                print(
                    f"Epoch {epoch} step {global_step} "
                    f"loss {loss.item():.4f}"
                )

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved model to {out_dir}")


def main():
    parser = argparse.ArgumentParser()

    # --- Data & transcript ---
    parser.add_argument(
        "--transcript_dir",
        type=str,
        required=True,
        help="Path to HF dataset saved with save_to_disk (must have 'index' and 'tokens').",
    )
    parser.add_argument(
        "--subset_frac",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use (prefix in index order).",
    )

    # --- Model source: scratch vs finetune ---
    parser.add_argument(
        "--base_model",
        type=str,
        default="scratch",
        help=(
            "Base model to start from. "
            "Use 'scratch' to build a small GPT-2 from scratch, "
            "or a path / HF ID to finetune an existing model."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the trained model.",
    )

    # --- Architecture params (used only if base_model == scratch) ---
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)

    # --- Training hyperparams ---
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help=(
            "If set, shuffle the dataset order before training "
            "(breaks transcript order; use for independent models)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading transcript from {args.transcript_dir} ...")
    ds = load_from_disk(args.transcript_dir)

    # Sanity: enforce int tokens
    def fix_tokens(example):
        return {"tokens": [int(t) for t in example["tokens"]]}

    ds = ds.map(fix_tokens)

    # First, define the canonical transcript order by the 'index' column.
    if "index" in ds.column_names:
        ds = ds.sort("index")

    # Now, if shuffle=True, we BREAK that order for independence.
    if args.shuffle:
        print(f"Shuffling dataset order with seed={args.seed} ...")
        ds = ds.shuffle(seed=args.seed)

    # Apply subset prefix (in *current* order)
    if args.subset_frac < 1.0:
        subset_len = int(len(ds) * args.subset_frac)
        print(f"Using first {subset_len}/{len(ds)} sequences (prefix) ...")
        ds = ds.select(range(subset_len))
    else:
        print(f"Using full dataset: {len(ds)} sequences.")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    block_size = len(ds[0]["tokens"])
    print(f"Block size (sequence length): {block_size}")

    # Build or load model (pretrain vs finetune)
    print(f"Building/loading model from base_model='{args.base_model}' ...")
    model = build_or_load_model(
        base_model=args.base_model,
        tokenizer=tokenizer,
        block_size=block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
    )

    print(
        f"Starting training: epochs={args.epochs}, "
        f"batch_size={args.batch_size}, lr={args.lr}, shuffle={args.shuffle}"
    )
    train_model(
        model=model,
        dataset=ds,
        tokenizer=tokenizer,
        out_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        shuffle=False,  # dataset order already decided; keep DataLoader deterministic
    )


if __name__ == "__main__":
    main()
