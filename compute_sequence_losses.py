import argparse
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
import math


def sequence_nll(model, input_ids, device):
    """
    Compute average negative log-likelihood (in nats) per token for a single sequence.
    """
    model.eval()
    with torch.no_grad():
        ids = input_ids.unsqueeze(0)  # [1, L]
        labels = ids.clone()
        outputs = model(input_ids=ids, labels=labels)
        # loss is mean over tokens, already reduced
        return outputs.loss.item()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading transcript from {args.transcript_dir} ...")
    ds = load_from_disk(args.transcript_dir)

    print(f"Loading tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    alice = GPT2LMHeadModel.from_pretrained(args.alice_dir).to(device)
    bob = GPT2LMHeadModel.from_pretrained(args.bob_dir).to(device)

    def add_losses(example):
        ids = torch.tensor(example["tokens"], dtype=torch.long, device=device)
        loss_alice = sequence_nll(alice, ids, device)
        loss_bob = sequence_nll(bob, ids, device)
        return {
            "loss_alice_main": loss_alice,
            "loss_bob_main": loss_bob,
        }

    print("Computing losses (this may take a while)...")
    ds_with_losses = ds.map(
        add_losses,
        batched=False,
        desc="Per-sequence NLL",
    )

    print(f"Saving updated transcript to {args.out_dir} ...")
    ds_with_losses.save_to_disk(args.out_dir)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transcript_dir",
        type=str,
        default="data/wiki_transcript",
        help="Original transcript (index, tokens).",
    )
    parser.add_argument(
        "--alice_dir",
        type=str,
        default="models/alice_small",
    )
    parser.add_argument(
        "--bob_dir",
        type=str,
        default="models/bob_from_alice",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/wiki_transcript_with_losses",
    )
    args = parser.parse_args()
    main(args)
