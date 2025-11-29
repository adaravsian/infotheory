import argparse
import numpy as np
from datasets import load_from_disk

def main(args):
    print(f"Loading original transcript from {args.source_dir} ...")
    ds = load_from_disk(args.source_dir)

    n = len(ds)
    print(f"Dataset has {n} sequences.")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)  # new “training steps” for each row

    # Replace the index column with a shuffled order
    def assign_new_index(example, idx):
        # idx is the row index (0..n-1); perm[idx] is its new training time
        return {"index": int(perm[idx])}

    print("Assigning shuffled indices (this defines D1.5 order)...")
    ds_shuf = ds.map(assign_new_index, with_indices=True)

    # Keep columns in the same order: index, tokens
    ds_shuf = ds_shuf.select_columns(["index", "tokens"])

    print(f"Saving D1.5 transcript to {args.out_dir} ...")
    ds_shuf.save_to_disk(args.out_dir)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str,
                        default="data/tinystories_transcript_50k",
                        help="Original D1 transcript directory.")
    parser.add_argument("--out_dir", type=str,
                        default="data/tinystories_transcript_50k_d1.5",
                        help="Where to save the shuffled D1.5 transcript.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for the permutation.")
    args = parser.parse_args()
    main(args)
