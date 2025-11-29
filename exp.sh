#!/bin/bash
set -euo pipefail

# -------------------------------------------
# Paths to transcripts (already created)
# -------------------------------------------
D1="data/tinystories_transcript_50k"
D2="data/wiki103_transcript_50k"
D1_5="data/tinystories_transcript_50k_d1.5"   # strict shuffled-index version of D1

# -------------------------------------------
# Model + results directories
# -------------------------------------------
MODELDIR="models"
mkdir -p "$MODELDIR"

RESULTS_FILE="results_all.txt"
: > "$RESULTS_FILE"   # truncate existing file

echo "================================================"
echo " 0. CREATE D1.5 (SHUFFLED-INDEX VERSION OF D1)"
echo "================================================"

if [ ! -d "$D1_5" ]; then
  python - <<'PY'
from datasets import load_from_disk
import numpy as np
import os

src = "data/tinystories_transcript_50k"
dst = "data/tinystories_transcript_50k_d1.5"

ds = load_from_disk(src)
n = len(ds)
rng = np.random.default_rng(123)
perm = rng.permutation(n)

# Assign a *new* random index, but keep the original row order.
# This way, the "order" column used in the test is a random permutation.
def assign_new_index(ex, idx):
    return {"index": int(perm[idx])}

ds2 = ds.map(assign_new_index, with_indices=True)
ds2 = ds2.select_columns(["index", "tokens"])
os.makedirs(os.path.dirname(dst), exist_ok=True)
ds2.save_to_disk(dst)
PY
else
  echo "D1.5 exists — skipping creation."
fi

echo "================================================"
echo " 1. TRAIN ALICE ON D1 (ordered transcript)"
echo "================================================"
python train.py \
  --transcript_dir "$D1" \
  --base_model scratch \
  --output_dir "$MODELDIR/alice_d1" \
  --epochs 2 \
  --batch_size 32 \
  --lr 3e-4 \
  --seed 0

echo "================================================"
echo " 2. TRAIN ALICE ON D2 (ordered transcript)"
echo "================================================"
python train.py \
  --transcript_dir "$D2" \
  --base_model scratch \
  --output_dir "$MODELDIR/alice_d2" \
  --epochs 2 \
  --batch_size 32 \
  --lr 3e-4 \
  --seed 0

echo "================================================"
echo " 3. TRAIN BOB_FROM_ALICE ON 30% OF D1 (IN-ORDER)"
echo "================================================"
python train.py \
  --transcript_dir "$D1" \
  --base_model "$MODELDIR/alice_d1" \
  --output_dir "$MODELDIR/bob_from_alice_d1" \
  --subset_frac 0.3 \
  --epochs 1 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 1

echo "================================================"
echo " 4. TRAIN BOB_INDEPENDENT ON D1 (SCRATCH + SHUFFLED DATASET ORDER)"
echo "================================================"
python train.py \
  --transcript_dir "$D1" \
  --base_model scratch \
  --output_dir "$MODELDIR/bob_independent_d1" \
  --epochs 1 \
  --batch_size 32 \
  --lr 3e-4 \
  --shuffle \
  --seed 123

echo "================================================"
echo " 4b. EXTRA BOB_INDEPENDENT REPLICATES (NULL MODELS)"
echo "================================================"

for SEED in 10 11 12; do
  OUTDIR="$MODELDIR/bob_independent_d1_seed${SEED}"
  python train.py \
    --transcript_dir "$D1" \
    --base_model scratch \
    --output_dir "$OUTDIR" \
    --epochs 1 \
    --batch_size 32 \
    --lr 3e-4 \
    --shuffle \
    --seed "$SEED"
done

echo "================================================"
echo " 5. PALIMPSEST: CONTINUE BOB_FROM_ALICE ON FRACTIONS OF D2"
echo "================================================"

python train.py \
  --transcript_dir "$D2" \
  --base_model "$MODELDIR/bob_from_alice_d1" \
  --output_dir "$MODELDIR/bob_from_alice_d1_then_d2_frac0.1" \
  --subset_frac 0.1 \
  --epochs 1 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 2

python train.py \
  --transcript_dir "$D2" \
  --base_model "$MODELDIR/bob_from_alice_d1" \
  --output_dir "$MODELDIR/bob_from_alice_d1_then_d2_frac0.3" \
  --subset_frac 0.3 \
  --epochs 1 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 3

python train.py \
  --transcript_dir "$D2" \
  --base_model "$MODELDIR/bob_from_alice_d1" \
  --output_dir "$MODELDIR/bob_from_alice_d1_then_d2_full" \
  --subset_frac 1.0 \
  --epochs 1 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 4

echo "================================================"
echo " 6. QUERY TESTS (NO REFERENCE MODEL, n_samples=20000)"
echo "================================================"

run_test() {
  local MODEL="$1"
  local TRANSCRIPT="$2"
  local LABEL="$3"
  local N_SAMPLES="${4:-20000}"

  echo "==== Running: $LABEL (n_samples=${N_SAMPLES}) ====" | tee -a "$RESULTS_FILE"

  local OUT
  OUT=$(python run_query_test.py \
    --model "$MODEL" \
    --n_samples "$N_SAMPLES" \
    --transcript "$TRANSCRIPT" | tail -n 1)

  echo "$LABEL: $OUT" | tee -a "$RESULTS_FILE"
  echo "" | tee -a "$RESULTS_FILE"
}

# 6.1 Alice self-correlations (main results)
run_test "$MODELDIR/alice_d1" "$D1"   "alice_d1_on_D1"
run_test "$MODELDIR/alice_d2" "$D2"   "alice_d2_on_D2"

# 6.1b On shuffled D1.5 (should be ~independent)
run_test "$MODELDIR/alice_d1" "$D1_5" "alice_d1_on_D1.5"
run_test "$MODELDIR/alice_d2" "$D1_5" "alice_d2_on_D1.5"

# 6.2 Derivative vs independent on D1
run_test "$MODELDIR/bob_from_alice_d1"  "$D1" "bob_from_alice_on_D1"
run_test "$MODELDIR/bob_independent_d1" "$D1" "bob_independent_on_D1"

# Extra: independent null replicates on D1
for SEED in 10 11 12; do
  run_test "$MODELDIR/bob_independent_d1_seed${SEED}" "$D1" "bob_independent_seed${SEED}_on_D1"
done

# 6.3 Cross dataset checks
run_test "$MODELDIR/alice_d1" "$D2" "alice_d1_on_D2"
run_test "$MODELDIR/alice_d2" "$D1" "alice_d2_on_D1"

# 6.4 Palimpsest models probed with D1 (original phase)
run_test "$MODELDIR/bob_from_alice_d1_then_d2_frac0.1" "$D1" "palimpsest_frac0.1_on_D1"
run_test "$MODELDIR/bob_from_alice_d1_then_d2_frac0.3" "$D1" "palimpsest_frac0.3_on_D1"
run_test "$MODELDIR/bob_from_alice_d1_then_d2_full"    "$D1" "palimpsest_full_on_D1"

# 6.5 Palimpsest models probed with D2 (new phase)
run_test "$MODELDIR/bob_from_alice_d1_then_d2_frac0.1" "$D2" "palimpsest_frac0.1_on_D2"
run_test "$MODELDIR/bob_from_alice_d1_then_d2_frac0.3" "$D2" "palimpsest_frac0.3_on_D2"
run_test "$MODELDIR/bob_from_alice_d1_then_d2_full"    "$D2" "palimpsest_full_on_D2"

# 6.6 Sanity: palimpsest model probed wrt shuffled order (should be ~independent)
run_test "$MODELDIR/bob_from_alice_d1_then_d2_full" "$D1_5" "palimpsest_full_on_D1.5"

echo "================================================"
echo " ALL EXPERIMENTS COMPLETE — RESULTS SAVED TO $RESULTS_FILE"
echo "================================================"
