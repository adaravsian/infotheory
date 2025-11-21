#!/bin/bash
set -euo pipefail

# -------------------------------------------
# Paths to transcripts
# -------------------------------------------
D1="data/tinystories_transcript_50k"
D2="data/wiki103_transcript_50k"

# -------------------------------------------
# Model + results directories
# -------------------------------------------
MODELDIR="models"
mkdir -p "$MODELDIR"

RESULTS_FILE="results_all.txt"
: > "$RESULTS_FILE"   # truncate existing file

echo "================================================"
echo " CLEAN START (optional manual cleanup)"
echo "================================================"
# If you really want to wipe everything each run, uncomment:
# rm -rf "$MODELDIR"
# mkdir -p "$MODELDIR"
# : > "$RESULTS_FILE"

echo "================================================"
echo " 1. TRAIN ALICE ON D1 (TinyStories)"
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
echo " 2. TRAIN ALICE ON D2 (Wiki103)"
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
echo " 3. TRAIN BOB_FROM_ALICE ON PREFIX OF D1"
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
echo " 4. TRAIN BOB_INDEPENDENT ON D1 (SHUFFLED)"
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
echo " 5. PALIMPSEST: BOB_FROM_ALICE_D1 CONTINUED ON FRACTIONS OF D2"
echo "================================================"

# ---- 5.1: 10% of D2 ----
python train.py \
  --transcript_dir "$D2" \
  --base_model "$MODELDIR/bob_from_alice_d1" \
  --output_dir "$MODELDIR/bob_from_alice_d1_then_d2_frac0.1" \
  --subset_frac 0.1 \
  --epochs 1 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 2

# ---- 5.2: 30% of D2 ----
python train.py \
  --transcript_dir "$D2" \
  --base_model "$MODELDIR/bob_from_alice_d1" \
  --output_dir "$MODELDIR/bob_from_alice_d1_then_d2_frac0.3" \
  --subset_frac 0.3 \
  --epochs 1 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 3

# ---- 5.3: 100% of D2 ----
python train.py \
  --transcript_dir "$D2" \
  --base_model "$MODELDIR/bob_from_alice_d1" \
  --output_dir "$MODELDIR/bob_from_alice_d1_then_d2_full" \
  --epochs 1 \
  --batch_size 32 \
  --lr 1e-4 \
  --seed 4

echo "================================================"
echo " 6. RUN ALL QUERY-SETTING TESTS (ϕ_query^ref)"
echo "================================================"

run_test() {
  local MODEL="$1"
  local REF="$2"
  local TRANSCRIPT="$3"
  local LABEL="$4"

  echo "==== Running test: $LABEL ===="
  # Grab only the final SignificanceResult line from run_query_test.py
  local STATLINE
  STATLINE=$(python run_query_test.py \
    --model "$MODEL" \
    --ref_model "$REF" \
    --n_samples 20000 \
    --transcript "$TRANSCRIPT" \
    | tail -n 1)

  # Append to one master results file
  echo "$LABEL: $STATLINE" | tee -a "$RESULTS_FILE"
}

# 6.1 Baseline self-correlations (Alice on own data)
run_test "$MODELDIR/alice_d1" "$MODELDIR/alice_d2" "$D1" "alice_d1_on_D1"
run_test "$MODELDIR/alice_d2" "$MODELDIR/alice_d1" "$D2" "alice_d2_on_D2"

# 6.2 Same-data vs independent on D1
run_test "$MODELDIR/bob_from_alice_d1"   "$MODELDIR/alice_d1" "$D1" "bob_from_alice_on_D1"
run_test "$MODELDIR/bob_independent_d1"  "$MODELDIR/alice_d1" "$D1" "bob_independent_on_D1"

# 6.3 Cross-dataset independence
run_test "$MODELDIR/alice_d1" "$MODELDIR/alice_d2" "$D2" "alice_d1_on_D2"
run_test "$MODELDIR/alice_d2" "$MODELDIR/alice_d1" "$D1" "alice_d2_on_D1"

# 6.4 Palimpsest: D1 → (partial D1) → D2, probed with both D1 and D2 transcripts
# Probed w.r.t. D1 ordering
run_test "$MODELDIR/bob_from_alice_d1_then_d2_frac0.1" "$MODELDIR/alice_d1" "$D1" "palimpsest_frac0.1_on_D1"
run_test "$MODELDIR/bob_from_alice_d1_then_d2_frac0.3" "$MODELDIR/alice_d1" "$D1" "palimpsest_frac0.3_on_D1"
run_test "$MODELDIR/bob_from_alice_d1_then_d2_full"    "$MODELDIR/alice_d1" "$D1" "palimpsest_full_on_D1"

# Probed w.r.t. D2 ordering
run_test "$MODELDIR/bob_from_alice_d1_then_d2_frac0.1" "$MODELDIR/alice_d2" "$D2" "palimpsest_frac0.1_on_D2"
run_test "$MODELDIR/bob_from_alice_d1_then_d2_frac0.3" "$MODELDIR/alice_d2" "$D2" "palimpsest_frac0.3_on_D2"
run_test "$MODELDIR/bob_from_alice_d1_then_d2_full"    "$MODELDIR/alice_d2" "$D2" "palimpsest_full_on_D2"

echo "================================================"
echo " ALL EXPERIMENTS COMPLETE!"
echo " Consolidated results in: $RESULTS_FILE"
echo "================================================"
