#!/bin/bash
# =============================================================================
# STEP 1: Download ds003643 from OpenNeuro S3
# Run on LOGIN NODE (no GPU needed, just network)
# Usage: bash 01_download_data.sh
# Takes ~20-40 min for 5 subjects per language
# =============================================================================

set -e

DATA_DIR="/share1/$USER/ds003643"
S3_BASE="s3://openneuro.org/ds003643"

# ── Updated subject list (agreed by pod, version 2.0.1) ──────────────────────
EN_SUBJECTS=("sub-EN057" "sub-EN058" "sub-EN059" "sub-EN061" "sub-EN062")
FR_SUBJECTS=("sub-FR001" "sub-FR002" "sub-FR003" "sub-FR004" "sub-FR005")
CN_SUBJECTS=("sub-CN001" "sub-CN002" "sub-CN003" "sub-CN004" "sub-CN005")

echo "=== Downloading ds003643 to $DATA_DIR ==="
echo "Subjects: EN(${#EN_SUBJECTS[@]}) FR(${#FR_SUBJECTS[@]}) CN(${#CN_SUBJECTS[@]})"
echo ""

# ── Dataset metadata, stimuli, annotations ───────────────────────────────────
echo "[1/3] Metadata + annotation + stimuli..."
aws s3 sync --no-sign-request "$S3_BASE" "$DATA_DIR" \
    --exclude "*" \
    --include "task-*.json" \
    --include "participants.tsv" \
    --include "participants.json" \
    --include "README*" \
    --include "dataset_description.json" \
    --include "annotation/*" \
    --include "stimuli/*" \
    --quiet
echo "  done."

# ── Per-subject: preprocessed BOLD + events ───────────────────────────────────
echo "[2/3] Downloading subject BOLD + events TSV..."

download_subject() {
    local sub=$1
    echo -n "  $sub ... "
    # Preprocessed BOLD from derivatives/ (MNI space, already preprocessed)
    aws s3 sync --no-sign-request \
        "$S3_BASE/derivatives/$sub" \
        "$DATA_DIR/derivatives/$sub" \
        --quiet
    # Events TSV (word timings) from raw subject folder
    aws s3 sync --no-sign-request \
        "$S3_BASE/$sub/func" \
        "$DATA_DIR/$sub/func" \
        --exclude "*" \
        --include "*events.tsv" \
        --quiet
    # Count downloaded BOLD runs
    local n_runs=$(ls "$DATA_DIR/derivatives/$sub/func/"*.nii.gz 2>/dev/null | wc -l)
    local n_evnt=$(ls "$DATA_DIR/$sub/func/"*events.tsv 2>/dev/null | wc -l)
    echo "done ($n_runs BOLD runs, $n_evnt events files)"
}

for sub in "${EN_SUBJECTS[@]}"; do download_subject "$sub"; done
for sub in "${FR_SUBJECTS[@]}"; do download_subject "$sub"; done
for sub in "${CN_SUBJECTS[@]}"; do download_subject "$sub"; done

echo ""
echo "[3/3] Verifying download..."
for sub in "${EN_SUBJECTS[@]}" "${FR_SUBJECTS[@]}" "${CN_SUBJECTS[@]}"; do
    bold_dir="$DATA_DIR/derivatives/$sub/func"
    evnt_dir="$DATA_DIR/$sub/func"
    n_bold=$(ls "$bold_dir/"*.nii.gz 2>/dev/null | wc -l)
    n_evnt=$(ls "$evnt_dir/"*events.tsv 2>/dev/null | wc -l)
    status="OK"
    if [ "$n_bold" -eq 0 ] || [ "$n_evnt" -eq 0 ]; then status="MISSING"; fi
    echo "  [$status] $sub  bold=$n_bold  events=$n_evnt"
done

echo ""
echo "=== Data download complete ==="
du -sh $DATA_DIR
echo "Next step: bash 02_download_model.sh"
