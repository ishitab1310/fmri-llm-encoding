#!/bin/bash
# =============================================================================
# STEP 2: Download Llama-3.2-3B model weights from HuggingFace
# Run directly on LOGIN NODE (no GPU needed, just network)
# Usage: bash 02_download_llama.sh
# Model is ~6GB total (2 safetensors shards)
# =============================================================================

set -e

TOKEN=$(cat ~/.cache/huggingface/token)
MODEL_DIR="/share1/ishita/llama_model"
BASE="https://huggingface.co/meta-llama/Llama-3.2-3B/resolve/main"

echo "=== Downloading Llama-3.2-3B to $MODEL_DIR ==="
mkdir -p "$MODEL_DIR"

# ── Config and tokenizer files (small, fast) ──────────────────────────────────
echo "[1/2] Config and tokenizer files..."
for fname in \
    "config.json" \
    "generation_config.json" \
    "tokenizer.json" \
    "tokenizer_config.json" \
    "special_tokens_map.json" \
    "model.safetensors.index.json"; do

    out="$MODEL_DIR/$fname"
    if [ -f "$out" ] && [ "$(stat -c%s "$out")" -gt 100 ]; then
        echo "  SKIP: $fname"
        continue
    fi
    echo -n "  DL: $fname ... "
    wget -q --header="Authorization: Bearer $TOKEN" \
         -O "$out" "$BASE/$fname" && echo "ok" || echo "FAILED"
done

# ── Model weight shards (~3GB each, 2 total) ──────────────────────────────────
echo ""
echo "[2/2] Model weight shards (2 x ~3GB)..."
for i in 1 2; do
    fname=$(printf "model-%05d-of-00002.safetensors" $i)
    out="$MODEL_DIR/$fname"
    sz=0
    [ -f "$out" ] && sz=$(stat -c%s "$out")
    if [ "$sz" -gt 2000000000 ]; then
        echo "  SKIP: $fname  ($(( sz / 1000000000 ))GB already present)"
        continue
    fi
    echo "  DL shard $i/2: $fname"
    # -c = resume if interrupted, --tries=3 = retry on failure
    wget -c --tries=3 --timeout=120 \
         --header="Authorization: Bearer $TOKEN" \
         -O "$out" "$BASE/$fname"
    echo "  done: $(du -sh $out | cut -f1)"
done

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Download complete ==="
ls -lh "$MODEL_DIR/"
du -sh "$MODEL_DIR"
echo ""
echo "Next step: sbatch 03_extract_features_llama.job"