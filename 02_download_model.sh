#!/bin/bash
# =============================================================================
# STEP 2: Download LLaDA-8B-Base model weights
# Run on LOGIN NODE — wget handles resume automatically
# Usage: bash 02_download_model.sh
# Model is ~16GB total (6 shards)
# =============================================================================

MODEL_DIR="/share1/$USER/llada_model"
BASE="https://huggingface.co/GSAI-ML/LLaDA-8B-Base/resolve/main"

echo "=== Downloading LLaDA-8B-Base to $MODEL_DIR ==="
mkdir -p "$MODEL_DIR"

# Small config files (fast)
SMALL_FILES=(
    "config.json"
    "configuration_llada.py"
    "modeling_llada.py"
    "tokenizer.json"
    "tokenizer_config.json"
    "special_tokens_map.json"
    "model.safetensors.index.json"
)

echo "[1/2] Config and tokenizer files..."
for fname in "${SMALL_FILES[@]}"; do
    out="$MODEL_DIR/$fname"
    if [ -f "$out" ] && [ "$(stat -c%s "$out")" -gt 100 ]; then
        echo "  SKIP: $fname"
        continue
    fi
    echo -n "  DL: $fname ... "
    wget -q -O "$out" "$BASE/$fname"
    echo "ok"
done

# Model shards (~2.7GB each)
echo ""
echo "[2/2] Model weight shards (6 x ~2.7GB)..."
for i in 1 2 3 4 5 6; do
    fname="model-0000${i}-of-00006.safetensors"
    out="$MODEL_DIR/$fname"
    sz=0
    [ -f "$out" ] && sz=$(stat -c%s "$out")
    if [ "$sz" -gt 2000000000 ]; then
        echo "  SKIP: $fname  ($(( sz / 1000000000 ))GB)"
        continue
    fi
    echo "  DL shard $i/6: $fname"
    # -c = resume, --tries=3 = retry on failure
    wget -c -q --show-progress --tries=3 --timeout=120 \
         -O "$out" "$BASE/$fname"
    echo "  done: $(du -sh $out | cut -f1)"
done

echo ""
echo "=== Model download complete ==="
ls -lh "$MODEL_DIR/"*.safetensors 2>/dev/null
du -sh "$MODEL_DIR"

# ── Patch __init__ to accept **kwargs (needed for loading) ────────────────────
echo ""
echo "Patching modeling_llada.py..."
python3 - << 'PYEOF'
import sys
MODEL_DIR = "/share1/" + __import__('os').environ['USER'] + "/llada_model"
fpath = f"{MODEL_DIR}/modeling_llada.py"
with open(fpath) as f:
    src = f.read()
OLD = ("def __init__(self, config: LLaDAConfig, "
       "model: Optional[LLaDAModel] = None, init_params: bool = False):")
NEW = ("def __init__(self, config: LLaDAConfig, "
       "model: Optional[LLaDAModel] = None, init_params: bool = False, **kwargs):")
if OLD in src:
    with open(fpath, "w") as f:
        f.write(src.replace(OLD, NEW))
    print("Patched OK")
else:
    print("Already patched or signature differs")
PYEOF

echo ""
echo "Next step: sbatch 03_extract_features.job"
