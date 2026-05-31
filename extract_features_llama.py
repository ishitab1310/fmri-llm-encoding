#!/usr/bin/env python3
import os, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",  required=True)
parser.add_argument("--data_dir",   required=True)
parser.add_argument("--feat_dir",   required=True)
parser.add_argument("--langs",      nargs="+", default=["EN","FR","CN"])
parser.add_argument("--subjects",   nargs="+", required=True)
args = parser.parse_args()

TR, TRIM_TRS, CONTEXT_WIN = 2.0, 4, 20
BOLD_SUFFIX = "space-MNIColin27_desc-preproc_bold.nii.gz"
TASK = {"EN":"lppEN","FR":"lppFR","CN":"lppCN"}
SUBJECTS_BY_LANG = {
    "EN":[s for s in args.subjects if "EN" in s],
    "FR":[s for s in args.subjects if "FR" in s],
    "CN":[s for s in args.subjects if "CN" in s],
}

print("="*60)
print("Llama Feature Extraction")
print(f"Model    : {args.model_dir}")
print(f"Data     : {args.data_dir}")
print(f"Features : {args.feat_dir}")
print("="*60)

# Load annotation
ANN_CACHE = {}
for lang in args.langs:
    task = TASK[lang]
    path = os.path.join(args.data_dir, "annotation", lang,
                        f"{task}_word_information.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        ANN_CACHE[lang] = df
        print(f"  Annotation {lang}: {len(df)} words")
    else:
        print(f"  WARNING: no annotation for {lang}")
        ANN_CACHE[lang] = None

def build_windows(lang, run_num, n_trs):
    df = ANN_CACHE.get(lang)
    if df is None:
        return ["[silence]"] * n_trs
    run_df = df[df["section"] == run_num].copy()
    if run_df.empty:
        return ["[silence]"] * n_trs
    run_df = run_df.sort_values("onset").reset_index(drop=True)
    words   = run_df["word"].tolist()
    onsets  = run_df["onset"].values
    offsets = run_df["offset"].values if "offset" in run_df.columns \
              else onsets + 0.3
    windows = []
    for t in range(n_trs):
        t0, t1 = t * TR, (t + 1) * TR
        ctx = [words[i] for i in range(len(words))
               if offsets[i] < t0][-CONTEXT_WIN:]
        cur = [words[i] for i in range(len(words))
               if onsets[i] >= t0 and onsets[i] < t1]
        txt = " ".join(str(w) for w in ctx + cur).strip()
        windows.append(txt if txt else "[silence]")
    non_sil = sum(1 for w in windows if w != "[silence]")
    print(f"    section {run_num}: {n_trs} TRs, {non_sil} with words")
    return windows

# Load Llama
print("\nLoading Llama (float16)...")
tokenizer = AutoTokenizer.from_pretrained(
    args.model_dir, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model_dir,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    local_files_only=True,
    low_cpu_mem_usage=True,
    device_map={"":0},
    output_hidden_states=True,
)
model.eval()

N_LAYERS   = model.config.num_hidden_layers
HIDDEN_DIM = model.config.hidden_size
EXTRACT_INDICES = sorted(set(list(range(1, N_LAYERS+1, 4)) + [N_LAYERS]))
LAYER_NAMES     = [f"L{i:02d}" for i in EXTRACT_INDICES]

print(f"Loaded: {N_LAYERS} layers, {HIDDEN_DIM} hidden")
print(f"GPU: {torch.cuda.memory_allocated(0)/1e9:.1f}GB")
print(f"Layers: {LAYER_NAMES}")

_t = tokenizer("the little prince", return_tensors="pt").to("cuda")
with torch.no_grad():
    _o = model(**_t)
print(f"Sanity: {len(_o.hidden_states)} hidden states OK")
del _t, _o; torch.cuda.empty_cache()

@torch.no_grad()
def get_hidden(text):
    inp = tokenizer(text, return_tensors="pt",
                    truncation=True, max_length=128).to("cuda")
    out = model(**inp)
    result = {}
    for name, idx in zip(LAYER_NAMES, EXTRACT_INDICES):
        real = min(idx, len(out.hidden_states)-1)
        result[name] = out.hidden_states[real][0].float().mean(0).cpu().numpy()
    return result

for lang in args.langs:
    if not SUBJECTS_BY_LANG.get(lang):
        continue
    task = TASK[lang]
    print(f"\n{'='*55}  {lang}")
    for sub in SUBJECTS_BY_LANG[lang]:
        out_dir  = os.path.join(args.feat_dir, lang, sub)
        bold_dir = os.path.join(args.data_dir, "derivatives", sub, "func")
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.exists(bold_dir):
            print(f"  [{sub}] SKIP — no bold_dir"); continue
        bolds = sorted([f for f in os.listdir(bold_dir)
                        if task in f and BOLD_SUFFIX in f])
        if not bolds:
            bolds = sorted([f for f in os.listdir(bold_dir)
                           if task in f and f.endswith(".nii.gz")])
        if not bolds:
            print(f"  [{sub}] SKIP — no BOLD"); continue
        print(f"\n  {sub}: {len(bolds)} runs", flush=True)
        layer_runs = {name: [] for name in LAYER_NAMES}
        for run_i, bf in enumerate(bolds):
            run_num = run_i + 1
            img     = nib.load(os.path.join(bold_dir, bf))
            n_trs   = img.shape[3] - TRIM_TRS
            windows = build_windows(lang, run_num, n_trs)
            run_feats = {name: [] for name in LAYER_NAMES}
            for txt in tqdm(windows, desc=f"    run{run_i+1}",
                            ncols=60, leave=False):
                d = get_hidden(txt)
                for name, vec in d.items():
                    run_feats[name].append(vec)
            for name in LAYER_NAMES:
                layer_runs[name].append(
                    np.array(run_feats[name], dtype=np.float32))
            print(f"    run {run_i+1}/{len(bolds)}: {len(windows)} TRs done",
                  flush=True)
        for name in LAYER_NAMES:
            concat = np.concatenate(layer_runs[name], axis=0)
            np.save(os.path.join(out_dir, f"{name}.npy"), concat)
        sample = np.load(os.path.join(out_dir, f"{LAYER_NAMES[4]}.npy"))
        var = np.var(sample, axis=0).mean()
        print(f"  Variance check: {var:.4f} {'OK' if var > 1e-6 else 'WARNING'}")
        json.dump({"sub": sub, "lang": lang, "layers": LAYER_NAMES, "tr": TR},
                  open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
        print(f"  Saved -> {out_dir}", flush=True)

print("\n=== DONE ===")
