#!/usr/bin/env python3
"""
encoding_model_llama.py
========================
Ridge regression encoding model for Llama features.
Identical pipeline to encoding_model.py (LLaDA version) —
only the default feature directory and print labels differ.

Phase 1 : English layer × lag search  -> BEST_LAYER, BEST_LAG
Phase 2 : Transfer experiment EN -> FR

Parcel-based approach (6 ROIs, 10mm spheres) — identical to teammate pipeline.
LORO cross-validation with ridge regression.
"""

import os, json, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nibabel as nib
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",    required=True)
parser.add_argument("--feat_dir",    required=True,
                    help="Root of Llama feature dir (e.g. /home2/ishita/features)")
parser.add_argument("--results_dir", required=True)
parser.add_argument("--n_jobs",      type=int, default=8)
args = parser.parse_args()

os.makedirs(args.results_dir, exist_ok=True)

# ── Shared constants (IDENTICAL to teammate / LLaDA pipeline) ─────────────────
TR, TRIM_TRS = 2.0, 4
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
RIDGE_CV     = 5
BOLD_SUFFIX  = "space-MNIColin27_desc-preproc_bold.nii.gz"
TASK = {"EN": "lppEN", "FR": "lppFR", "CN": "lppCN"}

# ── All subjects — full ds003643 dataset (uncomment to run all, comment block below) ──
TOTAL_SUBJECTS = {
    "EN": [
        "sub-EN057", "sub-EN058", "sub-EN059", "sub-EN061", "sub-EN062",
        "sub-EN063", "sub-EN064", "sub-EN065", "sub-EN067", "sub-EN068",
        "sub-EN069", "sub-EN070", "sub-EN072", "sub-EN073", "sub-EN074",
        "sub-EN075", "sub-EN076", "sub-EN077", "sub-EN078", "sub-EN079",
        "sub-EN081", "sub-EN082", "sub-EN083", "sub-EN084", "sub-EN086",
        "sub-EN087", "sub-EN088", "sub-EN089", "sub-EN091", "sub-EN092",
        "sub-EN093", "sub-EN094", "sub-EN095", "sub-EN096", "sub-EN097",
        "sub-EN098", "sub-EN099", "sub-EN100", "sub-EN101", "sub-EN103",
        "sub-EN104", "sub-EN105", "sub-EN106", "sub-EN108", "sub-EN109",
        "sub-EN110", "sub-EN113", "sub-EN114", "sub-EN115",
    ],
    "FR": [
        "sub-FR001", "sub-FR002", "sub-FR003", "sub-FR004", "sub-FR005",
        "sub-FR006", "sub-FR007", "sub-FR008", "sub-FR009", "sub-FR010",
        "sub-FR011", "sub-FR012", "sub-FR013", "sub-FR014", "sub-FR015",
        "sub-FR016", "sub-FR017", "sub-FR018", "sub-FR019", "sub-FR020",
        "sub-FR022", "sub-FR023", "sub-FR024", "sub-FR025", "sub-FR026",
        "sub-FR028", "sub-FR029", "sub-FR030",
    ],
    "CN": [
        "sub-CN001", "sub-CN002", "sub-CN003", "sub-CN004", "sub-CN005",
        "sub-CN006", "sub-CN007", "sub-CN008", "sub-CN009", "sub-CN010",
        "sub-CN011", "sub-CN013", "sub-CN014", "sub-CN015", "sub-CN016",
        "sub-CN017", "sub-CN018", "sub-CN019", "sub-CN020", "sub-CN021",
        "sub-CN022", "sub-CN023", "sub-CN024", "sub-CN025", "sub-CN026",
        "sub-CN027", "sub-CN028", "sub-CN029", "sub-CN030", "sub-CN031",
        "sub-CN032", "sub-CN033", "sub-CN034", "sub-CN036", "sub-CN037",
    ],
}

# ── Active: 5 subjects per language (pod-agreed subset) ──────────────────────
SUBJECTS = {
    "EN": ["sub-EN057", "sub-EN058", "sub-EN059", "sub-EN061", "sub-EN062"],
    "FR": ["sub-FR001", "sub-FR002", "sub-FR003", "sub-FR004", "sub-FR005"],
    "CN": ["sub-CN001", "sub-CN002", "sub-CN003", "sub-CN004", "sub-CN005"],
}

# ── Alt. — full ds003643 dataset (uncomment to run all, comment block below) ──
SUBJECTS = TOTAL_SUBJECTS

# Parcels — IDENTICAL to teammate (6 ROIs, 10mm spheres)
BUILTIN_ROIS = {
    "IFG":  {"mni_xyz": (-51,  28,   3), "radius_mm": 10.0},
    "STG":  {"mni_xyz": (-57, -16,   4), "radius_mm": 10.0},
    "MTG":  {"mni_xyz": (-60, -32,  -4), "radius_mm": 10.0},
    "AngG": {"mni_xyz": (-48, -58,  28), "radius_mm": 10.0},
    "STS":  {"mni_xyz": (-54, -40,   8), "radius_mm": 10.0},
    "mPFC": {"mni_xyz": (  -4, 52,  -4), "radius_mm": 10.0},
}
PARCEL_KEYS = list(BUILTIN_ROIS.keys())

# Lag grid — IDENTICAL to teammate
LAG_GRID = {
    "lag024":   [0.0, 2.0, 4.0],
    "lag246":   [2.0, 4.0, 6.0],
    "lag02468": [0.0, 2.0, 4.0, 6.0, 8.0],
}

print("="*60)
print("A3 Llama Encoding Model")
print(f"Data dir    : {args.data_dir}")
print(f"Feature dir : {args.feat_dir}")
print(f"Results dir : {args.results_dir}")
print(f"CPUs        : {args.n_jobs}")
print(f"Parcels     : {PARCEL_KEYS}")
print(f"Lag grid    : {list(LAG_GRID.keys())}")
print(f"Alphas      : {RIDGE_ALPHAS}")
print("="*60)

# ── Sphere mask ───────────────────────────────────────────────────────────────
def make_sphere_mask(affine, vol_shape, mni_center, radius_mm):
    i, j, k = np.mgrid[0:vol_shape[0], 0:vol_shape[1], 0:vol_shape[2]]
    coords   = np.stack([i.ravel(), j.ravel(), k.ravel(),
                         np.ones(i.size)], 0)
    mm   = (affine @ coords)[:3].T
    dist = np.linalg.norm(
        mm - np.asarray(mni_center, float)[None, :], axis=1)
    return (dist <= float(radius_mm)).reshape(vol_shape)

# ── Load BOLD as parcel time-series ──────────────────────────────────────────
def load_bold_parcels(sub, lang):
    """Returns list of (n_trs, 6) float32 arrays, one per run."""
    task     = TASK[lang]
    bold_dir = os.path.join(args.data_dir, "derivatives", sub, "func")

    bolds = sorted([f for f in os.listdir(bold_dir)
                    if task in f and BOLD_SUFFIX in f])
    if not bolds:
        bolds = sorted([f for f in os.listdir(bold_dir)
                        if task in f and f.endswith(".nii.gz")])
    if not bolds:
        raise FileNotFoundError(f"No BOLD for {sub} {task}")

    runs = []
    for bf in bolds:
        img    = nib.load(os.path.join(bold_dir, bf))
        data   = img.get_fdata(dtype=np.float32)
        affine = img.affine
        vshape = img.shape[:3]

        pts = []
        for spec in BUILTIN_ROIS.values():
            mask = make_sphere_mask(affine, vshape,
                                    spec["mni_xyz"], spec["radius_mm"])
            pts.append(data[mask].mean(0) if mask.sum() > 0
                       else np.zeros(data.shape[3], dtype=np.float32))

        bold = np.stack(pts, axis=1)[TRIM_TRS:]   # (n_trs, 6)
        mu   = bold.mean(0, keepdims=True)
        sig  = bold.std(0, keepdims=True) + 1e-8
        runs.append((bold - mu) / sig)

    return runs

# ── FIR lag stack ─────────────────────────────────────────────────────────────
def lag_stack_sec(feats, lag_seconds):
    """feats (T,D) -> (T, D*len(lag_seconds)), using lags in seconds."""
    T, D    = feats.shape
    lags_tr = [int(round(s / TR)) for s in lag_seconds]
    out     = np.zeros((T, D * len(lags_tr)), dtype=np.float32)
    for i, lag in enumerate(lags_tr):
        s = np.zeros_like(feats)
        if lag > 0:
            s[lag:] = feats[:-lag]
        else:
            s = feats.copy()
        out[:, i*D:(i+1)*D] = s
    return out

# ── LORO ridge ────────────────────────────────────────────────────────────────
def fit_loro_ridge(feat_runs, bold_runs, lag_seconds):
    """
    Leave-One-Run-Out cross-validated ridge regression.
    Returns mean Pearson r per parcel: (n_parcels,)
    """
    n_runs    = len(feat_runs)
    n_parcels = bold_runs[0].shape[1]

    if n_runs == 1:
        X = lag_stack_sec(feat_runs[0], lag_seconds)
        Y = bold_runs[0]
        n = min(len(X), len(Y)); X, Y = X[:n], Y[:n]
        sp = int(n * 0.8)
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[:sp]); Xte = sc.transform(X[sp:])
        ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=RIDGE_CV)
        ridge.fit(Xtr, Y[:sp])
        pred = ridge.predict(Xte)
        return np.array([
            pearsonr(Y[sp:, p], pred[:, p])[0]
            if Y[sp:, p].std() > 1e-8 and pred[:, p].std() > 1e-8 else 0.
            for p in range(n_parcels)
        ])

    r_folds = np.zeros((n_runs, n_parcels))
    for ti in range(n_runs):
        trX = np.concatenate([lag_stack_sec(feat_runs[i], lag_seconds)
                               for i in range(n_runs) if i != ti])
        trY = np.concatenate([bold_runs[i]
                               for i in range(n_runs) if i != ti])
        teX = lag_stack_sec(feat_runs[ti], lag_seconds)
        teY = bold_runs[ti]
        n   = min(len(teX), len(teY)); teX, teY = teX[:n], teY[:n]
        sc  = StandardScaler()
        trX = sc.fit_transform(trX); teX = sc.transform(teX)
        ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=RIDGE_CV)
        ridge.fit(trX, trY)
        pred = ridge.predict(teX)
        for p in range(n_parcels):
            if teY[:, p].std() > 1e-8 and pred[:, p].std() > 1e-8:
                r_folds[ti, p] = pearsonr(teY[:, p], pred[:, p])[0]

    return r_folds.mean(0)

# ── Helpers ───────────────────────────────────────────────────────────────────
def split_features(raw, bold_runs):
    """Split concatenated feature array back into per-run chunks."""
    feat_runs, cursor = [], 0
    for b in bold_runs:
        n     = len(b)
        chunk = raw[cursor:cursor + n]
        feat_runs.append(chunk[:min(len(chunk), n)])
        cursor += n
    return feat_runs

def get_layer_names(sub, lang):
    feat_sub = os.path.join(args.feat_dir, lang, sub)
    if not os.path.exists(feat_sub):
        return []
    return sorted([
        f.replace(".npy", "")
        for f in os.listdir(feat_sub)
        if f.endswith(".npy") and not f.startswith("r_")
    ])

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: English layer × lag search
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("PHASE 1: English Recipe Search (layer x lag x LORO)")
print("="*60)

en_rows = []
for sub in SUBJECTS["EN"]:
    print(f"\n  {sub}: loading BOLD...", end=" ", flush=True)
    try:
        bold_runs = load_bold_parcels(sub, "EN")
        print(f"{len(bold_runs)} runs  {bold_runs[0].shape[1]} parcels")
    except Exception as e:
        print(f"ERROR: {e}"); continue

    layer_names = get_layer_names(sub, "EN")
    if not layer_names:
        print(f"  No features found — run Step 3 first"); continue

    for layer_name in layer_names:
        raw       = np.load(os.path.join(args.feat_dir, "EN", sub,
                                         f"{layer_name}.npy"))
        feat_runs = split_features(raw, bold_runs)

        for lag_name, lag_secs in LAG_GRID.items():
            r      = fit_loro_ridge(feat_runs, bold_runs, lag_secs)
            mean_r = float(r.mean())
            en_rows.append({
                "sub": sub, "lang": "EN",
                "layer": layer_name, "lag": lag_name,
                "mean_r": mean_r,
                "top10_r": float(np.percentile(r, 90)),
                "pos_frac": float((r > 0).mean()),
                **{f"r_{roi}": float(r[i])
                   for i, roi in enumerate(PARCEL_KEYS)},
            })

    sub_rows = [x for x in en_rows if x["sub"] == sub]
    if sub_rows:
        best = max(sub_rows, key=lambda x: x["mean_r"])
        print(f"    Best: layer={best['layer']}  lag={best['lag']}  "
              f"mean_r={best['mean_r']:.4f}")

en_df  = pd.DataFrame(en_rows)
en_csv = os.path.join(args.results_dir, "EN_recipe_search.csv")
en_df.to_csv(en_csv, index=False)
print(f"\nSaved: {en_csv}")

# Find best recipe
if not en_df.empty:
    agg        = en_df.groupby(["layer","lag"])["mean_r"].mean().reset_index()
    agg        = agg.sort_values("mean_r", ascending=False)
    best_row   = agg.iloc[0]
    BEST_LAYER = best_row["layer"]
    BEST_LAG   = best_row["lag"]
    BEST_LAG_SECS = LAG_GRID[BEST_LAG]
    print(f"\n{'='*60}")
    print(f"BEST RECIPE: layer={BEST_LAYER}  lag={BEST_LAG}  "
          f"mean_r={best_row['mean_r']:.4f}")
    print(f"Top 5:\n{agg.head().to_string(index=False)}")
else:
    BEST_LAYER    = "L16"; BEST_LAG = "lag024"; BEST_LAG_SECS = [0., 2., 4.]
    print("No EN results — using defaults (L16 / lag024)")

json.dump({
    "best_layer": BEST_LAYER,
    "best_lag": BEST_LAG,
    "best_lag_secs": BEST_LAG_SECS,
}, open(os.path.join(args.results_dir, "best_recipe.json"), "w"), indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Transfer EN -> FR
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"PHASE 2: Transfer EN -> FR")
print(f"Recipe: layer={BEST_LAYER}  lag={BEST_LAG}  secs={BEST_LAG_SECS}")
print("="*60)

transfer_rows = []
for lang in ["EN", "FR", "CN"]:
    print(f"\n--- {lang} ---")
    for sub in SUBJECTS[lang]:
        feat_path = os.path.join(args.feat_dir, lang, sub,
                                 f"{BEST_LAYER}.npy")
        res_dir   = os.path.join(args.results_dir, lang, sub)
        os.makedirs(res_dir, exist_ok=True)

        if not os.path.exists(feat_path):
            print(f"  {sub}: no features for {BEST_LAYER}"); continue

        try:
            bold_runs = load_bold_parcels(sub, lang)
        except Exception as e:
            print(f"  {sub}: {e}"); continue

        raw       = np.load(feat_path)
        feat_runs = split_features(raw, bold_runs)
        r         = fit_loro_ridge(feat_runs, bold_runs, BEST_LAG_SECS)

        np.save(os.path.join(res_dir, f"r_transfer_{BEST_LAYER}.npy"), r)
        mean_r = float(r.mean())
        print(f"  {sub}: mean_r={mean_r:.4f}  " +
              "  ".join(f"{roi}={r[i]:.3f}"
                        for i, roi in enumerate(PARCEL_KEYS)))
        transfer_rows.append({
            "sub": sub, "lang": lang,
            "layer": BEST_LAYER, "lag": BEST_LAG,
            "mean_r": mean_r,
            "top10_r": float(np.percentile(r, 90)),
            **{f"r_{roi}": float(r[i])
               for i, roi in enumerate(PARCEL_KEYS)},
        })

tr_df  = pd.DataFrame(transfer_rows)
tr_csv = os.path.join(args.results_dir, "transfer_EN_FR.csv")
tr_df.to_csv(tr_csv, index=False)
print(f"\nSaved: {tr_csv}")

if not tr_df.empty:
    summary = tr_df.groupby("lang")[["mean_r", "top10_r"]].mean()
    print(f"\n{'='*60}\nTRANSFER SUMMARY  layer={BEST_LAYER}  lag={BEST_LAG}")
    print(summary.to_string())
    en_r = summary.loc["EN", "mean_r"] if "EN" in summary.index else 0
    fr_r = summary.loc["FR", "mean_r"] if "FR" in summary.index else 0
    drop = en_r - fr_r
    pct  = drop / en_r * 100 if en_r > 0 else 0
    print(f"\n  EN: {en_r:.4f}  FR: {fr_r:.4f}  "
          f"Drop: {drop:.4f} ({pct:.1f}%)")
    if   pct < 15: print("  -> Low drop: Llama transfers well EN->FR")
    elif pct < 35: print("  -> Moderate drop: partial transfer")
    else:          print("  -> High drop: EN recipe is language-specific")

    print("\nPer-parcel:")
    for roi in PARCEL_KEYS:
        col = f"r_{roi}"
        if col in tr_df.columns:
            en_p = tr_df[tr_df.lang == "EN"][col].mean() \
                   if "EN" in tr_df.lang.values else 0
            fr_p = tr_df[tr_df.lang == "FR"][col].mean() \
                   if "FR" in tr_df.lang.values else 0
            print(f"  {roi:6s}: EN={en_p:.4f}  FR={fr_p:.4f}  "
                  f"drop={en_p-fr_p:.4f}")

print("\n=== Llama encoding model complete ===")
