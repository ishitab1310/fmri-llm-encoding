import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

FEAT_DIR  = "/home2/ishita/features_llada"
DATA_DIR  = "/home2/ishita/ds003643"   # or scratch path if job is running
RESULTS   = "/home2/ishita/results_llada"
os.makedirs(RESULTS, exist_ok=True)

TR, TRIM = 2.0, 4
BOLD_SUFFIX = "space-MNIColin27_desc-preproc_bold.nii.gz"
TASK = {"EN":"lppEN","FR":"lppFR","CN":"lppCN"}
ALPHAS = [0.1, 1.0, 10.0, 100.0]
LAG_SECS = [0, 2, 4, 6, 8]

ROIS = {
    "IFG":  (-51,  28,   3),
    "STG":  (-57, -16,   4),
    "MTG":  (-60, -32,  -4),
    "AngG": (-48, -58,  28),
    "STS":  (-54, -40,   8),
    "mPFC": (  -4, 52,  -4),
}

def sphere_mask(affine, shape, center, r=10.0):
    i,j,k = np.mgrid[0:shape[0],0:shape[1],0:shape[2]]
    coords = np.stack([i.ravel(),j.ravel(),k.ravel(),np.ones(i.size)],0)
    xyz = (affine @ coords)[:3].T
    return (np.linalg.norm(xyz - np.array(center), axis=1) <= r).reshape(shape)

def load_bold(sub, lang, data_dir):
    task = TASK[lang]
    d = os.path.join(data_dir, "derivatives", sub, "func")
    if not os.path.exists(d): return None
    bolds = sorted([f for f in os.listdir(d) if task in f and f.endswith(".nii.gz")])
    if not bolds: return None
    runs = []
    for b in bolds:
        img  = nib.load(os.path.join(d, b))
        data = img.get_fdata(dtype=np.float32)
        aff  = img.affine
        sh   = img.shape[:3]
        parcels = []
        for roi, center in ROIS.items():
            mask = sphere_mask(aff, sh, center)
            if mask.sum() == 0: parcels.append(np.zeros(data.shape[3]))
            else: parcels.append(data[mask].mean(0))
        bold = np.stack(parcels, axis=1)[TRIM:]
        mu, sg = bold.mean(0), bold.std(0)+1e-8
        runs.append((bold-mu)/sg)
    return runs

def lag_stack(X, lags_trs):
    T, D = X.shape
    out = np.zeros((T, D*len(lags_trs)), dtype=np.float32)
    for i, lag in enumerate(lags_trs):
        s = np.zeros_like(X)
        if lag > 0: s[lag:] = X[:-lag]
        else: s = X.copy()
        out[:, i*D:(i+1)*D] = s
    return out

rows = []
for lang in ["EN","FR","CN"]:
    lang_feat = os.path.join(FEAT_DIR, lang)
    if not os.path.exists(lang_feat): continue
    for sub in sorted(os.listdir(lang_feat)):
        sub_dir = os.path.join(lang_feat, sub)
        meta_f  = os.path.join(sub_dir, "meta.json")
        if not os.path.exists(meta_f): continue
        meta = json.load(open(meta_f))
        layers = meta["layers"]
        print(f"\n{sub} ({lang}): {len(layers)} layers")

        bold_runs = load_bold(sub, lang, DATA_DIR)
        if bold_runs is None:
            print(f"  SKIP — no BOLD"); continue
        n_v = bold_runs[0].shape[1]
        lags_trs = [int(s/TR) for s in LAG_SECS]

        for layer in layers:
            feat_path = os.path.join(sub_dir, f"{layer}.npy")
            if not os.path.exists(feat_path): continue
            raw = np.load(feat_path)

            # split into runs
            cursor, feat_runs = 0, []
            for br in bold_runs:
                n = len(br)
                feat_runs.append(raw[cursor:cursor+n])
                cursor += n

            # LORO-CV
            r_vals = np.zeros(n_v)
            sc = StandardScaler()
            for ti in range(len(bold_runs)):
                tr_X = np.concatenate([lag_stack(feat_runs[i], lags_trs)
                                        for i in range(len(feat_runs)) if i!=ti])
                tr_Y = np.concatenate([bold_runs[i] for i in range(len(bold_runs)) if i!=ti])
                te_X = lag_stack(feat_runs[ti], lags_trs)
                te_Y = bold_runs[ti]
                n = min(len(te_X), len(te_Y))
                tr_X = sc.fit_transform(tr_X)
                te_X = sc.transform(te_X[:n])
                ridge = RidgeCV(alphas=ALPHAS, cv=5)
                ridge.fit(tr_X, tr_Y)
                pred = ridge.predict(te_X)
                for v in range(n_v):
                    if te_Y[:n,v].std()>1e-8 and pred[:,v].std()>1e-8:
                        r_vals[v] += pearsonr(te_Y[:n,v], pred[:,v])[0]
            r_vals /= len(bold_runs)

            mean_r = float(r_vals.mean())
            top10  = float(np.percentile(r_vals, 90))
            print(f"  {layer}: mean_r={mean_r:.4f}  top10={top10:.4f}")
            rows.append({"sub":sub,"lang":lang,"layer":layer,
                         "mean_r":mean_r,"top10_r":top10,
                         "pos_frac":float((r_vals>0).mean())})

df = pd.DataFrame(rows)
df.to_csv(f"{RESULTS}/encoding_results_partial.csv", index=False)
print(f"\nSaved: {RESULTS}/encoding_results_partial.csv")
print(df.groupby(["lang","layer"])["mean_r"].mean().to_string())
