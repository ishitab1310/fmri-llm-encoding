#!/usr/bin/env python3
"""plots.py — generates all figures from results CSVs"""

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", required=True)
parser.add_argument("--plots_dir",   required=True)
args = parser.parse_args()
os.makedirs(args.plots_dir, exist_ok=True)

plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":11,"axes.titlesize":12,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.3,
    "figure.facecolor":"white","figure.dpi":150,
})
COLORS = {"EN":"#2563EB","FR":"#16A34A","CN":"#DC2626"}
PARCEL_KEYS = ["IFG","STG","MTG","AngG","STS","mPFC"]
LAG_GRID = {"lag024":[0.,2.,4.],"lag246":[2.,4.,6.],"lag02468":[0.,2.,4.,6.,8.]}

en_csv = os.path.join(args.results_dir,"EN_recipe_search.csv")
tr_csv = os.path.join(args.results_dir,"transfer_EN_FR.csv")
recipe = os.path.join(args.results_dir,"best_recipe.json")

en_df = pd.read_csv(en_csv) if os.path.exists(en_csv) else pd.DataFrame()
tr_df = pd.read_csv(tr_csv) if os.path.exists(tr_csv) else pd.DataFrame()
BL, BLG = "L20","lag024"
if os.path.exists(recipe):
    r = json.load(open(recipe))
    BL, BLG = r["best_layer"], r["best_lag"]

# Fig 1: layer profile per lag
if not en_df.empty:
    lag_names = sorted(en_df["lag"].unique())
    fig,axes = plt.subplots(1,len(lag_names),figsize=(5*len(lag_names),4.5),sharey=True)
    if len(lag_names)==1: axes=[axes]
    for ax,lag_name in zip(axes,lag_names):
        sub_df = en_df[en_df["lag"]==lag_name]
        lo = sorted(sub_df["layer"].unique(),key=lambda x:int(x[1:]))
        for sub in sub_df["sub"].unique():
            vals=[sub_df[(sub_df["sub"]==sub)&(sub_df["layer"]==l)]["mean_r"].values for l in lo]
            vals=[v[0] if len(v)>0 else np.nan for v in vals]
            ax.plot(range(len(lo)),vals,"o-",color=COLORS["EN"],alpha=0.25,lw=1,ms=3)
        means=[sub_df[sub_df["layer"]==l]["mean_r"].mean() for l in lo]
        ax.plot(range(len(lo)),means,"o-",color=COLORS["EN"],lw=2.5,ms=6)
        if BL in lo: ax.axvline(lo.index(BL),color="red",lw=1.5,ls="--",alpha=0.7)
        ax.set_xticks(range(len(lo))); ax.set_xticklabels(lo,rotation=60,fontsize=7)
        ax.set_title(f"lag={lag_name}")
        ax.axhline(0,color="black",lw=0.7,ls="--",alpha=0.4)
    axes[0].set_ylabel("Mean Pearson r (parcels)")
    fig.suptitle(f"LLaDA-8B EN Layer Profile  [best: {BL}/{BLG}]",fontsize=12)
    plt.tight_layout()
    out = os.path.join(args.plots_dir,"fig1_layer_profile.png")
    plt.savefig(out,bbox_inches="tight"); plt.close()
    print(f"Saved {out}")

# Fig 2: transfer + per-parcel
if not tr_df.empty:
    fig,axes = plt.subplots(1,2,figsize=(12,4.5))
    ax=axes[0]
    for i,lang in enumerate(["EN","FR"]):
        s=tr_df[tr_df["lang"]==lang]
        if s.empty: continue
        m=s["mean_r"].mean(); se=s["mean_r"].sem()
        ax.bar(i,m,color=COLORS[lang],alpha=0.85,yerr=se,capsize=5)
        ax.text(i,m+0.002,f"{m:.3f}",ha="center",fontsize=10)
    ax.set_xticks([0,1]); ax.set_xticklabels(["English","French"])
    ax.set_ylabel("Mean Pearson r"); ax.set_title(f"EN->FR  layer={BL}  lag={BLG}")
    ax.axhline(0,color="black",lw=0.8,ls="--",alpha=0.4)
    ax=axes[1]
    x=np.arange(len(PARCEL_KEYS)); w=0.35
    for shift,lang in [(-w/2,"EN"),(w/2,"FR")]:
        s=tr_df[tr_df["lang"]==lang]
        if s.empty: continue
        vals=[s[f"r_{p}"].mean() if f"r_{p}" in s.columns else 0 for p in PARCEL_KEYS]
        errs=[s[f"r_{p}"].sem()  if f"r_{p}" in s.columns else 0 for p in PARCEL_KEYS]
        ax.bar(x+shift,vals,w,color=COLORS[lang],alpha=0.85,yerr=errs,capsize=3,label=lang)
    ax.set_xticks(x); ax.set_xticklabels(PARCEL_KEYS)
    ax.set_ylabel("Pearson r"); ax.set_title("Per-parcel breakdown"); ax.legend()
    ax.axhline(0,color="black",lw=0.8,ls="--",alpha=0.4)
    plt.tight_layout()
    out = os.path.join(args.plots_dir,"fig2_transfer.png")
    plt.savefig(out,bbox_inches="tight"); plt.close()
    print(f"Saved {out}")

# Fig 3: combined summary
fig=plt.figure(figsize=(14,5)); gs=gridspec.GridSpec(1,3,figure=fig,wspace=0.35)
ax=fig.add_subplot(gs[0,:2])
if not en_df.empty:
    sub_df=en_df[en_df["lag"]==BLG] if BLG in en_df["lag"].values else en_df
    lo=sorted(sub_df["layer"].unique(),key=lambda x:int(x[1:]))
    means=[sub_df[sub_df["layer"]==l]["mean_r"].mean() for l in lo]
    ax.plot(range(len(lo)),means,"o-",color=COLORS["EN"],lw=2)
    if BL in lo: ax.axvline(lo.index(BL),color="red",lw=1.5,ls="--",label=f"Best:{BL}")
    ax.set_xticks(range(len(lo))); ax.set_xticklabels(lo,rotation=45,fontsize=7)
    ax.legend()
ax.set_title("A. Layer profile (EN, best lag)"); ax.set_ylabel("Mean Pearson r")
ax.axhline(0,color="black",lw=0.8,ls="--",alpha=0.4)
ax=fig.add_subplot(gs[0,2])
if not tr_df.empty:
    for i,lang in enumerate(["EN","FR"]):
        s=tr_df[tr_df["lang"]==lang]
        if s.empty: continue
        m=s["mean_r"].mean()
        ax.bar(i,m,color=COLORS[lang],alpha=0.85)
        ax.text(i,m+0.001,f"{m:.3f}",ha="center",fontsize=9)
ax.set_xticks([0,1]); ax.set_xticklabels(["EN","FR"])
ax.set_title("B. EN->FR transfer"); ax.set_ylabel("Pearson r")
ax.axhline(0,color="black",lw=0.8,ls="--",alpha=0.4)
fig.suptitle(f"A3 LLaDA-8B Multilingual fMRI Encoding | {BL}/{BLG}",fontsize=12)
plt.tight_layout()
out = os.path.join(args.plots_dir,"fig3_summary.png")
plt.savefig(out,bbox_inches="tight",dpi=150); plt.close()
print(f"Saved {out}")
print("All plots done.")
