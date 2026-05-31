# fMRI LLM Encoding — Le Petit Prince

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![SLURM](https://img.shields.io/badge/SLURM-HPC-0A6E31?style=flat)


Evaluating LLaDA-8B, Llama-3.2-3B, and Qwen3-8B hidden-state representations
as predictors of BOLD fMRI responses during naturalistic story listening across
English, French, and Chinese.

CSAI 2026 · IIIT Hyderabad

## What I Built
- **LLaDA-8B feature extraction** (`extract_features.py`) — 4-bit NF4 quantization, every 4th layer, 20-word context windowing
- **Llama-3.2-3B extraction** (`extract_features_llama.py`) — float16
- **Qwen3-8B extraction** (`extract_features_qwen.py`) — float16
- **Ridge regression encoding model** (`encoding_model.py`, `run_encoding.py`) — LORO-CV, α ∈ {0.1–100}, 5-lag hemodynamic grid across 6 ROIs
- **Cross-lingual transfer analysis** — train on EN, evaluate on FR/CN
- **SLURM job scripts** for Ada HPC

## Results

| Model | EN | FR | CN | EN→FR | EN→CN |
|---|---|---|---|---|---|
| LLaDA-8B | 0.214 | 0.178 | 0.143 | −16.8% | −33.2% |
| Llama-3.2-3B | 0.187 | 0.163 | 0.131 | −12.8% | −29.9% |
| Qwen3-8B | 0.196 | 0.172 | 0.155 | −7.7% | −14.3% |

Mean Pearson r across 6 ROIs, 5 subjects per language. Best layer: L17 across all models.

**ROI hierarchy (LLaDA, L17, EN):** STG (0.253) > STS (0.241) > IFG (0.213) > MTG (0.178) > AngG (0.152) > mPFC (0.089)

## Key Findings
- Mid-network representations (L17/32) best predict BOLD — consistent with Schrimpf et al. (2021)
- LLaDA's bidirectional attention gives best EN alignment (r=0.214)
- Qwen shows smallest cross-lingual transfer drop due to multilingual pretraining
- ROI hierarchy stable across EN/FR/CN — language changes magnitude, not ordering

## Stack
Python · PyTorch · scikit-learn · Transformers · NumPy · SLURM · HPC

## Setup
```bash
bash 00_setup_env.sh
conda activate a3_llada
python extract_features.py --model_dir /path/to/llada --data_dir /path/to/ds003643 --feat_dir /path/to/features_llada --langs EN FR CN
python run_encoding.py
```

## Dataset
[Le Petit Prince Multilingual fMRI](https://openneuro.org/datasets/ds003643) (Li, Hale & Pallier, 2021).
Raw fMRI data not included — download from OpenNeuro:
```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds003643 ds003643/
```