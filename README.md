# AD/MCI/Control classification from blood gene expression (GSE63061)

Single-cohort baseline: 382 blood samples (AddNeuroMed cohort, batch 2, GEO
accession GSE63061) classified into Alzheimer's disease / MCI / control using
gene expression + age + sex. XGBoost, 5-fold stratified CV, SHAP for feature
attribution.

**This is baseline classification at time of blood draw, not future-onset
prediction.** GSE63061's diagnosis labels are current status, not confirmed
longitudinal conversion outcomes (see note below). Don't claim otherwise in
writeups.

## What's actually in the data (verified against the real file, not assumed)

- 388 samples total, 32,049 probes (Illumina HumanHT-12 V4 array), blood-derived RNA.
- `status` field has 7 raw values: AD (139), CTL (134), MCI (109), "borderline
  MCI" (3), "OTHER" (1), "CTL to AD" (1), "MCI to CTL" (1). The pipeline keeps
  only the three main classes (382 samples) — the two diagnosis-change flags
  are single samples each, not a usable converter cohort.
- `age` and `gender` are present per sample.
- No other clinical/cognitive variables are in this file.

## Results (first pass, no batch correction — single cohort so it's not needed)

5-fold stratified CV, top-500 genes by ANOVA F-test + age + sex:

- Mean accuracy: 0.573
- Mean macro-F1: 0.556
- Mean AUC (OvR): 0.738

This is a real first-pass number, not a target — it's well short of the
"90%" bar mentioned early in scoping. Room to improve: more genes, different
feature selection, hyperparameter tuning, ensembling — but don't inflate the
claim past what a single small cohort with no external validation supports.

## Run locally

```bash
pip install -r requirements.txt
python pipeline.py /path/to/GSE63061_series_matrix.txt.gz
# or, with internet access (e.g. in Colab):
python pipeline.py GSE63061
```

## Run in Colab

Open `run_in_colab.ipynb` in this repo via Colab (or use the badge once you've
pushed this repo), run all cells. It clones this repo, installs dependencies,
downloads GSE63061 directly from NCBI (Colab has full internet access), and
runs the pipeline.

## Data source

GSE63061, NCBI GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63061
(AddNeuroMed Cohort batch 2 — Alzheimer's, MCI, and age/gender-matched controls,
peripheral blood RNA.)
