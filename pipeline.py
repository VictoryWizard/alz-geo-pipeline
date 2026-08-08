"""
Alzheimer's / MCI / Control classification from GSE63061 blood gene expression
(AddNeuroMed cohort, batch 2, 388 samples, Illumina HumanHT-12 V4).

Single-cohort run (no batch correction needed): top-variance-filtered gene
features + age/gender, XGBoost with stratified 5-fold CV, SHAP explanations.

Usage:
    python pipeline.py /path/to/GSE63061_series_matrix.txt.gz
    python pipeline.py GSE63061          # downloads it first (needs internet - use in Colab)

No GEOparse dependency: GEO series matrix files are parsed directly here because
GEOparse's SOFT-family parser does not reliably handle the "series matrix" format
(verified: it silently returns 0 samples on this exact file).
"""

import sys
import os
import gzip
import re
import urllib.request
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_TOP_GENES = 500  # feature-selection cap to fight the p >> n problem (388 samples, ~32k probes)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

GEO_URL_TEMPLATE = "https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}nnn/{acc}/matrix/{acc}_series_matrix.txt.gz"


def resolve_input(arg):
    """Accept either a local file path or a bare GEO accession (e.g. GSE63061)."""
    if os.path.exists(arg):
        return arg
    if re.match(r"^GSE\d+$", arg.strip(), re.IGNORECASE):
        acc = arg.strip().upper()
        prefix = acc[:-3]  # e.g. GSE63061 -> GSE63
        url = GEO_URL_TEMPLATE.format(prefix=prefix, acc=acc)
        os.makedirs(DATA_DIR, exist_ok=True)
        dest = os.path.join(DATA_DIR, f"{acc}_series_matrix.txt.gz")
        print(f"'{arg}' not found locally, downloading from {url} ...")
        urllib.request.urlretrieve(url, dest)
        return dest
    raise FileNotFoundError(f"'{arg}' is not a local file and doesn't look like a GEO accession.")


def parse_series_matrix(path):
    """Parse a GEO series matrix file (gzip or plain text) into (expr, meta).
    expr: samples x probes DataFrame. meta: samples x characteristics DataFrame.
    """
    opener = gzip.open if path.endswith(".gz") else open
    header_rows = {}
    table_lines = []
    in_table = False
    with opener(path, "rt", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                in_table = False
                continue
            if in_table:
                table_lines.append(line)
            elif line.startswith("!Sample_"):
                parts = line.split("\t")
                key = parts[0].lstrip("!")
                vals = [v.strip('"') for v in parts[1:]]
                header_rows.setdefault(key, []).append(vals)

    samples = header_rows["Sample_geo_accession"][0]

    # characteristics_ch1 can repeat (one row per "key: value" field); pivot into named columns
    meta = pd.DataFrame(index=samples)
    for row in header_rows.get("Sample_characteristics_ch1", []):
        # infer the field name from the first non-empty cell, e.g. "age: 57" -> "age"
        field_name = None
        for cell in row:
            if ":" in cell:
                field_name = cell.split(":", 1)[0].strip().lower()
                break
        if field_name is None:
            continue
        values = []
        for cell in row:
            values.append(cell.split(":", 1)[1].strip() if ":" in cell else np.nan)
        meta[field_name] = values

    expr = pd.read_csv(StringIO("\n".join(table_lines)), sep="\t", quotechar='"', index_col=0)
    expr = expr.T  # -> samples x probes
    expr.index = samples

    return expr, meta


def build_features(expr, meta, n_top_genes=N_TOP_GENES):
    common = expr.index.intersection(meta.index)
    expr = expr.loc[common]
    meta = meta.loc[common]

    y_raw = meta["status"].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    selector = SelectKBest(f_classif, k=min(n_top_genes, expr.shape[1]))
    X_genes = selector.fit_transform(expr.values, y)
    gene_cols = expr.columns[selector.get_support()]
    X = pd.DataFrame(X_genes, index=expr.index, columns=gene_cols)

    X["age"] = pd.to_numeric(meta["age"], errors="coerce")
    X["sex"] = LabelEncoder().fit_transform(meta["gender"].astype(str))

    X = X.fillna(X.median(numeric_only=True))
    return X, y, le


def run_cv(X, y, n_classes):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            random_state=42,
        )
        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[te])
        proba = model.predict_proba(X.iloc[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
        try:
            auc = roc_auc_score(y[te], proba, multi_class="ovr") if n_classes > 2 else roc_auc_score(y[te], proba[:, 1])
        except ValueError:
            auc = float("nan")
        aucs.append(auc)
        print(f"Fold {fold}: acc={accs[-1]:.3f}  macro-F1={f1s[-1]:.3f}  AUC={auc:.3f}")
    print(f"\nMean acc={np.mean(accs):.3f}  Mean macro-F1={np.mean(f1s):.3f}  Mean AUC={np.nanmean(aucs):.3f}")
    return accs, f1s, aucs


def final_model_and_shap(X, y, class_names):
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X, class_names=class_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=150)
    print(f"Saved SHAP summary plot to {OUTPUT_DIR}/shap_summary.png")
    return model


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    source = resolve_input(sys.argv[1])

    expr, meta = parse_series_matrix(source)
    print(f"Expression matrix: {expr.shape[0]} samples x {expr.shape[1]} probes")
    print(f"Metadata columns found: {list(meta.columns)}")
    print(f"Raw class counts:\n{meta['status'].value_counts()}")

    # The 'status' field has 7 raw values, not just AD/MCI/CTL: 3 "borderline MCI",
    # 1 "OTHER", 1 "CTL to AD", 1 "MCI to CTL" (the last two are single-sample
    # diagnosis-change flags, not a usable converter cohort - far too small to
    # model). Keep only the three main classes for a clean 3-way task.
    keep = meta["status"].isin(["AD", "MCI", "CTL"])
    dropped = (~keep).sum()
    if dropped:
        print(f"Dropping {dropped} samples with ambiguous/edge-case status labels "
              f"({meta.loc[~keep, 'status'].value_counts().to_dict()})")
    expr, meta = expr.loc[keep], meta.loc[keep]

    X, y, le = build_features(expr, meta)
    print(f"Feature matrix: {X.shape}, classes: {dict(zip(le.classes_, range(len(le.classes_))))}")

    run_cv(X, y, n_classes=len(le.classes_))
    model = final_model_and_shap(X, y, class_names=list(le.classes_))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_model(os.path.join(OUTPUT_DIR, "xgb_model.json"))
    print("Done.")


if __name__ == "__main__":
    main()
