#!/usr/bin/env python3
"""
CONSOLIDATED ANALYSIS  -  every number the Results section cites, one command.

    python analysis.py --stages fast      # A1 A2 A3 A4 A5 R1 R2 R3   (minutes)
    python analysis.py --stages A6        # hyperparameter grid search (~30 min)
    python analysis.py --stages T1        # permutation null (hours, checkpointed)
    python analysis.py --stages all

Results accumulate into output/results.json.  Each stage writes its own key and
re-running a stage overwrites only that key, so stages can be run in any order,
in parallel processes, or resumed after a crash.

DESIGN NOTES THAT MATTER FOR METHODS
------------------------------------
* Detection floor, probe mask, feature selection and model fit are ALL derived
  from the training cohort only.  Nothing touches the test cohort before scoring.
* The primary split is by whole study (train GSE63060 -> test GSE63061 and the
  reverse).  There is no site/batch/scan-date field in either series matrix, so
  cross-series transfer is the only batch control this data admits.
* Headline model uses FIXED hyperparameters.  A6 grid-searches separately and is
  reported as "does tuning change the conclusion", NOT as the headline.  Reason:
  the permutation null (T1) must run the identical procedure on shuffled labels,
  and grid-searching inside 1000 permutations is infeasible; tuning the observed
  value but not the null would invalidate the p-value.
"""
import argparse
import gzip
import json
import os
import sys
import time
import warnings
from io import StringIO

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ====================================================================== CONFIG
CONFIG = dict(
    SERIES=["GSE63060", "GSE63061"],
    STATUS_KEEP=["AD", "CTL"],
    POSITIVE_CLASS="AD",
    FLOOR_K=3.0,
    KDE_BW=0.15,
    KDE_SUBSAMPLE=8000,
    KDE_SUBSAMPLE_SEED=0,
    KDE_GRID=1500,
    SD_WINDOW=0.5,
    N_TOP=500,
    N_SPLITS=5,
    SEED=42,
    N_BOOT=2000,
    BOOT_SEED=777,
    N_PERM=1000,
    PERM_CHECKPOINT=25,
    PERM_ARMS=[("GSE63060", "GSE63061", "STRICT-100"),
               ("GSE63061", "GSE63060", "STRICT-100")],
    FLOOR_KS=[2.0, 2.5, 3.0, 3.5, 4.0],
    N_SEEDS=10,
    KDE_STABILITY_SEEDS=list(range(10)),
    KDE_STABILITY_BWS=[0.10, 0.15, 0.20],
    XGB=dict(n_estimators=300, max_depth=4, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.8,
             eval_metric="logloss", tree_method="hist", verbosity=0),
    GRID=dict(max_depth=[2, 3, 4, 6],
              learning_rate=[0.03, 0.05, 0.10],
              n_estimators=[200, 400]),
    GRID_INNER_FOLDS=3,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "output")
RESULTS = os.path.join(OUT_DIR, "results.json")
os.makedirs(OUT_DIR, exist_ok=True)

T0 = time.time()
NTHREAD = int(os.environ.get("ALZ_NTHREAD", "2"))
DEVICE = os.environ.get("ALZ_DEVICE", "cpu")     # "cpu" or "cuda"
PERM_JOBS = int(os.environ.get("ALZ_PERM_JOBS", "1"))


def P(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


# ================================================================ persistence
def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {}


def save_stage(key, value):
    """Read-modify-write so parallel stage processes don't clobber each other."""
    r = load_results()
    r[key] = value
    r["config"] = CONFIG
    r["versions"] = versions()
    tmp = RESULTS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(r, f, indent=2, default=_jsonable)
    os.replace(tmp, RESULTS)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def versions():
    import sklearn
    import xgboost
    return dict(python=sys.version.split()[0], numpy=np.__version__,
                pandas=pd.__version__, scipy=__import__("scipy").__version__,
                sklearn=sklearn.__version__, xgboost=xgboost.__version__)


# ======================================================================= data
def parse_series_matrix(path):
    """GEO series matrix -> (expr [samples x probes], meta [samples x fields])."""
    opener = gzip.open if path.endswith(".gz") else open
    header_rows, table_lines, in_table = {}, [], False
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
                header_rows.setdefault(parts[0].lstrip("!"), []).append(
                    [v.strip('"') for v in parts[1:]])

    samples = header_rows["Sample_geo_accession"][0]
    meta = pd.DataFrame(index=samples)
    # BeadChip barcode and array position live in Sample_title (e.g. "4856076025_A");
    # the recruitment-centre code is the leading alpha run of Sample_description.
    if "Sample_title" in header_rows:
        t = header_rows["Sample_title"][0]
        meta["chip"] = [str(v).split("_")[0] for v in t]
        meta["array_pos"] = [str(v).split("_")[1] if "_" in str(v) else "" for v in t]
    if "Sample_description" in header_rows:
        import re as _re
        d = header_rows["Sample_description"][0]
        meta["subject_code"] = d
        meta["site"] = [(_re.match(r"[A-Za-z]+", str(v)) or _re.match(r"", "")).group(0)[:3]
                        for v in d]
    for row in header_rows.get("Sample_characteristics_ch1", []):
        field = next((c.split(":", 1)[0].strip().lower() for c in row if ":" in c), None)
        if field is None:
            continue
        meta[field] = [c.split(":", 1)[1].strip() if ":" in c else np.nan for c in row]

    expr = pd.read_csv(StringIO("\n".join(table_lines)), sep="\t",
                       quotechar='"', index_col=0).T
    expr.index = samples
    return expr, meta


FLAG_FIELD = "included in case -control study"   # note the GEO typo: space before '-'


def load_data():
    """Returns D[series] = dict(X, y, age, sex, flag) and the shared probe index."""
    exprs, metas, drops = {}, {}, {}
    for g in CONFIG["SERIES"]:
        e, m = parse_series_matrix(os.path.join(DATA_DIR, f"{g}_series_matrix.txt.gz"))
        exprs[g], metas[g] = e, m
        drops[g] = m.loc[~m["status"].isin(CONFIG["STATUS_KEEP"]),
                         "status"].value_counts().to_dict()

    shared = exprs[CONFIG["SERIES"][0]].columns.intersection(
        exprs[CONFIG["SERIES"][1]].columns)

    D, info = {}, dict(n_shared_probes=int(len(shared)), dropped=drops, n={})
    for g in CONFIG["SERIES"]:
        m = metas[g]
        keep = m["status"].isin(CONFIG["STATUS_KEEP"])
        mk = m.loc[keep]
        flag_raw = mk[FLAG_FIELD].astype(str).str.strip().str.lower() \
            if FLAG_FIELD in mk.columns else pd.Series("unknown", index=mk.index)
        D[g] = dict(
            X=exprs[g].loc[keep, shared].values.astype(float),
            y=(mk["status"] == CONFIG["POSITIVE_CLASS"]).astype(int).values,
            age=pd.to_numeric(mk["age"], errors="coerce").values,
            sex=(mk["gender"].astype(str).str.strip().str.lower()
                 == "female").astype(int).values,
            flag=(flag_raw == "yes").astype(int).values,
            chip=mk["chip"].values if "chip" in mk.columns else None,
            site=mk["site"].values if "site" in mk.columns else None,
        )
        info["n"][g] = dict(total=int(keep.sum()),
                            AD=int(D[g]["y"].sum()),
                            CTL=int((D[g]["y"] == 0).sum()),
                            flag_yes=int(D[g]["flag"].sum()))
    return D, info


# ==================================================================== machinery
def floor_of(train, k=None, bw=None, sub_seed=None):
    """Detection floor: KDE peak of per-probe means + k * SD of means near the peak."""
    k = CONFIG["FLOOR_K"] if k is None else k
    bw = CONFIG["KDE_BW"] if bw is None else bw
    sub_seed = CONFIG["KDE_SUBSAMPLE_SEED"] if sub_seed is None else sub_seed
    mu = train.mean(axis=0)
    n_sub = CONFIG["KDE_SUBSAMPLE"]
    v = mu if len(mu) <= n_sub else np.random.RandomState(sub_seed).choice(
        mu, n_sub, replace=False)
    kde = stats.gaussian_kde(v, bw_method=bw)
    grid = np.linspace(mu.min(), mu.max(), CONFIG["KDE_GRID"])
    peak = grid[int(np.argmax(kde(grid)))]
    return float(peak + k * mu[np.abs(mu - peak) < CONFIG["SD_WINDOW"]].std())


RULES = {
    "STRICT-100": lambda Xtr, T: (Xtr <= T).all(axis=0),
    "DETECTED":   lambda Xtr, T: Xtr.mean(axis=0) > T,
    "ALL":        lambda Xtr, T: np.ones(Xtr.shape[1], dtype=bool),
}


def rank_rows(A):
    """Within-sample rank transform. This is what makes V3 -> V4 transfer valid."""
    return np.apply_along_axis(stats.rankdata, 1, A) / A.shape[1]


def make_xgb(seed=None, **over):
    p = dict(CONFIG["XGB"])
    p.update(over)
    return XGBClassifier(random_state=CONFIG["SEED"] if seed is None else seed,
                         nthread=NTHREAD, device=DEVICE, **p)


def fit_score(Xtr, ytr, Xte, seed=None, **over):
    """Select-then-fit on train only; return P(AD) for every test subject."""
    sel = SelectKBest(f_classif, k=min(CONFIG["N_TOP"], Xtr.shape[1])).fit(Xtr, ytr)
    m = make_xgb(seed, **over).fit(sel.transform(Xtr), ytr)
    return m.predict_proba(sel.transform(Xte))[:, 1]


def prep(Xtr_full, Xte_full, rule, T):
    """Mask by rule (train-derived), then rank-normalise each sample."""
    mask = RULES[rule](Xtr_full, T)
    return rank_rows(Xtr_full[:, mask]), rank_rows(Xte_full[:, mask]), int(mask.sum())


def boot_indices(y, n_boot, seed):
    """Stratified subject resampling so both classes always survive."""
    rs = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    return [np.concatenate([rs.choice(pos, len(pos), True),
                            rs.choice(neg, len(neg), True)]) for _ in range(n_boot)]


def ci(a, lo=2.5, hi=97.5):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def directions():
    a, b = CONFIG["SERIES"]
    return [(a, b), (b, a)]


# ========================================================================= A1
def stage_A1(D):
    """Cross-cohort transfer, both directions, three probe sets, bootstrap CIs."""
    P("=" * 88)
    P("A1  CROSS-COHORT TRANSFER (primary)")
    P("=" * 88)
    P(f"  {'train->test':<16}{'probe set':<12}{'n probes':>9}{'AUC':>8}{'2.5%':>8}{'97.5%':>8}")
    out = {}
    for a, b in directions():
        Xa, ya, Xb, yb = D[a]["X"], D[a]["y"], D[b]["X"], D[b]["y"]
        T = floor_of(Xa)
        idxs = boot_indices(yb, CONFIG["N_BOOT"], CONFIG["BOOT_SEED"])
        scores = {}
        for rule in ("STRICT-100", "DETECTED", "ALL"):
            A, B, n_pr = prep(Xa, Xb, rule, T)
            s = fit_score(A, ya, B)
            scores[rule] = s
            obs = roc_auc_score(yb, s)
            bs = np.array([roc_auc_score(yb[i], s[i]) for i in idxs])
            lo, hi = ci(bs)
            out[f"{a[-2:]}->{b[-2:]}|{rule}"] = dict(
                obs=float(obs), lo=lo, hi=hi, n_probes=n_pr, floor=T,
                boot_mean=float(bs.mean()), boot_sd=float(bs.std()),
                frac_le_half=float((bs <= 0.5).mean()))
            P(f"  {a[-2:]+'->'+b[-2:]:<16}{rule:<12}{n_pr:>9}{obs:>8.3f}{lo:>8.3f}{hi:>8.3f}")
        d = np.array([roc_auc_score(yb[i], scores["DETECTED"][i])
                      - roc_auc_score(yb[i], scores["STRICT-100"][i]) for i in idxs])
        lo, hi = ci(d)
        out[f"{a[-2:]}->{b[-2:]}|DET_minus_DARK"] = dict(
            obs=float(roc_auc_score(yb, scores["DETECTED"])
                      - roc_auc_score(yb, scores["STRICT-100"])),
            lo=lo, hi=hi, frac_le_zero=float((d <= 0).mean()))
        P(f"  {a[-2:]+'->'+b[-2:]:<16}{'DET-DARK':<12}{'':>9}{d.mean():>8.3f}"
          f"{lo:>8.3f}{hi:>8.3f}   P(diff<=0)={float((d <= 0).mean()):.4f}")
    save_stage("A1", out)
    return out


# ========================================================================= A2
def cv_auc(X, y, rule, k=None, seed=None, cov=None):
    """Within-cohort CV. Floor, mask and selection re-derived inside every fold."""
    seed = CONFIG["SEED"] if seed is None else seed
    skf = StratifiedKFold(CONFIG["N_SPLITS"], shuffle=True, random_state=CONFIG["SEED"])
    aucs, ns = [], []
    for tr, te in skf.split(X, y):
        if rule is None:                       # covariates only, no expression
            A, B = cov[tr], cov[te]
            ns.append(cov.shape[1])
        else:
            T = floor_of(X[tr], k)
            mask = RULES[rule](X[tr], T)
            if mask.sum() < 10:
                aucs.append(np.nan)
                ns.append(int(mask.sum()))
                continue
            cols = np.where(mask)[0]
            A, B = rank_rows(X[np.ix_(tr, cols)]), rank_rows(X[np.ix_(te, cols)])
            ns.append(int(mask.sum()))
            if cov is not None:
                A = np.hstack([A, cov[tr]])
                B = np.hstack([B, cov[te]])
        aucs.append(roc_auc_score(y[te], fit_score(A, y[tr], B, seed=seed)))
    return float(np.nanmean(aucs)), [float(v) for v in aucs], int(np.mean(ns))


def stage_A2(D):
    P("=" * 88)
    P("A2  WITHIN-COHORT 5-FOLD CV (secondary)")
    P("=" * 88)
    out = {}
    for g in CONFIG["SERIES"]:
        for rule in ("STRICT-100", "DETECTED", "ALL"):
            m, folds, n_pr = cv_auc(D[g]["X"], D[g]["y"], rule)
            out[f"{g}|{rule}"] = dict(auc=m, folds=folds, n_probes=n_pr)
            P(f"  {g:<12}{rule:<12}{n_pr:>9}{m:>8.3f}")
    save_stage("A2", out)
    return out


# ========================================================================= A3
def stage_A3(D):
    """Demographics baseline. Without this the headline AUC cannot be interpreted."""
    P("=" * 88)
    P("A3  COVARIATE BASELINE  (age + sex)")
    P("=" * 88)
    out = {}
    for g in CONFIG["SERIES"]:
        X, y = D[g]["X"], D[g]["y"]
        age = np.nan_to_num(D[g]["age"], nan=float(np.nanmedian(D[g]["age"])))
        cov = np.column_stack([age, D[g]["sex"]])
        m, folds, _ = cv_auc(X, y, None, cov=cov)
        out[f"{g}|COVARIATES_ONLY"] = dict(auc=m, folds=folds)
        P(f"  {g:<12}{'age+sex':<22}{m:>8.3f}")
        for rule in ("STRICT-100", "DETECTED"):
            m2, folds2, _ = cv_auc(X, y, rule, cov=cov)
            out[f"{g}|{rule}+COV"] = dict(auc=m2, folds=folds2)
            P(f"  {g:<12}{rule + ' + age/sex':<22}{m2:>8.3f}")
        # is age itself different between groups?
        t, p = stats.ttest_ind(age[y == 1], age[y == 0], equal_var=False)
        out[f"{g}|age_AD_vs_CTL"] = dict(mean_AD=float(age[y == 1].mean()),
                                         mean_CTL=float(age[y == 0].mean()),
                                         t=float(t), p=float(p))
        P(f"  {g:<12}age AD {age[y==1].mean():.1f} vs CTL {age[y==0].mean():.1f}"
          f"   p={p:.3g}")
    save_stage("A3", out)
    return out


# ========================================================================= A4
def stage_A4(D):
    """Case-control flag: is the dark-probe signal actually processing structure?"""
    P("=" * 88)
    P("A4  CASE-CONTROL FLAG SENSITIVITY")
    P("=" * 88)
    out = {}

    # A4c  contingency + chi2
    for g in CONFIG["SERIES"]:
        y, fl = D[g]["y"], D[g]["flag"]
        tab = np.array([[int(((y == c) & (fl == f)).sum()) for f in (1, 0)] for c in (1, 0)])
        chi2, p, _, _ = stats.chi2_contingency(tab)
        out[f"{g}|flag_x_diagnosis"] = dict(
            table_AD_yes_no=[int(tab[0, 0]), int(tab[0, 1])],
            table_CTL_yes_no=[int(tab[1, 0]), int(tab[1, 1])],
            p_yes_given_AD=float(tab[0, 0] / tab[0].sum()),
            p_yes_given_CTL=float(tab[1, 0] / tab[1].sum()),
            chi2=float(chi2), p=float(p))
        P(f"  {g}  P(yes|AD)={tab[0,0]/tab[0].sum():.3f}  "
          f"P(yes|CTL)={tab[1,0]/tab[1].sum():.3f}  chi2={chi2:.1f}  p={p:.3g}")

    # A4b  predict the FLAG from dark probes, diagnosis ignored.
    #      A high AUC here is the mechanism: dark probes read processing groups.
    P("  --- predicting the flag itself from probes (diagnosis ignored) ---")
    for g in CONFIG["SERIES"]:
        for rule in ("STRICT-100", "DETECTED"):
            m, folds, n_pr = cv_auc(D[g]["X"], D[g]["flag"], rule)
            out[f"{g}|predict_flag|{rule}"] = dict(auc=m, folds=folds, n_probes=n_pr)
            P(f"  {g:<12}{'flag ~ ' + rule:<24}{m:>8.3f}")

    # A4a  restrict to flag == yes, redo the cross-cohort transfer
    P("  --- transfer restricted to flag == yes ---")
    for a, b in directions():
        ka, kb = D[a]["flag"] == 1, D[b]["flag"] == 1
        Xa, ya = D[a]["X"][ka], D[a]["y"][ka]
        Xb, yb = D[b]["X"][kb], D[b]["y"][kb]
        T = floor_of(Xa)
        idxs = boot_indices(yb, CONFIG["N_BOOT"], CONFIG["BOOT_SEED"])
        for rule in ("STRICT-100", "DETECTED"):
            A, B, n_pr = prep(Xa, Xb, rule, T)
            s = fit_score(A, ya, B)
            obs = roc_auc_score(yb, s)
            lo, hi = ci([roc_auc_score(yb[i], s[i]) for i in idxs])
            out[f"{a[-2:]}->{b[-2:]}|flagyes|{rule}"] = dict(
                obs=float(obs), lo=lo, hi=hi, n_train=int(ka.sum()),
                n_test=int(kb.sum()), n_probes=n_pr)
            P(f"  {a[-2:]+'->'+b[-2:]:<10}{rule:<12}n={int(ka.sum())}/{int(kb.sum())}"
              f"{obs:>8.3f}  [{lo:.3f}, {hi:.3f}]")
    save_stage("A4", out)
    return out


# ========================================================================= A5
def stage_A5(D):
    """Model family: is the finding an XGBoost artifact?"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    P("=" * 88)
    P("A5  MODEL FAMILY")
    P("=" * 88)

    def fam_cv(X, y, rule, maker):
        skf = StratifiedKFold(CONFIG["N_SPLITS"], shuffle=True,
                              random_state=CONFIG["SEED"])
        aucs = []
        for tr, te in skf.split(X, y):
            T = floor_of(X[tr])
            cols = np.where(RULES[rule](X[tr], T))[0]
            A, B = rank_rows(X[np.ix_(tr, cols)]), rank_rows(X[np.ix_(te, cols)])
            sel = SelectKBest(f_classif, k=min(CONFIG["N_TOP"], A.shape[1])).fit(A, y[tr])
            mdl = maker().fit(sel.transform(A), y[tr])
            aucs.append(roc_auc_score(y[te], mdl.predict_proba(sel.transform(B))[:, 1]))
        return float(np.mean(aucs))

    fams = {
        "XGBoost": lambda: make_xgb(),
        "L2 logistic": lambda: LogisticRegression(C=1.0, max_iter=2000,
                                                  random_state=CONFIG["SEED"]),
        "Random forest": lambda: RandomForestClassifier(
            n_estimators=500, random_state=CONFIG["SEED"], n_jobs=NTHREAD),
    }
    out = load_results().get("A5", {})
    for g in CONFIG["SERIES"]:
        for rule in ("STRICT-100", "DETECTED", "ALL"):
            if f"{g}|{rule}" in out:
                P(f"  {g:<12}{rule:<12}cached")
                continue
            row = {name: fam_cv(D[g]["X"], D[g]["y"], rule, mk)
                   for name, mk in fams.items()}
            out[f"{g}|{rule}"] = row
            save_stage("A5", out)
            P(f"  {g:<12}{rule:<12}" + "".join(f"{v:>9.3f}" for v in row.values())
              + "    (" + ", ".join(fams) + ")")
    save_stage("A5", out)
    return out


# ========================================================================= A6
def stage_A6(D):
    """Hyperparameter grid search, nested inside the TRAINING cohort only.

    Reported as a robustness arm, not as the headline: the permutation null must
    run an identical procedure on shuffled labels, and grid-searching inside 1000
    permutations is infeasible. Tuning the observed value while leaving the null
    untuned would inflate significance.
    """
    P("=" * 88)
    P("A6  HYPERPARAMETER GRID SEARCH  (nested in train cohort; robustness only)")
    P("=" * 88)
    grid = list(ParameterGrid(CONFIG["GRID"]))
    P(f"  {len(grid)} combinations x {CONFIG['GRID_INNER_FOLDS']} inner folds per arm")
    out = {}
    for a, b in directions():
        Xa, ya, Xb, yb = D[a]["X"], D[a]["y"], D[b]["X"], D[b]["y"]
        T = floor_of(Xa)
        for rule in ("STRICT-100", "DETECTED"):
            A, B, n_pr = prep(Xa, Xb, rule, T)
            inner = StratifiedKFold(CONFIG["GRID_INNER_FOLDS"], shuffle=True,
                                    random_state=CONFIG["SEED"])
            best, best_auc, scored = None, -1.0, []
            for params in grid:
                sc = []
                for tr, te in inner.split(A, ya):
                    sc.append(roc_auc_score(
                        ya[te], fit_score(A[tr], ya[tr], A[te], **params)))
                mean_sc = float(np.mean(sc))
                scored.append(dict(params=params, inner_auc=mean_sc))
                if mean_sc > best_auc:
                    best, best_auc = params, mean_sc
            tuned = float(roc_auc_score(yb, fit_score(A, ya, B, **best)))
            fixed = float(roc_auc_score(yb, fit_score(A, ya, B)))
            out[f"{a[-2:]}->{b[-2:]}|{rule}"] = dict(
                best_params=best, inner_cv_auc=best_auc,
                tuned_test_auc=tuned, fixed_test_auc=fixed,
                delta=tuned - fixed, n_probes=n_pr,
                all_combinations=scored)
            P(f"  {a[-2:]+'->'+b[-2:]:<10}{rule:<12} fixed {fixed:.3f} -> tuned "
              f"{tuned:.3f}  (delta {tuned - fixed:+.3f})   best={best}")
            save_stage("A6", out)
    save_stage("A6", out)
    return out


# ========================================================================= R1
def stage_R1(D):
    P("=" * 88)
    P("R1  MODEL-SEED REPEATS  (10 seeds, cross-cohort transfer)")
    P("=" * 88)
    out = {}
    for a, b in directions():
        Xa, ya, Xb, yb = D[a]["X"], D[a]["y"], D[b]["X"], D[b]["y"]
        T = floor_of(Xa)
        for rule in ("STRICT-100", "DETECTED"):
            A, B, _ = prep(Xa, Xb, rule, T)
            v = [roc_auc_score(yb, fit_score(A, ya, B, seed=s))
                 for s in range(CONFIG["N_SEEDS"])]
            out[f"{a[-2:]}->{b[-2:]}|{rule}"] = dict(
                mean=float(np.mean(v)), sd=float(np.std(v)),
                min=float(np.min(v)), max=float(np.max(v)), values=v)
            P(f"  {a[-2:]+'->'+b[-2:]:<10}{rule:<12}{np.mean(v):>8.3f} +/- {np.std(v):.3f}"
              f"   [{np.min(v):.3f}, {np.max(v):.3f}]")
    save_stage("R1", out)
    return out


# ========================================================================= R2
def stage_R2(D):
    P("=" * 88)
    P("R2  DETECTION-FLOOR SWEEP")
    P("=" * 88)
    out = load_results().get("R2", {})          # resume after a container restart
    for g in CONFIG["SERIES"]:
        for k in CONFIG["FLOOR_KS"]:
            if f"{g}|k={k}" in out:
                P(f"  {g:<12}k={k:<5}cached")
                continue
            row = {}
            for rule in ("STRICT-100", "DETECTED"):
                m, folds, n_pr = cv_auc(D[g]["X"], D[g]["y"], rule, k=k)
                row[rule] = dict(auc=m, folds=folds, n_probes=n_pr)
            out[f"{g}|k={k}"] = row
            save_stage("R2", out)               # checkpoint every cell
            P(f"  {g:<12}k={k:<5}n_dark={row['STRICT-100']['n_probes']:>7}"
              f"   dark {row['STRICT-100']['auc']:.3f}"
              f"   det {row['DETECTED']['auc']:.3f}")
    save_stage("R2", out)
    return out


# ========================================================================= R3
def stage_R3(D):
    """The three KDE constants that move the floor and were never varied."""
    P("=" * 88)
    P("R3  KDE CONSTANT STABILITY  (subsample seed and bandwidth)")
    P("=" * 88)
    out = {}
    for g in CONFIG["SERIES"]:
        X = D[g]["X"]
        for bw in CONFIG["KDE_STABILITY_BWS"]:
            floors, ndark = [], []
            for s in CONFIG["KDE_STABILITY_SEEDS"]:
                T = floor_of(X, bw=bw, sub_seed=s)
                floors.append(T)
                ndark.append(int(RULES["STRICT-100"](X, T).sum()))
            out[f"{g}|bw={bw}"] = dict(
                floor_mean=float(np.mean(floors)), floor_sd=float(np.std(floors)),
                floor_min=float(np.min(floors)), floor_max=float(np.max(floors)),
                ndark_mean=float(np.mean(ndark)), ndark_sd=float(np.std(ndark)),
                ndark_min=int(np.min(ndark)), ndark_max=int(np.max(ndark)),
                floors=floors, ndark=ndark)
            o = out[f"{g}|bw={bw}"]
            P(f"  {g:<12}bw={bw:<6}floor {o['floor_mean']:.4f} +/- {o['floor_sd']:.4f}"
              f"   n_dark {o['ndark_mean']:.0f} +/- {o['ndark_sd']:.0f}"
              f"   [{o['ndark_min']}, {o['ndark_max']}]")
    save_stage("R3", out)
    return out


# ========================================================================= T1
def _perm_one(A, yp, B, yb):
    """One permutation: refit selection+model on shuffled train labels, score on
    the true test labels. Top level so joblib can pickle it."""
    return float(roc_auc_score(yb, fit_score(A, yp, B)))


def stage_T1(D):
    """Permutation null. Labels shuffled in the TRAIN cohort; floor/mask/selection/
    fit all redone; scored against the TRUE test labels. Checkpointed."""
    P("=" * 88)
    P(f"T1  PERMUTATION NULL  ({CONFIG['N_PERM']} shuffles, checkpoint every "
      f"{CONFIG['PERM_CHECKPOINT']})")
    P("=" * 88)
    out = load_results().get("T1", {})
    for a, b, rule in CONFIG["PERM_ARMS"]:
        key = f"{a[-2:]}->{b[-2:]}|{rule}"
        Xa, ya, Xb, yb = D[a]["X"], D[a]["y"], D[b]["X"], D[b]["y"]
        T = floor_of(Xa)
        A, B, n_pr = prep(Xa, Xb, rule, T)
        obs = float(roc_auc_score(yb, fit_score(A, ya, B)))

        done = out.get(key, {}).get("null", [])
        P(f"  {key}   observed {obs:.4f}   n_probes {n_pr}   resuming at {len(done)}")
        rs = np.random.RandomState(20260808 + abs(hash(key)) % 10000)
        # burn the draws already consumed so a resume reproduces the same stream
        for _ in range(len(done)):
            rs.permutation(len(ya))

        chunk = CONFIG["PERM_CHECKPOINT"]
        while len(done) < CONFIG["N_PERM"]:
            n = min(chunk, CONFIG["N_PERM"] - len(done))
            # draw the permutations sequentially so the RNG stream (and therefore
            # the result) is identical no matter how many workers run them
            perms = [ya[rs.permutation(len(ya))] for _ in range(n)]
            if PERM_JOBS > 1:
                from joblib import Parallel, delayed
                vals = Parallel(n_jobs=PERM_JOBS, prefer="processes")(
                    delayed(_perm_one)(A, yp, B, yb) for yp in perms)
            else:
                vals = [_perm_one(A, yp, B, yb) for yp in perms]
            done.extend(vals)
            if True:
                nul = np.array(done)
                # +1 correction: p is never exactly zero with finite permutations
                pval = float((np.sum(nul >= obs) + 1) / (len(nul) + 1))
                out[key] = dict(observed=obs, n_probes=n_pr, n_perm=len(done),
                                null_mean=float(nul.mean()), null_sd=float(nul.std()),
                                null_max=float(nul.max()),
                                null_q95=float(np.percentile(nul, 95)),
                                p_value=pval, complete=len(done) >= CONFIG["N_PERM"],
                                null=done)
                save_stage("T1", out)
                P(f"    {key}  {len(done):>5}/{CONFIG['N_PERM']}  null "
                  f"{nul.mean():.3f}+/-{nul.std():.3f}  max {nul.max():.3f}  p={pval:.4f}")
    return out


# ========================================================================= A7
def stage_A7(D):
    """Technical batch: BeadChip barcode and recruitment centre.

    Both are recoverable from the series matrix (Sample_title carries the chip
    barcode and array position; Sample_description carries a subject code whose
    leading letters are the centre). Neither appears in the characteristics
    fields, which is why they were missed initially.
    """
    P("=" * 88)
    P("A7  BATCH STRUCTURE: BEADCHIP AND RECRUITMENT CENTRE")
    P("=" * 88)
    out = {}
    for g in CONFIG["SERIES"]:
        X, y = D[g]["X"], D[g]["y"]
        for var in ("chip", "site"):
            v = D[g][var]
            if v is None:
                continue
            lev = sorted(set(v))
            tab = np.array([[int(((v == L) & (y == c)).sum()) for L in lev] for c in (1, 0)])
            tab = tab[:, tab.sum(axis=0) > 0]
            chi2, pv, dof, _ = stats.chi2_contingency(tab)
            pure = int(((tab == 0).any(axis=0)).sum())
            n_pure = int(tab[:, (tab == 0).any(axis=0)].sum())
            out[f"{g}|{var}|balance"] = dict(
                n_levels=int(tab.shape[1]), chi2=float(chi2), dof=int(dof), p=float(pv),
                levels_single_diagnosis=pure, samples_in_those=n_pure,
                counts={str(L): [int(((v == L) & (y == 1)).sum()),
                                 int(((v == L) & (y == 0)).sum())] for L in lev})
            P(f"  {g}  {var:<5} {tab.shape[1]:>3} levels | chi2={chi2:>7.1f} dof={dof:>3} "
              f"p={pv:.3g} | single-diagnosis levels {pure} covering {n_pure} samples")

            # can the probes identify the batch variable itself?
            for rule in ("STRICT-100", "DETECTED"):
                codes = {L: i for i, L in enumerate(lev)}
                yv = np.array([codes[a] for a in v])
                keep = np.array([np.sum(yv == c) >= 8 for c in yv])
                if keep.sum() < 40 or len(set(yv[keep])) < 2:
                    continue
                acc = multiclass_cv(X[keep], yv[keep], rule)
                base = float(np.max(np.bincount(yv[keep])) / keep.sum())
                out[f"{g}|{var}|predict|{rule}"] = dict(
                    accuracy=acc, majority_baseline=base, n=int(keep.sum()),
                    n_classes=int(len(set(yv[keep]))))
                P(f"      predict {var:<5} from {rule:<11} acc {acc:.3f} "
                  f"(majority baseline {base:.3f}, {len(set(yv[keep]))} classes)")
        save_stage("A7", out)
    save_stage("A7", out)
    return out


def multiclass_cv(X, y, rule):
    """Accuracy at predicting a multi-level batch variable, 5-fold, floor per fold."""
    from sklearn.ensemble import RandomForestClassifier
    skf = StratifiedKFold(3, shuffle=True, random_state=CONFIG["SEED"])
    accs = []
    for tr, te in skf.split(X, y):
        T = floor_of(X[tr])
        cols = np.where(RULES[rule](X[tr], T))[0]
        A, B = rank_rows(X[np.ix_(tr, cols)]), rank_rows(X[np.ix_(te, cols)])
        sel = SelectKBest(f_classif, k=min(CONFIG["N_TOP"], A.shape[1])).fit(A, y[tr])
        m = RandomForestClassifier(n_estimators=300, random_state=CONFIG["SEED"],
                                   n_jobs=NTHREAD).fit(sel.transform(A), y[tr])
        accs.append(float((m.predict(sel.transform(B)) == y[te]).mean()))
    return float(np.mean(accs))


# ========================================================================= A8
def stage_A8(D):
    """Leave-one-clinic-out. Train on every recruitment centre but one, test on
    the held-out centre. If the signal survives a clinic the model never saw,
    clinic imbalance cannot be what produces it.

    Primary metric is the POOLED out-of-fold AUC: every subject is scored by a
    model that never saw their clinic, then all those scores are ranked together.
    Per-clinic AUCs are also reported but are noisy where a centre is small.
    """
    P("=" * 88)
    P("A8  LEAVE-ONE-CLINIC-OUT")
    P("=" * 88)
    out = load_results().get("A8", {})
    for g in CONFIG["SERIES"]:
        X, y, site = D[g]["X"], D[g]["y"], D[g]["site"]
        if site is None:
            P(f"  {g}: no site field, skipped")
            continue
        for rule in ("STRICT-100", "DETECTED"):
            if f"{g}|{rule}" in out:
                P(f"  {g:<12}{rule:<12}cached")
                continue
            oof = np.full(len(y), np.nan)
            per = {}
            for s in sorted(set(site)):
                te = site == s
                tr = ~te
                if tr.sum() < 40 or len(set(y[tr])) < 2 or te.sum() < 3:
                    continue
                T = floor_of(X[tr])
                cols = np.where(RULES[rule](X[tr], T))[0]
                if len(cols) < 10:
                    continue
                A = rank_rows(X[np.ix_(np.where(tr)[0], cols)])
                B = rank_rows(X[np.ix_(np.where(te)[0], cols)])
                sc = fit_score(A, y[tr], B)
                oof[te] = sc
                per[str(s)] = dict(n=int(te.sum()), n_AD=int(y[te].sum()),
                                   n_CTL=int((y[te] == 0).sum()), n_probes=int(len(cols)),
                                   auc=float(roc_auc_score(y[te], sc))
                                   if len(set(y[te])) == 2 else None)
            m = ~np.isnan(oof)
            pooled = float(roc_auc_score(y[m], oof[m])) if len(set(y[m])) == 2 else None
            # bootstrap the pooled estimate
            lo = hi = None
            if pooled is not None:
                ys, ss = y[m], oof[m]
                bs = [roc_auc_score(ys[i], ss[i])
                      for i in boot_indices(ys, 1000, CONFIG["BOOT_SEED"])]
                lo, hi = ci(bs)
            out[f"{g}|{rule}"] = dict(pooled_auc=pooled, lo=lo, hi=hi,
                                      n_scored=int(m.sum()), n_clinics=len(per),
                                      per_clinic=per)
            P(f"  {g:<12}{rule:<12}pooled {pooled:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"n={int(m.sum())} across {len(per)} clinics")
            for k, v in sorted(per.items()):
                a = f"{v['auc']:.3f}" if v["auc"] is not None else "  n/a"
                P(f"      {k:<8} n={v['n']:>3} ({v['n_AD']} AD / {v['n_CTL']} CTL)  auc {a}")
            save_stage("A8", out)
    save_stage("A8", out)
    return out


# ======================================================================== main
STAGES = dict(A1=stage_A1, A2=stage_A2, A3=stage_A3, A4=stage_A4, A5=stage_A5,
              A6=stage_A6, A7=stage_A7, A8=stage_A8, R1=stage_R1, R2=stage_R2,
              R3=stage_R3,
              T1=stage_T1)
GROUPS = dict(fast=["A1", "A2", "A3", "A4", "A5", "A7", "A8", "R1", "R2", "R3"],
              all=list(STAGES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="fast",
                    help="comma list of stage names, or 'fast' / 'all'")
    ap.add_argument("--nthread", type=int, default=None,
                    help="threads per XGBoost fit")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="XGBoost device. On this data (n~250, 500 features after "
                         "selection) cuda is usually SLOWER than cpu - benchmark it.")
    ap.add_argument("--perm-jobs", type=int, default=None,
                    help="parallel workers for the T1 permutation loop")
    ap.add_argument("--nperm", type=int, default=None, help="override N_PERM")
    args = ap.parse_args()
    global NTHREAD, DEVICE, PERM_JOBS
    if args.nthread:
        NTHREAD = args.nthread
    if args.device:
        DEVICE = args.device
    if args.perm_jobs:
        PERM_JOBS = args.perm_jobs
    if args.nperm:
        CONFIG["N_PERM"] = args.nperm

    req = GROUPS.get(args.stages, args.stages.split(","))
    bad = [s for s in req if s not in STAGES]
    if bad:
        sys.exit(f"unknown stage(s): {bad}. known: {list(STAGES)}")

    P(f"loading data   threads={NTHREAD}   stages={req}")
    D, info = load_data()
    save_stage("data", info)
    for g in CONFIG["SERIES"]:
        P(f"  {g}: n={info['n'][g]['total']}  AD={info['n'][g]['AD']}  "
          f"CTL={info['n'][g]['CTL']}  flag_yes={info['n'][g]['flag_yes']}  "
          f"dropped={info['dropped'][g]}")
    P(f"  shared probes: {info['n_shared_probes']}")

    for s in req:
        STAGES[s](D)
    P(f"DONE  ({', '.join(req)})  ->  {RESULTS}")


if __name__ == "__main__":
    main()
