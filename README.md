# Sub-floor probes in Alzheimer's blood expression data

Do probes that never rise above the array's background level carry signal that
separates Alzheimer's patients from controls? They are removed by standard
detection filtering on the assumption that they carry nothing.

Data: **GSE63060** (Illumina HT-12 V3, n = 249 AD+CTL) and **GSE63061** (V4,
n = 273 AD+CTL), AddNeuroMed whole blood. 25,549 probes shared between the two
chip versions.

## Running it

```bash
pip install -r requirements.txt

# put GSE6306{0,1}_series_matrix.txt.gz in data/  (see colab_run.ipynb cell 4)

python analysis.py --stages fast     # A1 A2 A3 A4 A5 R1 R2 R3   (~20 min, 2 cores)
python analysis.py --stages T1       # permutation null          (long, checkpointed)
python analysis.py --stages A6       # hyperparameter grid       (~30 min)
python build_report.py               # -> output/results_report.html
```

Everything lands in `output/results.json`. Each stage writes its own key, so
stages can run in any order, in separate processes, or be resumed after a crash.
`--stages T1` re-run picks up at the last checkpoint; `--stages R2` skips cells
already computed.

Useful flags: `--nthread N` (threads per fit), `--perm-jobs N` (parallel
permutation workers), `--nperm N` (shorter null), `--device cuda`.

**On GPU:** after feature selection each fit sees ~250 rows x 500 columns, well
below the size where GPU histogram building beats CPU. `--device cuda` is
available but is usually slower here. Benchmark it (notebook cell 5) rather than
assuming.

## Design

Detection floor, probe mask, feature selection and model fit are **all** derived
from the training cohort only; nothing touches the test cohort before scoring.

The primary split is by whole study — train on all of GSE63060, test on all of
GSE63061, then reverse. Neither series matrix carries a site, batch or scan-date
field (the only sample attributes are status, ethnicity, age, gender, an
inclusion flag and tissue), so cross-series transfer is the only batch control
this data admits.

Probe sets, all defined from training samples:

| name | rule |
|---|---|
| `STRICT-100` | probe at or below the floor in **every** training sample |
| `DETECTED` | training-set mean above the floor |
| `ALL` | no filter |

Floor = KDE peak of per-probe means + `FLOOR_K` x SD of means near the peak.
Within-sample rank transform is applied after masking, which is what makes
transfer across chip versions valid. Then `SelectKBest(f_classif, k=500)` and
XGBoost (300 trees, depth 4, lr 0.05).

## Stages

| key | what it answers |
|---|---|
| A1 | cross-cohort transfer, both directions, 2000-resample bootstrap CIs |
| A2 | within-cohort 5-fold CV, floor and mask re-derived inside each fold |
| A3 | age+sex baseline — is the signal just demographics? |
| A4 | the `included in case-control study` flag — is it processing structure? |
| A5 | XGBoost vs L2 logistic vs random forest |
| A6 | hyperparameter grid, nested in the training cohort (robustness, not headline) |
| R1 | 10 model seeds |
| R2 | detection-floor sweep, `FLOOR_K` 2.0 to 4.0 |
| R3 | KDE bandwidth and subsample-seed stability |
| T1 | permutation null — labels shuffled in train, refit end to end |

A6 is deliberately not the headline. The permutation null must run the identical
procedure on shuffled labels, and grid-searching inside 1000 permutations is not
feasible; tuning the observed value while leaving the null untuned would inflate
the p-value.

## Repository layout

```
analysis.py        the pipeline - every number Results cites
build_report.py    results.json -> self-contained HTML report
colab_run.ipynb    Colab runner (data download, resume-safe stages)
diagnostics/       exploratory scripts that preceded analysis.py
pipeline.py        original single-cohort script (superseded)
data/              series matrices - gitignored, ~60 MB each, never commit
output/            results.json and the report - gitignored
```
