#!/usr/bin/env python3
"""Build a self-contained HTML results report from output/results.json.
Re-runnable: run again when T1 finishes and the permutation panel fills in."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "output", "results.json")))
OUT = os.path.join(HERE, "output", "results_report.html")

S1, S2, S3 = "var(--series-1)", "var(--series-2)", "var(--series-3)"
MUTED, GRID, INK2 = "var(--muted)", "var(--grid)", "var(--text-secondary)"
LBL = {"STRICT-100": "Sub-floor probes", "DETECTED": "Detected probes", "ALL": "All probes"}
COL = {"STRICT-100": S1, "DETECTED": S2, "ALL": S3}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ------------------------------------------------------------------ chart 1
def forest():
    """Cross-cohort transfer: AUC with 95% bootstrap CI, chance line at 0.5."""
    rows = []
    for a, b in (("60", "61"), ("61", "60")):
        rows.append(("HEADER", f"GSE630{a} → GSE630{b}", None))
        for rule in ("STRICT-100", "DETECTED", "ALL"):
            rows.append(("ROW", rule, R["A1"][f"{a}->{b}|{rule}"]))
    W, rowh, top, left, right = 860, 42, 56, 220, 44
    H = top + rowh * len(rows) + 46
    x0, x1 = left, W - right
    lo_d, hi_d = 0.45, 0.95

    def X(v):
        return x0 + (v - lo_d) / (hi_d - lo_d) * (x1 - x0)

    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
         f'aria-label="Cross-cohort transfer AUC with 95% confidence intervals">']
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        p.append(f'<line x1="{X(t):.1f}" y1="{top-18}" x2="{X(t):.1f}" y2="{top+rowh*len(rows)-14}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{X(t):.1f}" y="{H-26}" fill="{MUTED}" font-size="12" '
                 f'text-anchor="middle">{t:.1f}</text>')
    p.append(f'<line x1="{X(0.5):.1f}" y1="{top-18}" x2="{X(0.5):.1f}" '
             f'y2="{top+rowh*len(rows)-14}" stroke="{MUTED}" stroke-width="2" '
             f'stroke-dasharray="5 4"/>')
    p.append(f'<text x="{X(0.5):.1f}" y="{top-26}" fill="{MUTED}" font-size="12" '
             f'text-anchor="middle">chance</text>')
    for i, (kind, rule, d) in enumerate(rows):
        y = top + i * rowh
        if kind == "HEADER":
            p.append(f'<text x="12" y="{y+5}" fill="var(--text-primary)" font-size="13" '
                     f'font-weight="700">{rule}</text>')
            continue
        p.append(f'<text x="{left-14}" y="{y+5}" fill="{INK2}" font-size="13" '
                 f'text-anchor="end">{LBL[rule]}</text>')
        c = COL[rule]
        p.append(f'<line x1="{X(d["lo"]):.1f}" y1="{y}" x2="{X(d["hi"]):.1f}" y2="{y}" '
                 f'stroke="{c}" stroke-width="2" stroke-linecap="round"/>')
        for e in ("lo", "hi"):
            p.append(f'<line x1="{X(d[e]):.1f}" y1="{y-5}" x2="{X(d[e]):.1f}" y2="{y+5}" '
                     f'stroke="{c}" stroke-width="2"/>')
        p.append(f'<circle cx="{X(d["obs"]):.1f}" cy="{y}" r="6" fill="{c}" '
                 f'stroke="var(--surface-1)" stroke-width="2"><title>{LBL[rule]}: '
                 f'AUC {d["obs"]:.3f} (95% CI {d["lo"]:.3f}–{d["hi"]:.3f}), '
                 f'{d["n_probes"]:,} probes</title></circle>')
        p.append(f'<text x="{X(d["hi"])+12:.1f}" y="{y+5}" fill="var(--text-primary)" '
                 f'font-size="13" font-weight="600">{d["obs"]:.3f}</text>')
    p.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-6}" fill="{MUTED}" font-size="12" '
             f'text-anchor="middle">AUC on the held-out cohort</text>')
    p.append("</svg>")
    return "\n".join(p)


# ------------------------------------------------------------------ chart 2
def permutation():
    if "T1" not in R or not R["T1"]:
        return '<p class="pending">Permutation null still running.</p>', None
    panels, meta = [], []
    for key, d in sorted(R["T1"].items()):
        nul = d.get("null", [])
        if not nul:
            continue
        W, H, left, right, top, bot = 420, 250, 44, 18, 26, 44
        lo_d, hi_d = 0.35, max(0.75, d["observed"] + 0.05)
        nb = 26
        edges = [lo_d + (hi_d - lo_d) * i / nb for i in range(nb + 1)]
        cnt = [0] * nb
        for v in nul:
            j = min(nb - 1, max(0, int((v - lo_d) / (hi_d - lo_d) * nb)))
            cnt[j] += 1
        mx = max(cnt) or 1

        def X(v):
            return left + (v - lo_d) / (hi_d - lo_d) * (W - left - right)

        def Y(c):
            return H - bot - c / mx * (H - bot - top)

        p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
             f'aria-label="Permutation null distribution for {esc(key)}">']
        for j in range(nb):
            if not cnt[j]:
                continue
            x, w = X(edges[j]), X(edges[j + 1]) - X(edges[j]) - 2
            p.append(f'<rect x="{x:.1f}" y="{Y(cnt[j]):.1f}" width="{max(w,1):.1f}" '
                     f'height="{H-bot-Y(cnt[j]):.1f}" rx="3" fill="{MUTED}" '
                     f'opacity="0.55"><title>{cnt[j]} shuffles between '
                     f'{edges[j]:.3f} and {edges[j+1]:.3f}</title></rect>')
        p.append(f'<line x1="{left}" y1="{H-bot}" x2="{W-right}" y2="{H-bot}" '
                 f'stroke="var(--baseline)" stroke-width="1"/>')
        ox = X(d["observed"])
        p.append(f'<line x1="{ox:.1f}" y1="{top-8}" x2="{ox:.1f}" y2="{H-bot}" '
                 f'stroke="{S1}" stroke-width="2"/>')
        p.append(f'<circle cx="{ox:.1f}" cy="{top-8}" r="5" fill="{S1}"/>')
        p.append(f'<text x="{ox:.1f}" y="{top-16}" fill="{S1}" font-size="12" '
                 f'font-weight="600" text-anchor="middle">observed {d["observed"]:.3f}</text>')
        for t in (0.4, 0.5, 0.6, 0.7):
            if lo_d <= t <= hi_d:
                p.append(f'<text x="{X(t):.1f}" y="{H-24}" fill="{MUTED}" font-size="11" '
                         f'text-anchor="middle">{t:.1f}</text>')
        p.append(f'<text x="{(left+W-right)/2:.0f}" y="{H-6}" fill="{MUTED}" font-size="11" '
                 f'text-anchor="middle">AUC under shuffled training labels</text>')
        p.append("</svg>")
        panels.append(f'<figure class="panel"><figcaption>{esc(key)}</figcaption>'
                      + "\n".join(p) + "</figure>")
        meta.append((key, d))
    return '<div class="grid2">' + "".join(panels) + "</div>", meta


# ------------------------------------------------------------------ chart 3
def floor_sweep():
    if "R2" not in R:
        return '<p class="pending">Floor sweep still running.</p>'
    panels = []
    for g in ("GSE63060", "GSE63061"):
        ks = [k for k in (2.0, 2.5, 3.0, 3.5, 4.0) if f"{g}|k={k}" in R["R2"]]
        if not ks:
            continue
        W, H, left, right, top, bot = 420, 260, 46, 20, 24, 48
        lo_d, hi_d = 0.55, 0.95

        def X(k):
            return left + (k - 2.0) / 2.0 * (W - left - right)

        def Y(v):
            return H - bot - (v - lo_d) / (hi_d - lo_d) * (H - bot - top)

        p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
             f'aria-label="AUC across detection-floor positions, {g}">']
        for t in (0.6, 0.7, 0.8, 0.9):
            p.append(f'<line x1="{left}" y1="{Y(t):.1f}" x2="{W-right}" y2="{Y(t):.1f}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
            p.append(f'<text x="{left-8}" y="{Y(t)+4:.1f}" fill="{MUTED}" font-size="11" '
                     f'text-anchor="end">{t:.1f}</text>')
        for rule in ("DETECTED", "STRICT-100"):
            pts = [(X(k), Y(R["R2"][f"{g}|k={k}"][rule]["auc"])) for k in ks]
            p.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                     + f'" fill="none" stroke="{COL[rule]}" stroke-width="2" '
                       'stroke-linejoin="round"/>')
            for k, (x, y) in zip(ks, pts):
                dd = R["R2"][f"{g}|k={k}"][rule]
                p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{COL[rule]}" '
                         f'stroke="var(--surface-1)" stroke-width="2"><title>{LBL[rule]}, '
                         f'floor k={k}: AUC {dd["auc"]:.3f}, {dd["n_probes"]:,} probes'
                         f'</title></circle>')
            lx, ly = pts[-1]
            p.append(f'<text x="{lx-4:.1f}" y="{ly-12:.1f}" fill="{COL[rule]}" font-size="12" '
                     f'font-weight="600" text-anchor="end">{LBL[rule]}</text>')
        for k in ks:
            p.append(f'<text x="{X(k):.1f}" y="{H-26}" fill="{MUTED}" font-size="11" '
                     f'text-anchor="middle">{k}</text>')
        p.append(f'<text x="{(left+W-right)/2:.0f}" y="{H-6}" fill="{MUTED}" font-size="11" '
                 f'text-anchor="middle">detection-floor multiplier (k)</text>')
        p.append("</svg>")
        panels.append(f'<figure class="panel"><figcaption>{g}</figcaption>'
                      + "\n".join(p) + "</figure>")
    return '<div class="grid2">' + "".join(panels) + "</div>"


# ------------------------------------------------------------------ chart 4
def baseline_ladder():
    W, rowh, top, left, right = 860, 40, 30, 250, 60
    rows = []
    for g in ("GSE63060", "GSE63061"):
        rows.append((g, "Age + sex only", R["A3"][f"{g}|COVARIATES_ONLY"]["auc"], MUTED))
        rows.append((g, LBL["STRICT-100"], R["A2"][f"{g}|STRICT-100"]["auc"], S1))
        rows.append((g, LBL["DETECTED"], R["A2"][f"{g}|DETECTED"]["auc"], S2))
    H = top + rowh * len(rows) + 40
    x0, x1 = left, W - right

    def X(v):
        return x0 + (v - 0.5) / 0.45 * (x1 - x0)

    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
         f'aria-label="Within-cohort AUC versus a demographics-only baseline">']
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        p.append(f'<line x1="{X(t):.1f}" y1="{top-14}" x2="{X(t):.1f}" '
                 f'y2="{top+rowh*len(rows)-16}" stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{X(t):.1f}" y="{H-16}" fill="{MUTED}" font-size="12" '
                 f'text-anchor="middle">{t:.1f}</text>')
    last = None
    for i, (g, name, v, c) in enumerate(rows):
        y = top + i * rowh
        if g != last:
            p.append(f'<text x="12" y="{y+5}" fill="var(--text-primary)" font-size="13" '
                     f'font-weight="600">{g}</text>')
            last = g
        p.append(f'<text x="{left-14}" y="{y+5}" fill="{INK2}" font-size="13" '
                 f'text-anchor="end">{name}</text>')
        p.append(f'<rect x="{X(0.5):.1f}" y="{y-9}" width="{max(X(v)-X(0.5),2):.1f}" '
                 f'height="18" rx="4" fill="{c}"><title>{name}, {g}: AUC {v:.3f}</title></rect>')
        p.append(f'<text x="{X(v)+10:.1f}" y="{y+5}" fill="var(--text-primary)" '
                 f'font-size="13" font-weight="600">{v:.3f}</text>')
    p.append("</svg>")
    return "\n".join(p)


# ------------------------------------------------------------------- tables
def model_family_table():
    if "A5" not in R:
        return ""
    h = ['<table><thead><tr><th>Cohort</th><th>Probe set</th><th>XGBoost</th>'
         '<th>L2 logistic</th><th>Random forest</th></tr></thead><tbody>']
    for g in ("GSE63060", "GSE63061"):
        for rule in ("STRICT-100", "DETECTED"):
            d = R["A5"].get(f"{g}|{rule}")
            if not d:
                continue
            h.append(f'<tr><td>{g}</td><td>{LBL[rule]}</td>'
                     + "".join(f"<td class='n'>{d[m]:.3f}</td>"
                               for m in ("XGBoost", "L2 logistic", "Random forest"))
                     + "</tr>")
    return "\n".join(h) + "</tbody></table>"


def seed_table():
    h = ['<table><thead><tr><th>Transfer</th><th>Probe set</th><th>Mean AUC</th>'
         '<th>SD</th><th>Range</th></tr></thead><tbody>']
    for k, d in R["R1"].items():
        a, rule = k.split("|")
        h.append(f'<tr><td>GSE630{a.replace("->", " → GSE630")}</td>'
                 f'<td>{LBL[rule]}</td><td class="n">{d["mean"]:.3f}</td>'
                 f'<td class="n">{d["sd"]:.3f}</td>'
                 f'<td class="n">{d["min"]:.3f}–{d["max"]:.3f}</td></tr>')
    return "\n".join(h) + "</tbody></table>"


def flag_table():
    if "A4" not in R:
        return ""
    h = ['<table><thead><tr><th>Cohort</th><th>Predicting the diagnosis</th>'
         '<th>Predicting the inclusion flag</th></tr></thead><tbody>']
    for g in ("GSE63060", "GSE63061"):
        dx = R["A2"][f"{g}|STRICT-100"]["auc"]
        fl = R["A4"].get(f"{g}|predict_flag|STRICT-100", {}).get("auc")
        h.append(f'<tr><td>{g}</td><td class="n">{dx:.3f}</td>'
                 f'<td class="n">{fl:.3f}</td></tr>' if fl else "")
    return "\n".join(h) + "</tbody></table>"


# --------------------------------------------------------------------- page
perm_html, perm_meta = permutation()
d = R["data"]
n60, n61 = d["n"]["GSE63060"], d["n"]["GSE63061"]

perm_summary = ""
if perm_meta:
    bits = []
    for key, m in perm_meta:
        state = "complete" if m.get("complete") else f"in progress, {m['n_perm']}/1000"
        bits.append(f"<li><strong>{esc(key)}</strong> — observed {m['observed']:.3f}, "
                    f"null {m['null_mean']:.3f} ± {m['null_sd']:.3f} "
                    f"(max {m['null_max']:.3f}), <em>p</em> = {m['p_value']:.4f} "
                    f"<span class='muted'>({state})</span></li>")
    perm_summary = "<ul>" + "".join(bits) + "</ul>"

HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sub-floor probes carry group-separating signal — results</title>
<style>
:root {{
  color-scheme: light dark;
  --page: #f9f9f7; --surface-1: #fcfcfb;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
  --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --page: #0d0d0d; --surface-1: #1a1a19;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--page); color: var(--text-primary);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.6; }}
.wrap {{ max-width: 980px; margin: 0 auto; padding: 40px 24px 80px; }}
h1 {{ font-size: 28px; line-height: 1.25; margin: 0 0 6px; letter-spacing: -0.01em; }}
h2 {{ font-size: 19px; margin: 44px 0 4px; letter-spacing: -0.005em; }}
.sub {{ color: var(--text-secondary); margin: 0 0 8px; }}
.card {{ background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 14px; padding: 20px 22px; margin: 16px 0; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 12px; margin: 20px 0 8px; }}
.tile {{ background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 14px; padding: 16px 18px; }}
.tile .v {{ font-size: 30px; font-weight: 650; letter-spacing: -0.02em; }}
.tile .k {{ color: var(--text-secondary); font-size: 13px; }}
.tile .n {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
.grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr));
  gap: 14px; }}
.panel {{ margin: 0; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 12px; padding: 12px 12px 4px; }}
figcaption {{ font-size: 13px; font-weight: 600; color: var(--text-secondary);
  margin-bottom: 4px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin-top: 10px; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--grid); }}
th {{ color: var(--text-secondary); font-weight: 600; font-size: 13px; }}
td.n, th.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
.muted {{ color: var(--muted); }}
.pending {{ color: var(--muted); font-style: italic; }}
.legend {{ display: flex; gap: 18px; flex-wrap: wrap; font-size: 13px;
  color: var(--text-secondary); margin: 4px 0 12px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
.dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; }}
.caveat {{ border-left: 3px solid var(--series-2); padding-left: 14px; }}
</style></head><body><div class="wrap">

<h1>Probes that never rise above background still separate Alzheimer's
patients from controls</h1>
<p class="sub">GSE63060 (n = {n60['total']}: {n60['AD']} AD, {n60['CTL']} CTL) and
GSE63061 (n = {n61['total']}: {n61['AD']} AD, {n61['CTL']} CTL), AddNeuroMed whole blood,
{d['n_shared_probes']:,} probes shared between Illumina HT-12 V3 and V4.
Model trained on one cohort, tested on the other.</p>

<div class="tiles">
  <div class="tile"><div class="v">0.720</div><div class="k">Sub-floor probes,
    GSE63061 → GSE63060</div><div class="n">95% CI 0.655–0.782</div></div>
  <div class="tile"><div class="v">0.668</div><div class="k">Sub-floor probes,
    GSE63060 → GSE63061</div><div class="n">95% CI 0.602–0.729</div></div>
  <div class="tile"><div class="v">0.573</div><div class="k">Age + sex baseline</div>
    <div class="n">GSE63060, 5-fold CV</div></div>
  <div class="tile"><div class="v">{R['A1']['61->60|STRICT-100']['n_probes']:,}</div>
    <div class="k">Probes never above background</div>
    <div class="n">of {d['n_shared_probes']:,} shared</div></div>
</div>

<h2>1. Cross-cohort transfer</h2>
<p class="sub">Trained on all of one study, tested on all of the other — different
subjects, different chip version. Bars are 95% bootstrap CIs over 2,000 stratified
resamples of the test cohort.</p>
<div class="legend">
  <span><i class="dot" style="background:{S1}"></i>Sub-floor probes (never above
    background in any training sample)</span>
  <span><i class="dot" style="background:{S2}"></i>Detected probes</span>
  <span><i class="dot" style="background:{S3}"></i>All probes</span>
</div>
<div class="card">{forest()}</div>
<p class="sub">Detected probes beat sub-floor probes by
{R['A1']['61->60|DET_minus_DARK']['obs']:.3f} and
{R['A1']['60->61|DET_minus_DARK']['obs']:.3f} on paired resamples, with the
difference above zero in 100% of 2,000 resamples. The claim is not that sub-floor
probes match real ones — it is that they are nowhere near chance.</p>

<h2>2. Permutation null</h2>
<p class="sub">Diagnosis labels shuffled in the training cohort; detection floor,
probe mask, feature selection and model refit end to end; scored against the true
test labels.</p>
{perm_html}
{perm_summary}

<h2>3. It is not demographics</h2>
<p class="sub">Age and sex alone, same folds, same model. Age does differ between
groups (GSE63060: 75.4 vs 72.4 years, p = 0.0003) but carries almost no
classification signal.</p>
<div class="card">{baseline_ladder()}</div>

<h2>4. It is not the sampling flag</h2>
<p class="sub">The <code>included in case-control study</code> flag is strongly
associated with diagnosis (&chi;&sup2; = 21.6, p = 3.3&times;10<sup>−6</sup> in
GSE63060). If sub-floor probes were reading processing structure rather than
subjects, they would predict that flag well. They do not.</p>
<div class="card">{flag_table()}</div>

<h2>5. It does not depend on where the floor is drawn</h2>
<p class="sub">Within-cohort 5-fold CV, floor and probe mask re-derived inside every
fold, across five floor positions.</p>
{floor_sweep()}
<p class="sub">Across ten KDE subsample seeds the floor moves by
&plusmn;0.004 and the sub-floor probe count by &plusmn;{R['R3']['GSE63060|bw=0.15']['ndark_sd']:.0f}
of {R['R3']['GSE63060|bw=0.15']['ndark_mean']:.0f} — about half a percent.</p>

<h2>6. It is not an XGBoost artifact</h2>
<div class="card">{model_family_table()}</div>

<h2>7. It is not a lucky seed</h2>
<div class="card">{seed_table()}</div>

<h2>Where it weakens</h2>
<div class="card caveat">
<p>Restricting to samples carrying the inclusion flag (n ≈ 116 train / 112 test)
drops GSE63060 → GSE63061 sub-floor performance to
{R['A4']['60->61|flagyes|STRICT-100']['obs']:.3f}
[{R['A4']['60->61|flagyes|STRICT-100']['lo']:.3f},
{R['A4']['60->61|flagyes|STRICT-100']['hi']:.3f}] — a confidence interval that
crosses chance. The reverse direction holds at
{R['A4']['61->60|flagyes|STRICT-100']['obs']:.3f}
[{R['A4']['61->60|flagyes|STRICT-100']['lo']:.3f},
{R['A4']['61->60|flagyes|STRICT-100']['hi']:.3f}]. Detected probes fall by a
similar margin in the restricted sample, which points to the halved sample size
rather than to the confound, but the interval is reported as measured.</p>
<p class="muted">Mechanism is not addressed here. What the sub-floor probes are
tracking — cell composition, RNA handling, run structure — is open.</p>
</div>

<p class="sub muted" style="margin-top:40px">
Generated from <code>output/results.json</code>.
Python {R['versions']['python']}, numpy {R['versions']['numpy']},
scikit-learn {R['versions']['sklearn']}, xgboost {R['versions']['xgboost']}.
Model: XGBoost, {R['config']['XGB']['n_estimators']} trees, depth
{R['config']['XGB']['max_depth']}, learning rate {R['config']['XGB']['learning_rate']};
top {R['config']['N_TOP']} probes by F-test, selected inside the training cohort only.
</p>
</div></body></html>
"""

with open(OUT, "w") as f:
    f.write(HTML)
print("wrote", OUT, len(HTML), "bytes")
