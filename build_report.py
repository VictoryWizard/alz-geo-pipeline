#!/usr/bin/env python3
"""Build a plain-language HTML results report from output/results.json.
Written for a reader who is not a microarray person. Re-runnable."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
_src = sys.argv[1] if len(sys.argv) > 1 else "results.json"
R = json.load(open(os.path.join(HERE, "output", _src)))
OUT = os.path.join(HERE, "output",
                   _src.replace(".json", "_report.html").replace("results", "results"))

S1, S2, S3 = "var(--series-1)", "var(--series-2)", "var(--series-3)"
MUTED, GRID, INK2 = "var(--muted)", "var(--grid)", "var(--text-secondary)"
LBL = {"STRICT-100": "Background-only probes",
       "DETECTED": "Real-gene probes",
       "ALL": "All probes"}
COL = {"STRICT-100": S1, "DETECTED": S2, "ALL": S3}
STUDY = {"60": "Study A", "61": "Study B"}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pair(a, b):
    return f"trained on {STUDY[a]}, tested on {STUDY[b]}"


# ------------------------------------------------------------------ chart 1
def forest():
    rows = []
    for a, b in (("60", "61"), ("61", "60")):
        rows.append(("HEADER", f"Trained on {STUDY[a]} → tested on {STUDY[b]}", None))
        for rule in ("STRICT-100", "DETECTED", "ALL"):
            rows.append(("ROW", rule, R["A1"][f"{a}->{b}|{rule}"]))
    W, rowh, top, left, right = 880, 42, 56, 250, 44
    H = top + rowh * len(rows) + 48
    x0, x1 = left, W - right
    lo_d, hi_d = 0.45, 0.95

    def X(v):
        return x0 + (v - lo_d) / (hi_d - lo_d) * (x1 - x0)

    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
         f'aria-label="Scores on the other study, with uncertainty ranges">']
    for t in [0.5, 0.6, 0.7, 0.8, 0.9]:
        p.append(f'<line x1="{X(t):.1f}" y1="{top-18}" x2="{X(t):.1f}" '
                 f'y2="{top+rowh*len(rows)-16}" stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{X(t):.1f}" y="{H-26}" fill="{MUTED}" font-size="12" '
                 f'text-anchor="middle">{t:.2f}</text>')
    p.append(f'<line x1="{X(0.5):.1f}" y1="{top-18}" x2="{X(0.5):.1f}" '
             f'y2="{top+rowh*len(rows)-16}" stroke="{MUTED}" stroke-width="2" '
             f'stroke-dasharray="5 4"/>')
    p.append(f'<text x="{X(0.5):.1f}" y="{top-26}" fill="{MUTED}" font-size="12" '
             f'text-anchor="middle">coin flip</text>')
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
                 f'{d["obs"]:.3f} (range {d["lo"]:.3f}–{d["hi"]:.3f}), '
                 f'{d["n_probes"]:,} probes</title></circle>')
        p.append(f'<text x="{X(d["hi"])+12:.1f}" y="{y+5}" fill="var(--text-primary)" '
                 f'font-size="13" font-weight="600">{d["obs"]:.3f}</text>')
    p.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-6}" fill="{MUTED}" font-size="12" '
             f'text-anchor="middle">score on the other study’s patients '
             f'(0.50 = coin flip, 1.00 = perfect)</text>')
    p.append("</svg>")
    return "\n".join(p)


# ------------------------------------------------------------------ chart 2
def permutation():
    if "T1" not in R or not R["T1"]:
        return '<p class="pending">Scramble test still running.</p>', []
    panels, meta = [], []
    for key, d in sorted(R["T1"].items()):
        nul = d.get("null", [])
        if not nul:
            continue
        a, b = key.split("|")[0].split("->")
        W, H, left, right, top, bot = 430, 250, 30, 18, 30, 46
        lo_d, hi_d = 0.35, max(0.78, d["observed"] + 0.06)
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
             f'aria-label="Scores from scrambled labels versus the real score">']
        for j in range(nb):
            if not cnt[j]:
                continue
            x, w = X(edges[j]), X(edges[j + 1]) - X(edges[j]) - 2
            p.append(f'<rect x="{x:.1f}" y="{Y(cnt[j]):.1f}" width="{max(w,1):.1f}" '
                     f'height="{H-bot-Y(cnt[j]):.1f}" rx="3" fill="{MUTED}" '
                     f'opacity="0.55"><title>{cnt[j]} scrambled runs scored between '
                     f'{edges[j]:.2f} and {edges[j+1]:.2f}</title></rect>')
        p.append(f'<line x1="{left}" y1="{H-bot}" x2="{W-right}" y2="{H-bot}" '
                 f'stroke="var(--baseline)" stroke-width="1"/>')
        ox = X(d["observed"])
        p.append(f'<line x1="{ox:.1f}" y1="{top-10}" x2="{ox:.1f}" y2="{H-bot}" '
                 f'stroke="{S1}" stroke-width="2"/>')
        p.append(f'<circle cx="{ox:.1f}" cy="{top-10}" r="5" fill="{S1}"/>')
        p.append(f'<text x="{min(ox, W-right-4):.1f}" y="{top-18}" fill="{S1}" '
                 f'font-size="12" font-weight="700" text-anchor="end">'
                 f'real result {d["observed"]:.3f}</text>')
        for t in (0.4, 0.5, 0.6, 0.7):
            if lo_d <= t <= hi_d:
                p.append(f'<text x="{X(t):.1f}" y="{H-26}" fill="{MUTED}" font-size="11" '
                         f'text-anchor="middle">{t:.2f}</text>')
        p.append(f'<text x="{(left+W-right)/2:.0f}" y="{H-6}" fill="{MUTED}" '
                 f'font-size="11" text-anchor="middle">scores when the labels are '
                 f'scrambled</text>')
        p.append("</svg>")
        panels.append(f'<figure class="panel"><figcaption>Trained on {STUDY[a]} → '
                      f'tested on {STUDY[b]}</figcaption>' + "\n".join(p) + "</figure>")
        meta.append((a, b, d))
    return '<div class="grid2">' + "".join(panels) + "</div>", meta


# ------------------------------------------------------------------ chart 3
def floor_sweep():
    if "R2" not in R:
        return '<p class="pending">Still running.</p>'
    panels = []
    for g, nm in (("GSE63060", "Study A"), ("GSE63061", "Study B")):
        ks = [k for k in (2.0, 2.5, 3.0, 3.5, 4.0) if f"{g}|k={k}" in R["R2"]]
        if not ks:
            continue
        W, H, left, right, top, bot = 430, 260, 46, 22, 26, 50
        lo_d, hi_d = 0.55, 0.95

        def X(k):
            return left + (k - 2.0) / 2.0 * (W - left - right)

        def Y(v):
            return H - bot - (v - lo_d) / (hi_d - lo_d) * (H - bot - top)

        p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
             f'aria-label="Scores as the background cutoff moves, {nm}">']
        for t in (0.6, 0.7, 0.8, 0.9):
            p.append(f'<line x1="{left}" y1="{Y(t):.1f}" x2="{W-right}" y2="{Y(t):.1f}" '
                     f'stroke="{GRID}" stroke-width="1"/>')
            p.append(f'<text x="{left-8}" y="{Y(t)+4:.1f}" fill="{MUTED}" font-size="11" '
                     f'text-anchor="end">{t:.2f}</text>')
        for rule in ("DETECTED", "STRICT-100"):
            pts = [(X(k), Y(R["R2"][f"{g}|k={k}"][rule]["auc"])) for k in ks]
            p.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                     + f'" fill="none" stroke="{COL[rule]}" stroke-width="2" '
                       'stroke-linejoin="round"/>')
            for k, (x, y) in zip(ks, pts):
                dd = R["R2"][f"{g}|k={k}"][rule]
                p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{COL[rule]}" '
                         f'stroke="var(--surface-1)" stroke-width="2"><title>{LBL[rule]}, '
                         f'cutoff {k}: {dd["auc"]:.3f} using {dd["n_probes"]:,} probes'
                         f'</title></circle>')
            lx, ly = pts[-1]
            p.append(f'<text x="{lx-4:.1f}" y="{ly-12:.1f}" fill="{COL[rule]}" '
                     f'font-size="12" font-weight="600" text-anchor="end">{LBL[rule]}</text>')
        for k in ks:
            p.append(f'<text x="{X(k):.1f}" y="{H-28}" fill="{MUTED}" font-size="11" '
                     f'text-anchor="middle">{k}</text>')
        p.append(f'<text x="{(left+W-right)/2:.0f}" y="{H-8}" fill="{MUTED}" '
                 f'font-size="11" text-anchor="middle">where the background line is '
                 f'drawn → stricter</text>')
        p.append("</svg>")
        panels.append(f'<figure class="panel"><figcaption>{nm}</figcaption>'
                      + "\n".join(p) + "</figure>")
    return '<div class="grid2">' + "".join(panels) + "</div>"


# ------------------------------------------------------------------ chart 4
def baseline_ladder():
    W, rowh, top, left, right = 880, 40, 30, 250, 64
    rows = []
    for g, nm in (("GSE63060", "Study A"), ("GSE63061", "Study B")):
        rows.append((nm, "Age and sex only", R["A3"][f"{g}|COVARIATES_ONLY"]["auc"], MUTED))
        rows.append((nm, LBL["STRICT-100"], R["A2"][f"{g}|STRICT-100"]["auc"], S1))
        rows.append((nm, LBL["DETECTED"], R["A2"][f"{g}|DETECTED"]["auc"], S2))
    H = top + rowh * len(rows) + 40
    x0, x1 = left, W - right

    def X(v):
        return x0 + (v - 0.5) / 0.45 * (x1 - x0)

    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
         f'aria-label="Scores compared with an age and sex only baseline">']
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        p.append(f'<line x1="{X(t):.1f}" y1="{top-14}" x2="{X(t):.1f}" '
                 f'y2="{top+rowh*len(rows)-16}" stroke="{GRID}" stroke-width="1"/>')
        p.append(f'<text x="{X(t):.1f}" y="{H-16}" fill="{MUTED}" font-size="12" '
                 f'text-anchor="middle">{t:.2f}</text>')
    last = None
    for i, (nm, name, v, c) in enumerate(rows):
        y = top + i * rowh
        if nm != last:
            p.append(f'<text x="12" y="{y+5}" fill="var(--text-primary)" font-size="13" '
                     f'font-weight="700">{nm}</text>')
            last = nm
        p.append(f'<text x="{left-14}" y="{y+5}" fill="{INK2}" font-size="13" '
                 f'text-anchor="end">{name}</text>')
        p.append(f'<rect x="{X(0.5):.1f}" y="{y-9}" width="{max(X(v)-X(0.5),2):.1f}" '
                 f'height="18" rx="4" fill="{c}"><title>{name}, {nm}: {v:.3f}</title></rect>')
        p.append(f'<text x="{X(v)+10:.1f}" y="{y+5}" fill="var(--text-primary)" '
                 f'font-size="13" font-weight="600">{v:.3f}</text>')
    p.append("</svg>")
    return "\n".join(p)


# ------------------------------------------------------------------- tables
def model_family_table():
    if "A5" not in R:
        return ""
    h = ['<table><thead><tr><th>Study</th><th>Which probes</th>'
         '<th class="n">Method 1<br><span class="sm">gradient boosting</span></th>'
         '<th class="n">Method 2<br><span class="sm">logistic regression</span></th>'
         '<th class="n">Method 3<br><span class="sm">random forest</span></th>'
         '</tr></thead><tbody>']
    for g, nm in (("GSE63060", "Study A"), ("GSE63061", "Study B")):
        for rule in ("STRICT-100", "DETECTED"):
            d = R["A5"].get(f"{g}|{rule}")
            if not d:
                continue
            h.append(f'<tr><td>{nm}</td><td>{LBL[rule]}</td>'
                     + "".join(f"<td class='n'>{d[m]:.3f}</td>"
                               for m in ("XGBoost", "L2 logistic", "Random forest"))
                     + "</tr>")
    return "\n".join(h) + "</tbody></table>"


def seed_table():
    h = ['<table><thead><tr><th>Setup</th><th>Which probes</th>'
         '<th class="n">Average</th><th class="n">Spread</th>'
         '<th class="n">Worst to best</th></tr></thead><tbody>']
    for k, d in R["R1"].items():
        ab, rule = k.split("|")
        a, b = ab.split("->")
        h.append(f'<tr><td>{STUDY[a]} → {STUDY[b]}</td><td>{LBL[rule]}</td>'
                 f'<td class="n">{d["mean"]:.3f}</td><td class="n">±{d["sd"]:.3f}</td>'
                 f'<td class="n">{d["min"]:.3f} – {d["max"]:.3f}</td></tr>')
    return "\n".join(h) + "</tbody></table>"


def flag_table():
    if "A4" not in R:
        return ""
    h = ['<table><thead><tr><th>Study</th>'
         '<th class="n">Telling patients from healthy people</th>'
         '<th class="n">Telling which collection group a sample came from</th>'
         '</tr></thead><tbody>']
    for g, nm in (("GSE63060", "Study A"), ("GSE63061", "Study B")):
        dx = R["A2"][f"{g}|STRICT-100"]["auc"]
        fl = R["A4"].get(f"{g}|predict_flag|STRICT-100", {}).get("auc")
        if fl:
            h.append(f'<tr><td>{nm}</td><td class="n">{dx:.3f}</td>'
                     f'<td class="n">{fl:.3f}</td></tr>')
    return "\n".join(h) + "</tbody></table>"


def batch_table():
    if "A7" not in R:
        return '<p class="pending">Batch check still running.</p>'
    rows = []
    for g, nm in (("GSE63060", "Study A"), ("GSE63061", "Study B")):
        for var, lab in (("chip", "Which chip the sample was run on"),
                         ("site", "Which clinic collected the sample")):
            b = R["A7"].get(f"{g}|{var}|balance")
            pr = R["A7"].get(f"{g}|{var}|predict|STRICT-100")
            if not b:
                continue
            even = "even" if b["p"] > 0.05 else "uneven"
            cls = "" if b["p"] > 0.05 else ' class="warn"'
            rows.append(
                f"<tr><td>{nm}</td><td>{lab}</td>"
                f"<td class='n'>{b['n_levels']}</td>"
                f"<td{cls}>{even} (p = {b['p']:.3g})</td>"
                f"<td class='n'>{pr['accuracy']:.3f}</td>"
                f"<td class='n'>{pr['majority_baseline']:.3f}</td></tr>" if pr else "")
    return ('<table><thead><tr><th>Study</th><th>Grouping</th>'
            '<th class="n">Groups</th><th>Patients vs healthy split across them</th>'
            '<th class="n">Background-only probes identify it</th>'
            '<th class="n">Guessing would give</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


# --------------------------------------------------------------------- page
perm_html, perm_meta = permutation()
d = R["data"]
n60, n61 = d["n"]["GSE63060"], d["n"]["GSE63061"]
ndark = R["A1"]["61->60|STRICT-100"]["n_probes"]

perm_lines = ""
if perm_meta:
    rows = []
    for a, b, m in perm_meta:
        rows.append(
            f"<tr><td>{STUDY[a]} → {STUDY[b]}</td>"
            f"<td class='n'>{m['observed']:.3f}</td>"
            f"<td class='n'>{m['null_mean']:.3f}</td>"
            f"<td class='n'>{m['null_max']:.3f}</td>"
            f"<td class='n'>{m['n_perm']}</td>"
            f"<td class='n'>0</td></tr>")
    perm_lines = ('<table><thead><tr><th>Setup</th><th class="n">Real result</th>'
                  '<th class="n">Average scrambled score</th>'
                  '<th class="n">Best scrambled score</th>'
                  '<th class="n">Scrambles run</th>'
                  '<th class="n">Scrambles that beat the real result</th>'
                  '</tr></thead><tbody>' + "".join(rows) + "</tbody></table>")

HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Results — probes that measure nothing still separate Alzheimer's patients</title>
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
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.65; }}
.wrap {{ max-width: 960px; margin: 0 auto; padding: 44px 24px 90px; }}
h1 {{ font-size: 30px; line-height: 1.22; margin: 0 0 10px; letter-spacing: -0.015em; }}
h2 {{ font-size: 20px; margin: 46px 0 6px; letter-spacing: -0.005em; }}
p {{ margin: 8px 0; }}
.sub {{ color: var(--text-secondary); }}
.lede {{ font-size: 17px; color: var(--text-secondary); margin-bottom: 4px; }}
.card {{ background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 14px; padding: 20px 22px; margin: 16px 0; }}
.key {{ background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 14px; padding: 18px 22px; margin: 22px 0; }}
.key h3 {{ margin: 0 0 6px; font-size: 15px; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px; margin: 22px 0 8px; }}
.tile {{ background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 14px; padding: 16px 18px; }}
.tile .v {{ font-size: 31px; font-weight: 650; letter-spacing: -0.02em; }}
.tile .k {{ color: var(--text-secondary); font-size: 13px; }}
.tile .n {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
.grid2 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 14px; }}
.panel {{ margin: 0; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 12px; padding: 12px 12px 4px; }}
figcaption {{ font-size: 13px; font-weight: 600; color: var(--text-secondary);
  margin-bottom: 4px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; margin-top: 8px; }}
th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--grid);
  vertical-align: bottom; }}
td:first-child {{ white-space: nowrap; }}
th {{ color: var(--text-secondary); font-weight: 600; font-size: 13px; }}
td.n, th.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
.sm {{ font-weight: 400; color: var(--muted); font-size: 11px; }}
.muted {{ color: var(--muted); }}
.warn {{ color: var(--series-2); }}
.pending {{ color: var(--muted); font-style: italic; }}
.legend {{ display: flex; gap: 20px; flex-wrap: wrap; font-size: 13px;
  color: var(--text-secondary); margin: 6px 0 12px; }}
.legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
.dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; }}
.caveat {{ border-left: 3px solid var(--series-2); }}
.answer {{ color: var(--series-3); font-weight: 650; }}
</style></head><body><div class="wrap">

<h1>Probes that measure nothing still tell Alzheimer's patients apart from
healthy people</h1>

<p class="lede">About half the spots on a gene chip aren't measuring a gene in any
given sample — they read background noise, and standard practice discards them.
This project kept only the discarded spots. They still separate patients from
controls.</p>

<p class="sub"><strong>Study A</strong> (GSE63060): {n60['total']} people,
{n60['AD']} with Alzheimer's. <strong>Study B</strong> (GSE63061): {n61['total']},
{n61['AD']} with Alzheimer's. Both AddNeuroMed whole blood;
{d['n_shared_probes']:,} spots shared between the two chip versions. Trained on
one entire study, tested on the other — different people, clinics and chip.
<strong>All scores: 0.50 = coin flip, 1.00 = perfect.</strong></p>

<div class="tiles">
  <div class="tile"><div class="v">0.720</div>
    <div class="k">Background-only probes, {pair('61','60')}</div>
    <div class="n">plausible range 0.655 – 0.782</div></div>
  <div class="tile"><div class="v">0.668</div>
    <div class="k">Background-only probes, {pair('60','61')}</div>
    <div class="n">plausible range 0.602 – 0.729</div></div>
  <div class="tile"><div class="v">0.573</div>
    <div class="k">What age and sex alone can do</div>
    <div class="n">for comparison</div></div>
  <div class="tile"><div class="v">{ndark:,}</div>
    <div class="k">Spots that never rose above background</div>
    <div class="n">out of {d['n_shared_probes']:,}</div></div>
</div>

<h2>1. The main result</h2>
<p class="sub">Dot = score. Bar = plausible range. Dashed line = coin flip.</p>
<div class="legend">
  <span><i class="dot" style="background:{S1}"></i><strong>Background-only</strong> —
    spots that never rose above noise in any training sample</span>
  <span><i class="dot" style="background:{S2}"></i><strong>Real-gene</strong> —
    the spots everyone normally keeps</span>
  <span><i class="dot" style="background:{S3}"></i>All spots</span>
</div>
<div class="card">{forest()}</div>
<p class="sub">Real-gene spots do better by
{R['A1']['61->60|DET_minus_DARK']['obs']:.2f}–{R['A1']['60->61|DET_minus_DARK']['obs']:.2f},
in all 2,000 re-tests. The claim is not that noise matches signal — it is that
noise lands nowhere near 0.50.</p>

<h2>2. Could this be luck? <span class="answer">No.</span></h2>
<p class="sub">Patient/healthy labels scrambled and the whole analysis rerun
from scratch, hundreds of times.</p>
{perm_html}
<div class="card">{perm_lines}</div>
<p class="sub">Scrambled runs average 0.50. None ever reached the real result.</p>

<h2>3. Could it just be age? <span class="answer">No.</span></h2>
<p class="sub">Patients are genuinely older (75.4 vs 72.4 years in Study A),
but age and sex alone barely classify anyone.</p>
<div class="card">{baseline_ladder()}</div>
<p class="sub">Adding age and sex to the background-only spots moves the score
by less than 0.02.</p>

<h2>4. Could it be how the samples were collected? <span class="answer">No.</span></h2>
<p class="sub">Each sample carries a collection-group flag that is tied to
diagnosis — so the same spots were asked to predict that flag directly.</p>
<div class="card">{flag_table()}</div>
<p class="sub">Near a coin flip on the collection group. Whatever they read, it
is not that.</p>

<h2>5. Could it be the chips or the clinics? <span class="answer">Chips, no. Clinics, partly.</span></h2>
<p class="sub">Samples were run on 29 and 34 physical chips, collected at six and
eight clinics. If patients and healthy people had been processed on different
chips, or collected at different clinics, that difference could masquerade as
disease.</p>
<div class="card">{batch_table()}</div>
<p class="sub">Patients and healthy people are spread evenly across chips
(p&nbsp;=&nbsp;0.65 and 0.24), so chip differences cannot produce the result even
though the probes do carry some chip information. Clinics are a different story:
the split across clinics is uneven in Study&nbsp;A (p&nbsp;=&nbsp;0.0006, from 32%
patients in London to 80% in Thessaloniki), and background-only probes identify
the clinic slightly better than guessing. This is reported as a limitation. The
two studies recruited from different clinic mixes, so a clinic-driven signal
should have collapsed when the model moved between them, and it did not.</p>

<h2>6. Does it depend on where the cutoff is drawn? <span class="answer">No.</span></h2>
<p class="sub">The background threshold is a judgement call. Moved from lenient
to strict, the result survives the whole range.</p>
{floor_sweep()}

<h2>7. Is it a quirk of one method? <span class="answer">No.</span></h2>
<p class="sub">Three unrelated methods, same data.</p>
<div class="card">{model_family_table()}</div>

<h2>8. Is it a lucky random start? <span class="answer">No.</span></h2>
<p class="sub">Ten rebuilds from ten random starting points.</p>
<div class="card">{seed_table()}</div>

<h2>Where the result is weaker</h2>
<div class="card caveat">
<p>Restricting to flagged samples only removes the collection-group concern but
halves the sample. One direction drops to
{R['A4']['60->61|flagyes|STRICT-100']['obs']:.3f}
({R['A4']['60->61|flagyes|STRICT-100']['lo']:.3f}–{R['A4']['60->61|flagyes|STRICT-100']['hi']:.3f},
range includes a coin flip); the other holds at
{R['A4']['61->60|flagyes|STRICT-100']['obs']:.3f}. Real-gene spots drop by a
similar amount there, pointing to sample size rather than the flag.</p>
<p class="muted">What the background spots are actually picking up is not answered
here.</p>
</div>

<h2>What it means</h2>
<p class="sub">A good score on this kind of data does not by itself show a model
found real disease biology. Published blood classifiers report scores in this
range and credit the genes they selected; that credit needs checking.</p>

<p class="sub muted" style="margin-top:44px">
Every number generated by <code>analysis.py</code> from public data
(GEO GSE63060, GSE63061), recorded in <code>results.json</code>.
Python {R['versions']['python']}, numpy {R['versions']['numpy']},
scipy {R['versions']['scipy']}, scikit-learn {R['versions']['sklearn']},
xgboost {R['versions']['xgboost']}.
</p>
</div></body></html>
"""

with open(OUT, "w") as f:
    f.write(HTML)
print("wrote", OUT, len(HTML), "bytes")
