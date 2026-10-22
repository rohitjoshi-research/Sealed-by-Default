"""Phase 7 analysis: bootstrap CIs, paired tests with Holm correction, effect sizes, figures.
Usage: python analyze.py   (reads results/main_episodes.csv and results/sweep_episodes.csv)"""
import json, numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(12345)
B = 2000
main = pd.read_csv("results/main_episodes.csv")
sweep = pd.read_csv("results/sweep_episodes.csv")
sweep_p = pd.read_csv("results/sweep_person_episodes.csv")
for df in (main, sweep, sweep_p):
    df["aie_total_s"] = df["aie_face_s"] + df["aie_plate_s"]

def boot_mean(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(axis=1)
    return (x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5))

def wide(df, metric):
    return df.pivot_table(index="episode_key", columns="design", values=metric)

def wilcoxon_paired(a, b):
    d = np.asarray(a) - np.asarray(b); d = d[~np.isnan(d)]
    nz = d[d != 0]
    if len(nz) == 0:
        return dict(n=len(d), n_nonzero=0, p=1.0, r_rb=0.0, mean_diff=0.0, ci=(0.0, 0.0), note="all paired differences are zero")
    res = stats.wilcoxon(nz)
    ranks = stats.rankdata(np.abs(nz))
    wpos = ranks[nz > 0].sum(); wneg = ranks[nz < 0].sum()
    r_rb = (wpos - wneg) / (wpos + wneg)
    idx = RNG.integers(0, len(d), (B, len(d)))
    md = d[idx].mean(axis=1)
    return dict(n=len(d), n_nonzero=len(nz), p=float(res.pvalue), r_rb=float(r_rb), mean_diff=float(d.mean()),
                ci=(float(np.percentile(md, 2.5)), float(np.percentile(md, 97.5))), note="")

def mcnemar(a, b):
    a = np.asarray(a); b = np.asarray(b); m = ~(np.isnan(a) | np.isnan(b)); a = a[m]; b = b[m]
    n10 = int(((a == 1) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    p = 1.0 if n10 + n01 == 0 else stats.binomtest(n10, n10 + n01, 0.5).pvalue
    d = a - b; idx = RNG.integers(0, len(d), (B, len(d))); md = d[idx].mean(axis=1)
    return dict(n=len(d), n10=n10, n01=n01, p=float(p), diff=float(d.mean()),
                ci=(float(np.percentile(md, 2.5)), float(np.percentile(md, 97.5))))

def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); running = 0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i])); adj[i] = running
    return adj

out = {}
# ---------------- primary tests (all main episodes) ----------------
A = wide(main, "aie_total_s"); E = wide(main, "evidence"); C = wide(main, "collateral_s")
tests = [
    ("T1 AIE: proposed vs event-raw (B1)", "wilcoxon", A["P_proposed"], A["B1_event_raw"]),
    ("T2 Evidence: proposed vs fail-closed blur (B2fc)", "mcnemar", E["P_proposed"], E["B2fc_event_blur_fc"]),
    ("T3 AIE: proposed vs fail-closed blur (B2fc)", "wilcoxon", A["P_proposed"], A["B2fc_event_blur_fc"]),
    ("T4 Evidence: proposed vs event-raw (B1)", "mcnemar", E["P_proposed"], E["B1_event_raw"]),
    ("T5 AIE: ablation no fail-closed (A1) vs proposed", "wilcoxon", A["A1_no_failclosed"], A["P_proposed"]),
    ("T6 Collateral: single key (A3) vs proposed", "wilcoxon", C["A3_single_key"], C["P_proposed"]),
]
rows = []
for name, kind, a, b in tests:
    r = wilcoxon_paired(a, b) if kind == "wilcoxon" else mcnemar(a, b)
    r.update(test=name, kind=kind); rows.append(r)
adj = holm(np.array([r["p"] for r in rows]))
for r, pa in zip(rows, adj): r["p_holm"] = float(pa)
out["primary_tests"] = rows

# ---------------- means with CIs by design ----------------
designs = ["B0_continuous_raw", "B1_event_raw", "B2_event_blur", "B2fc_event_blur_fc", "P_proposed",
           "A1_no_failclosed", "A2_continuous_sealed", "A3_single_key"]
overall = []
for d in designs:
    sub = main[main.design == d]
    overall.append(dict(design=d, aie=boot_mean(sub.aie_total_s), aie_face=boot_mean(sub.aie_face_s),
                        aie_plate=boot_mean(sub.aie_plate_s), evidence=boot_mean(sub.evidence),
                        collateral=boot_mean(sub.collateral_s), recorded=float(sub.recorded_frac.mean())))
out["overall"] = overall

# by condition and by density (selected designs)
sel = ["B1_event_raw", "B2fc_event_blur_fc", "P_proposed", "A1_no_failclosed"]
bycond = {}
for c in main.condition.unique():
    for d in sel:
        sub = main[(main.condition == c) & (main.design == d)]
        bycond[f"{c}|{d}"] = dict(aie=boot_mean(sub.aie_total_s), evidence=boot_mean(sub.evidence))
out["by_condition"] = bycond
bydens = {}
for dens in sorted(main.ped_rate_per_min.unique()):
    for d in sel + ["A3_single_key"]:
        sub = main[(main.ped_rate_per_min == dens) & (main.design == d)]
        bydens[f"{dens}|{d}"] = dict(aie_face=boot_mean(sub.aie_face_s), aie=boot_mean(sub.aie_total_s),
                                    evidence=boot_mean(sub.evidence), collateral=boot_mean(sub.collateral_s))
out["by_density"] = bydens
bylay = {}
for lay in main.layout.unique():
    for d in sel:
        sub = main[(main.layout == lay) & (main.design == d)]
        bylay[f"{lay}|{d}"] = dict(aie_face=boot_mean(sub.aie_face_s), aie_plate=boot_mean(sub.aie_plate_s))
out["by_layout"] = bylay

# ---------------- sweep ----------------
sw = {}
for (c, f, ts), g in sweep.groupby(["condition", "face_det_c50_pxm", "track_sigma"]):
    for d in ["B1_event_raw", "B2fc_event_blur_fc", "P_proposed", "A1_no_failclosed"]:
        sub = g[g.design == d]
        sw[f"{c}|{f}|{ts}|{d}"] = dict(aie_face=boot_mean(sub.aie_face_s), aie=boot_mean(sub.aie_total_s),
                                      evidence=boot_mean(sub.evidence))
out["sweep"] = sw
swp = {}
for (c, f), g in sweep_p.groupby(["condition", "person_det_c50_pxm"]):
    for d in ["B1_event_raw", "B2fc_event_blur_fc", "P_proposed", "A1_no_failclosed"]:
        sub = g[g.design == d]
        swp[f"{c}|{f}|{d}"] = dict(aie=boot_mean(sub.aie_total_s), aie_face=boot_mean(sub.aie_face_s), evidence=boot_mean(sub.evidence))
out["sweep_person"] = swp
# v1-flaw check: evidence must vary within conditions
ev_var = {}
for c, g in main[main.design == "P_proposed"].groupby("condition"):
    ev_var[c] = dict(mean=float(g.evidence.mean()), n=int(len(g)), n_pos=int(g.evidence.sum()))
out["evidence_within_condition"] = ev_var
out["n_sweep_person_episodes"] = int(sweep_p.episode_key.nunique())
out["n_main_episodes"] = int(main.episode_key.nunique()); out["n_sweep_episodes"] = int(sweep.episode_key.nunique())
json.dump(out, open("results/summary.json", "w"), indent=1, default=float)

# ---------------- figures ----------------
conds = [f"{t}_{w}" for t in ("day", "dusk", "night") for w in ("clear", "rain", "fog")]
labels = {"B1_event_raw": "Event raw (B1)", "B2fc_event_blur_fc": "Blur, fail-closed (B2fc)",
          "P_proposed": "Proposed", "A1_no_failclosed": "Proposed w/o fail-closed (A1)"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, metric, title in [(axes[0], "evidence", "Evidence utility (share of incidents)"),
                          (axes[1], "aie", "Accessible identifiable exposure (s / 10 min)")]:
    M = np.array([[bycond[f"{c}|{d}"][metric][0] for d in sel] for c in conds])
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(sel))); ax.set_xticklabels([labels[d] for d in sel], rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(conds))); ax.set_yticklabels([c.replace("_", ", ") for c in conds], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.2f}" if metric == "evidence" else f"{M[i,j]:.0f}", ha="center", va="center",
                    fontsize=7, color="white" if M[i, j] > M.max() * 0.6 else "black")
    ax.set_title(title, fontsize=10)
fig.suptitle("Simulated, geometric ground truth; detector and condition factors are assumptions", fontsize=8, y=0.02)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig("results/fig_condition_heatmap.png", dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
dens = sorted(main.ped_rate_per_min.unique())
for d in sel:
    y = [bydens[f"{x}|{d}"]["aie_face"] for x in dens]
    ax[0].errorbar(dens, [v[0] for v in y], yerr=[[v[0]-v[1] for v in y], [v[2]-v[0] for v in y]], marker="o", capsize=3, label=labels[d])
    y = [bydens[f"{x}|{d}"]["evidence"] for x in dens]
    ax[1].errorbar(dens, [v[0] for v in y], yerr=[[v[0]-v[1] for v in y], [v[2]-v[0] for v in y]], marker="o", capsize=3, label=labels[d])
ax[0].set_xlabel("Pedestrians per minute"); ax[0].set_ylabel("Face AIE (s / 10 min)")
ax[1].set_xlabel("Pedestrians per minute"); ax[1].set_ylabel("Evidence utility")
ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig("results/fig_density.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, c in zip(axes, ["day_clear", "night_clear"]):
    for d, ls in [("B1_event_raw", "-"), ("B2fc_event_blur_fc", "--"), ("P_proposed", "-"), ("A1_no_failclosed", ":")]:
        for ts, mk in [(0.6, "o"), (1.5, "s")]:
            xs = sorted(sweep.face_det_c50_pxm.unique())
            ys = [sw[f"{c}|{x}|{ts}|{d}"]["aie_face"][0] for x in xs]
            ax.plot(xs, ys, ls, marker=mk, label=f"{labels[d]}, σ={ts}")
    ax.axvline(250, color="grey", lw=0.8); ax.text(252, ax.get_ylim()[1]*0.9 if ax.get_ylim()[1] > 0 else 1, "ID threshold", fontsize=7)
    ax.set_title(c.replace("_", ", "), fontsize=10); ax.set_xlabel("Detector 50 % point (px/m)")
axes[0].set_ylabel("Face AIE (s / 10 min)"); axes[1].legend(fontsize=6)
fig.tight_layout(); fig.savefig("results/fig_detector_sweep.png", dpi=150)
print("analysis written: results/summary.json + 3 figures")
