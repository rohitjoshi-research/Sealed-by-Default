"""Phase 11 batch 2 analysis: cluster bootstrap (R3), evidence-threshold (R7), trigger (R6), condition factors (R16), A1 (R18)."""
import json, numpy as np, pandas as pd
from scipy import stats
rng = np.random.default_rng(2026); B = 4000
m = pd.read_csv("results/main_episodes.csv"); m["seedc"] = m["seed"]
out = {}
def wide(metric): return m.pivot_table(index=["seedc", "episode_key"], columns="design", values=metric)
def cluster_ci(diff_series):
    g = diff_series.groupby(level=0); seeds = np.array(list(g.groups.keys()))
    sums = g.sum().values; cnt = g.count().values
    est = sums.sum() / cnt.sum(); bs = []
    for _ in range(B):
        idx = rng.integers(0, len(seeds), len(seeds)); bs.append(sums[idx].sum() / cnt[idx].sum())
    seed_means = sums / cnt
    w = stats.wilcoxon(seed_means) if np.any(seed_means != 0) else None
    return dict(est=float(est), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                seed_level_p=float(w.pvalue) if w else 1.0, n_clusters=int(len(seeds)))
A, E, C = wide("aie_total_s"), wide("evidence"), wide("collateral_s")
out["cluster"] = {"T1_AIE_P_minus_B1": cluster_ci(A["P_proposed"] - A["B1_event_raw"]),
                  "T2_Ev_P_minus_B2fc": cluster_ci(E["P_proposed"] - E["B2fc_event_blur_fc"]),
                  "T6_Coll_A3_minus_P": cluster_ci(C["A3_single_key"] - C["P_proposed"]),
                  "B1_AIE_mean": cluster_ci(A["B1_event_raw"]), "P_evidence_mean": cluster_ci(E["P_proposed"])}
ps = [out["cluster"][k]["seed_level_p"] for k in ("T1_AIE_P_minus_B1", "T2_Ev_P_minus_B2fc", "T6_Coll_A3_minus_P")]
order = np.argsort(ps); adj = [0]*3; run = 0
for r, i in enumerate(order): run = max(run, min(1, (3 - r) * ps[i])); adj[i] = run
for k, a in zip(("T1_AIE_P_minus_B1", "T2_Ev_P_minus_B2fc", "T6_Coll_A3_minus_P"), adj): out["cluster"][k]["seed_level_p_holm"] = float(a)
b = pd.read_csv("results/batch2_rows.csv")
base = b[(b.exp == "trig") & (b.variant == "10/20/1.0")]
tau = {}
for t in (0.5, 1.0, 2.0, 3.0):
    tau[str(t)] = {d: float((base[base.design == d].vandal_usable_s >= t).mean()) for d in ("B1_event_raw", "B2fc_event_blur_fc", "P_proposed")}
out["evidence_threshold"] = tau
trig = {}
for v, g in b[b.exp == "trig"].groupby("variant"):
    trig[v] = dict(B1_aie=float(g[g.design == "B1_event_raw"].aie_total_s.mean()), P_aie=float(g[g.design == "P_proposed"].aie_total_s.mean()),
                   P_evidence=float(g[g.design == "P_proposed"].evidence.mean()), B2fc_evidence=float(g[g.design == "B2fc_event_blur_fc"].evidence.mean()),
                   A3_collateral=float(g[g.design == "A3_single_key"].collateral_s.mean()), recorded=float(g.recorded_frac.mean()))
out["trigger"] = trig
cf = {}
for (e, v), g in b[b.exp.isin(["night_q", "rain_q"])].groupby(["exp", "variant"]):
    cf[f"{e}={v}"] = dict(B1_aie=float(g[g.design == "B1_event_raw"].aie_total_s.mean()), P_aie=float(g[g.design == "P_proposed"].aie_total_s.mean()),
                          P_evidence=float(g[g.design == "P_proposed"].evidence.mean()))
out["condition_factor"] = cf
a1 = pd.read_csv("results/batch2_a1.csv")
out["A1"] = dict(overall=float(a1.a1_viol.sum() / max(1, a1.a1_tot.sum())), total_face_camera_steps=int(a1.a1_tot.sum()),
                 by_condition={c: float(g.a1_viol.sum() / max(1, g.a1_tot.sum())) for c, g in a1.groupby("condition")})
json.dump(out, open("results/batch2_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
