"""Summarise batch 3 (Table VIII). Bootstrap seed 7, 2,000 resamples."""
import pandas as pd, numpy as np, json
d = pd.read_csv("results/batch3_rows.csv"); rng = np.random.default_rng(7)
def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]; bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return [round(float(x.mean()), 3), round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3), len(x)]
ov = d[d.metric == "owner_view"]
out = {"owner_view": {v: ci(g.value) for v, g in ov.groupby("variant")},
       "owner_view_by_time": {t: ci(ov[(ov.variant == "seal_whole_person") & ov.condition.str.startswith(t)].value) for t in ["day", "dusk", "night"]}}
for m in ["swap_collateral", "swap_evidence", "drop_evidence_sealframe", "drop_evidence_discard", "drop_owner_view"]:
    out[m] = {str(v): ci(g.value) for v, g in d[d.metric == m].groupby("variant")}
json.dump(out, open("results/batch3_summary.json", "w"), indent=1); print("written results/batch3_summary.json")
