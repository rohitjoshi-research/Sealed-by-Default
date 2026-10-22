"""Phase 11 batch 2 sensitivity runs (resumable). Usage: python batch2.py BUDGET_S
Writes results/batch2_rows.csv (one row per experiment variant x episode x design) and results/batch2_a1.csv."""
import csv, os, sys, time, dataclasses as dc, numpy as np, sim
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
OUT, OUT_A1 = "results/batch2_rows.csv", "results/batch2_a1.csv"
done = set()
if os.path.exists(OUT):
    with open(OUT) as fh: done = {r["job"] for r in csv.DictReader(fh)}
def near_mask(scene, dist):
    x = np.abs(scene.ped_pos[:, :, 0]); y = np.abs(scene.ped_pos[:, :, 1])
    dx = np.clip(x - sim.EGO_HALF[0], 0, None); dy = np.clip(y - sim.EGO_HALF[1], 0, None)
    d = np.hypot(dx, dy); return np.nan_to_num(d, nan=1e9) < dist
def write(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new: w.writeheader()
        w.writerows(rows)
jobs = []
for c in sim.CONDITIONS:
    for s in range(15): jobs.append(("base", c, s, None))
for v in (0.35, 0.5, 0.65):
    for s in range(15): jobs.append(("night_q", "night_clear", s, v))
for v in (0.7, 0.8, 0.9):
    for s in range(15): jobs.append(("rain_q", "day_rain", s, v))
TRIG = [(5, 10, 1.0), (10, 20, 1.0), (20, 40, 1.0), (10, 20, 0.5), (10, 20, 1.5)]
t0 = time.time(); n = 0
for exp, cond, s, v in jobs:
    job = f"{exp}|{cond}|{s}|{v}"
    if job in done: continue
    if time.time() - t0 > BUDGET: print("BUDGET STOP", n); break
    tq, wq = dict(sim.TIME_Q), dict(sim.WEATHER_Q)
    if exp == "night_q": sim.TIME_Q["night"] = v
    if exp == "rain_q": sim.WEATHER_Q["rain"] = v
    try:
        p = sim.Params(condition=cond, ped_rate_per_min=6.0, layout="opposite_row", seed=1000 * s + 7)
        rng = np.random.default_rng(p.seed); scene = sim.build_scene(p, rng); obs = sim.observe(scene, p, rng)
    finally:
        sim.TIME_Q.update(tq); sim.WEATHER_Q.update(wq)
    rows = []
    variants = [("trig", f"{a}/{b}/{d}", dc.replace(p, pre_s=a, post_s=b, trigger_dist_m=d)) for a, b, d in TRIG] if exp == "base" else [(exp, str(v), p)]
    for ename, vname, pv in variants:
        o2 = dict(obs); o2["near"] = near_mask(scene, pv.trigger_dist_m)
        for d in ("B1_event_raw", "B2fc_event_blur_fc", "P_proposed", "A3_single_key"):
            r = sim.evaluate(scene, o2, pv, d)
            rows.append(dict(job=job, exp=ename, variant=vname, condition=cond, seed=s, design=d,
                             aie_total_s=r["aie_total_s"], aie_face_s=r["aie_face_s"], evidence=r["evidence"],
                             vandal_usable_s=r["vandal_usable_s"], collateral_s=r["collateral_s"], recorded_frac=r["recorded_frac"]))
    write(OUT, rows)
    if exp == "base": write(OUT_A1, [dict(condition=cond, seed=s, a1_tot=obs["a1_tot"], a1_viol=obs["a1_viol"])])
    n += 1
print("DONE", n, "jobs in", round(time.time() - t0), "s")
