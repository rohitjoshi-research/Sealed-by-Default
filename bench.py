"""Phase 7 benchmark. Usage: python bench.py main|sweep
Writes one CSV row per (episode, design) to results/<mode>_episodes.csv (appends; resumable)."""
import csv, os, sys, time, itertools, dataclasses as dc
import sim

MODE = sys.argv[1]
BUDGET = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
SEEDS = range(15)
if MODE == "main":
    grid = [dict(condition=c, ped_rate_per_min=d, layout=l, seed=s)
            for c in sim.CONDITIONS for d in (2.0, 6.0, 15.0) for l in ("opposite_row", "no_opposite") for s in SEEDS]
elif MODE == "sweep":
    grid = [dict(condition=c, face_det_c50_pxm=f, plate_det_c50_pxm=f * 90 / 110, track_sigma=ts, seed=s)
            for c in ("day_clear", "night_clear") for f in (110.0, 180.0, 250.0, 320.0) for ts in (0.6, 1.5) for s in SEEDS]
elif MODE == "sweep_person":
    # Stress the fail-closed fallback: weaker person/vehicle detection
    grid = [dict(condition=c, person_det_c50_pxm=f, vehicle_det_c50_pxm=f / 2, seed=s)
            for c in ("day_clear", "night_clear") for f in (20.0, 120.0, 250.0, 400.0) for s in SEEDS]
else:
    raise SystemExit("mode must be main, sweep or sweep_person")

out = f"results/{MODE}_episodes.csv"
done = set()
if os.path.exists(out):
    with open(out) as fh:
        for r in csv.DictReader(fh):
            done.add(r["episode_key"])
fields = None; t0 = time.time(); n_new = 0
for g in grid:
    # seed offset per scenario so scenarios are not copies of each other, but identical across designs
    key = "|".join(f"{k}={v}" for k, v in sorted(g.items()))
    if key in done: continue
    if time.time() - t0 > BUDGET:
        print(f"BUDGET STOP {MODE}: {n_new} new episodes", flush=True); break
    p = sim.Params(**{**g, "seed": 1000 * g["seed"] + 7})
    _, _, res = sim.run_episode(p)
    rows = []
    for r in res:
        row = dict(episode_key=key, **{k: v for k, v in g.items()}, **r)
        rows.append(row)
    new_file = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new_file: w.writeheader()
        w.writerows(rows)
    n_new += 1
    if n_new % 25 == 0:
        print(f"{n_new} new episodes, {time.time()-t0:.0f}s", flush=True)
print(f"DONE {MODE}: {n_new} new episodes in {time.time()-t0:.0f}s", flush=True)
