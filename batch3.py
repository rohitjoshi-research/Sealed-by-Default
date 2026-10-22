"""Phase 11 batch 3: owner-view utility (R2), tracking ID swaps (R8), dropped frames (R10). Resumable.
Usage: python batch3.py BUDGET_S  -> results/batch3_rows.csv"""
import csv, os, sys, time, numpy as np, sim
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
OUT = "results/batch3_rows.csv"
done = set()
if os.path.exists(OUT):
    with open(OUT) as fh: done = {r["job"] for r in csv.DictReader(fh)}
SWAP = [0.0, 0.01, 0.05, 0.1, 0.2]       # ID-swap probability per step while another person is within 2 m of the vandal (ASSUMPTION)
DROP = [0.0, 0.05, 0.2, 0.5]             # frame-deadline miss probability per step (ASSUMPTION)
t0 = time.time(); n = 0
for cond in sim.CONDITIONS:
    for s in range(15):
        job = f"{cond}|{s}"
        if job in done: continue
        if time.time() - t0 > BUDGET: print("BUDGET STOP", n); break
        p = sim.Params(condition=cond, ped_rate_per_min=6.0, layout="opposite_row", seed=1000 * s + 7)
        rng = np.random.default_rng(p.seed); scene = sim.build_scene(p, rng); obs = sim.observe(scene, p, rng)
        rec = sim.event_mask(obs["near"], p)
        a = int(np.where(scene.ped_is_vandal)[0][0])
        fg, fd, pd, bv = obs["face_gt"], obs["face_det"], obs["person_det"], obs["body_vis"]
        present = ~np.isnan(scene.ped_pos[:, a, 0])
        inc = present & rec & bv[:, a]                       # incident frames the raw owner view shows
        whole = pd[:, a] & ~fd[:, a]                         # fail-closed seals the whole person
        rows = []
        base = dict(job=job, condition=cond, seed=s, n_incident_frames=int(inc.sum()))
        if inc.sum():
            rows.append(dict(base, metric="owner_view", variant="raw", value=1.0))
            rows.append(dict(base, metric="owner_view", variant="seal_whole_person", value=float((inc & ~whole).sum() / inc.sum())))
            rows.append(dict(base, metric="owner_view", variant="seal_head_only", value=1.0))
        # tracking ID swaps (Monte Carlo, fixed seed per episode)
        mrng = np.random.default_rng(p.seed + 99)
        win = np.where(present)[0]; w0, w1 = win.min(), win.max() + 1
        others = [j for j in range(fg.shape[1]) if j != a]
        for sw in SWAP:
            coll = []; ev = []
            for rep in range(20):
                swapped = np.zeros(fg.shape[1], bool); lost_from = None; c = 0.0
                for k in range(w0, w1):
                    for j in others:
                        if np.isnan(scene.ped_pos[k, j, 0]): continue
                        if not swapped[j] and np.hypot(*(scene.ped_pos[k, j, :2] - scene.ped_pos[k, a, :2])) < 2.0 and mrng.random() < sw:
                            swapped[j] = True
                            if lost_from is None: lost_from = k
                        if swapped[j] and rec[k] and fg[k, j]: c += p.dt
                usable = fg[w0:w1, a] & rec[w0:w1]
                if lost_from is not None: usable[lost_from - w0:] = False
                coll.append(c); ev.append(float(usable.sum() * p.dt >= p.evidence_min_s))
            rows.append(dict(base, metric="swap_collateral", variant=sw, value=float(np.mean(coll))))
            rows.append(dict(base, metric="swap_evidence", variant=sw, value=float(np.mean(ev))))
        # dropped frames: seal-whole-frame keeps evidence; dropping loses it
        drng = np.random.default_rng(p.seed + 7)
        for dr in DROP:
            evk = []; evd = []; ovl = []
            for rep in range(20):
                dropped = drng.random(fg.shape[0]) < dr
                u = fg[:, a] & rec
                evk.append(float(u.sum() * p.dt >= p.evidence_min_s))
                evd.append(float((u & ~dropped).sum() * p.dt >= p.evidence_min_s))
                ovl.append(float((inc & ~whole & ~dropped).sum() / inc.sum()) if inc.sum() else np.nan)
            rows.append(dict(base, metric="drop_evidence_sealframe", variant=dr, value=float(np.mean(evk))))
            rows.append(dict(base, metric="drop_evidence_discard", variant=dr, value=float(np.mean(evd))))
            rows.append(dict(base, metric="drop_owner_view", variant=dr, value=float(np.nanmean(ovl)) if inc.sum() else np.nan))
        new = not os.path.exists(OUT)
        with open(OUT, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if new: wr.writeheader()
            wr.writerows(rows)
        n += 1
print("DONE", n, "in", round(time.time() - t0), "s")
