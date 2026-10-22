"""Export simulator replays (sim v3) for the Phase 9 video. Every frame shown is simulator output."""
import json, numpy as np, sys
sys.path.insert(0, "..")
import sim

def window(cond, dens, s, span_before, span_after, layout="opposite_row"):
    p = sim.Params(condition=cond, ped_rate_per_min=dens, layout=layout, seed=1000 * s + 7)
    scene, obs, res = sim.run_episode(p)
    rec = sim.event_mask(obs["near"], p)
    a = int(np.where(scene.ped_is_vandal)[0][0])
    pres = np.where(~np.isnan(scene.ped_pos[:, a, 0]))[0]
    k0 = max(0, pres.min() - int(span_before / p.dt)); k1 = min(scene.t.size, pres.min() + int(span_after / p.dt))
    ks = range(k0, k1)
    peds = []
    for j in range(scene.ped_pos.shape[1]):
        if np.all(np.isnan(scene.ped_pos[k0:k1, j, 0])): continue
        x = scene.ped_pos[k0:k1, j, 0]; y = scene.ped_pos[k0:k1, j, 1]
        if np.nanmin(np.abs(y)) > 13 and np.nanmin(x) > 12: continue
        peds.append(dict(v=bool(j == a),
                         p=[None if np.isnan(x[i]) else [round(float(x[i]), 2), round(float(y[i]), 2)] for i in range(len(x))],
                         g=[int(b) for b in obs["face_gt"][k0:k1, j]],
                         d=[int(b) for b in (obs["face_det"][k0:k1, j] | obs["person_det"][k0:k1, j])]))
    vehs = []
    for v in range(scene.veh_pos.shape[1]):
        x = scene.veh_pos[k0:k1, v, 0]; y = scene.veh_pos[k0:k1, v, 1]
        if np.all(np.isnan(x)): continue
        vehs.append([None if np.isnan(x[i]) else [round(float(x[i]), 2), round(float(y[i]), 2)] for i in range(len(x))])
    ev = {r["design"]: r["evidence"] for r in res}
    return dict(cond=cond, dens=dens, seed_index=s, layout=layout, dt=p.dt, n=int(k1 - k0),
                rec=[int(b) for b in rec[k0:k1]], peds=peds, vehs=vehs,
                opposite=layout == "opposite_row",
                evidence=dict(raw=ev["B1_event_raw"], blur=ev["B2fc_event_blur_fc"], prop=ev["P_proposed"]),
                episode_key=f"condition={cond}|layout={layout}|ped_rate_per_min={dens}|seed={s}")

out = {"main": window("day_clear", 6.0, 0, 14, 40)}
dens_cycle = [2.0, 6.0, 15.0]
out["montage"] = [window(c, dens_cycle[i % 3], 0, 4, 22) for i, c in enumerate(sim.CONDITIONS)]
json.dump(out, open("video_data.json", "w"), separators=(",", ":"), default=float)
print("main", out["main"]["n"], "steps, peds", len(out["main"]["peds"]), "evidence", out["main"]["evidence"])
for m in out["montage"]: print(m["cond"], m["dens"], m["n"], len(m["peds"]), m["evidence"])
