"""Joint stress: weak face/plate detector (c50=320 px/m) AND weakening person/vehicle detector."""
import csv, numpy as np, sim
rows = []
for c in ("day_clear", "night_clear"):
    for f in (20.0, 120.0, 250.0, 400.0):
        for s in range(10):
            p = sim.Params(condition=c, face_det_c50_pxm=320.0, plate_det_c50_pxm=320.0 * 90 / 110,
                           person_det_c50_pxm=f, vehicle_det_c50_pxm=f / 2, seed=1000 * s + 7)
            _, _, res = sim.run_episode(p)
            for r in res:
                if r["design"] in ("B1_event_raw", "P_proposed", "A1_no_failclosed", "B2fc_event_blur_fc"):
                    rows.append(dict(condition=c, person_det_c50_pxm=f, seed=s, **r))
with open("results/sweep_joint_episodes.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
import pandas as pd
d = pd.DataFrame(rows)
print(d.groupby(["condition", "person_det_c50_pxm", "design"])[["aie_face_s", "aie_total_s", "evidence"]].mean().round(2).to_string())
