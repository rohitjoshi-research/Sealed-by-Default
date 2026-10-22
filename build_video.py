"""Build the Phase 9 results video (single HTML) from simulator replays and results/summary.json."""
import json, pandas as pd
S = json.load(open("../results/summary.json")); V = json.load(open("video_data.json"))
J = pd.read_csv("../results/sweep_joint_episodes.csv")
j = J[J.condition == "day_clear"].groupby(["person_det_c50_pxm", "design"]).aie_face_s.mean().round(2)
joint = {str(int(k)): {d: float(j[(k, d)]) for d in ["B1_event_raw", "A1_no_failclosed", "P_proposed"]} for k in sorted(J.person_det_c50_pxm.unique())}
ov = {o["design"]: o for o in S["overall"]}
R = dict(
  overall={d: dict(aie=ov[d]["aie"], face=ov[d]["aie_face"], ev=ov[d]["evidence"], coll=ov[d]["collateral"]) for d in ov},
  cond={k: v for k, v in S["by_condition"].items()},
  dens={k: v for k, v in S["by_density"].items()},
  tests=[dict(test=t["test"], diff=t.get("mean_diff", t.get("diff")), ci=t["ci"], p=t["p_holm"]) for t in S["primary_tests"]],
  joint=joint, n=S["n_main_episodes"])
html = open("template.html").read().replace("/*VIDEO_DATA*/null", json.dumps(V, separators=(",", ":"))).replace("/*RESULTS*/null", json.dumps(R, separators=(",", ":")))
open("../results_video.html", "w").write(html)
print("built", len(html) // 1024, "KB")
