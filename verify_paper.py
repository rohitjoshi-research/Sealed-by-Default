"""Compare every number reported in the manuscript (draft v0.1, sim v3) with the values recomputed in results/.
A value matches when the recomputed number, rounded to the same decimals as printed in the paper, equals it.
Usage: python verify_paper.py   (after run_all.sh, or on the shipped results/ files)."""
import json, sys
import pandas as pd

R = "results/"
S = json.load(open(R + "summary.json")); B2 = json.load(open(R + "batch2_summary.json")); B3 = json.load(open(R + "batch3_summary.json"))
ov = {o["design"]: o for o in S["overall"]}
T = {t["test"][:2]: t for t in S["primary_tests"]}
J = pd.read_csv(R + "sweep_joint_episodes.csv")
jm = J[J.condition == "day_clear"].groupby(["person_det_c50_pxm", "design"]).aie_face_s.mean()

# (section, label, paper value as printed, recomputed value)
REF = [
 ("Abstract/VII-A Tab.IV", "B1 raw AIE", "59.5", ov["B1_event_raw"]["aie"][0]),
 ("Tab.IV", "B1 AIE CI low", "49.5", ov["B1_event_raw"]["aie"][1]),
 ("Tab.IV", "B1 AIE CI high", "69.6", ov["B1_event_raw"]["aie"][2]),
 ("Tab.IV", "B1 face exposure", "10.5", ov["B1_event_raw"]["aie_face"][0]),
 ("Tab.IV", "B0 continuous AIE", "178.1", ov["B0_continuous_raw"]["aie"][0]),
 ("Tab.IV", "B2 blur AIE", "0.20", ov["B2_event_blur"]["aie"][0]),
 ("Tab.IV", "B2fc AIE", "0.00", ov["B2fc_event_blur_fc"]["aie"][0]),
 ("Tab.IV", "P AIE", "0.00", ov["P_proposed"]["aie"][0]),
 ("Tab.IV", "P evidence", "0.52", ov["P_proposed"]["evidence"][0]),
 ("Tab.IV", "B2fc evidence", "0.00", ov["B2fc_event_blur_fc"]["evidence"][0]),
 ("Tab.IV", "A1 AIE", "0.20", ov["A1_no_failclosed"]["aie"][0]),
 ("Tab.IV", "A3 collateral", "1.08", ov["A3_single_key"]["collateral"][0]),
 ("Tab.V", "T1 mean diff", "-59.5", T["T1"]["mean_diff"]),
 ("Tab.V", "T1 CI low", "-70.0", T["T1"]["ci"][0]),
 ("Tab.V", "T1 CI high", "-50.4", T["T1"]["ci"][1]),
 ("Tab.V", "T2 evidence diff", "0.52", T["T2"]["diff"]),
 ("Tab.V", "T2 discordant", "418", T["T2"]["n10"]),
 ("Tab.V", "T5 A1-P", "0.20", T["T5"]["mean_diff"]),
 ("Tab.V", "T6 A3-P", "1.08", T["T6"]["mean_diff"]),
 ("VII-A cluster", "T1 cluster CI low", "-64.0", B2["cluster"]["T1_AIE_P_minus_B1"]["ci"][0]),
 ("VII-A cluster", "T1 cluster CI high", "-55.4", B2["cluster"]["T1_AIE_P_minus_B1"]["ci"][1]),
 ("VII-A cluster", "T2 cluster CI low", "0.45", B2["cluster"]["T2_Ev_P_minus_B2fc"]["ci"][0]),
 ("VII-A cluster", "T6 cluster CI high", "1.30", B2["cluster"]["T6_Coll_A3_minus_P"]["ci"][1]),
] + [
 ("VII-B Fig.3", f"evidence {c}", v, S["by_condition"][f"{c}|P_proposed"]["evidence"][0])
 for c, v in [("day_clear", "1.00"), ("day_fog", "0.93"), ("day_rain", "0.67"), ("dusk_clear", "0.67"), ("dusk_fog", "0.62"),
              ("dusk_rain", "0.62"), ("night_clear", "0.13"), ("night_rain", "0.00"), ("night_fog", "0.00")]
] + [
 ("VII-B layout", "plate exposure, facing row", "98.0", S["by_layout"]["opposite_row|B1_event_raw"]["aie_plate"][0]),
 ("VII-C Fig.4", "B1 face, 2/min", "3.4", S["by_density"]["2.0|B1_event_raw"]["aie_face"][0]),
 ("VII-C Fig.4", "B1 face, 6/min", "7.4", S["by_density"]["6.0|B1_event_raw"]["aie_face"][0]),
 ("VII-C Fig.4", "B1 face, 15/min", "20.7", S["by_density"]["15.0|B1_event_raw"]["aie_face"][0]),
 ("VII-C Fig.4", "A3 collateral, 2/min", "0.44", S["by_density"]["2.0|A3_single_key"]["collateral"][0]),
 ("VII-C Fig.4", "A3 collateral, 15/min", "2.09", S["by_density"]["15.0|A3_single_key"]["collateral"][0]),
 ("VII-D Fig.5", "A1 face, c50=320, day", "6.6", S["sweep"]["day_clear|320.0|0.6|A1_no_failclosed"]["aie_face"][0]),
 ("VII-D Fig.5", "raw face, day sweep", "15.5", S["sweep"]["day_clear|320.0|0.6|B1_event_raw"]["aie_face"][0]),
 ("VII-D Tab.VI", "P, person c50=250", "0.05", jm[(250.0, "P_proposed")]),
 ("VII-D Tab.VI", "P, person c50=400", "5.50", jm[(400.0, "P_proposed")]),
 ("VII-D Tab.VI", "B1 raw", "16.4", jm[(400.0, "B1_event_raw")]),
 ("VII-D Tab.VI", "A1", "6.6", jm[(400.0, "A1_no_failclosed")]),
 ("VII-F Tab.VIII", "evidence tau=0.5", "0.59", B2["evidence_threshold"]["0.5"]["P_proposed"]),
 ("VII-F Tab.VIII", "evidence tau=3", "0.31", B2["evidence_threshold"]["3.0"]["P_proposed"]),
 ("VII-F Tab.VIII", "raw AIE clip 20/40", "156.7", B2["trigger"]["20/40/1.0"]["B1_aie"]),
 ("VII-F Tab.VIII", "raw AIE trigger 0.5 m", "7.5", B2["trigger"]["10/20/0.5"]["B1_aie"]),
 ("VII-F Tab.VIII", "evidence trigger 0.5 m", "0.17", B2["trigger"]["10/20/0.5"]["P_evidence"]),
 ("VII-F Tab.VIII", "evidence night q=0.65", "0.67", B2["condition_factor"]["night_q=0.65"]["P_evidence"]),
 ("VII-F Tab.VIII", "raw AIE rain q=0.9", "499.7", B2["condition_factor"]["rain_q=0.9"]["B1_aie"]),
 ("IV/VII-F", "A1 violation rate (%)", "1.2", 100 * B2["A1"]["overall"]),
 ("VII-G Tab.IX", "owner view, whole person", "0.21", B3["owner_view"]["seal_whole_person"][0]),
 ("VII-G Tab.IX", "owner view, night", "0.13", B3["owner_view_by_time"]["night"][0]),
 ("VII-G Tab.IX", "swap collateral p=0.2", "0.03", B3["swap_collateral"]["0.2"][0]),
 ("VII-G Tab.IX", "evidence, 50% frames discarded", "0.41", B3["drop_evidence_discard"]["0.5"][0]),
]

def fmt_like(ref, x):
    d = len(ref.split(".")[1]) if "." in ref else 0
    return f"{x:.{d}f}"

bad = 0
print(f"{'section':16s} {'quantity':34s} {'paper':>8s} {'recomputed':>11s}  status")
for sec, lab, ref, val in REF:
    got = fmt_like(ref, float(val))
    ok = got.lstrip("-") == ref.lstrip("-") and (got.startswith("-") == ref.startswith("-") or float(got) == 0)
    bad += (not ok)
    print(f"{sec:16s} {lab:34s} {ref:>8s} {got:>11s}  {'OK' if ok else 'MISMATCH'}")
print(f"\n{len(REF) - bad}/{len(REF)} paper values reproduced.")
print("Not reproducible from code: Table VII (real-image check needs the author's photographs, which are not released).")
sys.exit(1 if bad else 0)
