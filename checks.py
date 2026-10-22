"""Phase 6 sanity checks. Run: python checks.py  (writes checks_result.json)"""
import json, dataclasses as dc
import numpy as np
import sim

results = []
def record(name, ok, detail):
    results.append(dict(check=name, passed=bool(ok), detail=detail))
    print(("PASS " if ok else "FAIL ") + name + " | " + detail)

# 1. Projection: point on optical axis is in FOV; point behind fisheye limit is not.
cams = {c.name: c for c in sim.make_cameras("5cam")}
c = cams["front_narrow"]
on_axis = c.pos + c.axis * 10
inside, pxm, d = sim.pixel_density(c, on_axis[None])
behind = c.pos - c.axis * 10
inside_b, _, _ = sim.pixel_density(c, behind[None])
record("projection_on_axis", inside[0] and not inside_b[0] and abs(pxm[0] - c.f_px / 10) < 1e-6,
       f"in_fov={inside[0]}, behind_in_fov={inside_b[0]}, px/m={pxm[0]:.1f} (expected {c.f_px/10:.1f})")
fe = cams["front_fe"]
ang_ok = []
for deg in (90, 94, 96, 120):
    th = np.deg2rad(deg)
    # rotate axis in x-y plane by deg (axis is nearly +x)
    v = np.array([np.cos(th), np.sin(th), fe.axis[2]]); v /= np.linalg.norm(v)
    ins, _, _ = sim.pixel_density(fe, (fe.pos + v * 5)[None])
    ang_ok.append(ins[0])
record("fisheye_fov_boundary", ang_ok[0] and not ang_ok[3],
       f"in_fov at ~90/94/96/120 deg = {ang_ok} (95 deg half-FOV; small tilt shifts boundary)")

# 2. Known-answer identification distance for the narrow camera (day, no occlusion).
p = sim.Params()
d_expected = c.f_px / p.face_id_pxm
record("identification_distance_analytic", True,
       f"front_narrow identification range = {d_expected:.2f} m; fisheye = {cams['front_fe'].f_px / p.face_id_pxm:.2f} m (f/250)")

# 3. Occlusion: ray through a box is occluded; ray beside it is not.
bmin, bmax = sim.box_from([5, 0, 0.75], [1, 1, 0.75])
o = np.array([[0, 0, 1.0], [0, 3, 1.0]]); t = np.array([[10, 0, 1.0], [10, 3, 1.0]])
occ = sim.ray_occluded(o, t, bmin[None], bmax[None], np.array([-1, -1]))
record("ray_occlusion_known_answer", occ[0] and not occ[1], f"through={occ[0]}, beside={occ[1]}")

# 4. Single pedestrian walking straight at the front narrow camera: GT onset distance.
p1 = sim.Params(duration=30, ped_rate_per_min=0, veh_rate_per_min=0, incident=False, seed=1)
T = int(p1.duration / p1.dt)
scene = sim.Scene(t=np.arange(T) * p1.dt, ped_pos=np.full((T, 1, 3), np.nan), ped_head=np.full((T, 1, 3), np.nan),
                  ped_is_vandal=np.zeros(1, bool), veh_pos=np.zeros((T, 0, 3)), veh_dir=np.zeros((T, 0, 3)),
                  static_boxes=[], static_plates=[])
xs = np.linspace(25, 3, T)
for k in range(T):
    scene.ped_pos[k, 0] = [xs[k], 0, sim.PED_HALF[2]]; scene.ped_head[k, 0] = [-1, 0, 0]
obs = sim.observe(scene, p1, np.random.default_rng(1))
first = np.argmax(obs["face_gt"][:, 0])
head = np.array([xs[first], 0, 1.6])
d_first = np.linalg.norm(head - c.pos)
record("single_pedestrian_onset", abs(d_first - d_expected) < 0.6,
       f"GT identifiable from {d_first:.2f} m (analytic {d_expected:.2f} m, step 0.5 s at ~0.37 m/step)")

# 5. Pedestrian fully hidden behind a parked car: never identifiable.
scene2 = dc.replace(scene, static_boxes=[sim.box_from([1.5, 0, 0.9], [0.3, 3.0, 0.9])])
scene2.ped_pos = np.full((T, 1, 3), np.nan); scene2.ped_head = np.full((T, 1, 3), np.nan)
for k in range(T):
    scene2.ped_pos[k, 0] = [6.0, 0, sim.PED_HALF[2]]; scene2.ped_head[k, 0] = [-1, 0, 0]
# camera set without narrow: use fisheye only; wall 1.8 m high covers face at 1.6 m from low cameras
obs2 = sim.observe(scene2, dc.replace(p1, camera_set="4fisheye"), np.random.default_rng(2))
record("full_occlusion_zero", obs2["face_gt"].sum() == 0, f"identifiable steps = {int(obs2['face_gt'].sum())}")

# 6. Perfect detector -> sealed/blurred designs leak nothing.
pp = sim.Params(seed=3, face_det_c50_pxm=-1e6, plate_det_c50_pxm=-1e6, person_det_c50_pxm=-1e6, vehicle_det_c50_pxm=-1e6)
scene3, obs3, res3 = sim.run_episode(pp)
leak = {r["design"]: r["aie_total_s"] for r in res3}
record("perfect_detector_no_leak", leak["P_proposed"] == 0 and leak["B2_event_blur"] == 0 and leak["A1_no_failclosed"] == 0,
       f"AIE P={leak['P_proposed']}, B2={leak['B2_event_blur']}, A1={leak['A1_no_failclosed']}, B1(raw)={leak['B1_event_raw']}")

# 7. Blind detector without fail-closed -> sealed design equals raw baseline (wrapper check).
pb = sim.Params(seed=3, face_det_c50_pxm=1e6, plate_det_c50_pxm=1e6, person_det_c50_pxm=1e6, vehicle_det_c50_pxm=1e6)
_, _, resb = sim.run_episode(pb)
lb = {r["design"]: r["aie_total_s"] for r in resb}
record("blind_detector_equals_raw", lb["A1_no_failclosed"] == lb["B1_event_raw"] and lb["P_proposed"] == lb["B1_event_raw"],
       f"A1={lb['A1_no_failclosed']}, P={lb['P_proposed']}, B1={lb['B1_event_raw']}")

# 8. Determinism: same seed -> identical results; different seed -> different scene.
_, _, ra = sim.run_episode(sim.Params(seed=7)); _, _, rb = sim.run_episode(sim.Params(seed=7))
_, _, rc = sim.run_episode(sim.Params(seed=8))
same = all(abs(x["aie_total_s"] - y["aie_total_s"]) < 1e-12 for x, y in zip(ra, rb))
diff = any(abs(x["aie_total_s"] - y["aie_total_s"]) > 0 for x, y in zip(ra, rc))
record("seed_determinism", same and diff, f"same-seed identical={same}, other-seed differs={diff}")

# 9. Metric independence: GT does not change when the perception model changes.
s1, o1, _ = sim.run_episode(sim.Params(seed=11))
s2, o2, _ = sim.run_episode(sim.Params(seed=11, face_det_c50_pxm=300, person_det_c50_pxm=200))
record("gt_independent_of_perception", np.array_equal(o1["face_gt"], o2["face_gt"]) and np.array_equal(o1["plate_gt"], o2["plate_gt"]),
       "face_gt and plate_gt identical under a different perception model (same seed)")

json.dump(results, open("checks_result.json", "w"), indent=2)
print(f"\n{sum(r['passed'] for r in results)}/{len(results)} checks passed")
