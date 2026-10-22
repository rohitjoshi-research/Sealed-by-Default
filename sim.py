"""
Privacy-Adaptive Sentry -- Phase 6 simulator (CPU-only, geometric).

What this is
------------
A 2.5D geometric simulator of a parked ego vehicle with surround cameras.
Ground truth (GT) identifiability of every face and number plate is computed
from true geometry: ray-traced occlusion, camera field of view, facing angle
and pixel density, using the IEC 62676-4 DORI "identification" level.

The method under test (and every baseline) only sees the output of a separate,
noisy perception model. Primary metrics are computed on GT, never on the
perception model (principle 4).

What this is NOT
----------------
No images are rendered. Identifiability is a geometric proxy, not the output of
a re-identification network. See VALIDATION.md for assumptions and limits.
"""
from __future__ import annotations
import dataclasses as dc
import numpy as np

# --------------------------------------------------------------------------
# Parameters (every value here is an ASSUMPTION unless marked otherwise)
# --------------------------------------------------------------------------

@dc.dataclass
class Params:
    dt: float = 0.5                 # s per simulation step
    duration: float = 600.0         # s parked episode
    ped_rate_per_min: float = 6.0   # pedestrian arrivals per minute
    veh_rate_per_min: float = 2.0   # passing vehicles per minute
    incident: bool = True           # one vandal incident in the episode
    condition: str = "day_clear"
    camera_set: str = "5cam"        # "4fisheye" or "5cam" (+ front narrow)
    layout: str = "opposite_row"    # "opposite_row" or "no_opposite"
    # GT identifiability thresholds, pixels per metre at the target
    # IEC 62676-4 DORI: identification = 250 px/m (REQUIRES SOURCE VERIFICATION
    # of exact wording; value widely quoted).
    face_id_pxm: float = 250.0
    plate_id_pxm: float = 200.0     # ASSUMPTION: ~15 px character height for 75 mm characters
    face_max_angle_deg: float = 70.0   # face must point within this angle of the camera
    plate_max_angle_deg: float = 60.0
    min_visible_frac: float = 0.5
    # Event trigger (strongest realistic baseline: event-triggered clips)
    trigger_dist_m: float = 1.0
    trigger_hold_s: float = 1.0
    pre_s: float = 10.0
    post_s: float = 20.0
    # Perception model (method's internal model) -- ASSUMPTIONS, swept in Phase 7
    face_det_c50_pxm: float = 110.0   # 50 % face detection at this effective density
    face_det_scale: float = 25.0
    plate_det_c50_pxm: float = 90.0
    plate_det_scale: float = 25.0
    person_det_c50_pxm: float = 20.0
    person_det_scale: float = 8.0
    vehicle_det_c50_pxm: float = 10.0
    vehicle_det_scale: float = 5.0
    track_sigma: float = 0.6          # per-track latent offset (logit units) -> correlated misses
    frame_sigma: float = 0.4          # per-frame noise (logit units)
    occl_penalty: float = 1.5         # logit penalty per unit occluded fraction
    yaw_free_deg: float = 40.0        # detector unaffected up to this face yaw (ASSUMPTION)
    yaw_penalty_per_30deg: float = 2.0  # logit penalty per 30 deg beyond yaw_free (ASSUMPTION)
    evidence_min_s: float = 1.0       # perpetrator identifiable for >= this long
    seed: int = 0

# Image-quality factor by condition, multiplies effective pixel density.
# ASSUMPTIONS (to be replaced by measured MTF / SNR data in Phase 8).
TIME_Q = {"day": 1.00, "dusk": 0.80, "night": 0.50}        # ASSUMPTION
WEATHER_Q = {"clear": 1.00, "rain": 0.80, "fog": 0.60}      # ASSUMPTION
TIME_SHIFT = {"day": 0.0, "dusk": -0.3, "night": -0.8}      # detector logit shift, ASSUMPTION
WEATHER_SHIFT = {"clear": 0.0, "rain": -0.3, "fog": -0.6}
CONDITIONS = [f"{t}_{w}" for t in TIME_Q for w in WEATHER_Q]
# Morning and afternoon are not distinguished: the model has no sun direction or glare.

def condition_factors(cond: str):
    t, w = cond.split("_")
    return TIME_Q[t] * WEATHER_Q[w], TIME_SHIFT[t] + WEATHER_SHIFT[w]

# sim v3: fog is distance-dependent (Koschmieder). Contrast transmission T(d) = exp(-beta d),
# beta = 3.912 / V. ASSUMPTIONS: visibility V = 100 m for 'fog'; effective density scales with T(d);
# detector logit shift = -FOG_SHIFT_K * (1 - T(d)).
FOG_VISIBILITY_M = 100.0
FOG_SHIFT_K = 3.0

def condition_arrays(cond: str, d: np.ndarray):
    """Per-target (q, detector shift) for distances d (m)."""
    t, w = cond.split("_")
    if w == "fog":
        T = np.exp(-3.912 / FOG_VISIBILITY_M * d)
        return TIME_Q[t] * T, TIME_SHIFT[t] - FOG_SHIFT_K * (1 - T)
    return np.full_like(d, TIME_Q[t] * WEATHER_Q[w]), np.full_like(d, TIME_SHIFT[t] + WEATHER_SHIFT[w])

# Backwards-compatible tables (identical values to the Phase 6 smoke run)
CONDITION_Q = {c: condition_factors(c)[0] for c in CONDITIONS}
CONDITION_DET_SHIFT = {c: condition_factors(c)[1] for c in CONDITIONS}

# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------
EGO_HALF = np.array([2.35, 0.95, 0.75])   # half-extents (m); box centre z=0.75
EGO_CENTER = np.array([0.0, 0.0, 0.75])

def box_from(center, half):
    c = np.asarray(center, float); h = np.asarray(half, float)
    return c - h, c + h

@dc.dataclass
class Camera:
    name: str
    pos: np.ndarray
    axis: np.ndarray          # unit optical axis
    model: str                # "fisheye" (equidistant) or "pinhole"
    f_px: float               # focal length in px/rad (fisheye) or px (pinhole)
    half_fov_rad: float       # fisheye: max angle from axis; pinhole: half HFOV
    half_vfov_rad: float = np.pi  # pinhole only
    up: np.ndarray = dc.field(default_factory=lambda: np.array([0, 0, 1.0]))

def _unit(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)

def make_cameras(camera_set: str):
    # Fisheye: 1920 px across 190 deg, equidistant -> f = 960 / (95 deg in rad)
    f_fe = 960.0 / np.deg2rad(95.0)
    tilt = np.deg2rad(30.0)
    cams = [
        Camera("front_fe", np.array([2.35, 0.0, 0.60]), _unit([np.cos(tilt*0.3), 0, -np.sin(tilt*0.3)]), "fisheye", f_fe, np.deg2rad(95)),
        Camera("rear_fe",  np.array([-2.35, 0.0, 0.95]), _unit([-np.cos(tilt*0.3), 0, -np.sin(tilt*0.3)]), "fisheye", f_fe, np.deg2rad(95)),
        Camera("left_fe",  np.array([0.95, 1.00, 1.00]), _unit([0, np.cos(tilt*2), -np.sin(tilt*2)]), "fisheye", f_fe, np.deg2rad(95)),
        Camera("right_fe", np.array([0.95, -1.00, 1.00]), _unit([0, -np.cos(tilt*2), -np.sin(tilt*2)]), "fisheye", f_fe, np.deg2rad(95)),
    ]
    if camera_set == "5cam":
        # Front narrow camera behind windscreen: 1920 px across 60 deg HFOV
        f_n = 960.0 / np.tan(np.deg2rad(30.0))
        cams.append(Camera("front_narrow", np.array([0.6, 0.0, 1.35]), _unit([1, 0, -0.03]), "pinhole",
                           f_n, np.deg2rad(30.0), np.deg2rad(20.0)))
    return cams

def pixel_density(cam: Camera, pts: np.ndarray):
    """Return (in_fov bool, px_per_m) for points (N,3). px/m at the target
    for a small object facing the camera (fronto-parallel approximation)."""
    v = pts - cam.pos
    d = np.linalg.norm(v, axis=1)
    u = v / d[:, None]
    cosang = u @ cam.axis
    ang = np.arccos(np.clip(cosang, -1, 1))
    if cam.model == "fisheye":
        in_fov = ang < cam.half_fov_rad
        pxm = cam.f_px / d           # equidistant: angular density f px/rad
    else:
        right = _unit(np.cross(cam.axis, cam.up)); up = np.cross(right, cam.axis)
        x = u @ right; y = u @ up; z = cosang
        with np.errstate(divide="ignore", invalid="ignore"):
            az = np.arctan2(x, z); el = np.arctan2(y, z)
        in_fov = (z > 0) & (np.abs(az) < cam.half_fov_rad) & (np.abs(el) < cam.half_vfov_rad)
        # pinhole density ~ f / (d * cos(angle)) ; keep conservative f/d
        pxm = cam.f_px / d
    return in_fov, pxm, d

def ray_occluded(origins, targets, bmin, bmax, exclude):
    """Vectorised slab test. origins/targets (R,3); boxes (B,3);
    exclude (R,) index of box to ignore (-1 = none). Returns (R,) bool."""
    R = origins.shape[0]; B = bmin.shape[0]
    if B == 0:
        return np.zeros(R, bool)
    d = targets - origins
    L = np.linalg.norm(d, axis=1)
    dn = d / L[:, None]
    inv = 1.0 / np.where(np.abs(dn) < 1e-12, 1e-12, dn)
    t1 = (bmin[None] - origins[:, None]) * inv[:, None]
    t2 = (bmax[None] - origins[:, None]) * inv[:, None]
    tnear = np.max(np.minimum(t1, t2), axis=2)
    tfar = np.min(np.maximum(t1, t2), axis=2)
    hit = (tnear <= tfar) & (tfar > 1e-3) & (tnear < (L[:, None] - 0.05))
    if exclude is not None:
        idx = np.arange(B)[None, :]
        hit &= idx != exclude[:, None]
    return hit.any(axis=1)

# --------------------------------------------------------------------------
# Scene
# --------------------------------------------------------------------------
PED_HALF = np.array([0.25, 0.25, 0.875])   # body box, 1.75 m tall
VEH_HALF = np.array([2.3, 0.92, 0.75])

@dc.dataclass
class Scene:
    t: np.ndarray
    # per actor arrays over time: positions (T,A,3) NaN when absent
    ped_pos: np.ndarray
    ped_head: np.ndarray        # facing unit vector (T,A,3)
    ped_is_vandal: np.ndarray   # (A,)
    veh_pos: np.ndarray         # (T,V,3) centre
    veh_dir: np.ndarray         # (T,V,3) heading unit
    static_boxes: list          # list of (min,max) occluders (parked neighbours)
    static_plates: list         # list of (centre, normal) for neighbours' plates
    ped_head_z: np.ndarray = None  # (A,) face height above ground (m)

def build_scene(p: Params, rng: np.random.Generator) -> Scene:
    T = int(round(p.duration / p.dt)); t = np.arange(T) * p.dt
    # Parked neighbours left and right (nose-in), occlude and expose plates.
    neigh_centres = [np.array([0.0, 2.6, 0.75]), np.array([0.0, -2.6, 0.75])]
    # Opposite row across the front aisle, parked nose-in towards the ego (plates face -x)
    opp_centres = [np.array([9.5, y, 0.75]) for y in (-5.2, -2.6, 0.0, 2.6, 5.2)] if p.layout == "opposite_row" else []
    static_boxes = [box_from(c, VEH_HALF) for c in neigh_centres + opp_centres]
    static_plates = []
    for c in neigh_centres:
        static_plates.append((c + np.array([2.32, 0, -0.35]), np.array([1.0, 0, 0])))   # front plate
        static_plates.append((c + np.array([-2.32, 0, -0.25]), np.array([-1.0, 0, 0])))  # rear plate
    for c in opp_centres:
        static_plates.append((c + np.array([-2.32, 0, -0.35]), np.array([-1.0, 0, 0])))  # front plate facing ego
        static_plates.append((c + np.array([2.32, 0, -0.25]), np.array([1.0, 0, 0])))    # rear plate (away)
    # Pedestrians: sidewalk in front of the parked row (x ~ 4.5 m), plus
    # some crossing behind (x ~ -4.5 m aisle).
    n_ped = rng.poisson(p.ped_rate_per_min * p.duration / 60.0)
    A = n_ped + (1 if p.incident else 0)
    ped_pos = np.full((T, A, 3), np.nan); ped_head = np.full((T, A, 3), np.nan)
    vandal = np.zeros(A, bool)
    head_z = np.clip(rng.normal(1.62, 0.08, A), 1.45, 1.85)
    for a in range(n_ped):
        t0 = rng.uniform(0, p.duration)
        lane_x = rng.choice([4.5, -4.8]) + rng.normal(0, 0.4)
        direction = rng.choice([-1, 1])
        speed = np.clip(rng.normal(1.35, 0.2), 0.8, 2.0)
        y0 = -15.0 * direction
        # Some walkers pass close to the ego car (cut through between cars)
        close = rng.random() < 0.08
        for k in range(T):
            tau = t[k] - t0
            if tau < 0: continue
            y = y0 + direction * speed * tau
            if abs(y) > 15.5: continue
            x = lane_x
            if close and abs(y) < 3.0:
                x = 2.35 + 0.6 if lane_x > 0 else -2.35 - 0.6
            ped_pos[k, a] = [x, y, PED_HALF[2]]
            ped_head[k, a] = [0, direction, 0]
    if p.incident:
        a = A - 1; vandal[a] = True
        t_in = rng.uniform(120, p.duration - 120)
        side = rng.choice([-1, 1])
        xs = rng.uniform(-1.8, 1.8)                      # where along the car the damage happens
        ys = side * (EGO_HALF[1] + rng.uniform(0.35, 0.9))
        yaw = np.deg2rad(rng.uniform(-60, 60))           # face not always towards the car
        stand = rng.uniform(6, 20)
        come_from = rng.choice([-1, 1])                  # approach from rear (-1) or front (+1) aisle
        leave_to = rng.choice([-1, 1])
        start = np.array([come_from * 6.5, ys, PED_HALF[2]]); spot = np.array([xs, ys, PED_HALF[2]])
        end = np.array([leave_to * 7.0, ys + side * 0.3, PED_HALF[2]])
        face_car = np.array([0, -side, 0.0])
        c, s_ = np.cos(yaw), np.sin(yaw)
        stand_head = np.array([c * face_car[0] - s_ * face_car[1], s_ * face_car[0] + c * face_car[1], 0])
        head_z[a] = rng.uniform(1.5, 1.85)
        for k in range(T):
            tau = t[k] - t_in
            if -10 <= tau < 0:
                frac = (tau + 10) / 10; pos = start + (spot - start) * frac; head = _unit(spot - start)
            elif 0 <= tau < stand:
                pos = spot + np.array([0.08 * np.sin(tau), 0, 0]); head = stand_head
            elif stand <= tau < stand + 12:
                frac = (tau - stand) / 12; pos = spot + (end - spot) * frac; head = _unit(end - spot)
            else:
                continue
            ped_pos[k, a] = pos; ped_head[k, a] = head
    # Passing vehicles in the aisle behind (x ~ -6.5)
    n_veh = rng.poisson(p.veh_rate_per_min * p.duration / 60.0)
    veh_pos = np.full((T, n_veh, 3), np.nan); veh_dir = np.full((T, n_veh, 3), np.nan)
    for v in range(n_veh):
        t0 = rng.uniform(0, p.duration); direction = rng.choice([-1, 1])
        speed = rng.uniform(2.0, 5.0)
        for k in range(T):
            tau = t[k] - t0
            if tau < 0: continue
            y = -25 * direction + direction * speed * tau
            if abs(y) > 25.5: continue
            veh_pos[k, v] = [6.3 if v % 2 == 0 else -7.5, y, 0.75]; veh_dir[k, v] = [0, direction, 0]
    return Scene(t, ped_pos, ped_head, vandal, veh_pos, veh_dir, static_boxes, static_plates, head_z)

# --------------------------------------------------------------------------
# Ground truth + perception per step
# --------------------------------------------------------------------------
FACE_OFFS = np.array([[0, 0, 0], [0, 0.07, 0.06], [0, -0.07, 0.06], [0, 0.07, -0.08], [0, -0.07, -0.08]])

def observe(scene: Scene, p: Params, rng: np.random.Generator):
    """Returns dict of per-step arrays:
    face_gt[T,A]  : face identifiable by >=1 camera (GT)
    face_det[T,A] : face detected by perception in a camera where it is visible
    person_det[T,A]
    plate_gt[T,P], plate_det[T,P], vehicle_det[T,P]  (P = static + moving plates)
    near[T,A]     : distance to ego body < trigger distance
    """
    cams = make_cameras(p.camera_set)
    T, A = scene.ped_pos.shape[:2]
    V = scene.veh_pos.shape[1]
    S = len(scene.static_plates)
    P = S + 2 * V
    face_gt = np.zeros((T, A), bool); face_det = np.zeros((T, A), bool)
    person_det = np.zeros((T, A), bool); near = np.zeros((T, A), bool)
    body_vis = np.zeros((T, A), bool)   # instrumentation: body at least partly visible in some camera
    face_pxm_best = np.zeros((T, A))
    plate_gt = np.zeros((T, P), bool); plate_det = np.zeros((T, P), bool)
    vehicle_det = np.zeros((T, P), bool)
    track_face = rng.normal(0, p.track_sigma, A); track_person = rng.normal(0, p.track_sigma, A)
    track_plate = rng.normal(0, p.track_sigma, P)
    a1_tot = 0; a1_viol = 0   # instrumentation for Proposition 3, assumption A1 (no effect on results)
    cos_face = np.cos(np.deg2rad(p.face_max_angle_deg)); cos_plate = np.cos(np.deg2rad(p.plate_max_angle_deg))
    sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -60, 60)))
    for k in range(T):
        present = ~np.isnan(scene.ped_pos[k, :, 0])
        pidx = np.where(present)[0]
        vpresent = ~np.isnan(scene.veh_pos[k, :, 0]); vidx = np.where(vpresent)[0]
        # occluder boxes: neighbours, moving vehicles, pedestrians
        boxes = list(scene.static_boxes)
        boxes += [box_from(scene.veh_pos[k, v], VEH_HALF) for v in vidx]
        ped_box_start = len(boxes)
        boxes += [box_from(scene.ped_pos[k, a], PED_HALF) for a in pidx]
        bmin = np.array([b[0] for b in boxes]); bmax = np.array([b[1] for b in boxes])
        # distance to ego body (2D) for trigger
        for j, a in enumerate(pidx):
            pos = scene.ped_pos[k, a]
            dx = max(abs(pos[0]) - EGO_HALF[0], 0); dy = max(abs(pos[1]) - EGO_HALF[1], 0)
            near[k, a] = np.hypot(dx, dy) < p.trigger_dist_m
        # ---- faces ----
        if len(pidx):
            hz = scene.ped_head_z[pidx] if scene.ped_head_z is not None else np.full(len(pidx), 1.6)
            heads = scene.ped_pos[k, pidx].copy(); heads[:, 2] = hz
            facing = scene.ped_head[k, pidx]
            for cam in cams:
                in_fov, pxm, d = pixel_density(cam, heads)
                q, dshift = condition_arrays(p.condition, d)
                to_cam = cam.pos - heads; to_cam /= np.linalg.norm(to_cam, axis=1)[:, None]
                cos_yaw = np.sum(to_cam * facing, axis=1)
                face_ok = cos_yaw > cos_face
                yaw_deg = np.rad2deg(np.arccos(np.clip(cos_yaw, -1, 1)))
                yaw_pen = p.yaw_penalty_per_30deg * np.maximum(yaw_deg - p.yaw_free_deg, 0) / 30.0
                # occlusion: rays to 5 face points; exclude own body box
                pts = (heads[:, None, :] + FACE_OFFS[None]).reshape(-1, 3)
                excl = np.repeat(ped_box_start + np.arange(len(pidx)), len(FACE_OFFS))
                occ = ray_occluded(np.repeat(cam.pos[None], len(pts), 0), pts, bmin, bmax, excl)
                vis_frac = 1 - occ.reshape(len(pidx), -1).mean(axis=1)
                eff = pxm * q
                gt = in_fov & face_ok & (vis_frac >= p.min_visible_frac) & (eff >= p.face_id_pxm)
                face_gt[k, pidx] |= gt
                face_pxm_best[k, pidx] = np.maximum(face_pxm_best[k, pidx], np.where(in_fov & face_ok, eff * vis_frac, 0))
                # perception (method's model): face
                z = (eff - p.face_det_c50_pxm) / p.face_det_scale + dshift + track_face[pidx] \
                    - p.occl_penalty * (1 - vis_frac) - yaw_pen + rng.normal(0, p.frame_sigma, len(pidx))
                fd = in_fov & face_ok & (vis_frac > 0) & (rng.random(len(pidx)) < sig(z))
                face_det[k, pidx] |= fd
                # person detection uses body size (~1.75 m) -> much easier
                body_pts = scene.ped_pos[k, pidx]
                in_fov_b, pxm_b, d_b = pixel_density(cam, body_pts)
                q_b, dshift_b = condition_arrays(p.condition, d_b)
                bp = np.concatenate([body_pts[:, None, :] + np.array([[0, 0, dz]]) for dz in (-0.45, 0.0, 0.6)], axis=1).reshape(-1, 3)
                occ_b3 = ray_occluded(np.repeat(cam.pos[None], len(bp), 0), bp, bmin, bmax,
                                      np.repeat(ped_box_start + np.arange(len(pidx)), 3))
                occ_b = occ_b3.reshape(len(pidx), 3).mean(axis=1)
                zb = (pxm_b * q_b - p.person_det_c50_pxm) / p.person_det_scale + dshift_b + track_person[pidx] \
                     - p.occl_penalty * occ_b + rng.normal(0, p.frame_sigma, len(pidx))
                person_det[k, pidx] |= in_fov_b & (rng.random(len(pidx)) < sig(zb))
                body_vis[k, pidx] |= in_fov_b & (occ_b < 1.0)
                a1_tot += int(gt.sum())
                a1_viol += int((gt & ((pxm_b * q_b < p.face_id_pxm) | ~in_fov_b | (occ_b > (1 - vis_frac) + 1e-9))).sum())
        # ---- plates ----
        plates = list(scene.static_plates)
        for v in vidx:
            c = scene.veh_pos[k, v]; h = scene.veh_dir[k, v]
            plates.append((c + h * 2.32 + np.array([0, 0, -0.35]), h))
            plates.append((c - h * 2.32 + np.array([0, 0, -0.25]), -h))
        plate_ids = list(range(S)) + [S + 2 * v + s for v in vidx for s in (0, 1)]
        if plates:
            pc = np.array([pl[0] for pl in plates]); pn = np.array([pl[1] for pl in plates])
            # owning vehicle box index per plate (exclude it from occlusion)
            own = np.array(list(np.repeat(np.arange(len(scene.static_boxes)), 2)) +
                           [len(scene.static_boxes) + j for j in range(len(vidx)) for _ in (0, 1)])
            pid = np.array(plate_ids)
            for cam in cams:
                in_fov, pxm, d = pixel_density(cam, pc)
                q, dshift = condition_arrays(p.condition, d)
                to_cam = cam.pos - pc; to_cam /= np.linalg.norm(to_cam, axis=1)[:, None]
                ok = np.sum(to_cam * pn, axis=1) > cos_plate
                occ = ray_occluded(np.repeat(cam.pos[None], len(pc), 0), pc + pn * 0.02, bmin, bmax, own)
                eff = pxm * q
                gt = in_fov & ok & ~occ & (eff >= p.plate_id_pxm)
                plate_gt[k, pid] |= gt
                z = (eff - p.plate_det_c50_pxm) / p.plate_det_scale + dshift + track_plate[pid] \
                    - p.occl_penalty * occ + rng.normal(0, p.frame_sigma, len(pid))
                plate_det[k, pid] |= in_fov & ok & (rng.random(len(pid)) < sig(z))
                zv = (eff * 4 - p.vehicle_det_c50_pxm) / p.vehicle_det_scale + dshift  # vehicle ~4x plate width
                vehicle_det[k, pid] |= in_fov & (rng.random(len(pid)) < sig(zv))
    return dict(face_gt=face_gt, face_det=face_det, person_det=person_det, near=near,
                face_pxm_best=face_pxm_best, plate_gt=plate_gt, plate_det=plate_det,
                vehicle_det=vehicle_det, a1_tot=a1_tot, a1_viol=a1_viol, body_vis=body_vis)

# --------------------------------------------------------------------------
# Designs (method + baselines + ablations)
# --------------------------------------------------------------------------

def event_mask(near: np.ndarray, p: Params):
    """Recording mask [T] from proximity trigger held >= trigger_hold_s."""
    T = near.shape[0]; hold = int(round(p.trigger_hold_s / p.dt))
    any_near = near.any(axis=1).astype(int)
    run = np.convolve(any_near, np.ones(hold, int), mode="full")[:T]
    trig = np.where(run >= hold)[0]
    rec = np.zeros(T, bool)
    pre = int(round(p.pre_s / p.dt)); post = int(round(p.post_s / p.dt))
    for k in trig:
        rec[max(0, k - pre): min(T, k + post + 1)] = True
    return rec

DESIGNS = {
    #  name              recording     identity handling      fail_closed  per_person_keys
    "B0_continuous_raw": ("continuous", "raw",        False, False),
    "B1_event_raw":      ("event",      "raw",        False, False),
    "B2_event_blur":     ("event",      "irreversible", False, False),
    "B2fc_event_blur_fc":("event",      "irreversible", True,  False),
    "P_proposed":        ("event",      "sealed",     True,  True),
    "A1_no_failclosed":  ("event",      "sealed",     False, True),
    "A2_continuous_sealed":("continuous","sealed",    True,  True),
    "A3_single_key":     ("event",      "sealed",     True,  False),
}

def evaluate(scene: Scene, obs: dict, p: Params, design: str):
    rec_mode, ident, fail_closed, per_person = DESIGNS[design]
    T = scene.t.size
    rec = np.ones(T, bool) if rec_mode == "continuous" else event_mask(obs["near"], p)
    fg, fd, pd = obs["face_gt"], obs["face_det"], obs["person_det"]
    plg, pld, vd = obs["plate_gt"], obs["plate_det"], obs["vehicle_det"]
    if ident == "raw":
        face_leak = fg; plate_leak = plg
    else:
        protected_face = fd | (pd if fail_closed else False)
        protected_plate = pld | (vd if fail_closed else False)
        face_leak = fg & ~protected_face; plate_leak = plg & ~protected_plate
    R = rec[:, None]
    aie_face = (face_leak & R).sum() * p.dt          # identifiable person-seconds accessible
    aie_plate = (plate_leak & R).sum() * p.dt
    captured_face = (fg & R)                          # GT identifiable within recorded time
    # Evidence utility: perpetrator identifiable (after authorised unseal) >= evidence_min_s
    vand = np.where(scene.ped_is_vandal)[0]
    ev = np.nan; collateral = 0.0; vandal_usable_s = np.nan
    if len(vand):
        a = vand[0]
        if ident in ("raw", "sealed"):
            usable = captured_face[:, a]
        else:  # irreversible: only leaked (missed) frames remain identifiable
            usable = (face_leak & R)[:, a]
        vandal_usable_s = float(usable.sum() * p.dt)
        ev = float(vandal_usable_s >= p.evidence_min_s)
        if ident == "sealed":
            # Unseal for the incident window. Per-person keys: only the vandal.
            inc = rec & scene_window(scene, a)
            others = np.ones(fg.shape[1], bool); others[a] = False
            if not per_person:
                collateral = (fg[inc][:, others]).sum() * p.dt
    total_gt_face = fg.sum() * p.dt
    return dict(design=design, aie_face_s=aie_face, aie_plate_s=aie_plate,
                aie_total_s=aie_face + aie_plate, evidence=ev, collateral_s=collateral, vandal_usable_s=vandal_usable_s,
                recorded_frac=float(rec.mean()), gt_face_s=total_gt_face)

def scene_window(scene: Scene, a: int):
    present = ~np.isnan(scene.ped_pos[:, a, 0])
    idx = np.where(present)[0]
    w = np.zeros(scene.t.size, bool)
    if len(idx): w[idx.min(): idx.max() + 1] = True
    return w

def run_episode(p: Params):
    rng = np.random.default_rng(p.seed)
    scene = build_scene(p, rng)
    obs = observe(scene, p, rng)
    return scene, obs, [evaluate(scene, obs, p, d) for d in DESIGNS]
