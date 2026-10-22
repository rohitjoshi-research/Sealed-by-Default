# Phase 6 — Simulator validation report

Status: **simulator and method implemented; sanity checks pass (10/10); no benchmark results yet.**
The smoke run below only checks plausibility and runtime. It is not evidence for any claim.

## What is simulated
| Part | Model | Proven / simulated / assumed |
|---|---|---|
| Scene | Ego car parked between two neighbours; opposite row of 5 cars across the front aisle; pedestrians on a front sidewalk and rear aisle (Poisson arrivals); vehicles passing in both aisles; one vandal incident per episode | Simulated; layout assumed |
| Cameras | 4 equidistant fisheyes (1920 px, 190°) + optional front pinhole (1920 px, 60° HFOV) | Typical values, assumed |
| Occlusion | Ray tracing (slab test) from each camera to 5 points on each face and to each plate, against vehicle and pedestrian boxes | Simulated (exact for box geometry) |
| **Ground truth identifiability (primary metric)** | Face: in FOV, facing ≤ 70° towards camera, ≥ 50 % visible, effective density ≥ 250 px/m (IEC 62676-4 "identification" level — REQUIRES SOURCE VERIFICATION). Plate: ≤ 60°, unoccluded, ≥ 200 px/m (assumed) | Thresholds assumed; geometry simulated |
| Conditions | Image-quality factor q multiplies effective density (day 1.0 … night-rain 0.4) | **Assumed**, to be replaced by measured MTF/SNR |
| Method's perception model | Logistic detection probability in effective density, with condition shift, yaw penalty, occlusion penalty, per-track latent offset (correlated misses) and per-frame noise | **Assumed**; independent of GT by construction |
| Designs | B0 continuous raw; B1 event-triggered raw (strongest realistic baseline); B2 event + irreversible blur; B2fc same with fail-closed (best configuration of B2); P proposed; ablations A1 no fail-closed, A2 continuous sealed, A3 single key per clip | Implemented |

## Metrics
- **AIE (primary):** GT-identifiable face- and plate-seconds inside recorded time that a person can see without an authorised unseal. Computed from GT only.
- **Evidence utility:** vandal GT-identifiable for ≥ 1 s in footage available after an authorised unseal.
- **Collateral exposure:** GT-identifiable seconds of *other* people revealed by that unseal.
- **Deviation from plan (Section 9 of the paper):** AIE uses a geometric identifiability oracle, not an independent re-identification network, because no images are rendered on a single CPU. This is weaker than planned and must be stated as a threat to validity.

## Sanity checks (`python checks.py`)
| # | Check | Result |
|---|---|---|
| 1 | Point on optical axis in FOV, behind not; px/m = f/d | PASS |
| 2 | Fisheye FOV boundary | PASS (boundary shifts ~1° with camera tilt) |
| 3 | Analytic identification range: narrow 6.65 m, fisheye 2.32 m | Recorded |
| 4 | Ray through box occluded, beside not | PASS |
| 5 | Single pedestrian walking at narrow camera: GT onset 6.51 m vs analytic 6.65 m | PASS (step quantisation) |
| 6 | Pedestrian fully behind a wall: never identifiable | PASS |
| 7 | Perfect detector → all sealed/blurred designs leak 0 | PASS |
| 8 | Blind detector, no fail-closed → sealed = raw baseline (wrapper check) | PASS |
| 9 | Same seed identical, other seed differs | PASS |
| 10 | GT unchanged when perception model changes | PASS |

## Smoke run (5 seeds × 2 conditions, 10-min episodes, 6 pedestrians/min) — plausibility only
Mean per episode. Runtime about 2.5 s per episode on one CPU core.

| Condition | Design | AIE face (s) | AIE plate (s) | Evidence | Collateral (s) |
|---|---|---|---|---|---|
| Day | B0 continuous raw | 26.3 | 1736.6 | 1.0 | 0 |
| Day | B1 event raw | 13.2 | 367.1 | 1.0 | 0 |
| Day | B2fc blur, fail-closed | 0.0 | 0.0 | 0.0 | 0 |
| Day | P proposed | 0.0 | 0.0 | 1.0 | 0 |
| Day | A1 no fail-closed | 0.0 | 0.4 | 1.0 | 0 |
| Day | A3 single key | 0.0 | 0.0 | 1.0 | 2.2 |
| Night | B1 event raw | 1.1 | 0.0 | 0.0 | 0 |
| Night | P proposed | 0.0 | 0.0 | 0.0 | 0 |

## What the smoke run reveals about the simulator (not about the method)
1. **Zero face leaks for P and B2fc are a consequence of the assumed detector**, which reaches 50 % detection at 110 px/m while identification needs 250 px/m. Phase 7 must sweep the detector (c50 up to and above 250 px/m, larger correlated misses, fisheye-periphery penalty) before any privacy claim.
2. **Parked cars' number plates dominate exposure** (opposite row in view of the front camera). This depends on the assumed layout and must be varied (angle parking, no opposite row).
3. **Night evidence is zero for every design**: at q = 0.5 the fisheyes cannot identify the vandal. This is a limit of optics, not of privacy design; it needs real night data (Phase 8).
4. **The proposed design's advantage over B2fc is evidence utility, not privacy.** With fail-closed blurring, the irreversible baseline already leaks nothing in this model. The paper must frame the contribution as "privacy of B2fc with evidence of B1", not "more private than everything".

## Not yet implemented (planned)
Zone classes and masks (Z1–Z3), policy engine, cooperative sentry (E6), private zone lookup (E7), eHMI study (E4), overhead on target hardware (E5).

## Phase 7 run v1 — superseded (5 Oct 2026)
The first full benchmark (810 + 240 episodes) exposed two simulator flaws, so its numbers are archived in
`results/superseded_v1/` and must not be cited:
1. The vandal incident had a nearly fixed geometry, so evidence utility became 1.0 or 0.0 per condition.
2. The fail-closed fallback detector was never stressed.

**Sim v2 changes:** randomised incident (position along the car, standing distance 0.35–0.9 m, face yaw ±60°,
duration 6–20 s, approach and exit directions), per-person face height 1.45–1.85 m, three-point body
occlusion for person detection, and a new `sweep_person` stress test. All 10 sanity checks pass on v2.
A first probe shows the fail-closed fallback stays effective even with a much weaker person detector,
because a person whose face is identifiable is, by geometry, close enough for the body to be detected;
this must be confirmed by the full sweep and, ultimately, by real data.

## sim v3 — distance-dependent fog (5 Oct 2026)
Prompted by the Phase 8 fog photo. Fog now uses Koschmieder transmission T(d) = exp(-3.912 d / V) with an
assumed visibility V = 100 m; effective density is multiplied by T(d) and the detector logit shift is
-3.0 * (1 - T(d)) (both ASSUMPTIONS). Rain remains a distance-independent factor (lens droplets, ASSUMPTION).
Checks: 10/10 pass; non-fog episodes are numerically identical to v2 (verified on 6 episodes); the 270 fog
episodes were rerun and all statistics recomputed. v2 results archived in results/superseded_v2/.
Detector and joint stress sweeps use day/night clear only and are unaffected.
