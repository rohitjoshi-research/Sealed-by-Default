# Sealed by Default — privacy-preserving parked-vehicle camera recording (simulation)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23575914.svg)](https://doi.org/10.5281/zenodo.23575914)

Code and data (archived at https://doi.org/10.5281/zenodo.23575914) for the manuscript *"Sealed by Default: Privacy-Preserving Parked-Vehicle Camera Recording with
Fail-Closed Sealing and Owner-or-Court Unsealing"* (R. Joshi).

## Problem
Parked cars with cameras ("sentry" functions) record every passer-by. Raw footage keeps evidence of damage but
exposes every face and number plate; irreversible blurring protects bystanders but destroys the evidence.

## What this repository contains
A CPU-only geometric simulator that measures **accessible identifiable exposure (AIE)** — identifiable face- and
plate-seconds anyone can see without an authorized unseal — on simulator ground truth (ray-traced occlusion,
IEC 62676-4 identification density), plus every benchmark, sweep and analysis used in the paper.

## Main results (simulated, sim v3, 810 paired episodes)
| Design | AIE (s / 10 min) | Evidence utility |
|---|---|---|
| Event-triggered raw recording | 59.5 [49.5, 69.6] | 0.52 |
| Fail-closed blurring | 0.00 | 0.00 |
| **Sealed by default (proposed)** | **0.00** | **0.52** |

Per-person keys avoid 1.08 s of bystander exposure per unseal. Fail-closed protection holds until the person
detector needs the identification resolution itself.

## Limitations (read before citing)
- Simulation only; identifiability is a geometric proxy, not a face-recognition result.
- Detector, rain, fog and night factors are **assumptions**; absolute magnitudes change by more than 10x with them
  (paper Section VII-F). Only the relative conclusions held in every sensitivity run.
- One incident type; no real automotive camera data; head-region sealing assumes the head estimate covers the face.

## Quick start
```bash
pip install -r requirements.txt
python checks.py          # 10 sanity checks
python verify_paper.py    # compares all 56 paper values with the shipped results/ (~2 s)
./run_all.sh              # reruns everything from scratch (~1.5 h, one CPU core), then verifies
```
Or open `validate_colab.ipynb` in Google Colab (QUICK mode ~1 min, FULL mode ~1.5 h).

## Layout
| Path | Content |
|---|---|
| `sim.py` | scene, cameras, ray tracing, ground truth, perception model, designs, metrics |
| `checks.py` | sanity checks |
| `bench.py`, `joint_stress.py`, `batch2.py`, `batch3.py` | experiment runners (resumable, fixed seeds) |
| `analyze.py`, `analyze_batch2.py`, `summarize_batch3.py` | statistics and figures |
| `verify_paper.py` | paper-value check |
| `.github/workflows/verify.yml` | CI: sanity checks and paper-value check on every push |
| `make_figures.py` | architecture and logic diagrams, colour result figures, official-zone and driver-guidance concept diagrams (Figs. 1-8) |
| `validate_colab.ipynb`, `make_notebook.py` | validation notebook and its generator |
| `results/` | per-episode CSVs, summaries, figures (see `results/README.md`) |
| `phase8/` | real-image check code and derived numbers (photos not released) |
| `phase9/` | results video build (HTML) |
| `VALIDATION.md`, `params.json` | simulator validation log and all parameters |

## Licence
Code: MIT (`LICENSE`). Data and figures in `results/` and `phase9/video_data.json`: CC BY 4.0 (`LICENSE-DATA`).

## Citation
R. Joshi, *Sealed by Default: simulator and data for privacy-preserving parked-vehicle camera recording*, v0.1.0, Zenodo, 2026. doi:[10.5281/zenodo.23575914](https://doi.org/10.5281/zenodo.23575914). Author ORCID: [0009-0007-4201-7862](https://orcid.org/0009-0007-4201-7862). Machine-readable metadata: `CITATION.cff`.

## Disclosure
This is an independent personal research project; views are the author's own and do not represent any employer. 
