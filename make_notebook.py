"""Generate validate_colab.ipynb: self-contained (writes the exact experiment code), QUICK and FULL modes."""
import json, pandas as pd
CODE = ["sim.py", "checks.py", "bench.py", "analyze.py", "joint_stress.py", "batch2.py", "analyze_batch2.py",
        "batch3.py", "summarize_batch3.py", "verify_paper.py"]
m = pd.read_csv("results/main_episodes.csv")
SUB = [("day_clear", 6.0, "opposite_row", 0), ("night_clear", 15.0, "no_opposite", 1), ("dusk_fog", 2.0, "opposite_row", 2),
       ("day_rain", 15.0, "opposite_row", 3), ("night_fog", 6.0, "no_opposite", 4), ("dusk_clear", 2.0, "no_opposite", 5)]
exp = []
for c, d, l, s in SUB:
    key = f"condition={c}|layout={l}|ped_rate_per_min={d}|seed={s}"
    g = m[m.episode_key == key]
    assert len(g) == 8, key
    for _, r in g.iterrows():
        exp.append(dict(condition=c, dens=d, layout=l, seed=s, design=r.design, aie_face_s=round(r.aie_face_s, 6),
                        aie_plate_s=round(r.aie_plate_s, 6), evidence=r.evidence, collateral_s=round(r.collateral_s, 6)))
def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t}
cells = [md("# Sealed by Default \u2014 validation notebook\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OWNER/REPO/blob/main/validate_colab.ipynb) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23575914.svg)](https://doi.org/10.5281/zenodo.23575914)\n\nReproduces the simulation results of the manuscript *Sealed by Default: Privacy-Preserving Parked-Vehicle Camera Recording with Fail-Closed Sealing and Owner-or-Court Unsealing* (R. Joshi, ORCID [0009-0007-4201-7862](https://orcid.org/0009-0007-4201-7862)). Code and data: [doi:10.5281/zenodo.23575914](https://doi.org/10.5281/zenodo.23575914).\n\n"
            "* **QUICK** (about 1 minute): ten sanity checks, then six benchmark episodes recomputed and compared value-by-value with the published per-episode results.\n"
            "* **FULL** (about 1.5 h on one CPU core): every benchmark, sweep, sensitivity and mechanism run, then all 56 paper values compared with the recomputed ones (Tables IV\u2013VI, VIII, IX; Figs. 3\u20135; Sections IV and VII).\n\n"
            "Not reproducible here: Table VII (real-image check), because the photographs are personal data and are not released.\n\n"
            "All results are **simulated**; see the paper's threats to validity."),
         code('MODE = "QUICK"   # set to "FULL" to rerun everything'),
         code("!pip -q install numpy==2.4.4 scipy==1.17.1 pandas matplotlib\nimport os; os.makedirs('results', exist_ok=True)")]
for f in CODE:
    cells.append(code(f"%%writefile {f}\n" + open(f).read()))
cells += [md("## 1. Sanity checks (Section VI)"), code("!python checks.py"),
          md("## 2. QUICK: six episodes against published per-episode values (Section VII-A data)"),
          code("import json, numpy as np, sim\nEXPECTED = json.loads('''" + json.dumps(exp) + "''')\n"
               "bad = 0\nfor c, d, l, s in sorted({(e['condition'], e['dens'], e['layout'], e['seed']) for e in EXPECTED}):\n"
               "    _, _, res = sim.run_episode(sim.Params(condition=c, ped_rate_per_min=d, layout=l, seed=1000 * s + 7))\n"
               "    got = {r['design']: r for r in res}\n"
               "    for e in [e for e in EXPECTED if (e['condition'], e['dens'], e['layout'], e['seed']) == (c, d, l, s)]:\n"
               "        r = got[e['design']]\n"
               "        same = all(abs(r[k] - e[k]) < 1e-6 for k in ('aie_face_s', 'aie_plate_s', 'collateral_s')) and r['evidence'] == e['evidence']\n"
               "        bad += not same\n"
               "    print(f'{c:12s} {d:5.1f}/min {l:13s} seed {s}:', 'all 8 designs match' if bad == 0 else f'{bad} mismatches so far')\n"
               "print('QUICK result:', 'PASS' if bad == 0 else f'FAIL ({bad} mismatches)')"),
          md("## 3. FULL: rerun everything and compare every paper value"),
          code('if MODE == "FULL":\n'
               "    !python bench.py main && python bench.py sweep && python bench.py sweep_person\n"
               "    !python joint_stress.py && python analyze.py\n"
               "    !python batch2.py && python analyze_batch2.py && python batch3.py && python summarize_batch3.py\n"
               "    !python verify_paper.py\n"
               "else:\n    print('Set MODE = \"FULL\" to rerun all experiments (about 1.5 h).')")]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("validate_colab.ipynb", "w"), indent=1)
print("notebook cells:", len(cells), "expected rows:", len(exp))
