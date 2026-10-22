"""Colour figures for the paper (Figs. 1-4). Reads results/*.json|csv only; no statistics are computed here
except plotting the bootstrap CIs already stored in summary.json."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

C = {"raw": "#D1495B", "blur": "#6C7A89", "prop": "#00A19A", "a1": "#F28E2B", "a3": "#7B5EA7", "ink": "#1F2933"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 9.5, "legend.frameon": False})
S = json.load(open("results/summary.json"))
NOTE = "Simulated (sim v3), geometric ground truth; detector and condition factors are assumptions."

# ---------------- Fig. 1 system architecture (overview) ----------------
from matplotlib.patches import Circle, Polygon
def rbox(ax, x, y, w, h, t, s="", fc="white", ec="#6C7A89", lw=1.3, ts=7.0, ss=6.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0", lw=lw, ec=ec, fc=fc, ls=ls))
    if s:
        ax.text(x + w / 2, y + h * 0.68, t, ha="center", va="center", fontsize=ts, weight="bold", color=C["ink"])
        ax.text(x + w / 2, y + h * 0.30, s, ha="center", va="center", fontsize=ss, color="#374151")
    else:
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=ts, weight="bold", color=C["ink"])
def arr(ax, a, b, col="#4B5563", ls="-", lw=1.1, txt=None, tx=0, ty=1.2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=lw, color=col, ls=ls))
    if txt: ax.text((a[0] + b[0]) / 2 + tx, (a[1] + b[1]) / 2 + ty, txt, ha="center", fontsize=5.8, color="#374151", style="italic")
fig, ax = plt.subplots(figsize=(7.16, 4.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
# zones
ax.add_patch(FancyBboxPatch((0.5, 2), 20, 52, boxstyle="round,pad=0.3,rounding_size=1.5", lw=1, ec="#9CA3AF", fc="#F9FAFB"))
ax.add_patch(FancyBboxPatch((23, 2), 49, 52, boxstyle="round,pad=0.3,rounding_size=1.5", lw=1.2, ec="#00A19A", fc="#F3FBFA", ls="--"))
ax.add_patch(FancyBboxPatch((74.5, 2), 25, 52, boxstyle="round,pad=0.3,rounding_size=1.5", lw=1, ec="#9CA3AF", fc="#F9FAFB"))
for x, t in [(10.5, "Public space"), (47.5, "In-vehicle camera ECU"), (87, "People and key holders")]:
    ax.text(x, 56.5, t, ha="center", fontsize=8, weight="bold", color=C["ink"])
# street scene: parked car with cameras, bystanders, incident
ax.add_patch(FancyBboxPatch((5, 22), 11, 17, boxstyle="round,pad=0.3,rounding_size=2.5", lw=1.2, ec="#374151", fc="#E5E7EB"))
ax.text(10.5, 30.5, "parked\nvehicle", ha="center", va="center", fontsize=6.3, color=C["ink"])
for cx, cy in [(10.5, 39.3), (10.5, 21.7), (4.7, 30.5), (16.3, 30.5), (10.5, 36)]:
    ax.add_patch(Circle((cx, cy), 0.9, fc="#1F2933", ec="white", lw=0.6))
def person(x, y, col):
    ax.add_patch(Circle((x, y + 2.6), 1.0, fc=col, ec="none")); ax.add_patch(FancyBboxPatch((x - 0.9, y - 1.2), 1.8, 3.0, boxstyle="round,pad=0.1,rounding_size=0.6", fc=col, ec="none"))
for x, y in [(4, 46), (9, 47), (15, 45.5)]: person(x, y, "#9CA3AF")
ax.text(9.5, 51.3, "bystanders", ha="center", fontsize=6, color="#374151")
person(17.5, 14, C["a1"]); ax.text(13, 9, "incident\n(damage)", ha="center", fontsize=6, color="#374151")
# ECU blocks
rbox(ax, 25, 47, 45, 4.5, "Safety perception path (parking, driving): never modified", fc="#E8F5E9", ec="#2E7D32", ts=6.4)
rbox(ax, 25, 35, 13, 8, "Image buffer", "frames held\nbefore writing", fc="#EEF2F6")
rbox(ax, 40.5, 35, 13.5, 8, "Detection", "face, plate,\nperson, vehicle", fc="#FFF4E5", ec="#F28E2B")
rbox(ax, 56.5, 35, 13.5, 8, "Sealing policy", "Fig. 2 (logic)", fc="#FFF4E5", ec="#F28E2B")
rbox(ax, 25, 20, 20, 9, "Sealed vault", "exact pixels,\none key per person", fc="#E0F5F3", ec="#00A19A", lw=1.8)
rbox(ax, 50, 20, 20, 9, "Owner view", "anonymized scene,\nactions visible", fc="#E3F2FD", ec="#1E88E5")
rbox(ax, 25, 5, 20, 9, "Append-only log", "policy, unseals,\nevery access", fc="#F1ECF8", ec="#7B5EA7")
rbox(ax, 50, 5, 20, 9, "Unseal rule", "(owner OR court)\nAND one more key", fc="#FDF2F3", ec=C["raw"])
arr(ax, (16.5, 34), (25, 39)); arr(ax, (16.5, 37), (25, 48.5), "#2E7D32")
arr(ax, (38, 39), (40.5, 39)); arr(ax, (54, 39), (56.5, 39))
arr(ax, (59, 35), (40, 29), "#00A19A", txt="identity regions", tx=-2, ty=1.5)
arr(ax, (66, 35), (63, 29), "#1E88E5", txt="scene", tx=4, ty=0)
arr(ax, (40, 20), (55, 14), "#00A19A", txt="incident only", tx=4, ty=1.2)
arr(ax, (50, 9.5), (45, 9.5), "#7B5EA7")
# key holders
rbox(ax, 76.5, 42, 21, 8, "Owner app", "sees scene, may\nrequest an unseal", fc="#E3F2FD", ec="#1E88E5")
rbox(ax, 76.5, 30, 21, 8, "Court", "court key /\ncourt order", fc="white")
rbox(ax, 76.5, 18, 21, 8, "OEM trustee", "third key holder", fc="white")
rbox(ax, 76.5, 6, 21, 8, "Police", "request only\nvia a court", fc="white")
arr(ax, (70, 24.5), (76.5, 45), "#1E88E5"); arr(ax, (76.5, 43), (70, 13), "#1E88E5", ls="--")
for y in (34, 22): arr(ax, (76.5, y), (70, 11), "#4B5563")
arr(ax, (87, 14), (87, 18) if False else (87, 30), "#6C7A89", ls="--", txt="request", tx=-4.5, ty=-6)
fig.tight_layout(); fig.savefig("results/fig1_architecture.png", dpi=300); plt.close(fig)

# ---------------- Fig. 2 logic diagram ----------------
def dia(ax, cx, cy, t, w=15, h=8):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], closed=True, fc="#FFF4E5", ec="#F28E2B", lw=1.2))
    ax.text(cx, cy, t, ha="center", va="center", fontsize=6.0, color=C["ink"])
def lab(ax, x, y, t): ax.text(x, y, t, fontsize=6.0, color="#374151", weight="bold")
fig, ax = plt.subplots(figsize=(7.16, 4.7)); ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis("off")
ax.text(28, 64, "(a) Sealing logic, per frame and camera", ha="center", fontsize=8, weight="bold", color=C["ink"])
ax.text(80, 64, "(b) Unseal logic, per request", ha="center", fontsize=8, weight="bold", color=C["ink"])
ax.plot([57, 57], [1, 62], color="#D1D5DB", lw=0.8)
rbox(ax, 7, 56, 14, 4.5, "New frame", fc="#EEF2F6", ts=6.5)
dia(ax, 14, 47.5, "Processed in\ntime?", w=16)
dia(ax, 14, 35.5, "Face found\nfor person?", w=16)
dia(ax, 14, 23.5, "Person\nfound?", w=16)
rbox(ax, 27, 44.5, 14, 6, "Seal whole frame", fc="#E0F5F3", ec="#00A19A", ts=6.2)
rbox(ax, 27, 32.5, 14, 6, "Seal face box", fc="#E0F5F3", ec="#00A19A", ts=6.2)
rbox(ax, 27, 20.5, 14, 6, "Seal head region", "fail-closed", fc="#E0F5F3", ec="#00A19A", ts=6.2, ss=5.6)
rbox(ax, 5, 6.5, 18, 8, "Not sealed", "residual leak,\ncounted in AIE", fc="#F3F4F6", ec="#9CA3AF", ts=6.2, ss=5.6)
rbox(ax, 44, 12, 11.5, 40, "", fc="#E0F5F3", ec="#00A19A", lw=1.6)
ax.text(49.75, 32, "Encrypt original\npixels with a\nper-person key\n\n\u2193\n\nWrite blurred\nowner view\n\n\u2193\n\nAppend hash\nto the log", ha="center", va="center", fontsize=6.0, color=C["ink"])
arr(ax, (14, 56), (14, 51.5)); arr(ax, (22, 47.5), (27, 47.5)); lab(ax, 22.6, 48.6, "no")
arr(ax, (14, 43.5), (14, 39.5)); lab(ax, 15, 41, "yes")
arr(ax, (22, 35.5), (27, 35.5)); lab(ax, 22.6, 36.6, "yes")
arr(ax, (14, 31.5), (14, 27.5)); lab(ax, 15, 29, "no")
arr(ax, (22, 23.5), (27, 23.5)); lab(ax, 22.6, 24.6, "yes")
arr(ax, (14, 19.5), (14, 14.9)); lab(ax, 15, 16.7, "no")
for y in (47.5, 35.5, 23.5): arr(ax, (41, y), (44, y))
ax.text(28, 2.0, "Number plates follow the same logic with plate and vehicle detection.", ha="center", fontsize=5.8, color="#374151", style="italic")
# unseal
rbox(ax, 67, 56, 26, 4.5, "Unseal request (incident a)", fc="#EEF2F6", ts=6.0)
dia(ax, 79, 47.5, "Owner key OR\ncourt order?", w=18)
dia(ax, 79, 35.5, "At least one\nfurther key holder?", w=18)
dia(ax, 79, 23.5, "Bulk request,\nno court order?", w=18)
for y in (47.5, 35.5, 23.5):
    rbox(ax, 91, y - 3, 8, 6, "Deny,\nlog", fc="#F3F4F6", ec="#6C7A89", ts=6.0)
arr(ax, (88, 47.5), (91, 47.5)); lab(ax, 88.2, 48.6, "no")
arr(ax, (88, 35.5), (91, 35.5)); lab(ax, 88.3, 36.6, "no")
arr(ax, (88, 23.5), (91, 23.5)); lab(ax, 88.3, 24.6, "yes")
arr(ax, (79, 56), (79, 51.5)); arr(ax, (79, 43.5), (79, 39.5)); lab(ax, 80, 41, "yes")
arr(ax, (79, 31.5), (79, 27.5)); lab(ax, 80, 29, "yes")
arr(ax, (79, 19.5), (79, 15.5)); lab(ax, 80, 17, "no")
rbox(ax, 65, 9, 30, 6, "Decrypt only the tracks of incident a", fc="#E0F5F3", ec="#00A19A", ts=6.0)
rbox(ax, 65, 1.5, 30, 5, "Log request, decision and every view", fc="#F1ECF8", ec="#7B5EA7", ts=6.0)
arr(ax, (80, 9), (80, 6.5))
fig.tight_layout(); fig.savefig("results/fig2_logic.png", dpi=300); plt.close(fig)

# ---------------- Fig. 2 condition heatmaps ----------------
conds = [f"{t}_{w}" for t in ("day", "dusk", "night") for w in ("clear", "rain", "fog")]
des = [("B1_event_raw", "Raw event\n(B1)"), ("B2fc_event_blur_fc", "Blur, fail-\nclosed (B2fc)"),
       ("P_proposed", "Proposed\n(P)"), ("A1_no_failclosed", "P without\nfail-closed (A1)")]
E = np.array([[S["by_condition"][f"{c}|{d}"]["evidence"][0] for d, _ in des] for c in conds])
A = np.array([[S["by_condition"][f"{c}|{d}"]["aie"][0] for d, _ in des] for c in conds])
fig, axs = plt.subplots(1, 2, figsize=(7.16, 3.6))
cm_e = LinearSegmentedColormap.from_list("e", ["#F7FBFB", "#7FD3CB", "#00A19A", "#00594F"])
cm_a = LinearSegmentedColormap.from_list("a", ["#FFF8F8", "#F4A8B2", "#D1495B", "#7A1F2B"])
for ax, M, cm, title, fmt, norm in [(axs[0], E, cm_e, "Evidence utility (share of incidents)", "{:.2f}", None),
                                    (axs[1], A, cm_a, "Exposure AIE (s per 10 min, log colour)", "{:.1f}", matplotlib.colors.SymLogNorm(1, vmin=0, vmax=A.max()))]:
    im = ax.imshow(M, cmap=cm, aspect="auto", norm=norm, vmin=None if norm else 0, vmax=None if norm else 1)
    ax.set_xticks(range(len(des))); ax.set_xticklabels([n for _, n in des], fontsize=7)
    ax.set_yticks(range(len(conds))); ax.set_yticklabels([c.replace("_", ", ") for c in conds], fontsize=7.5)
    ax.tick_params(length=0); ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]; dark = (v > 0.55) if norm is None else (v > 30)
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7, color="white" if dark else C["ink"], weight="bold" if j == 2 else None)
    for s in ax.spines.values(): s.set_visible(False)
fig.text(0.5, 0.01, NOTE, ha="center", fontsize=6.5, color="#4B5563")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("results/fig_condition_heatmap.png", dpi=300); plt.close(fig)

# ---------------- Fig. 3 density (face AIE, evidence, collateral) ----------------
dens = [2.0, 6.0, 15.0]
def series(d, m): return np.array([S["by_density"][f"{x}|{d}"][m] for x in dens])
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.6))
def eb(ax, y, col, lab, mk="o", ls="-", lw=1.8):
    ax.errorbar(dens, y[:, 0], yerr=[y[:, 0] - y[:, 1], y[:, 2] - y[:, 0]], color=col, marker=mk, ls=ls, lw=lw, ms=4.5, capsize=3, label=lab)
eb(axs[0], series("B1_event_raw", "aie_face"), C["raw"], "Raw event (B1)")
eb(axs[0], series("B2fc_event_blur_fc", "aie_face"), C["blur"], "Blur, fail-closed (B2fc)", "s", "--")
eb(axs[0], series("P_proposed", "aie_face"), C["prop"], "Proposed (P)", "D", "-", 2.4)
for i, x in enumerate(dens):
    axs[0].annotate(f"{series('B1_event_raw','aie_face')[i,0]:.1f}", (x, series('B1_event_raw','aie_face')[i,0]), textcoords="offset points", xytext=(5, -10) if i < 2 else (-22, 2), fontsize=7, color=C["ink"])
axs[0].set_title("Face exposure (s / 10 min)"); axs[0].set_xlabel("Pedestrians per minute")
eb(axs[1], series("B1_event_raw", "evidence"), C["raw"], "Raw event (B1)", "o", "-", 3.5)
eb(axs[1], series("P_proposed", "evidence"), C["prop"], "Proposed (P)", "D", "--", 1.6)
eb(axs[1], series("B2fc_event_blur_fc", "evidence"), C["blur"], "Blur, fail-closed (B2fc)", "s", "--")
axs[1].set_ylim(-0.03, 0.7); axs[1].set_title("Evidence utility"); axs[1].set_xlabel("Pedestrians per minute")
axs[1].text(8.5, 0.6, "P = B1 (lines overlap)", fontsize=6.8, color=C["ink"], ha="center")
eb(axs[2], series("A3_single_key", "collateral"), C["a3"], "One key per clip (A3)", "^")
eb(axs[2], series("P_proposed", "collateral"), C["prop"], "Per-person keys (P)", "D", "-", 2.4)
for i, x in enumerate(dens):
    axs[2].annotate(f"{series('A3_single_key','collateral')[i,0]:.2f}", (x, series('A3_single_key','collateral')[i,0]), textcoords="offset points", xytext=(5, -10) if i < 2 else (-26, 2), fontsize=7, color=C["ink"])
axs[2].set_title("Exposed per unseal (s)"); axs[2].set_xlabel("Pedestrians per minute")
for ax in axs: ax.set_xticks(dens); ax.grid(axis="y", color="#E5E7EB", lw=0.6)
h0, l0 = axs[0].get_legend_handles_labels(); h2, l2 = axs[2].get_legend_handles_labels()
fig.legend(h0 + [h2[0]], l0 + [l2[0]], loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig("results/fig_density.png", dpi=300); plt.close(fig)

# ---------------- Fig. 4 detector sweep ----------------
sw = S["sweep"]; xs = [110.0, 180.0, 250.0, 320.0]
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.8), sharey=True)
for ax, c in zip(axs, ["day_clear", "night_clear"]):
    g = lambda d, s=0.6: [sw[f"{c}|{x}|{s}|{d}"]["aie_face"][0] for x in xs]
    ax.axvspan(250, 335, color="#FDECEE", zorder=0); ax.axvline(250, color=C["raw"], lw=0.9, ls=":")
    ax.text(292, 12.5, "face detector\nweaker than\nID level", fontsize=6.3, color="#4B5563", ha="center")
    ax.plot(xs, g("B1_event_raw"), color=C["raw"], marker="o", lw=2, label="Raw event (B1)")
    ax.plot(xs, g("A1_no_failclosed"), color=C["a1"], marker="o", lw=2, label="P without fail-closed (A1), \u03c3=0.6")
    ax.plot(xs, g("A1_no_failclosed", 1.5), color=C["a1"], marker="s", lw=1.4, ls=":", label="A1, \u03c3=1.5 (more correlated misses)")
    ax.plot(xs, g("B2fc_event_blur_fc"), color=C["blur"], marker="s", lw=1.2, ls="--", label="Blur, fail-closed (B2fc)")
    ax.plot(xs, g("P_proposed"), color=C["prop"], marker="D", lw=2.4, label="Proposed (P)")
    if c == "day_clear":
        for x, y, off in zip(xs, g("A1_no_failclosed"), [(-4, 8), (-6, 8), (-22, 4), (-22, 4)]):
            ax.annotate(f"{y:.2f}" if y < 1 else f"{y:.1f}", (x, y), textcoords="offset points", xytext=off, fontsize=6.8, color=C["ink"], weight="bold")
    ax.annotate(f"raw {g('B1_event_raw')[0]:.1f}", (xs[0], g("B1_event_raw")[0]), textcoords="offset points", xytext=(4, 5), fontsize=6.8, color=C["ink"], weight="bold")
    ax.set_title(c.replace("_", ", ")); ax.set_xlabel("Face detector 50 % point (px/m)"); ax.set_xlim(100, 335)
    ax.grid(axis="y", color="#E5E7EB", lw=0.6)
axs[0].set_ylabel("Face exposure (s / 10 min)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig("results/fig_detector_sweep.png", dpi=300); plt.close(fig)
print("figures written")

# ---------------- Fig. 6 official zones: sign board and schematic map (concept) ----------------
from matplotlib.patches import Rectangle, Wedge
from matplotlib.lines import Line2D
ZC = {"Z1": ("#D6EAF8", "#1E88E5"), "Z2": ("#FFF1D6", "#F28E2B"), "Z3": ("#E5E7EB", "#374151")}
fig = plt.figure(figsize=(7.16, 3.7))
# (a) official sign
ax = fig.add_axes([0.01, 0.04, 0.33, 0.88]); ax.set_xlim(0, 40); ax.set_ylim(0, 50); ax.axis("off")
ax.text(20, 48.5, "(a) Official sign at the zone entrance", ha="center", fontsize=8, weight="bold", color=C["ink"])
ax.add_patch(Rectangle((19.3, 9.5), 1.4, 7, fc="#6B7280", ec="none"))
ax.add_patch(FancyBboxPatch((6, 16), 28, 29, boxstyle="round,pad=0.4,rounding_size=1.5", fc="white", ec="#1F2933", lw=1.6))
ax.add_patch(Circle((20, 35.5), 7.2, fc="white", ec="#C62828", lw=3.2))          # prohibition ring (graphic)
ax.add_patch(FancyBboxPatch((15.2, 32.4), 7.6, 5.6, boxstyle="round,pad=0.2,rounding_size=0.8", fc="#1F2933", ec="none"))
ax.add_patch(Polygon([(22.8, 34.0), (25.4, 32.6), (25.4, 38.4), (22.8, 37.0)], fc="#1F2933", ec="none"))
ax.add_patch(Circle((18.4, 35.2), 1.4, fc="white", ec="none"))
ax.plot([14.9, 25.1], [30.4, 40.6], color="#C62828", lw=3.2, solid_capstyle="round")
ax.text(20, 25.8, "No vehicle camera recording", ha="center", fontsize=6.8, weight="bold", color=C["ink"])
ax.text(20, 22.6, "Zone class Z3 \u00b7 statutory site", ha="center", fontsize=6.2, color=C["ink"])
ax.text(18.5, 19.6, "Register entry DE-Z3-0417", ha="center", fontsize=5.6, color="#374151")
qr = np.array([[1,1,1,0,1,0,1],[1,0,1,1,0,1,1],[1,1,1,0,1,0,0],[0,1,0,1,1,1,0],[1,0,1,0,0,1,1],[0,1,1,1,0,0,1],[1,1,0,1,1,1,1]])
for i in range(7):
    for j in range(7):
        if qr[i, j]: ax.add_patch(Rectangle((29.2 + j * 0.55, 17.4 + (6 - i) * 0.55), 0.55, 0.55, fc="#1F2933", ec="none"))
ax.add_patch(FancyBboxPatch((11, 11.0), 18, 4.2, boxstyle="round,pad=0.25,rounding_size=0.6", fc="white", ec="#1F2933", lw=1.2))
ax.text(20, 13.1, "next 600 m", ha="center", va="center", fontsize=6.6, weight="bold", color=C["ink"])
ax.text(20, 4.0, "Machine-readable code links the sign to its\nsigned register entry; signs without an\nentry are ignored and reported.", ha="center", fontsize=5.6, color="#374151")
# (b) schematic map
ax = fig.add_axes([0.36, 0.04, 0.63, 0.88]); ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
ax.text(50, 60.5, "(b) Schematic map: official zones reach the car before it arrives", ha="center", fontsize=8, weight="bold", color=C["ink"])
ax.add_patch(Rectangle((0, 2), 100, 55, fc="#F3F4F6", ec="#D1D5DB", lw=0.8))
for y in (14, 30, 46): ax.add_line(Line2D([0, 100], [y, y], color="white", lw=5.5, zorder=1))
for x in (20, 48, 76): ax.add_line(Line2D([x, x], [2, 57], color="white", lw=5.5, zorder=1))
def zone(x, y, w, h, cls, name):
    fc, ec = ZC[cls]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2", fc=fc, ec=ec, lw=1.4, ls="--", zorder=2, alpha=0.95))
    ax.text(x + w / 2, y + h - 2.6, f"{cls}  {name}", ha="center", fontsize=6.3, weight="bold", color=C["ink"], zorder=3)
zone(52, 33, 22, 12, "Z1", "clinic / hospital"); zone(23, 33, 23, 12, "Z2", "school")
zone(52, 3.5, 46, 9, "Z3", "military site")
ax.add_patch(Rectangle((52, 3.5), 46, 9, fill=False, hatch="////", ec="#9CA3AF", lw=0, zorder=2))
route = np.array([[4, 2], [4, 30], [48, 30], [48, 14], [60, 14], [60, 8]])
ax.plot(route[:, 0], route[:, 1], color="#1E88E5", lw=3, zorder=4, solid_capstyle="round")
ax.plot([48, 48, 60], [22, 14, 14], color="#F28E2B", lw=6, alpha=0.45, zorder=3, solid_capstyle="round")
ax.text(39, 18.2, "500 m look-ahead", fontsize=6.0, color=C["ink"], ha="center", style="italic", zorder=5)
ax.add_patch(FancyBboxPatch((46.3, 20.2), 3.4, 5.2, boxstyle="round,pad=0.2,rounding_size=0.8", fc="#1F2933", ec="white", lw=0.8, zorder=6))
ax.add_patch(FancyBboxPatch((4, 49), 44, 7, boxstyle="round,pad=0.3,rounding_size=1.0", fc="white", ec="#374151", lw=1.0, zorder=6))
ax.text(26, 54.2, "Recording zone Z3 ahead in 480 m", ha="center", fontsize=6.4, weight="bold", color=C["ink"], zorder=7)
ax.text(26, 51.0, "recording stops on entry; driver assistance stays on", ha="center", fontsize=5.8, color="#374151", zorder=7)
ax.add_patch(FancyArrowPatch((30, 49), (46.8, 26), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#374151", zorder=6))
ax.add_patch(Rectangle((61.3, 15.0), 3.4, 3.8, fc="white", ec="#1F2933", lw=0.9, zorder=6))
ax.text(63, 19.6, "sign", fontsize=5.6, ha="center", color=C["ink"], zorder=7)
ax.add_patch(Rectangle((78, 47), 18, 7.5, fc="#FDE68A", ec="#B45309", lw=1, zorder=4))
ax.text(87, 50.7, "underground\ncar park", ha="center", va="center", fontsize=5.6, color=C["ink"], zorder=5)
ax.text(87, 44.6, "GNSS lost: keeps last\nverified entry state", ha="center", va="top", fontsize=5.4, color="#374151", zorder=5)
ax.text(99, 57.6, "schematic, no real locations", ha="right", va="bottom", fontsize=5.4, color="#4B5563", style="italic", zorder=5)
ax.add_line(Line2D([3, 23], [4.5, 4.5], color=C["ink"], lw=1.4, zorder=5)); ax.text(13, 5.4, "200 m", ha="center", fontsize=5.6, color=C["ink"], zorder=5)
fig.savefig("results/fig6_zones_sign_map.png", dpi=300); plt.close(fig)

# ---------------- Fig. 7 zone classes and zone-verification logic (concept) ----------------
fig = plt.figure(figsize=(7.16, 3.9))
ax = fig.add_axes([0.01, 0.03, 0.47, 0.90]); ax.set_xlim(0, 50); ax.set_ylim(0, 52); ax.axis("off")
ax.text(25, 50.5, "(a) Zone classes tighten the default, never loosen it", ha="center", fontsize=7.6, weight="bold", color=C["ink"])
cols = [(1, "Class"), (8, "Typical site"), (23, "Recording"), (37.5, "Unseal")]
for x, t in cols: ax.text(x, 45.5, t, fontsize=6.4, weight="bold", color=C["ink"])
rows = [("Z0", "#E0F5F3", "#00A19A", "ordinary street,\ncar park", "sealed by default\n(evaluated design)", "owner or court\n+ one more key"),
        ("Z1", ZC["Z1"][0], ZC["Z1"][1], "clinic, hospital,\nplace of worship", "sealed; entrance\nmasked; short keep", "court only"),
        ("Z2", ZC["Z2"][0], ZC["Z2"][1], "school,\nplayground", "sealed; short keep;\nminors protected", "court only"),
        ("Z3", ZC["Z3"][0], ZC["Z3"][1], "military site, critical\ninfra. (by statute)", "no recording, no\nremote view*", "nothing stored")]
for k, (z, fc, ec, site, rec, uns) in enumerate(rows):
    y = 34 - k * 10.5
    ax.add_patch(FancyBboxPatch((0.5, y), 49, 9, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc, ec=ec, lw=1.2))
    ax.text(3.6, y + 4.5, z, ha="center", va="center", fontsize=8, weight="bold", color=C["ink"])
    for x, t in [(8, site), (23, rec), (37.5, uns)]:
        ax.text(x, y + 4.5, t, va="center", fontsize=5.9, color=C["ink"])
ax.text(25, 0.8, "*Driving assistance stays live (frames never stored). Only authorities declare zones.", ha="center", fontsize=5.6, color="#374151", style="italic")
ax = fig.add_axes([0.50, 0.03, 0.49, 0.90]); ax.set_xlim(0, 52); ax.set_ylim(0, 52); ax.axis("off")
ax.text(26, 50.5, "(b) How the car decides its zone class", ha="center", fontsize=7.6, weight="bold", color=C["ink"])
rbox(ax, 1, 41, 22, 6, "Signed register", "private tile query", fc="#EEF2F6", ts=6.0, ss=5.4)
rbox(ax, 29, 41, 22, 6, "Sign recognition", "front camera", fc="#EEF2F6", ts=6.0, ss=5.4)
dia(ax, 40, 32.5, "Sign has a\nregister entry?", w=17, h=8)
rbox(ax, 41.5, 20.5, 10, 5.5, "Ignore,\nreport", fc="#F3F4F6", ec="#6C7A89", ts=5.6)
dia(ax, 14, 26, "Sources disagree\nor GNSS uncertain?", w=19, h=8.5)
rbox(ax, 37, 12, 14, 5.5, "Stricter\nclass wins", fc="#FFF4E5", ec="#F28E2B", ts=5.6)
dia(ax, 14, 13, "GNSS lost\n(garage)?", w=15, h=7.5)
rbox(ax, 26, 0.5, 13, 5.5, "Keep last verified\nentry state", fc="#FFF4E5", ec="#F28E2B", ts=5.2)
rbox(ax, 1, 0.5, 21, 5.5, "Apply class to sealing\npolicy (Fig. 2)", fc="#E0F5F3", ec="#00A19A", ts=5.6)
arr(ax, (12, 41), (13.5, 30.3)); arr(ax, (40, 41), (40, 36.5))
arr(ax, (45, 30.5), (46.5, 26)); lab(ax, 46.8, 28.3, "no")
arr(ax, (31.5, 32.5), (22.5, 27.8)); lab(ax, 26.2, 31.2, "yes")
arr(ax, (23.5, 26), (40, 17.5)); lab(ax, 31, 23.4, "yes")
arr(ax, (14, 21.75), (14, 16.75)); lab(ax, 15, 19.2, "no")
arr(ax, (21.5, 13), (30, 6)); lab(ax, 24.2, 11.2, "yes")
arr(ax, (14, 9.25), (12, 6)); lab(ax, 14.4, 7.3, "no")
arr(ax, (40, 12), (22, 5.6)); arr(ax, (26, 3.2), (22, 3.2))
fig.savefig("results/fig7_zone_logic.png", dpi=300); plt.close(fig)
print("zone figures written")


# ---------------- Fig. 8 driver guidance in a Z3 zone (HMI concept) ----------------
fig, axs = plt.subplots(2, 2, figsize=(7.16, 4.6)); axs = axs.ravel()
HB, HT, HQ, OK, AM = "#0F1A25", "#F3F4F6", "#9CA3AF", "#34D399", "#FBBF24"
def cam_icon(ax, x, y, col, slash=False, s=1.0):
    ax.add_patch(FancyBboxPatch((x - 1.6 * s, y - 1.0 * s), 2.6 * s, 2.0 * s, boxstyle="round,pad=0.1,rounding_size=0.3", fc=col, ec="none"))
    ax.add_patch(Polygon([(x + 1.0 * s, y - 0.5 * s), (x + 2.0 * s, y - 1.0 * s), (x + 2.0 * s, y + 1.0 * s), (x + 1.0 * s, y + 0.5 * s)], fc=col, ec="none"))
    if slash: ax.plot([x - 2.0 * s, x + 2.3 * s], [y - 1.6 * s, y + 1.6 * s], color=HT, lw=1.6)
panels = [
 ("(a) Approaching", [("cam_off", AM, "Recording zone Z3 in 480 m"), ("txt", HQ, "Recording and remote view\nstop on entry"), ("ok", OK, "Driver assistance\nstays active")]),
 ("(b) Driving in Z3", [("cam_off", HQ, "Recording OFF"), ("ok", OK, "Assistance ON\n(live frames, never stored)"), ("txt", HQ, "AEB, lane keeping and\nparking aids unchanged")]),
 ("(c) Parked in Z3", [("cam_off", HQ, "Sentry cameras\nunavailable here"), ("ok", OK, "Alarm active: motion, tilt,\nultrasonic, radar"), ("txt", HQ, "Owner notified\nwithout images")]),
 ("(d) Exception: a law bans camera operation", [("warn", AM, "Camera assistance\nunavailable"), ("txt", HT, "Parking: ultrasonic\nguidance only"), ("txt", HQ, "Check mirrors,\nfollow site staff")]),
]
for ax, (title, items) in zip(axs, panels):
    ax.set_xlim(0, 52); ax.set_ylim(0, 30); ax.axis("off"); ax.set_aspect("equal")
    ax.add_patch(FancyBboxPatch((0.4, 0.6), 51.2, 24.6, boxstyle="round,pad=0.2,rounding_size=1.6", fc=HB, ec="#374151", lw=1.2))
    ax.text(26, 27.6, title, ha="center", va="center", fontsize=7.4, weight="bold", color=C["ink"])
    for k, (kind, col, t) in enumerate(items):
        y = 20.5 - k * 7.2
        if kind == "cam_off": cam_icon(ax, 5.5, y, col, slash=True, s=0.9)
        elif kind == "ok": ax.add_patch(Circle((5.5, y), 1.5, fc=col, ec="none")); ax.text(5.5, y - 0.1, "\u2713", ha="center", va="center", fontsize=7, color=HB, weight="bold")
        elif kind == "warn": ax.add_patch(Polygon([(5.5, y + 1.7), (7.2, y - 1.3), (3.8, y - 1.3)], fc=col, ec="none")); ax.text(5.5, y - 0.3, "!", ha="center", va="center", fontsize=7, color=HB, weight="bold")
        else: ax.add_patch(Circle((5.5, y), 0.5, fc=col, ec="none"))
        ax.text(10.0, y, t.replace("\n", " ") if len(t) < 34 else t, va="center", fontsize=6.6, color=HT if kind != "txt" else col)
fig.text(0.5, 0.01, "HMI concept, not evaluated. Privacy rules act only on recording, storage and remote viewing; live perception for safety is never switched off by the privacy policy.",
         ha="center", fontsize=5.8, color="#4B5563", style="italic")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig("results/fig8_driver_guidance.png", dpi=300); plt.close(fig)
print("fig8 written")


# ---------------- Graphical abstract (16:9) ----------------
fig = plt.figure(figsize=(12.8, 7.2)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 128); ax.set_ylim(0, 72); ax.axis("off")
ax.add_patch(Rectangle((0, 0), 128, 72, fc="#0B1117"))
ax.text(64, 66, "Sealed by Default: privacy-preserving parked-vehicle cameras", ha="center", fontsize=21, weight="bold", color="white")
ax.text(64, 61.5, "Seal, don't blur: bystander identity locked by default, evidence kept, released only to the owner or a court", ha="center", fontsize=12, color="#C7D2DC")
steps = [("Parked-car camera", "records an incident", "#6C7A89"), ("Seal identity in the ISP", "exact pixels, 1 key per person;\nhead region if no face found", "#00A19A"),
         ("Owner view", "anonymized scene,\nactions visible", "#1E88E5"), ("Unseal one incident", "(owner OR court)\nAND one more key", "#7B5EA7")]
for k, (t1, t2, col) in enumerate(steps):
    y = 50 - k * 12
    ax.add_patch(FancyBboxPatch((4, y - 4.5), 46, 9, boxstyle="round,pad=0.3,rounding_size=1.2", fc="#111A23", ec=col, lw=2.2))
    ax.add_patch(Circle((9, y), 2.6, fc=col, ec="none")); ax.text(9, y, str(k + 1), ha="center", va="center", fontsize=13, weight="bold", color="white")
    ax.text(14, y + 1.6, t1, fontsize=13, weight="bold", color="white", va="center"); ax.text(14, y - 2.0, t2, fontsize=9.5, color="#C7D2DC", va="center")
    if k < 3: ax.add_patch(FancyArrowPatch((27, y - 4.8), (27, y - 7.2), arrowstyle="-|>", mutation_scale=12, color="#9AA6B2", lw=1.4))
tiles = [("59.5 s \u2192 0 s", "faces & plates exposed\nper 10 min vs raw recording", C["raw"]), ("0.52 kept", "evidence utility\n(blurring keeps 0)", C["prop"]),
         ("1.08 s \u2192 0", "bystanders revealed\nper unseal", C["a3"]), ("100 %*", "incident visible to owner\n(vs 21 % whole-person)", "#1E88E5")]
for k, (big, small, col) in enumerate(tiles):
    x = 56 + (k % 2) * 35; y = 33 - (k // 2) * 24
    ax.add_patch(FancyBboxPatch((x, y), 32, 20, boxstyle="round,pad=0.3,rounding_size=1.4", fc="#111A23", ec=col, lw=2.4))
    ax.text(x + 16, y + 13.2, big, ha="center", fontsize=20, weight="bold", color="white"); ax.text(x + 16, y + 5.0, small, ha="center", fontsize=10, color="#C7D2DC", va="center")
ax.text(64, 4.2, "Simulation: 810 paired episodes, 9 light & weather conditions. *Assumes the sealed head region covers the face. Real-vehicle validation pending.",
        ha="center", fontsize=9.5, color="#9FB0C0")
fig.savefig("results/graphical_abstract.png", dpi=150, facecolor="#0B1117"); plt.close(fig)
print("graphical abstract written")
