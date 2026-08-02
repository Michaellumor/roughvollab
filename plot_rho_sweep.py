"""Dissertation Figure 4.1: rho-sweep cost ratio, generated from the committed rho_sweep.csv.
Usage: py plot_rho_sweep.py   (expects rho_sweep.csv in the working directory)
Outputs: rho_sweep.pdf, rho_sweep.png"""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("rho_sweep.csv")))
rho = [float(r["rho"]) for r in rows]
ratio = [float(r["ratio"]) for r in rows]
Lstar = [int(r["Lstar"]) for r in rows]

fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(rho, ratio, marker="o", color="#1a5276", lw=1.6, ms=5, zorder=3)
ax.axhline(1.0, color="#888", lw=0.9, ls="--", zorder=1)
ax.text(-0.985, 1.015, "parity (ratio = 1)", fontsize=8, color="#666")
for x, y, L in zip(rho, ratio, Lstar):
    ax.annotate(f"$L^*={L}$", (x, y), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=7.5, color="#444")
ax.set_xlabel(r"leverage correlation $\rho$")
ax.set_ylabel(r"$\mathrm{Cost}_{\mathrm{cond\text{-}SG}}\,/\,\mathrm{Cost}_{\mathrm{cond\text{-}MLMC}}$")
ax.set_ylim(0.30, 1.08)
ax.invert_xaxis()
ax.grid(alpha=0.25, lw=0.5)
fig.tight_layout()
fig.savefig("rho_sweep.pdf")
fig.savefig("rho_sweep.png", dpi=180)
