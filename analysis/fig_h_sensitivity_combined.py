"""Combined BTC + ETH per-maturity H-sensitivity overlay for the calibration paper.

Reads the {btc,eth}_hsens.json emitted by layer4_span_identifiability.py (the D41/D42
jacobian at each currency's documented θ̂) and overlays both ||∂IV/∂H|| profiles on one
axes → output/layer4_h_sensitivity.png. This is the figure referenced by fig:hsens: the
H-signal is concentrated at the short end and decays to near-zero at the one-year tenor,
and the Ethereum profile is nearly identical to Bitcoin's (cross-market mechanism).

Pure plotting — no re-run of the jacobian. Regenerate the inputs with:
  python analysis/layer4_span_identifiability.py --currency BTC
  python analysis/layer4_span_identifiability.py --currency ETH
"""
import json, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")


def load(cur):
    with open(os.path.join(OUT, f"{cur}_hsens.json")) as f:
        return json.load(f)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b, e = load("btc"), load("eth")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(b["months"], b["sens"], "o-", color="#D85A30", lw=2.2, ms=8, label="BTC")
    ax.plot(e["months"], e["sens"], "s--", color="#7F77DD", lw=2.2, ms=8, label="ETH")
    for m, v in zip(b["months"], b["sens"]):
        ax.annotate(f"{v:.1f}", (m, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color="#D85A30")
    for m, v in zip(e["months"], e["sens"]):
        ax.annotate(f"{v:.1f}", (m, v), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color="#7F77DD")
    ax.set_xlabel("maturity (months)")
    ax.set_ylabel("H-sensitivity  ||∂IV/∂H||  (vol-pts per unit H)")
    ax.set_title("Per-maturity H-signal: short end peaks, long tenor ≈ flat\n"
                 "BTC and ETH profiles nearly identical (cross-market mechanism)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(max(b["sens"]), max(e["sens"])) * 1.18)
    fig.tight_layout()
    path = os.path.join(OUT, "layer4_h_sensitivity.png")
    fig.savefig(path, dpi=140)
    print(f"figure -> {os.path.normpath(path)}")
    print(f"  BTC {b['sens'][0]:.1f} -> {b['sens'][-1]:.1f}  "
          f"({b['sens'][0] / b['sens'][-1]:.1f}x fall over {b['months'][0]:.2f}-{b['months'][-1]:.1f} mo)")
    print(f"  ETH {e['sens'][0]:.1f} -> {e['sens'][-1]:.1f}  "
          f"({e['sens'][0] / e['sens'][-1]:.1f}x fall over {e['months'][0]:.2f}-{e['months'][-1]:.1f} mo)")


if __name__ == "__main__":
    main()
