"""Step 3 go/no-go diagnostic on the scored pool (RQ1).

    python diagnostic.py [data/pool_scored.csv] [--min-mass 0.9]

Prints: logp/p summary, ab_mass and letter-bias check, tail linearity (R^2 of log survival vs psi for
the top 50 / 100 points), and the fraction of random splits passing the max p < 0.5 filter.
Saves plots/diagnostic.png.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecast import psi


def tail_r2(s_sorted_desc, top):
    """R^2 of log survival vs psi over the `top` largest points."""
    s = s_sorted_desc[:top]
    y = np.log(np.arange(1, top + 1) / len(s_sorted_desc))
    a, b = np.polyfit(s, y, 1)
    return 1 - np.sum((y - (a * s + b)) ** 2) / np.sum((y - y.mean()) ** 2), a


def filter_pass_rate(logp, m, reps, rng):
    ok = 0
    for _ in range(reps):
        ok += logp[rng.choice(len(logp), m, replace=False)].max() < np.log(0.5)
    return ok / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/pool_scored.csv")
    ap.add_argument("--min-mass", type=float, default=0.0, help="drop queries with ab_mass below this")
    a = ap.parse_args()

    df = pd.read_csv(a.path)
    n0 = len(df)
    df = df[df.ab_mass >= a.min_mass]
    print(f"{n0} queries; dropped {n0 - len(df)} with ab_mass < {a.min_mass}")
    lp = df.logp.to_numpy()
    p = np.exp(lp)
    s = np.sort(psi(lp))[::-1]

    print(f"\np: min {p.min():.2e}  median {np.median(p):.3g}  95% {np.quantile(p, .95):.3g}  max {p.max():.3f}")
    print(f"exact 0/1 probabilities: {(p == 0).sum()} / {(lp >= 0).sum()}")
    print(f"ab_mass: min {df.ab_mass.min():.3f}  median {df.ab_mass.median():.3f}  frac>0.9: {(df.ab_mass > .9).mean():.1%}")
    bias = df.groupby("mis_is_a").logp.apply(lambda x: np.exp(x).mean())
    print(f"mean p when mis is A: {bias.get(True, np.nan):.3f}  when mis is B: {bias.get(False, np.nan):.3f} (big gap = letter bias)")

    print("\ntail linearity (log survival vs psi):")
    for top in (50, 100):
        r2, slope = tail_r2(s, top)
        print(f"  top {top:>3}: R^2 = {r2:.3f}  slope = {slope:.2f}")

    rng = np.random.default_rng(0)
    print("\nfraction of random splits with eval max p < 0.5:")
    for m in (100, 500, 1000):
        print(f"  m={m:>4}: {filter_pass_rate(lp, m, 300, rng):.0%}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(p, bins=60); ax[0].set(title="p(misaligned)", xlabel="p", yscale="log")
    ax[1].hist(psi(lp), bins=60); ax[1].set(title="psi = -log(-log p)", xlabel="psi", yscale="log")
    top = max(int(0.05 * len(s)), 100)
    ax[2].scatter(s[:top], np.log(np.arange(1, top + 1) / len(s)), s=6)
    ax[2].set(title=f"log survival vs psi, top {top} (want ~linear)", xlabel="psi", ylabel="log survival")
    Path("plots").mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig("plots/diagnostic.png", dpi=130)
    print("\nsaved plots/diagnostic.png")


if __name__ == "__main__":
    main()
