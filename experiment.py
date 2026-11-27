"""RQ1 experiment loop: paired Gumbel-tail vs log-normal on random disjoint eval/deployment splits.

    python experiment.py [data/pool_scored.csv] [--reps 100] [--min-mass 0.0]

For each m in {100, 500, 1000} and n in {1k, 5k, 10k, ~20k}: draw `reps` disjoint (eval, deployment)
splits, apply the paper's filter (eval max p < 0.5), score both methods on the same splits.
n is capped at (pool size - m); the actual n is recorded. Writes results/*.csv and prints summaries.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import forecast as fc

LN10 = np.log(10)
FREQ_OFFSETS = (0.5, 1.0, 1.5, 2.0, 3.0)   # threshold psi = eval k-th largest psi + offset
MIN_HITS = 5


def run(logp, ms=(100, 500, 1000), ns=(1_000, 5_000, 10_000, 20_000), reps=100, k=10, seed=0):
    rng = np.random.default_rng(seed)
    N = len(logp)
    rows, freq_rows = [], []
    for m in ms:
        for n_req in ns:
            n = min(n_req, N - m)
            for split in range(reps):
                perm = rng.permutation(N)           # disjoint by construction
                ev, dep = logp[perm[:m]], logp[perm[m:m + n]]
                passed = ev.max() < np.log(0.5)
                row = dict(m=m, n_req=n_req, n=n, split=split, passed=passed)
                if passed:
                    row.update(
                        truth=dep.max(),
                        gumbel=fc.gumbel_tail(ev, n, k),
                        lognormal=fc.lognormal(ev, n),
                        agg_truth=fc.log_aggregate(dep),
                        agg_gumbel=fc.aggregate(ev, n, rng, k),
                        agg_lognormal=fc.aggregate_lognormal(ev, n, rng),
                    )
                    fit = fc.tail_fit(ev, k)
                    if fit is not None:
                        for d in FREQ_OFFSETS:
                            tau = fc.inv_psi(fit[2] + d)
                            hits = int(np.sum(dep > tau))
                            if hits >= MIN_HITS:
                                freq_rows.append(dict(m=m, n_req=n_req, n=n, split=split, offset=d,
                                                      hits=hits, truth=np.log(hits / n),
                                                      forecast=float(fc.frequency(ev, tau, k))))
                rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(freq_rows)


def boot_ci(x, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    means = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return np.quantile(means, [0.025, 0.975])


def summarise(res, freq):
    out = []
    for (m, n_req), g in res.groupby(["m", "n_req"]):
        r = dict(m=m, n_req=n_req, n=g.n.iloc[0], splits=len(g), passed=int(g.passed.sum()))
        p = g[g.passed]
        if len(p):
            ge = (p.gumbel - p.truth) / LN10
            le = (p.lognormal - p.truth) / LN10
            r.update(gumbel_nan=int(p.gumbel.isna().sum()),
                     gumbel_err=ge.abs().mean(), gumbel_under=(ge < 0).mean(),
                     lognormal_err=le.abs().mean(), lognormal_under=(le < 0).mean())
            both = p.dropna(subset=["gumbel", "lognormal"])
            if len(both) > 1:
                d = ((both.gumbel - both.truth).abs() - (both.lognormal - both.truth).abs()) / LN10
                lo, hi = boot_ci(d)
                r.update(paired_diff=d.mean(), diff_lo=lo, diff_hi=hi)  # negative = Gumbel better
            for name in ("gumbel", "lognormal"):
                a = (p["agg_" + name] - p.agg_truth) / LN10
                r[f"agg_{name}_err"] = a.abs().mean()
        out.append(r)
    freq_out = []
    if len(freq):
        freq = freq.assign(err=(freq.forecast - freq.truth) / LN10)
        for (m, n_req), g in freq.groupby(["m", "n_req"]):
            freq_out.append(dict(m=m, n_req=n_req, points=len(g), err=g.err.abs().mean(), under=(g.err < 0).mean()))
    return pd.DataFrame(out), pd.DataFrame(freq_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/pool_scored.csv")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--k", type=int, default=10, help="fixed at 10 for RQ1; the k sweep belongs to RQ2")
    ap.add_argument("--min-mass", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    df = pd.read_csv(a.path)
    n0 = len(df)
    df = df[df.ab_mass >= a.min_mass]
    print(f"pool {n0}; dropped {n0 - len(df)} with ab_mass < {a.min_mass}; using {len(df)}")
    res, freq = run(df.logp.to_numpy(), reps=a.reps, k=a.k, seed=a.seed)
    summ, fsumm = summarise(res, freq)

    out = Path(a.outdir); out.mkdir(exist_ok=True)
    res.to_csv(out / "splits.csv", index=False); freq.to_csv(out / "frequency_points.csv", index=False)
    summ.to_csv(out / "summary_worst_query.csv", index=False); fsumm.to_csv(out / "summary_frequency.csv", index=False)
    pd.set_option("display.width", 200, "display.float_format", "{:.2f}".format)
    print("\nWorst-query risk (abs log10 error; under = underestimate rate; diff<0 means Gumbel better):")
    print(summ.to_string(index=False))
    print("\nFrequency (Gumbel only; points with >= %d deployment hits):" % MIN_HITS)
    print(fsumm.to_string(index=False) if len(fsumm) else "  no points")


if __name__ == "__main__":
    main()
