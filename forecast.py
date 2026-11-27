"""Tail forecasters from Jones et al. 2025 plus a synthetic sanity test (RQ1, step 0).

Everything works on natural-log probabilities (logp), not p: tiny probabilities
underflow or get clipped otherwise, which silently wrecks the bulk-fitting baseline.
"""
import numpy as np
from scipy.stats import norm

LN10 = np.log(10)


def psi(logp):
    """Elicitation score -log(-log p), computed from logp. Monotone in p."""
    return -np.log(-np.minimum(logp, -1e-15))


def inv_psi(s):
    """Back to logp."""
    return -np.exp(-s)


def gumbel_tail(logp_eval, n, k=10):
    """Forecast log Q_p(n), the 1/n-th largest elicitation probability.
    OLS of log survival on psi over the k largest eval scores, extrapolated to n."""
    s = np.sort(psi(logp_eval))[::-1][:k]
    if np.ptp(s) == 0:
        return np.nan
    a, b = np.polyfit(s, np.log(np.arange(1, k + 1) / len(logp_eval)), 1)
    return inv_psi((-np.log(n) - b) / a) if a < 0 else np.nan


def lognormal(logp_eval, n):
    """Baseline: psi ~ Normal; forecast is its 1/n upper quantile."""
    s = psi(logp_eval)
    return inv_psi(s.mean() + s.std(ddof=1) * norm.isf(1 / n))


def tail_fit(logp_eval, k=10):
    """Shared Gumbel-tail line fit. Returns (a, b, psi_k) or None if degenerate."""
    s = np.sort(psi(logp_eval))[::-1][:k]
    if np.ptp(s) == 0:
        return None
    a, b = np.polyfit(s, np.log(np.arange(1, k + 1) / len(logp_eval)), 1)
    return (a, b, s[-1]) if a < 0 else None


def frequency(logp_eval, tau_logp, k=10):
    """Forecast log S(tau): fraction of deployment queries with logp > tau_logp.
    Only meaningful when psi(tau) is above the eval k-th largest psi; otherwise nan."""
    fit = tail_fit(logp_eval, k)
    if fit is None:
        return np.nan
    a, b, psi_k = fit
    st = psi(np.asarray(tau_logp, dtype=float))
    return np.where(st > psi_k, a * st + b, np.nan)


def log_aggregate(logp):
    """log R, R = 1 - prod(1 - p_i), stable for tiny p (log(1-p) via expm1)."""
    L = np.sum(np.log(-np.expm1(logp)))
    return np.log(-np.expm1(L)) if L < 0 else -np.inf


def aggregate(logp_eval, n, rng, k=10):
    """Forecast log R for n queries. NOTE: my construction, check against the paper's appendix.
    Simulate n draws: with probability S(psi_k) a draw comes from the fitted line above psi_k,
    otherwise it is resampled from the eval scores below the top k (the bulk)."""
    fit = tail_fit(logp_eval, k)
    if fit is None:
        return np.nan
    a, b, psi_k = fit
    s_k = min(np.exp(a * psi_k + b), 1.0)
    s_eval = np.sort(psi(logp_eval))[::-1]
    bulk = s_eval[k:]
    in_tail = rng.random(n) < s_k
    tail = (np.log(rng.random(n) * s_k) - b) / a  # inverse of S(psi) = exp(a psi + b)
    draws = np.where(in_tail, tail, rng.choice(bulk, n))
    return log_aggregate(inv_psi(draws))


def aggregate_lognormal(logp_eval, n, rng):
    """Baseline for aggregate risk: n draws with psi ~ Normal fitted to all eval scores."""
    s = psi(logp_eval)
    return log_aggregate(inv_psi(rng.normal(s.mean(), s.std(ddof=1), n)))


def score(truth, fc):
    """truth, fc are logp. Returns (mean abs log10 error, underestimate rate, n used)."""
    ok = ~np.isnan(fc)
    t, f = truth[ok], fc[ok]
    return np.mean(np.abs(f - t)) / LN10, np.mean(f < t), ok.sum()


def simulate(draw, m, n, reps, rng):
    """draw(rng, N) -> N psi-scores. Forecast only if eval max p < 0.5 (paper protocol)."""
    rows = []
    while len(rows) < reps:
        lp = inv_psi(draw(rng, m + n))
        ev, dep = lp[:m], lp[m:]
        if ev.max() < np.log(0.5):
            rows.append((dep.max(), gumbel_tail(ev, n), lognormal(ev, n)))
    t, g, l = np.array(rows).T
    return score(t, g), score(t, l)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    worlds = {
        "Gumbel-tail world": lambda r, N: r.gumbel(-6, 0.8, N),
        "Normal-psi world": lambda r, N: r.normal(-8, 2, N),
    }
    for name, draw in worlds.items():
        print(f"\n{name}  (abs log10 error | underestimate rate)")
        print(f"{'m':>5} {'n':>6} | {'gumbel-tail':>18} | {'log-normal':>18}")
        for m in (100, 500, 1000):
            for n in (5_000, 20_000):
                (ge, gu, _), (le, lu, _) = simulate(draw, m, n, 200, rng)
                print(f"{m:>5} {n:>6} | {ge:8.2f} {gu:8.0%} | {le:8.2f} {lu:8.0%}")
