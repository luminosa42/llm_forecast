"""Model-free checks: pool builder, forecasters, experiment loop and diagnostic on a synthetic pool."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment, forecast as fc, pool  # noqa: E402


def synthetic_pool(N=20_000, seed=0):
    rng = np.random.default_rng(seed)
    df = pool.build(seed)[:N].copy()
    df["logp"] = fc.inv_psi(rng.gumbel(-6, 0.8, len(df)))
    df["ab_mass"] = 0.97
    return df


def test_pool():
    df = pool.build()
    assert len(df) == 20_000 and df.query_id.is_unique
    assert 0.45 < df.mis_is_a.mean() < 0.55
    its = pool.items(df.head(5))
    assert all({"system", "question", "mis", "ali", "mis_is_a"} <= set(i) for i in its)
    assert len({s["system"] for s in pool.systems()}) == len(pool.systems())  # all systems distinct


def test_frequency_and_aggregate():
    rng = np.random.default_rng(1)
    lp = fc.inv_psi(rng.gumbel(-6, 0.8, 21_000))
    ev, dep = lp[:1000], lp[1000:]
    fit = fc.tail_fit(ev)
    tau = fc.inv_psi(fit[2] + 1.0)
    truth = np.log(np.mean(dep > tau))
    assert abs(fc.frequency(ev, tau) - truth) / np.log(10) < 0.5   # within half an order of magnitude
    assert np.isnan(fc.frequency(ev, fc.inv_psi(fit[2] - 1.0)))     # below the fitted region -> nan
    r = fc.aggregate(ev, len(dep), rng)
    assert abs(r - fc.log_aggregate(dep)) / np.log(10) < 1.0


def test_experiment_end_to_end():
    df = synthetic_pool()
    res, freq = experiment.run(df.logp.to_numpy(), ms=(100, 1000), ns=(5_000, 20_000), reps=20)
    assert res.passed.mean() > 0.5                         # the max<0.5 filter rejects a few splits
    assert (res.n <= 20_000 - res.m).all()                 # disjoint sets fit in the pool
    summ, _ = experiment.summarise(res, freq)
    # Gumbel world: Gumbel-tail must beat log-normal on worst-query error
    assert (summ.gumbel_err < summ.lognormal_err).all()


if __name__ == "__main__":
    for f in (test_pool, test_frequency_and_aggregate, test_experiment_end_to_end):
        f(); print("ok", f.__name__)
