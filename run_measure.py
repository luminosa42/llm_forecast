"""Score the pool with the model (RQ1, step 2). Saves logp and ab_mass once; never recompute.

    python run_measure.py --bench 200     # time 200 queries, print extrapolated full-pool time
    python run_measure.py                 # full run, resumable; writes data/pool_scored.csv
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pool as poolmod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/pool.csv")
    ap.add_argument("--out", default="data/pool_scored.csv")
    ap.add_argument("--bench", type=int, default=0, help="time this many queries and exit")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=800, help="queries per checkpoint file")
    a = ap.parse_args()

    import measure  # heavy imports (torch) only when actually scoring

    df = pd.read_csv(a.pool)
    its = poolmod.items(df)
    tok, model, ab_ids = measure.load()

    if a.bench:
        idx = np.random.default_rng(0).choice(len(df), a.bench, replace=False)
        t0 = time.time()
        lp, mass = measure.elicit(tok, model, ab_ids, [its[i] for i in idx], a.batch_size)
        dt = time.time() - t0
        print(f"{a.bench} queries in {dt:.1f}s = {a.bench / dt:.1f}/s; full pool ({len(df)}) ~ {len(df) / (a.bench / dt) / 3600:.2f} h")
        print(f"p(mis): min {np.exp(lp).min():.2e}  median {np.median(np.exp(lp)):.3f}  max {np.exp(lp).max():.3f}")
        print(f"ab_mass: min {mass.min():.3f}  median {np.median(mass):.3f}")
        return

    ck = Path(a.out).with_suffix(".chunks")
    ck.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(df), a.chunk):
        f = ck / f"{start:07d}.npz"
        if f.exists():
            continue
        t0 = time.time()
        lp, mass = measure.elicit(tok, model, ab_ids, its[start:start + a.chunk], a.batch_size)
        np.savez(f, logp=lp, ab_mass=mass)
        print(f"{start + len(lp)}/{len(df)}  {time.time() - t0:.0f}s/chunk", flush=True)

    parts = [np.load(ck / f"{s:07d}.npz") for s in range(0, len(df), a.chunk)]
    df["logp"] = np.concatenate([p["logp"] for p in parts])
    df["ab_mass"] = np.concatenate([p["ab_mass"] for p in parts])
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
