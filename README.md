# How Much Can We Trust Tail-Risk Forecasts for LLM Failures? Uncertainty and Clustering in Extreme-Value Forecasting of Rare Model Behaviors

Rare model failures are invisible in small tests but can still become near-certain once a model handles millions of queries. Jones et al. (2025) showed that extreme-value statistics can forecast those deployment-scale risks from a modest evaluation. This project first replicates that method on a small open model using a harmless target behavior. It then tests two things the original paper leaves open: whether the forecasts can carry calibrated error bars, and how badly the usual assumption of independent queries breaks when real traffic arrives in clusters, such as the same system prompt or attack template. The aim is a tail-risk forecast that reports its own uncertainty and holds up under clustered traffic. That is the same accumulation problem that drives catastrophe modeling, and it gives the frequency inputs a later AI-enabled cyber risk catalog would need.

### Current: RQ1 replication (Gumbel-tail vs log-normal)

Run insturctions:

    pip install -r requirements.txt
    python forecast.py                       # synthetic sanity test (no model needed)
    python tests/test_pipeline.py            # model-free pipeline tests
    python measure.py                        # smoke test on Qwen2.5-1.5B-Instruct (check letter bias, ab_mass)
    python pool.py                           # data/pool.csv (400 systems x 50 questions = 20k)
    python run_measure.py --bench 200        # timing; then
    python run_measure.py                    # full scoring, resumable -> data/pool_scored.csv
    python diagnostic.py --min-mass 0.9      # step 3 go/no-go gate, plots/diagnostic.png
    python experiment.py                     # paired splits -> results/

| File | Role |
|---|---|
| `forecast.py` | psi, Gumbel-tail, log-normal, frequency, aggregate forecasters; scoring; synthetic test |
| `measure.py` | log p(misaligned) from a local model |
| `pool.py` | template query pool (decisions 1-2: self-written questions, templates only) |
| `run_measure.py` | benchmark and checkpointed scoring of the pool |
| `diagnostic.py` | histograms, tail linearity, filter pass rate, ab_mass |
| `experiment.py` | split loop, both methods on the same splits, paired bootstrap |

Aggregate-risk forecaster is my own construction (spec section 3): verify against the paper's appendix.


### Extention

to be updated...
