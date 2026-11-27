# Project Scope: Forecasting Rare LLM Failures, With Honest Error Bars

*Based on Jones et al. (2025), "Forecasting Rare Language Model Behaviors" (arXiv:2502.16797). Draft scope, October 2026.*

## 1. Summary

Replicate the paper's forecasting method on a small local model using a harmless target behavior. Then extend it in two ways the paper leaves open: **(a)** confidence intervals for the forecasts, and **(b)** what happens when deployment queries arrive in correlated clusters instead of independently.

## 2. The paper in plain language

- Sampling a model once per query hides rare failures. A query's **elicitation probability** is the chance that one sampled response shows the behavior.
- At deployment scale, risk is driven by the *top tail* of these probabilities across queries.
- **Gumbel-tail method:** transform each probability as ψ = −log(−log p), fit a straight line to the log survival of the ten largest ψ values, and extrapolate to larger query counts.
- It forecasts three quantities: worst-query risk, behavior frequency above a threshold, and aggregate risk (1 − ∏(1 − pᵢ)).
- On worst-query risk for misuse behaviors it beat a log-normal baseline (mean absolute log error 1.7 vs 2.4) and underestimated far less often (34% vs 72% of the time). Forecasts reach up to about three orders of magnitude beyond the evaluation size.
- Gaps the authors name themselves: no uncertainty estimates, no adaptive check of which extreme-value family fits, no study of distribution shift, and forecasts that are sensitive to the specific top-ten values.

## 3. Research questions

| # | Question | Hypothesis |
|---|----------|------------|
| RQ1 | **Replication.** Does Gumbel-tail beat a log-normal baseline on a small open model with a benign behavior? | Yes on absolute log error, and it underestimates less often. |
| RQ2 | **Uncertainty.** Can we attach calibrated intervals to the forecasts? Does a free-shape tail fit (GPD) change the picture? | Naive intervals under-cover; a bootstrap over the evaluation set gets close to nominal coverage. A free shape parameter shows where the Gumbel assumption bends. |
| RQ3 | **Dependence.** If deployment queries come in clusters (shared system prompt, domain, template), what breaks? | Point forecasts roughly hold, but interval coverage collapses under independence. A cluster-level bootstrap restores it. |

RQ3 is the main novel contribution. It is the same accumulation/clustering effect that inflates tail losses in catastrophe modeling.

## 4. Design

| Component | Choice |
|-----------|--------|
| Model | Qwen2.5-1.5B-Instruct, already running locally (CPU, float32) |
| Primary behavior | Misaligned-action multiple choice (power-seeking, self-preservation) under many generated system prompts, in the style of the paper's Section 5. Elicitation probability is the model's probability of the misaligned option: one forward pass, no sampling cost. |
| Question source | Public Anthropic model-written evals (confirm availability and licence) or a small self-written set |
| Stretch behavior | Canary-string leak under injected instructions: a cyber-flavored setting that links to the cyber peril-catalog project |
| Query pool | About 20–50k (system prompt × question) pairs built from templates plus LLM paraphrase. Hold out 10–20k as ground-truth deployment sets. |
| Protocol | As in the paper: split into non-overlapping evaluation (m = 100, 500, 1000) and deployment (n up to about 10–20k) sets. Forecast only when the evaluation maximum is below 0.5. |
| Metrics | Mean absolute log10 error, underestimation rate (ideal is 50%), and, new here, interval coverage and width |
| Scope limit | Benign behaviors only. The paper's chem/bio misuse setting is **not** replicated. |

## 5. Phases (rough, part-time)

| Phase | Work | Output | Gate before moving on |
|-------|------|--------|-----------------------|
| 0. Feasibility (3–4 days) | Benchmark forward-pass throughput. Build the query pool. Plot log survival against ψ. Quick literature check. | Pool, diagnostic plot | Is the tail roughly linear? If not, change behavior or model. |
| 1. Replicate (~1 week) | Gumbel-tail vs log-normal on all three quantities | Figure and table | Gumbel-tail at least matches log-normal. If not, investigate first; that is a finding. |
| 2. Uncertainty (~1 week) | Top-k sweep (5, 10, 20, 50). GPD peaks-over-threshold with free shape. Bootstrap intervals. Coverage over many partitions. | Coverage plot | Report under-coverage honestly if it appears. |
| 3. Dependence (~1 week) | Build deployments by cluster sampling. Compare independent bootstrap vs cluster bootstrap. | Coverage under clustering | Test the RQ3 hypothesis either way. |
| 4. Writeup (~1 week) | 4–6 pages, README, reproducible repo | Final deliverable | |

Total: about 4–5 weeks.

## 6. Positioning and portfolio link

- Jones et al. list uncertainty estimates and distribution shift as future work.
- Angell et al. (ICML 2026, arXiv:2604.22167) improve per-query probability estimates using importance sampling with steered "unsafe" proposal models. From the main text, they still treat deployment queries as independent and do not calibrate intervals on the extrapolation. Do a proper literature check in Phase 0 (including Dorman et al. 2026 and Wu & Hilton 2024, arXiv:2410.13211) before claiming novelty.
- **Bridge to the cyber peril-catalog project:** the outputs here (tail-frequency curve, calibrated interval, clustering correction) are the frequency-side inputs a per-peril catalog would need.

## 7. Risks

| Risk | Mitigation |
|------|------------|
| A small model's tail may be too fat or too thin for extreme-value extrapolation | Phase 0 diagnostic gate; switch behavior or model if it fails |
| Ground truth reaches only 1–2 orders of magnitude beyond evaluation size (the paper reaches ~3) | State the limit plainly; do not claim larger extrapolation |
| Synthetic query pool may not resemble real traffic | Caveat in the writeup; no real-world claims |
| Single model and single behavior | Add a second model or the canary behavior if time allows |
| Overlap with concurrent work | Phase 0 literature check |
| CPU throughput | Batch the forward passes; cap the pool at about 20k if needed |

## 8. Open decisions

1. Which behavior first: misaligned-action multiple choice (cheaper, closer to the paper) or canary-leak (closer to cyber work)?
2. How much query generation to run through an LLM API vs local templates?
3. Writeup target: short fellowship-portfolio piece, or a fuller paper-style report?

## References

- Jones, Tong, Mu, Mahfoud, Leike, Grosse, Kaplan, Fithian, Perez, Sharma. *Forecasting Rare Language Model Behaviors.* arXiv:2502.16797, 2025.
- Angell, Singhal, Horvitz, Yu, Ranganath, McKeown, He. *Estimating Tail Risks in Language Model Output Distributions.* ICML 2026, arXiv:2604.22167.
- Wu & Hilton. *Estimating the Probabilities of Rare Outputs in Language Models.* arXiv:2410.13211, 2024.
