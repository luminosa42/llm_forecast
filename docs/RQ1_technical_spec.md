# RQ1 Technical Spec: Replicating Gumbel-Tail Forecasting

**Question.** Does the Gumbel-tail method (Jones et al. 2025, arXiv:2502.16797) forecast tail risk better than a log-normal baseline on a small open model with a benign target behavior?

**Status.** Forecasters and the synthetic test are written and verified (`forecast.py`). The measurement function is written and syntax-checked but not yet run against the model (`measure.py`). The pool builder, the diagnostic, the experiment loop, and the frequency and aggregate forecasters are still to do.

---

## 1. Definitions

- **Query:** a (system prompt, two-option question) pair.
- **Elicitation probability p:** the model's probability of picking the misaligned option, renormalised over the two answer letters. One forward pass, no sampling.
- **Everything is stored and computed as log p** (natural log). Plain probabilities underflow or get clipped, which silently corrupts any method that uses the bulk of the distribution. The synthetic test caught exactly this bug in my first version.
- **ψ score:** ψ = −log(−log p). Monotone in p. Small p gives very negative ψ; p near 1 gives large positive ψ.

## 2. Methods compared

**Gumbel-tail (paper's method).**
1. Take the k = 10 largest ψ values from the evaluation set (size m).
2. The j-th largest has empirical survival j/m. Fit log(j/m) = a·ψ + b by ordinary least squares (a should be negative).
3. To forecast for n queries, solve log(1/n) = a·ψ + b for ψ, then convert back to p. That is the forecast of the worst-query probability, Q_p(n).

**Log-normal baseline.** Fit a normal to all evaluation ψ values (mean, standard deviation). The forecast is its upper 1/n quantile, converted back to p.

Both return `nan` for degenerate fits (tied top values, non-negative slope). These are counted and reported, not silently dropped.

## 3. Quantities forecast

| Quantity | Forecast | Truth (from the deployment set) | Status |
|---|---|---|---|
| Worst-query risk | Q_p(n) | max p over n deployment queries | implemented |
| Behavior frequency | S(τ) = exp(a·ψ(τ) + b) for a threshold τ above the evaluation's k-th largest value | fraction of deployment queries with p > τ (require at least about 5 hits) | to do |
| Aggregate risk | simulate n draws, bulk from the evaluation empirical distribution and tail from the fitted line; R = 1 − ∏(1 − pᵢ) | R computed from the deployment p's via `log(-expm1(logp))` for stability | to do; my construction, check against the paper's appendix |

## 4. Protocol

1. Build the pool (about 20k queries). Store: `query_id, system_id, question_id, domain, tone, mis_is_a, logp, ab_mass`. Keep `system_id` for the later dependence work (RQ3).
2. For each setting m ∈ {100, 500, 1000} and n ∈ {1k, 5k, 10k, ~20k}: draw disjoint evaluation and deployment sets at random, repeat for at least 100 splits.
3. **Filter (paper protocol):** forecast only if the evaluation maximum is below p = 0.5. Report how many splits pass.
4. Score every split with both methods on the **same** splits, so comparisons are paired.

## 5. Metrics and acceptance criteria

- **Mean absolute log10 error** between forecast and truth.
- **Underestimation rate** (fraction of splits where forecast < truth). Ideal is 50%. Underestimating is the unsafe direction.
- **Paired bootstrap** over splits for the difference in error between methods, with a 95% interval.
- **"Replicated" means direction, not exact numbers.** Gumbel-tail has lower error than log-normal, underestimates less often, and improves as m grows. A different model and behavior will not reproduce the paper's figures. If the direction fails, the diagnostics come before any method changes, and the failure is itself a result.

## 6. Verified so far: synthetic test (`python forecast.py`)

200 splits per cell, same filter as above.

| World | m | n | Gumbel-tail error / underest. | Log-normal error / underest. |
|---|---|---|---|---|
| Gumbel tail | 100 | 5,000 | 0.26 / 38% | 3.01 / 100% |
| Gumbel tail | 1000 | 20,000 | 0.10 / 60% | 2.07 / 100% |
| Normal ψ | 100 | 5,000 | 0.94 / 19% | 0.74 / 66% |
| Normal ψ | 1000 | 20,000 | 0.43 / 31% | 0.24 / 62% |

The harness behaves correctly: each method does well in its own world, and the baseline wins when its assumption holds. This validates the code, not the LLM result.

## 7. Step 3 diagnostic (go/no-go gate)

Before running the experiment on real data:
- Histogram of p and of ψ.
- Log survival against ψ for the top 5% of queries; the top 50–100 points should be roughly linear.
- Check what fraction of splits pass the max < 0.5 filter. If most fail, the behavior is too common for this method; change the question set or the model.
- Check `ab_mass` (want above about 0.9). Decide before splitting whether to drop low-mass queries, and report how many were dropped.

## 8. Measurement details (`measure.py`)

- Chat template, then the fixed assistant prefix `My answer is option (`, then read the next-token logits for `A` and `B`.
- log p(misaligned) = log-sigmoid of (logit_mis − logit_ali), computed in float64.
- **Randomise which letter holds the misaligned option** per query. Small models have strong letter bias that otherwise swamps the signal. The smoke test prints the bias gap.
- Cut the prompt at `(` and check that `A` and `B` carry most of the probability mass; if the tokenizer merges `(A` into one token, mass will be low.
- Run the smoke test first (`python measure.py`), then time 200 queries before committing to the full pool. Save all log-probabilities to disk once; never recompute.

## 9. Pitfalls

- Reusing queries across evaluation and deployment sets.
- Fitting on probabilities rather than log-probabilities.
- Probabilities of exactly 0 or 1 (guarded in `psi`, but check the pool for them).
- Choosing k after seeing results. Fix k = 10 for RQ1; the k sweep belongs to RQ2.
- Small-model tails may be too fat or too thin for the method. The Step 3 gate exists to catch this early.

## 10. Schedule (rough)

| Days | Work |
|---|---|
| 1 | Step 0 done. Run the smoke test. |
| 2–3 | Build the pool, benchmark, run the full measurement. |
| 3 | Step 3 diagnostic (gate). |
| 4–6 | Experiment loop; add frequency and aggregate forecasters. |
| 6–7 | Scoring, plots, short results note. |

## 11. Open decisions

1. Question source: public model-written evals or a self-written set of about 100 two-option questions.
2. Whether to generate system prompts with an LLM API or from templates only.
3. Exact aggregate-risk construction (see section 3).

## Files

- `forecast.py`: forecasters, scoring, synthetic test
- `measure.py`: elicitation log-probability from a local model
