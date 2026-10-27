# CP4.7.0 Evaluation and Predictor Results

## Evaluator version and isolation

CP4.7.0 evaluates the frozen generation cohort with
`gemini-3.5-flash-lite`. The version was selected after the persisted
Gemini 2.5 Flash-Lite retirement response suggested this model. The evidence
instructions, structured response schema, and RELIABLE / UNRELIABLE /
INADEQUATE definitions were retained. Model and prompt provenance are recorded
as `gemini-3.5-flash-lite` / `CP4.7.0`.

Results are isolated in
`data/production/cp4_7_0_gemini_3_5_flash_lite/`. They are not merged with
CP4.5.0 or any earlier evaluator version. Evaluation reused the 310 existing
Qwen3-1.7B generations and their recorded prompts; it did not regenerate
answers or hidden states.

## Evaluation outcome

The resumable run completed all 310 generation IDs. The latest version-matched
record for every generation is `SUCCESS`:

| Outcome | Generations |
| --- | ---: |
| RELIABLE | 248 |
| UNRELIABLE | 61 |
| INADEQUATE | 1 |

The INADEQUATE result is not a binary correctness label. The run history
contains 320 API attempts across 311 evaluation iterations. One iteration
failed with a 429 quota error on credential slot `GEMINI_API_KEY_10`; the
failed record remained unlabeled and was later successfully retried, with
attempt history preserved. The successful resume took 982.36 seconds. The
worker is no longer running. No credentials or key values are written in the
report.

The full run log is
[`evaluation_run.json`](../data/production/cp4_7_0_gemini_3_5_flash_lite/evaluation_run.json).
Individual records are under
[`evaluations/`](../data/production/cp4_7_0_gemini_3_5_flash_lite/evaluations/).

## Empirical-risk targets

A target is emitted only when all ten generations for a prompt have resolved
binary RELIABLE / UNRELIABLE labels. The target is the number of UNRELIABLE
generations divided by ten. INADEQUATE, failed, manual-review, and missing
records are excluded; they never become binary labels.

Thirty of 31 prompt groups had complete targets; one group was excluded due to
its INADEQUATE outcome. The 30 target values have mean 0.2033 and range
0.0–1.0:

| Risk | Prompt groups |
| ---: | ---: |
| 0.0 | 22 |
| 0.3 | 1 |
| 0.6 | 1 |
| 0.7 | 2 |
| 0.8 | 1 |
| 1.0 | 3 |

The target file is
[`targets.json`](../data/production/cp4_7_0_gemini_3_5_flash_lite/targets.json).

## Predictor comparison

All three baselines use the same prompt-identity split: 21 training prompts
and 9 test prompts. Feature transformations, including hidden-state PCA, are
fitted using training prompts only. Generated answers and evaluator outputs
are not predictor features. The test distribution is notably shifted from
the training distribution, and all three test-set R² values are negative.

| Baseline | Test MAE | Test RMSE | R² | MAE change vs. train-mean baseline |
| --- | ---: | ---: | ---: | ---: |
| Prompt-only | 0.4355 | 0.5345 | -0.5853 | +0.0095 |
| Hidden-state-only | 0.4387 | 0.5379 | -0.6050 | +0.0063 |
| Fusion | 0.4519 | 0.5311 | -0.5646 | -0.0069 |

The training-mean baseline test MAE is 0.4450. Paired bootstrap 95% intervals
for the MAE changes are:

| Baseline | 95% interval |
| --- | --- |
| Prompt-only | [-0.0229, 0.0364] |
| Hidden-state-only | [-0.0745, 0.0739] |
| Fusion | [-0.1128, 0.0755] |

Every interval includes zero. With only nine held-out prompts and a
model-judged target, these estimates are descriptive and unstable. Prompt-only
and hidden-state-only have small point improvements, but neither is supported
by the uncertainty interval; fusion has a worse point MAE than the baseline.
This experiment does not demonstrate predictive improvement or validate the
research hypothesis.

Detailed metrics, split IDs, distributions, and bootstrap intervals are in
[`comparison_results.json`](../data/production/cp4_7_0_gemini_3_5_flash_lite/predictors/comparison_results.json).
