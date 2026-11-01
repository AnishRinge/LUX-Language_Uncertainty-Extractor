# Final Report

## Executive Summary

LUX completed a rigorous research pipeline for estimating answer unreliability before generation. The project established a validated SQuAD v2 canonical prompt dataset, preserved dataset integrity, and built a careful production evaluation workflow. The finished benchmark used 31 compatible prompts and ten Qwen3 1.7B generations per prompt. The resulting data set provides a stable foundation for analysis, but the evidence did not support a trustworthy individualized predictor.

The project validated the canonical dataset at 130,319 normalized scenarios, including 86,821 answerable and 43,498 unanswerable rows. The repository also includes a strict audit that confirmed structural integrity and leakage safeguards. These were important engineering outcomes. They created a reproducible base for model generation and evaluation without mixing future labels into the input data.

## Completed Research and Evidence

The production cohort run used a frozen Qwen3 1.7B configuration and a fixed Gemini 3.5 Flash Lite evaluator. Across the completed generation set, the evaluator produced 248 reliable labels, 61 unreliable labels, and 1 inadequate label. Among the eligible prompt groups, 30 had complete binary targets. The empirical target mean was 0.2033, the range was 0.0 to 1.0, and 22 prompt groups landed at zero risk. This means the cohort was dominated by low observed unreliability, which placed a high burden on any learned predictor.

The final model comparison followed a common prompt-level split and only used question, context, and hidden-state features as predictors. Generated answers and evaluator outputs were not used as predictors. The best learned model was a prompt-only support vector regressor with a test mean absolute error of 0.2142. The median baseline produced a test mean absolute error of 0.2086. The hidden-state and fusion models did not consistently outperform the simple baseline, and their bootstrap confidence intervals included zero. The evidence therefore describes a small empirical signal at best, not a proven deployment-ready risk estimator.

## Engineering Outcome

The repository also includes a strict engineering interface for risk inference. It exposes cohort mean and median baselines, validates prompt inputs, records provenance, and routes uncertain or missing estimates to escalation. This interface is conservative by design. It is designed for auditability and safe failure behavior, not for claiming individualized calibration or downstream reliability gains.

The project delivered a production-like research artifact that is explicit about what it can and cannot establish. It is valuable as an inspectable experimental foundation and as a measured negative result. The completed evidence supports the conclusion that the LUX project created a sound dataset foundation, a reproducible evaluation pipeline, and a careful baseline-based inference layer, while showing that a dependable learned pre-generation hallucination risk predictor was not demonstrated in this completed benchmark.

## Final Assessment

The strongest final conclusion is practical and grounded in the saved results. LUX succeeded in building the dataset, the evaluation protocol, the comparison pipeline, and the conservative inference interface. It also succeeded in producing a careful negative result. The completed empirical analysis did not show that prompt-only, hidden-state-only, or fusion models reliably improve on simple baseline estimates for prompt-level unreliability risk. The repository therefore stands as a credible research artifact and a disciplined engineering baseline, rather than as a validated factory-grade hallucination detector.

This is the final status of the project as represented by the repository evidence.
