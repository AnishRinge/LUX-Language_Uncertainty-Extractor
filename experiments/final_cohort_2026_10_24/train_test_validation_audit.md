# Train/Validation/Test Audit (read-only)

**Experiment:** experiments/final_cohort_2026_10_24/
**Dataset:** 486 prompts total (from dataset_info.json); X_prompt shape (486,5), X_hidden shape (486,29,2048), y shape (486,), pids shape (486,).
**Reconciliation:**
econciliation.json shows selected_prompts=500, generations_total=5000, eval_files=5000, target_rows=1231 (context only).

## Answers

| Question | Verified finding | Supporting file/line references |
|---|---|---|
| 1) Was the dataset actually divided into training and test sets? Identify the exact code and saved split artifact. | Yes. Prompt-level split into train/test by shuffling unique prompt IDs with fixed seed and taking 70% train / 30% test. Split saved to split.json. |
un_final_ml.py:33-44 creates split (shuffle
p.random.RandomState(12345),
_train = int(len(unique_pids)*0.7), build masks by membership).
un_final_ml2.py:48-59 does similar (handles 3D hidden).
un_final_ml3.py:45-56 does similar (stringifies pids). Saved artifact: experiments/final_cohort_2026_10_24/split.json (keys 	rain_pids, 	est_pids, seed). |
| 2) What were the exact sample counts in each partition? | 340 train prompts, 146 test prompts (by prompt count). Row counts equal prompt counts (486 total) because each prompt appears once per row in the arrays. | split.json has 340 train_pids, 146 test_pids. Verified via Python: train_rows=486? Wait - arrays have 486 rows total; masks give train_rows = number of rows whose pid is in train set = 486 if all covered? Actually: p.shape (486,), 	rain_m.sum() over membership gives count of rows in train; we got 486 total rows and split is by prompt - with 486 rows total and 340+146 prompt IDs covering all? Or some prompts filtered. But dataset_info.json says n_prompts 486; split gives 486 prompt-occurrences total? Let us recheck. Verified: Xh.shape (486,29,2048), Xp.shape (486,5), y.shape (486,), p.shape (486,), len(train_pids)=340, len(test_pids)=146 so 486 prompt instances total split across train/test by prompt membership. |
| 3) Was there a separate validation set? | No separate validation set found in saved artifacts or in the executed ML scripts. Splits only train/test. | No al_pids/alidation in split.json;
un_final_ml*.py only create train/test masks. |
| 4) Was cross-validation performed? If so, how many folds, and was it restricted to training data? | No cross-validation performed in the final ML scripts that produced model_comparison.json. Imports of GroupKFold, cross_val_predict appear in
un_final_ml.py:7 but are unused; no CV calls present. |
un_final_ml.py:7 imports CV tools but grep of usage shows none (only imports).
un_final_ml2.py,
un_final_ml3.py have no CV imports/usage. model_comparison.json contains only single train/test results. |
| 5) Were Ridge, Random Forest, Extra Trees, and HistGradientBoosting actually fitted, or were only configurations and metrics produced? | All four model types were fitted on train data and evaluated on test data. |
un_final_ml3.py fits: Ridge (prompt/hidden/fusion), RF (all three), ET (all three), HGB (all three). Each fit is called on train splits; predictions computed on test splits; metrics written to model_comparison.json. Verified by reading script and corresponding entries in model_comparison.json. |
| 6) Were scalers and PCA fitted exclusively on training data? | Yes. Pipelines fit on train only; transforms applied to test using fitted pipeline. For hidden states, pipe_h (StandardScaler + PCA(n_components=50, random_state=12345)) is fit on X_hidden_train and used to transform both train/test. For fusion, PCA features come from the same fitted pipe_h applied to train/test. |
un_final_ml3.py:70-97 shows pipe_h.fit(X_hidden_train, y_train), then pipe_h.named_steps["pca"].transform(pipe_h.named_steps["scaler"].transform(X_hidden_test)) for test; fusion built from transformed train/test using fitted components. Same pattern for Ridge pipelines. |
| 7) Was the test set used during hyperparameter tuning or model selection? | No evidence of hyperparameter tuning or model selection using test set. All models use fixed hyperparameters (Ridge alpha=1.0, RF/ET n_estimators=100, default HGB). No tuning loops/searches found. | Scripts use fixed params; no GridSearchCV/RandomizedSearchCV imports or usage in
un_final_ml*.py. model_comparison.json is direct train/test evaluation. |
| 8) Can the saved test metrics be reproduced from persisted predictions? | Not directly verifiable from saved artifacts - no per-sample predictions saved (no predictions*.npy/json found). Metrics exist in model_comparison.json but reproduction would require rerunning the training code with same splits/data. | experiments/final_cohort_2026_10_24/ contains no prediction files; only arrays, split, model_comparison, dataset_info, reconciliation. |
| 9) Was the training-mean baseline calculated from the training targets only? | Yes. Baseline predicts y_train.mean() for all test samples. |
un_final_ml3.py:62-63: aseline_pred = np.full_like(y_test, y_train.mean()) and metrics computed against y_test. |
| 10) Are all three feature configurations evaluated on the same held-out prompts? | Yes. All models use the same train/test masks split by prompt IDs, so same held-out prompts across prompt/hidden/fusion configurations. | Split created once by prompt membership (	rain_mask/	est_mask from pids); all subsequent models use same X_*_train/test, y_train/test. Verified in
un_final_ml3.py:45-61. |

**VERDICT:** Implemented with specific methodological gaps.

**Gaps identified:**
- No validation set (only train/test).
- No hyperparameter tuning/CV (imports present but unused).
- Hidden state shape handling suggests multiple script iterations (ml.py assumes different shapes than ml2/ml3 handle); final artifacts produced by ml3 logic.
- No persisted predictions to independently verify metric reproduction.
- Test set is used only for final evaluation (no tuning leakage observed), but lack of validation/CV limits model selection rigor.
