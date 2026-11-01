import json, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

src = Path("experiments/final_cohort_2026_10_24")
out = Path("experiments/cv_model_comparison_2026_10_24")
out.mkdir(parents=True, exist_ok=True)

X_prompt = np.load(src / "X_prompt.npy")
X_hidden = np.load(src / "X_hidden.npy")
y = np.load(src / "y.npy")
pids = np.load(src / "pids.npy")

if X_hidden.ndim == 3:
    X_hidden_pool = X_hidden.mean(axis=1)
else:
    X_hidden_pool = X_hidden.copy()

split = json.load(open(src / "split.json"))
train_pids_set = set(split["train_pids"])
test_pids_set = set(split["test_pids"])
seed = split.get("seed", 12345)

def pid_in(s, x):
    sx = str(x)
    if sx in s:
        return True
    if x in s:
        return True
    return False

train_mask = np.array([pid_in(train_pids_set, x) for x in pids])
test_mask = np.array([pid_in(test_pids_set, x) for x in pids])

X_prompt_train, X_prompt_test = X_prompt[train_mask], X_prompt[test_mask]
X_hidden_train, X_hidden_test = X_hidden_pool[train_mask], X_hidden_pool[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
pids_train, pids_test = pids[train_mask], pids[test_mask]

json.dump({"train_pids": split["train_pids"], "test_pids": split["test_pids"], "seed": seed}, open(out / "split.json", "w"))

gkf = GroupKFold(n_splits=5)
groups = pids_train
fold_membership = []
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_prompt_train, y_train, groups)):
    fold_membership.append({
        "fold": fold,
        "train_prompt_count": int(len(np.unique(groups[tr_idx]))),
        "val_prompt_count": int(len(np.unique(groups[val_idx]))),
        "train_row_count": int(len(tr_idx)),
        "val_row_count": int(len(val_idx)),
    })

cv_results = {}

# ridge prompt
maes = []
for tr_idx, val_idx in gkf.split(X_prompt_train, y_train, groups):
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(X_prompt_train[tr_idx], y_train[tr_idx])
    pred = pipe.predict(X_prompt_train[val_idx])
    maes.append(mean_absolute_error(y_train[val_idx], pred))
cv_results["ridge_prompt"] = {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)), "maes": [float(x) for x in maes]}

# ridge hidden
maes = []
for tr_idx, val_idx in gkf.split(X_prompt_train, y_train, groups):
    pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=seed)), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(X_hidden_train[tr_idx], y_train[tr_idx])
    pred = pipe.predict(X_hidden_train[val_idx])
    maes.append(mean_absolute_error(y_train[val_idx], pred))
cv_results["ridge_hidden"] = {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)), "maes": [float(x) for x in maes]}

# ridge fusion
maes = []
for tr_idx, val_idx in gkf.split(X_prompt_train, y_train, groups):
    pipe_h = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=seed))])
    pipe_h.fit(X_hidden_train[tr_idx], y_train[tr_idx])
    Zh_tr = pipe_h.transform(X_hidden_train[tr_idx])
    Zh_val = pipe_h.transform(X_hidden_train[val_idx])
    Xf_tr = np.concatenate([X_prompt_train[tr_idx], Zh_tr], axis=1)
    Xf_val = np.concatenate([X_prompt_train[val_idx], Zh_val], axis=1)
    pipe_f = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe_f.fit(Xf_tr, y_train[tr_idx])
    pred = pipe_f.predict(Xf_val)
    maes.append(mean_absolute_error(y_train[val_idx], pred))
cv_results["ridge_fusion"] = {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)), "maes": [float(x) for x in maes]}

# rf prompt
maes = []
for tr_idx, val_idx in gkf.split(X_prompt_train, y_train, groups):
    m = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    m.fit(X_prompt_train[tr_idx], y_train[tr_idx])
    pred = m.predict(X_prompt_train[val_idx])
    maes.append(mean_absolute_error(y_train[val_idx], pred))
cv_results["rf_prompt"] = {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)), "maes": [float(x) for x in maes]}

# rf hidden
maes = []
for tr_idx, val_idx in gkf.split(X_prompt_train, y_train, groups):
    pipe_h = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=seed))])
    pipe_h.fit(X_hidden_train[tr_idx], y_train[tr_idx])
    Zh_tr = pipe_h.transform(X_hidden_train[tr_idx])
    Zh_val = pipe_h.transform(X_hidden_train[val_idx])
    m = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    m.fit(Zh_tr, y_train[tr_idx])
    pred = m.predict(Zh_val)
    maes.append(mean_absolute_error(y_train[val_idx], pred))
cv_results["rf_hidden"] = {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)), "maes": [float(x) for x in maes]}

# rf fusion
maes = []
for tr_idx, val_idx in gkf.split(X_prompt_train, y_train, groups):
    pipe_h = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=seed))])
    pipe_h.fit(X_hidden_train[tr_idx], y_train[tr_idx])
    Zh_tr = pipe_h.transform(X_hidden_train[tr_idx])
    Zh_val = pipe_h.transform(X_hidden_train[val_idx])
    Xf_tr = np.concatenate([X_prompt_train[tr_idx], Zh_tr], axis=1)
    Xf_val = np.concatenate([X_prompt_train[val_idx], Zh_val], axis=1)
    m = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    m.fit(Xf_tr, y_train[tr_idx])
    pred = m.predict(Xf_val)
    maes.append(mean_absolute_error(y_train[val_idx], pred))
cv_results["rf_fusion"] = {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)), "maes": [float(x) for x in maes]}

# et prompt
maes = []
for tr_idx, val_idx in gkf.split(X_prompt_train, y_train, groups):
    m = ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    m.fit(X_prompt_train[tr_idx], y_train[tr_idx])
    pred = m.predict(X_prompt_train[val_idx])
    maes.append(mean_absolute_error(y_train[val_idx], pred))
cv_results["et_prompt"] = {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)), "maes": [float(x) for x in maes]}

# et hidden
maes = []
for tr_idx, val_idx in gkf.split(X_prompt_train, y_train, groups):
    pipe_h = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=seed))])
    pipe_h.fit(X_hidden_train[tr_idx], y_train[tr_idx])
    Zh_tr = pipe_h.transform(X_hidden_train[tr_idx])
    Zh_val = pipe_h.transform(X_hidden_train[val_idx])
    m = ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    m.fit(Zh_tr, y_train[tr_idx])
    pred = m.predict(Zh_val)
    maes.append(mean_absolute_error(y_train[val_idx], pred))
cv_results["et_hidden"] = {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)), "maes": [float(x) for x in maes]}

# et fusion
maes = []
for tr_idx, val_idx in gkf.split(X_prompt_train, y_train, groups):
    pipe_h = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=seed))])
    pipe_h.fit(X_hidden_train[tr_idx], y_train[tr_idx])
    Zh_tr = pipe_h.transform(X_hidden_train[tr_idx])
    Zh_val = pipe_h.transform(X_hidden_train[val_idx])
    Xf_tr = np.concatenate([X_prompt_train[tr_idx], Zh_tr], axis=1)
    Xf_val = np.concatenate([X_prompt_train[val_idx], Zh_val], axis=1)
    m = ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    m.fit(Xf_tr, y_train[tr_idx])
    pred = m.predict(Xf_val)
    maes.append(mean_absolute_error(y_train[val_idx], pred))
cv_results["et_fusion"] = {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)), "maes": [float(x) for x in maes]}

# hgb prompt
maes = []
for tr_idx, val_idx in gkf.split(X_prompt_train, y_train, groups):
    m = HistGradientBoostingRegressor(random_state=seed)
    m.fit(X_prompt_train[tr_idx], y_train[tr_idx])
    pred = m.predict(X_prompt_train[val_idx])
    maes.append(mean_absolute_error(y_train[val_idx], pred))
cv_results["hgb_prompt"] = {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)), "maes": [float(x) for x in maes]}

# hgb hidden
maes = []
for tr_idx, val_idx in gkf.split(X_prompt_train, y_train, groups):
    pipe_h = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=seed))])
    pipe_h.fit(X_hidden_train[tr_idx], y_train[tr_idx])
    Zh_tr = pipe_h.transform(X_hidden_train[tr_idx])
    Zh_val = pipe_h.transform(X_hidden_train[val_idx])
    m = HistGradientBoostingRegressor(random_state=seed)
    m.fit(Zh_tr, y_train[tr_idx])
    pred = m.predict(Zh_val)
    maes.append(mean_absolute_error(y_train[val_idx], pred))
cv_results["hgb_hidden"] = {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)), "maes": [float(x) for x in maes]}

# hgb fusion
maes = []
for tr_idx, val_idx in gkf.split(X_prompt_train, y_train, groups):
    pipe_h = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=seed))])
    pipe_h.fit(X_hidden_train[tr_idx], y_train[tr_idx])
    Zh_tr = pipe_h.transform(X_hidden_train[tr_idx])
    Zh_val = pipe_h.transform(X_hidden_train[val_idx])
    Xf_tr = np.concatenate([X_prompt_train[tr_idx], Zh_tr], axis=1)
    Xf_val = np.concatenate([X_prompt_train[val_idx], Zh_val], axis=1)
    m = HistGradientBoostingRegressor(random_state=seed)
    m.fit(Xf_tr, y_train[tr_idx])
    pred = m.predict(Xf_val)
    maes.append(mean_absolute_error(y_train[val_idx], pred))
cv_results["hgb_fusion"] = {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)), "maes": [float(x) for x in maes]}

json.dump({"folds": fold_membership, "cv_results": cv_results}, open(out / "cv_results.json", "w"), indent=2)

best_key = min(cv_results.keys(), key=lambda k: cv_results[k]["mae_mean"])

if best_key == "rf_fusion":
    pipe_h = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=seed))])
    pipe_h.fit(X_hidden_train, y_train)
    Zh_tr = pipe_h.transform(X_hidden_train)
    Zh_te = pipe_h.transform(X_hidden_test)
    Xf_tr = np.concatenate([X_prompt_train, Zh_tr], axis=1)
    Xf_te = np.concatenate([X_prompt_test, Zh_te], axis=1)
    m = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    m.fit(Xf_tr, y_train)
    test_pred = m.predict(Xf_te)
    selected = {"model": "rf", "config": "fusion", "params": {"n_estimators": 100, "pca_components": 50}, "pipe": "pipe_h+rf"}
elif best_key == "et_fusion":
    pipe_h = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=seed))])
    pipe_h.fit(X_hidden_train, y_train)
    Zh_tr = pipe_h.transform(X_hidden_train)
    Zh_te = pipe_h.transform(X_hidden_test)
    Xf_tr = np.concatenate([X_prompt_train, Zh_tr], axis=1)
    Xf_te = np.concatenate([X_prompt_test, Zh_te], axis=1)
    m = ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    m.fit(Xf_tr, y_train)
    test_pred = m.predict(Xf_te)
    selected = {"model": "et", "config": "fusion", "params": {"n_estimators": 100, "pca_components": 50}, "pipe": "pipe_h+et"}
elif best_key == "rf_hidden":
    pipe_h = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=seed))])
    pipe_h.fit(X_hidden_train, y_train)
    Zh_tr = pipe_h.transform(X_hidden_train)
    Zh_te = pipe_h.transform(X_hidden_test)
    m = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    m.fit(Zh_tr, y_train)
    test_pred = m.predict(Zh_te)
    selected = {"model": "rf", "config": "hidden", "params": {"n_estimators": 100, "pca_components": 50}, "pipe": "pipe_h+rf"}
elif best_key == "et_hidden":
    pipe_h = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=seed))])
    pipe_h.fit(X_hidden_train, y_train)
    Zh_tr = pipe_h.transform(X_hidden_train)
    Zh_te = pipe_h.transform(X_hidden_test)
    m = ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    m.fit(Zh_tr, y_train)
    test_pred = m.predict(Zh_te)
    selected = {"model": "et", "config": "hidden", "params": {"n_estimators": 100, "pca_components": 50}, "pipe": "pipe_h+et"}
elif best_key == "ridge_fusion":
    pipe_h = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=seed))])
    pipe_h.fit(X_hidden_train, y_train)
    Zh_tr = pipe_h.transform(X_hidden_train)
    Zh_te = pipe_h.transform(X_hidden_test)
    Xf_tr = np.concatenate([X_prompt_train, Zh_tr], axis=1)
    Xf_te = np.concatenate([X_prompt_test, Zh_te], axis=1)
    m = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    m.fit(Xf_tr, y_train)
    test_pred = m.predict(Xf_te)
    selected = {"model": "ridge", "config": "fusion", "params": {"alpha": 1.0, "pca_components": 50}, "pipe": "pipe_h+scaler+ridge"}
elif best_key == "ridge_prompt":
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe.fit(X_prompt_train, y_train)
    test_pred = pipe.predict(X_prompt_test)
    selected = {"model": "ridge", "config": "prompt", "params": {"alpha": 1.0}, "pipe": "pipeline"}
elif best_key == "rf_prompt":
    m = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    m.fit(X_prompt_train, y_train)
    test_pred = m.predict(X_prompt_test)
    selected = {"model": "rf", "config": "prompt", "params": {"n_estimators": 100}, "pipe": None}
elif best_key == "et_prompt":
    m = ExtraTreesRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    m.fit(X_prompt_train, y_train)
    test_pred = m.predict(X_prompt_test)
    selected = {"model": "et", "config": "prompt", "params": {"n_estimators": 100}, "pipe": None}
elif best_key == "hgb_prompt":
    m = HistGradientBoostingRegressor(random_state=seed)
    m.fit(X_prompt_train, y_train)
    test_pred = m.predict(X_prompt_test)
    selected = {"model": "hgb", "config": "prompt", "params": {}, "pipe": None}
elif best_key == "hgb_hidden":
    pipe_h = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=seed))])
    pipe_h.fit(X_hidden_train, y_train)
    Zh_tr = pipe_h.transform(X_hidden_train)
    Zh_te = pipe_h.transform(X_hidden_test)
    m = HistGradientBoostingRegressor(random_state=seed)
    m.fit(Zh_tr, y_train)
    test_pred = m.predict(Zh_te)
    selected = {"model": "hgb", "config": "hidden", "params": {"pca_components": 50}, "pipe": "pipe_h+hgb"}
else:
    pipe_h = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=seed))])
    pipe_h.fit(X_hidden_train, y_train)
    Zh_tr = pipe_h.transform(X_hidden_train)
    Zh_te = pipe_h.transform(X_hidden_test)
    Xf_tr = np.concatenate([X_prompt_train, Zh_tr], axis=1)
    Xf_te = np.concatenate([X_prompt_test, Zh_te], axis=1)
    m = HistGradientBoostingRegressor(random_state=seed)
    m.fit(Xf_tr, y_train)
    test_pred = m.predict(Xf_te)
    selected = {"model": "hgb", "config": "fusion", "params": {"pca_components": 50}, "pipe": "pipe_h+hgb"}

baseline_pred = np.full_like(y_test, y_train.mean())
baseline_metrics = {"mae": float(mean_absolute_error(y_test, baseline_pred)), "rmse": float(np.sqrt(mean_squared_error(y_test, baseline_pred))), "r2": float(r2_score(y_test, baseline_pred))}
test_metrics = {"mae": float(mean_absolute_error(y_test, test_pred)), "rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))), "r2": float(r2_score(y_test, test_pred))}

rng = np.random.RandomState(seed)
n_boot = 1000
boot_diffs = []
for i in range(n_boot):
    idx = rng.choice(len(y_test), len(y_test), replace=True)
    boot_diffs.append(mean_absolute_error(y_test[idx], baseline_pred[idx]) - mean_absolute_error(y_test[idx], test_pred[idx]))
boot_diffs = np.array(boot_diffs)
ci_low = float(np.percentile(boot_diffs, 2.5))
ci_high = float(np.percentile(boot_diffs, 97.5))
p_better = float(np.mean(boot_diffs > 0))

results = {
    "config": {
        "source_dir": str(src),
        "out_dir": str(out),
        "seed": int(seed),
        "n_train_prompts": int(len(train_pids_set)),
        "n_test_prompts": int(len(test_pids_set)),
        "pooling": "mean over layers for 3D hidden",
        "cv": {"type": "GroupKFold", "n_splits": 5, "grouped_by_prompt": True},
    },
    "fold_membership": fold_membership,
    "cv_results": cv_results,
    "selected": {"key": best_key, **selected},
    "baseline": baseline_metrics,
    "test": test_metrics,
    "bootstrap": {"n": n_boot, "mae_diff_baseline_minus_model_ci": [ci_low, ci_high], "p_diff>0": p_better},
}

json.dump(results, open(out / "experiment_results.json", "w"), indent=2)
np.save(out / "test_predictions.npy", test_pred)
np.save(out / "baseline_predictions.npy", baseline_pred)
np.save(out / "test_y.npy", y_test)
np.save(out / "test_pids.npy", pids_test)

report = []
report.append("# CV Model Comparison (Training-Only CV)")
report.append("Source: " + str(src))
report.append("Output: " + str(out))
report.append("")
report.append("## Feature pooling")
report.append("Hidden states (486,29,2048) mean-pooled over layers (axis=1) -> (486,2048). Prompt features (5 dims). Fusion = prompt + PCA(50) of pooled hidden fit per fold on train only.")
report.append("")
report.append("## Split")
report.append("Prompt-level GroupKFold by prompt (5 folds), 340 train prompts / 146 test prompts (seed " + str(seed) + "). Test set untouched during CV/tuning.")
report.append("")
report.append("## Selected")
report.append("Best by CV MAE: " + best_key + " -> " + str(selected))
report.append("")
report.append("## Metrics")
report.append("Baseline (train mean): MAE=" + str(baseline_metrics["mae"]) + ", RMSE=" + str(baseline_metrics["rmse"]) + ", R2=" + str(baseline_metrics["r2"]))
report.append("Test (" + selected["model"] + "/" + selected["config"] + "): MAE=" + str(test_metrics["mae"]) + ", RMSE=" + str(test_metrics["rmse"]) + ", R2=" + str(test_metrics["r2"]))
report.append("")
report.append("## Bootstrap (paired)")
report.append("MAE_diff (baseline-model), 95% CI: [" + str(ci_low) + ", " + str(ci_high) + "], P(diff>0)=" + str(p_better))
report.append("")
report.append("## Notes")
report.append("Negative R2 indicates worse than train-mean baseline on test set. Differences small/uncertain; no test data used for selection.")

open(out / "report.md", "w").write("\n".join(report))
print(best_key, baseline_metrics["mae"], test_metrics["mae"])
