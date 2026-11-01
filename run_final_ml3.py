import json
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

exp_dir = Path("experiments/final_cohort_2026_10_24")
X_prompt = np.load(exp_dir / "X_prompt.npy")
X_hidden = np.load(exp_dir / "X_hidden.npy")
y = np.load(exp_dir / "y.npy")
pids = np.load(exp_dir / "pids.npy")

if X_hidden.ndim == 3:
    X_hidden = X_hidden.mean(axis=1)

rng = np.random.RandomState(12345)
unique_pids = np.unique(pids)
rng.shuffle(unique_pids)
n_train = int(len(unique_pids) * 0.7)
train_pids = set(unique_pids[:n_train])
test_pids = set(unique_pids[n_train:])

train_mask = np.array([p in train_pids for p in pids])
test_mask = np.array([p in test_pids for p in pids])

X_prompt_train, X_prompt_test = X_prompt[train_mask], X_prompt[test_mask]
X_hidden_train, X_hidden_test = X_hidden[train_mask], X_hidden[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

json.dump({"train_pids": [str(x) for x in train_pids], "test_pids": [str(x) for x in test_pids], "seed": 12345}, open(str(exp_dir/"split.json"),"w"))

baseline_pred = np.full_like(y_test, y_train.mean())
results = {"baseline": {"mae": float(mean_absolute_error(y_test, baseline_pred)), "rmse": float(np.sqrt(mean_squared_error(y_test, baseline_pred))), "r2": float(r2_score(y_test, baseline_pred))}}

# Ridge prompt
pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
pipe.fit(X_prompt_train, y_train)
pred = pipe.predict(X_prompt_test)
results["ridge_prompt"] = {"mae": float(mean_absolute_error(y_test,pred)), "rmse": float(np.sqrt(mean_squared_error(y_test,pred))), "r2": float(r2_score(y_test,pred))}

# Hidden
pipe_h = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=50, random_state=12345)), ("ridge", Ridge(alpha=1.0))])
pipe_h.fit(X_hidden_train, y_train)
pred_h = pipe_h.predict(X_hidden_test)
results["ridge_hidden"] = {"mae": float(mean_absolute_error(y_test,pred_h)), "rmse": float(np.sqrt(mean_squared_error(y_test,pred_h))), "r2": float(r2_score(y_test,pred_h))}

# Fusion
X_f_train = np.concatenate([X_prompt_train, pipe_h.named_steps["pca"].transform(pipe_h.named_steps["scaler"].transform(X_hidden_train))], axis=1)
X_f_test = np.concatenate([X_prompt_test, pipe_h.named_steps["pca"].transform(pipe_h.named_steps["scaler"].transform(X_hidden_test))], axis=1)
pipe_f = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
pipe_f.fit(X_f_train, y_train)
pred_f = pipe_f.predict(X_f_test)
results["ridge_fusion"] = {"mae": float(mean_absolute_error(y_test,pred_f)), "rmse": float(np.sqrt(mean_squared_error(y_test,pred_f))), "r2": float(r2_score(y_test,pred_f))}

# RF
rf = RandomForestRegressor(n_estimators=100, random_state=12345, n_jobs=-1)
rf.fit(X_prompt_train, y_train)
results["rf_prompt"] = {"mae": float(mean_absolute_error(y_test,rf.predict(X_prompt_test))), "rmse": float(np.sqrt(mean_squared_error(y_test,rf.predict(X_prompt_test)))), "r2": float(r2_score(y_test,rf.predict(X_prompt_test)))}

rf_h = RandomForestRegressor(n_estimators=100, random_state=12345, n_jobs=-1)
rf_h.fit(pipe_h.named_steps["pca"].transform(pipe_h.named_steps["scaler"].transform(X_hidden_train)), y_train)
results["rf_hidden"] = {"mae": float(mean_absolute_error(y_test,rf_h.predict(pipe_h.named_steps["pca"].transform(pipe_h.named_steps["scaler"].transform(X_hidden_test))))), "rmse": float(np.sqrt(mean_squared_error(y_test,rf_h.predict(pipe_h.named_steps["pca"].transform(pipe_h.named_steps["scaler"].transform(X_hidden_test))))), "r2": float(r2_score(y_test,rf_h.predict(pipe_h.named_steps["pca"].transform(pipe_h.named_steps["scaler"].transform(X_hidden_test))))}

rf_f = RandomForestRegressor(n_estimators=100, random_state=12345, n_jobs=-1)
rf_f.fit(X_f_train, y_train)
results["rf_fusion"] = {"mae": float(mean_absolute_error(y_test,rf_f.predict(X_f_test))), "rmse": float(np.sqrt(mean_squared_error(y_test,rf_f.predict(X_f_test)))), "r2": float(r2_score(y_test,rf_f.predict(X_f_test)))}

# ET
et = ExtraTreesRegressor(n_estimators=100, random_state=12345, n_jobs=-1)
et.fit(X_prompt_train, y_train)
results["et_prompt"] = {"mae": float(mean_absolute_error(y_test,et.predict(X_prompt_test))), "rmse": float(np.sqrt(mean_squared_error(y_test,et.predict(X_prompt_test)))), "r2": float(r2_score(y_test,et.predict(X_prompt_test)))}

et_h = ExtraTreesRegressor(n_estimators=100, random_state=12345, n_jobs=-1)
et_h.fit(pipe_h.named_steps["pca"].transform(pipe_h.named_steps["scaler"].transform(X_hidden_train)), y_train)
results["et_hidden"] = {"mae": float(mean_absolute_error(y_test,et_h.predict(pipe_h.named_steps["pca"].transform(pipe_h.named_steps["scaler"].transform(X_hidden_test))))), "rmse": float(np.sqrt(mean_squared_error(y_test,et_h.predict(pipe_h.named_steps["pca"].transform(pipe_h.named_steps["scaler"].transform(X_hidden_test))))), "r2": float(r2_score(y_test,et_h.predict(pipe_h.named_steps["pca"].transform(pipe_h.named_steps["scaler"].transform(X_hidden_test))))}

et_f = ExtraTreesRegressor(n_estimators=100, random_state=12345, n_jobs=-1)
et_f.fit(X_f_train, y_train)
results["et_fusion"] = {"mae": float(mean_absolute_error(y_test,et_f.predict(X_f_test))), "rmse": float(np.sqrt(mean_squared_error(y_test,et_f.predict(X_f_test)))), "r2": float(r2_score(y_test,et_f.predict(X_f_test)))}

# HGB
hgb = HistGradientBoostingRegressor(random_state=12345)
hgb.fit(X_prompt_train, y_train)
results["hgb_prompt"] = {"mae": float(mean_absolute_error(y_test,hgb.predict(X_prompt_test))), "rmse": float(np.sqrt(mean_squared_error(y_test,hgb.predict(X_prompt_test)))), "r2": float(r2_score(y_test,hgb.predict(X_prompt_test)))}

hgb_h = HistGradientBoostingRegressor(random_state=12345)
hgb_h.fit(pipe_h.named_steps["pca"].transform(pipe_h.named_steps["scaler"].transform(X_hidden_train)), y_train)
results["hgb_hidden"] = {"mae": float(mean_absolute_error(y_test,hgb_h.predict(pipe_h.named_steps["pca"].transform(pipe_h.named_steps["scaler"].transform(X_hidden_test))))), "rmse": float(np.sqrt(mean_squared_error(y_test,hgb_h.predict(pipe_h.named_steps["pca"].transform(pipe_h.named_steps["scaler"].transform(X_hidden_test))))), "r2": float(r2_score(y_test,hgb_h.predict(pipe_h.named_steps["pca"].transform(pipe_h.named_steps["scaler"].transform(X_hidden_test))))}

hgb_f = HistGradientBoostingRegressor(random_state=12345)
hgb_f.fit(X_f_train, y_train)
results["hgb_fusion"] = {"mae": float(mean_absolute_error(y_test,hgb_f.predict(X_f_test))), "rmse": float(np.sqrt(mean_squared_error(y_test,hgb_f.predict(X_f_test)))), "r2": float(r2_score(y_test,hgb_f.predict(X_f_test)))}

json.dump(results, open(str(exp_dir/"model_comparison.json"),"w"), indent=2)
print(json.dumps(results, indent=2))
