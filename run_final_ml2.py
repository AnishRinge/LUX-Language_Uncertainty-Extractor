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

# X_hidden is (n, num_layers, hidden) - mean pool already done? wait - check
# in our build we took mean of last layer, so should be 2D; but shape shows 3D
# flatten properly
if X_hidden.ndim == 3:
    X_hidden = X_hidden.mean(axis=1)  # mean over layers

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

json.dump({"train_pids": [int(x) if hasattr(x,'item') else str(x) for x in train_pids], "test_pids": [int(x) if hasattr(x,'item') else str(x) for x in test_pids], "seed": 12345}, open(exp_dir/"split.json","w"))

baseline_pred = np.full_like(y_test, y_train.mean())
results = {"baseline": {"mae": float(mean_absolute_error(y_test, baseline_pred)), "rmse": float(np.sqrt(mean_squared_error(y_test, baseline_pred))), "r2": float(r2_score(y_test, baseline_pred))}}
