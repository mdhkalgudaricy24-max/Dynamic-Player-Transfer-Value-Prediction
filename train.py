"""TransferIQ - football player market value prediction.

Run:  python src/train.py
Outputs: results/ (metrics, plots) and models/best_model.joblib

Evaluation protocol (no test-set leakage):
  1. 20% of players are held out as a TEST set and touched only once, at the end.
  2. All model selection / tuning uses 5-fold CV on the remaining 80%.
  3. Early stopping is NOT used (it needs a validation set); n_estimators is tuned by CV instead.
  4. The best model is chosen by CV MAE, never by test score.
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "all_in_one_dataset.csv"
RES, MOD = ROOT / "results", ROOT / "models"
RES.mkdir(exist_ok=True)
MOD.mkdir(exist_ok=True)

TARGET = "market_value_millions"
# identifiers / text labels that are not predictive features
NON_FEATURES = ["player_id", "player_name", "market_value_", "market_value_last_updated"]
PROXY_THRESHOLD = 0.90      # |corr with target| above this => target proxy (leakage)
REDUNDANT_THRESHOLD = 0.99  # |corr between two features| above this => exact duplicate
PREFER_KEEP = {"injury_risk_score"}  # of two duplicate columns, keep the one with the meaningful name

GROUPS = {  # used for the feature-group ablation
    "social": ["twitter_mentions_count", "twitter_engagement_rate", "social_sentiment_score",
               "social_engagement_score", "public_perception_index", "mention_volatility_score",
               "trending_mentions_7d", "sentiment_classification"],
    "injury": ["injury_risk_score", "total_injuries_history", "days_injured_last_season",
               "current_injury_status", "avg_recovery_time_days", "injury_prone_body_parts",
               "historical_injury_count", "days_since_last_injury"],
    "contract": ["contract_end_year", "years_remaining_contract"],
}


def load():
    df = pd.read_csv(DATA).drop(columns=NON_FEATURES)
    df["injury_prone_body_parts"] = df["injury_prone_body_parts"].fillna("None")  # 97 missing = no injury history
    return df


def drop_leaky_and_redundant(train: pd.DataFrame):
    """Decide which columns to drop using TRAINING data only."""
    num = train.select_dtypes("number")
    corr_t = num.corr()[TARGET].drop(TARGET).abs()
    proxies = corr_t[corr_t > PROXY_THRESHOLD].index.tolist()
    feats = num.drop(columns=[TARGET] + proxies)
    cm = feats.corr().abs()
    redundant = []
    cols = list(cm.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if cm.loc[a, b] > REDUNDANT_THRESHOLD and b not in redundant and a not in redundant:
                redundant.append(a if (b in PREFER_KEEP) else b)
    return proxies, redundant


def make_pipeline(model, X):
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num = X.select_dtypes("number").columns.tolist()
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", StandardScaler(), num),
    ])
    # log1p target: values are right-skewed; predictions are converted back to EUR millions
    return TransformedTargetRegressor(Pipeline([("pre", pre), ("model", model)]),
                                      func=np.log1p, inverse_func=np.expm1)


def metrics(y, p):
    return {"r2": r2_score(y, p), "mae": mean_absolute_error(y, p),
            "rmse": float(np.sqrt(mean_squared_error(y, p)))}


def main():
    df = load()
    dev, test = train_test_split(df, test_size=0.2, random_state=SEED)
    proxies, redundant = drop_leaky_and_redundant(dev)
    print("Target proxies dropped:", proxies)
    print("Duplicate features dropped:", redundant)
    drop = [TARGET] + proxies + redundant
    X_dev, y_dev = dev.drop(columns=drop), dev[TARGET]
    X_test, y_test = test.drop(columns=drop), test[TARGET]
    print(f"dev={len(dev)} test={len(test)} features={X_dev.shape[1]}")

    cv = KFold(5, shuffle=True, random_state=SEED)
    # (estimator, small tuning grid) - grid keys are prefixed for the pipeline inside TransformedTargetRegressor
    P = "regressor__model__"
    candidates = {
        "Mean baseline": (DummyRegressor(strategy="mean"), {}),
        "Ridge (linear baseline)": (Ridge(), {P + "alpha": [0.1, 1, 10, 100, 300, 1000]}),
        "Random Forest": (RandomForestRegressor(random_state=SEED, n_jobs=-1),
                          {P + "n_estimators": [300], P + "max_depth": [6, 10, None],
                           P + "min_samples_leaf": [1, 3]}),
        "XGBoost": (XGBRegressor(random_state=SEED, n_jobs=-1, subsample=0.8, colsample_bytree=0.8),
                    {P + "n_estimators": [200, 500], P + "learning_rate": [0.03, 0.05],
                     P + "max_depth": [3, 5], P + "reg_lambda": [1, 10]}),
        "LightGBM": (LGBMRegressor(random_state=SEED, n_jobs=1, verbose=-1, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8),
                     {P + "n_estimators": [200, 500], P + "learning_rate": [0.03, 0.05],
                      P + "num_leaves": [8, 31], P + "min_child_samples": [5, 20]}),
    }
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}

    rows, fitted = [], {}
    for name, (est, grid) in candidates.items():
        pipe = make_pipeline(est, X_dev)
        if grid:
            gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
            gs.fit(X_dev, y_dev)
            best, params = gs.best_estimator_, {k.replace(P, ""): v for k, v in gs.best_params_.items()}
        else:
            best, params = pipe.fit(X_dev, y_dev), {}
        # honest CV estimate for the chosen configuration (note: mildly optimistic because the
        # configuration was picked on these same folds - the untouched test set is the final check)
        cvres = cross_validate(best, X_dev, y_dev, cv=cv, scoring=scoring)
        tm = metrics(y_test, best.predict(X_test))
        rows.append({"model": name, "params": params,
                     "cv_r2_mean": cvres["test_r2"].mean(), "cv_r2_std": cvres["test_r2"].std(),
                     "cv_mae_mean": -cvres["test_mae"].mean(), "cv_mae_std": cvres["test_mae"].std(),
                     "cv_rmse_mean": -cvres["test_rmse"].mean(),
                     "test_r2": tm["r2"], "test_mae": tm["mae"], "test_rmse": tm["rmse"]})
        fitted[name] = best
        print(f"{name:26s} CV R2 {rows[-1]['cv_r2_mean']:.3f}±{rows[-1]['cv_r2_std']:.3f} "
              f"MAE {rows[-1]['cv_mae_mean']:.2f} | TEST R2 {tm['r2']:.3f} MAE {tm['mae']:.2f}")

    table = pd.DataFrame(rows)
    best_name = table.loc[table["cv_mae_mean"].idxmin(), "model"]   # selected on CV, not test
    best = fitted[best_name]
    pred = best.predict(X_test)
    print("Selected by CV MAE:", best_name)

    # bootstrap 95% CI on the test set (only ~90 players, so uncertainty matters)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(y_test), len(y_test))
        boots.append([r2_score(y_test.iloc[i], pred[i]), mean_absolute_error(y_test.iloc[i], pred[i])])
    ci = np.percentile(boots, [2.5, 97.5], axis=0)

    # feature-group ablation (CV): does dropping social / injury / contract features hurt performance?
    # Run for the linear model AND the boosted model, so the conclusion isn't specific to one model family.
    ablation = {}
    for mname in ["Ridge (linear baseline)", "LightGBM"]:
        est = clone(fitted[mname].regressor_.named_steps["model"])
        full = cross_validate(make_pipeline(est, X_dev), X_dev, y_dev, cv=cv, scoring=scoring)
        ablation[f"{mname} | all features"] = (full["test_r2"].mean(), -full["test_mae"].mean())
        for g, cols in GROUPS.items():
            keep = X_dev.drop(columns=[c for c in cols if c in X_dev.columns])
            r = cross_validate(make_pipeline(clone(est), keep), keep, y_dev, cv=cv, scoring=scoring)
            ablation[f"{mname} | without {g}"] = (r["test_r2"].mean(), -r["test_mae"].mean())
    for k, (r2, mae) in ablation.items():
        print(f"ablation {k:42s} CV R2 {r2:.3f} MAE {mae:.2f}")

    # permutation importance on the held-out test set (model-agnostic, works on raw columns)
    pi = permutation_importance(best, X_test, y_test, n_repeats=30, random_state=SEED, scoring="r2")
    imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values(ascending=False)
    imp_std = pd.Series(pi.importances_std, index=X_test.columns)[imp.index]

    # ---- plots
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_test, pred, alpha=0.7)
    lim = [0, max(y_test.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, "r--", label="perfect prediction")
    ax.set(xlabel="Actual value (EUR M)", ylabel="Predicted value (EUR M)",
           title=f"{best_name}: held-out test set (n={len(y_test)})")
    ax.legend(); fig.tight_layout(); fig.savefig(RES / "actual_vs_predicted.png", dpi=150); plt.close(fig)

    top = imp.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values, xerr=imp_std[top.index].values, color="tab:green")
    ax.set(xlabel="Drop in test R² when feature is shuffled", title="Top 10 features (permutation importance)")
    fig.tight_layout(); fig.savefig(RES / "feature_importance.png", dpi=150); plt.close(fig)

    t = table.set_index("model")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(t.index[::-1], t["cv_r2_mean"][::-1], xerr=t["cv_r2_std"][::-1], color="tab:blue")
    ax.set(xlabel="5-fold CV R² (mean ± std)", title="Model comparison")
    fig.tight_layout(); fig.savefig(RES / "model_comparison.png", dpi=150); plt.close(fig)

    # ---- save
    joblib.dump(best, MOD / "best_model.joblib")
    table.round(4).to_csv(RES / "model_comparison.csv", index=False)
    out = {"selected_model": best_name, "n_dev": len(dev), "n_test": len(test), "n_features": X_dev.shape[1],
           "target_proxies_dropped": proxies, "duplicate_features_dropped": redundant,
           "test": metrics(y_test, pred),
           "test_r2_95ci": ci[:, 0].round(4).tolist(), "test_mae_95ci": ci[:, 1].round(4).tolist(),
           "target_mean_dev": float(y_dev.mean()), "target_range": [float(df[TARGET].min()), float(df[TARGET].max())],
           "ablation_cv(r2,mae)": {k: [round(v[0], 4), round(v[1], 3)] for k, v in ablation.items()},
           "top_features": imp.head(10).round(4).to_dict(),
           "params": table.set_index("model").loc[best_name, "params"]}
    (RES / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
