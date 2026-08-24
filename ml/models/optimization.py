"""
Hyperparameter Optimisation: Grid Search vs Random Search vs Bayesian (Optuna).

All three searchers share the SAME 5-fold stratified cross-validation splitter
and the SAME objective (mean F1 on the training split) so that the only
variable is the search strategy:

    GridSearchCV        - exhaustive: evaluates every point of a fixed grid.
    RandomizedSearchCV  - samples a fixed budget of random points from ranges.
    Optuna (TPE)        - Bayesian: a Tree-structured Parzen Estimator models
                          P(params | good score) vs P(params | bad score) and
                          proposes the next trial where the ratio is highest.

The test split is never touched during the search; it is used exactly once
per final model by the experiment runner.
"""

import time
from typing import Any, Dict, Tuple

import numpy as np
import optuna
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# A small, deliberately coarse grid: 3 x 3 x 2 = 18 fits per fold.
RF_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "max_features": ["sqrt", "log2"],
}

RF_RANDOM_DISTRIBUTIONS = {
    "n_estimators": randint(50, 301),
    "max_depth": randint(6, 31),
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 7),
    "max_features": ["sqrt", "log2", 0.5],
}

XGB_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
}

XGB_RANDOM_DISTRIBUTIONS = {
    "n_estimators": randint(60, 301),
    "max_depth": randint(3, 11),
    "learning_rate": uniform(0.01, 0.24),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "min_child_weight": randint(1, 8),
}


def _history_from_cv_results(cv_results: Dict[str, Any]) -> list:
    hist = []
    for i, (params, mean_score) in enumerate(zip(cv_results["params"], cv_results["mean_test_score"])):
        clean = {k: (None if v is None else (v if isinstance(v, (int, float, str)) else str(v))) for k, v in params.items()}
        hist.append({"number": i, "value": float(mean_score), "params": clean})
    return hist


class HyperparameterOptimizer:
    def __init__(self, n_trials: int = 25, cv_folds: int = 5, random_state: int = 42):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.study_results_: Dict[str, Any] = {}

    def _cv(self) -> StratifiedKFold:
        return StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

    # ------------------------------------------------------------------ #
    # Random Forest
    # ------------------------------------------------------------------ #
    def _rf_base(self) -> RandomForestClassifier:
        return RandomForestClassifier(class_weight="balanced", random_state=self.random_state, n_jobs=-1)

    def grid_search_random_forest(self, X_train, y_train) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        t0 = time.perf_counter()
        gs = GridSearchCV(self._rf_base(), RF_GRID, scoring="f1", cv=self._cv(), n_jobs=1, refit=True)
        gs.fit(X_train, y_train)
        wall = time.perf_counter() - t0
        result = {
            "method": "GridSearchCV",
            "best_params": {k: (v if v is not None else None) for k, v in gs.best_params_.items()},
            "best_cv_f1": float(gs.best_score_),
            "n_trials": int(len(gs.cv_results_["params"])),
            "wall_time_sec": round(wall, 2),
            "search_space": {k: [str(x) for x in v] for k, v in RF_GRID.items()},
            "trials_history": _history_from_cv_results(gs.cv_results_),
        }
        self.study_results_["grid_random_forest"] = result
        return gs.best_estimator_, result

    def random_search_random_forest(self, X_train, y_train, n_iter: int = 18) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        t0 = time.perf_counter()
        rs = RandomizedSearchCV(
            self._rf_base(), RF_RANDOM_DISTRIBUTIONS, n_iter=n_iter, scoring="f1",
            cv=self._cv(), n_jobs=1, refit=True, random_state=self.random_state,
        )
        rs.fit(X_train, y_train)
        wall = time.perf_counter() - t0
        result = {
            "method": "RandomizedSearchCV",
            "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in rs.best_params_.items()},
            "best_cv_f1": float(rs.best_score_),
            "n_trials": int(n_iter),
            "wall_time_sec": round(wall, 2),
            "trials_history": _history_from_cv_results(rs.cv_results_),
        }
        self.study_results_["random_random_forest"] = result
        return rs.best_estimator_, result

    def optimize_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """Optuna (TPE) search maximising 5-fold CV F1."""
        cv = self._cv()

        def objective(trial: optuna.Trial) -> float:
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 50, 300, step=25),
                max_depth=trial.suggest_int("max_depth", 6, 30),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            )
            clf = RandomForestClassifier(**params, class_weight="balanced", random_state=self.random_state, n_jobs=-1)
            scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
            return float(scores.mean())

        t0 = time.perf_counter()
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        wall = time.perf_counter() - t0

        best_params = study.best_params
        optimized_rf = RandomForestClassifier(**best_params, class_weight="balanced", random_state=self.random_state, n_jobs=-1)
        optimized_rf.fit(X_train, y_train)

        self.study_results_["optimized_random_forest"] = {
            "method": "Optuna TPE",
            "best_params": best_params,
            "best_cv_f1": float(study.best_value),
            "n_trials": len(study.trials),
            "wall_time_sec": round(wall, 2),
            "trials_history": [{"number": t.number, "value": t.value, "params": t.params} for t in study.trials if t.value is not None],
        }
        return optimized_rf, self.study_results_["optimized_random_forest"]

    # ------------------------------------------------------------------ #
    # XGBoost
    # ------------------------------------------------------------------ #
    def _scale_pos_weight(self, y_train) -> float:
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        return float(num_neg / max(1, num_pos))

    def _xgb_base(self, y_train) -> XGBClassifier:
        return XGBClassifier(
            scale_pos_weight=self._scale_pos_weight(y_train), eval_metric="logloss",
            random_state=self.random_state, n_jobs=-1, tree_method="hist",
        )

    def grid_search_xgboost(self, X_train, y_train) -> Tuple[XGBClassifier, Dict[str, Any]]:
        t0 = time.perf_counter()
        gs = GridSearchCV(self._xgb_base(y_train), XGB_GRID, scoring="f1", cv=self._cv(), n_jobs=1, refit=True)
        gs.fit(X_train, y_train)
        wall = time.perf_counter() - t0
        result = {
            "method": "GridSearchCV",
            "best_params": gs.best_params_,
            "best_cv_f1": float(gs.best_score_),
            "n_trials": int(len(gs.cv_results_["params"])),
            "wall_time_sec": round(wall, 2),
            "search_space": {k: [str(x) for x in v] for k, v in XGB_GRID.items()},
            "trials_history": _history_from_cv_results(gs.cv_results_),
        }
        self.study_results_["grid_xgboost"] = result
        return gs.best_estimator_, result

    def random_search_xgboost(self, X_train, y_train, n_iter: int = 18) -> Tuple[XGBClassifier, Dict[str, Any]]:
        t0 = time.perf_counter()
        rs = RandomizedSearchCV(
            self._xgb_base(y_train), XGB_RANDOM_DISTRIBUTIONS, n_iter=n_iter, scoring="f1",
            cv=self._cv(), n_jobs=1, refit=True, random_state=self.random_state,
        )
        rs.fit(X_train, y_train)
        wall = time.perf_counter() - t0
        result = {
            "method": "RandomizedSearchCV",
            "best_params": {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) for k, v in rs.best_params_.items()},
            "best_cv_f1": float(rs.best_score_),
            "n_trials": int(n_iter),
            "wall_time_sec": round(wall, 2),
            "trials_history": _history_from_cv_results(rs.cv_results_),
        }
        self.study_results_["random_xgboost"] = result
        return rs.best_estimator_, result

    def optimize_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[XGBClassifier, Dict[str, Any]]:
        """Optuna (TPE) search maximising 5-fold CV F1."""
        cv = self._cv()
        spw = self._scale_pos_weight(y_train)

        def objective(trial: optuna.Trial) -> float:
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 60, 300, step=20),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 7),
                gamma=trial.suggest_float("gamma", 0.0, 2.0),
            )
            clf = XGBClassifier(**params, scale_pos_weight=spw, eval_metric="logloss",
                                random_state=self.random_state, n_jobs=-1, tree_method="hist")
            scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
            return float(scores.mean())

        t0 = time.perf_counter()
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        wall = time.perf_counter() - t0

        best_params = study.best_params
        optimized_xgb = XGBClassifier(**best_params, scale_pos_weight=spw, eval_metric="logloss",
                                      random_state=self.random_state, n_jobs=-1, tree_method="hist")
        optimized_xgb.fit(X_train, y_train)

        self.study_results_["optimized_xgboost"] = {
            "method": "Optuna TPE",
            "best_params": best_params,
            "best_cv_f1": float(study.best_value),
            "n_trials": len(study.trials),
            "wall_time_sec": round(wall, 2),
            "trials_history": [{"number": t.number, "value": t.value, "params": t.params} for t in study.trials if t.value is not None],
        }
        return optimized_xgb, self.study_results_["optimized_xgboost"]
