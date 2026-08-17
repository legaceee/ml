"""
Hyperparameter Optimization Engine using Optuna with Stratified Cross-Validation.

Optimizes candidate classifiers maximizing F1-score on the training split only.
"""

from typing import Any, Dict, Tuple
import optuna
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

# Suppress verbose Optuna logging during automated runs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    def __init__(self, n_trials: int = 25, cv_folds: int = 5, random_state: int = 42):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.study_results_: Dict[str, Any] = {}

    def optimize_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
        """
        Optimize Random Forest hyperparameters via Optuna maximizing CV F1-score.
        """
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        def objective(trial: optuna.Trial) -> float:
            n_estimators = trial.suggest_int("n_estimators", 50, 250, step=25)
            max_depth = trial.suggest_int("max_depth", 6, 25)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 6)
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5])

            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            )
            scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
            return float(scores.mean())

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_f1 = float(study.best_value)

        # Train final optimized model on complete training set
        optimized_rf = RandomForestClassifier(
            **best_params,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1
        )
        optimized_rf.fit(X_train, y_train)

        self.study_results_["optimized_random_forest"] = {
            "best_params": best_params,
            "best_cv_f1": best_f1,
            "n_trials": len(study.trials),
            "trials_history": [{"number": t.number, "value": t.value, "params": t.params} for t in study.trials if t.value is not None]
        }

        return optimized_rf, self.study_results_["optimized_random_forest"]

    def optimize_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[XGBClassifier, Dict[str, Any]]:
        """
        Optimize XGBoost hyperparameters via Optuna maximizing CV F1-score.
        """
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # Compute positive class weight scale
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        scale_pos_weight = float(num_neg / max(1, num_pos))

        def objective(trial: optuna.Trial) -> float:
            n_estimators = trial.suggest_int("n_estimators", 60, 250, step=25)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.25, log=True)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
            min_child_weight = trial.suggest_int("min_child_weight", 1, 7)
            gamma = trial.suggest_float("gamma", 0.0, 2.0)

            clf = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                gamma=gamma,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1
            )
            scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
            return float(scores.mean())

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_f1 = float(study.best_value)

        optimized_xgb = XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1
        )
        optimized_xgb.fit(X_train, y_train)

        self.study_results_["optimized_xgboost"] = {
            "best_params": best_params,
            "best_cv_f1": best_f1,
            "n_trials": len(study.trials),
            "trials_history": [{"number": t.number, "value": t.value, "params": t.params} for t in study.trials if t.value is not None]
        }

        return optimized_xgb, self.study_results_["optimized_xgboost"]
