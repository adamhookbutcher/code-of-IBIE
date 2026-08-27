# -*- coding: utf-8 -*-
"""
IBIE Full Experimental Reproduction Script

This single-file program reproduces the main empirical workflow described in the
attached IBIE paper, including:
    1) Ten classical balancing algorithms + IBIE.
    2) Eleven credit-risk models.
    3) Five-fold train/validation/test evaluation.
    4) Optuna Bayesian hyperparameter optimization.
    5) Six evaluation metrics: AUC, Sensitivity, Specificity, Precision, GMean, FScore.
    6) Friedman test, paired t-test, Wilcoxon signed-rank test with Holm correction,
       and Cohen's d effect size.
    7) IBIE robustness experiments: sub-model replacement, alpha/multi sensitivity,
       imbalance-ratio sensitivity, exploration mechanism, CTGAN/VAE comparison,
       ablation experiments, runtime comparison, and generation trace export.
    8) TXT parameter export for direct reuse in the PARA AREA.

Important design choices:
    - The program is intentionally sequential and uses n_jobs=1 everywhere possible.
    - QUICK_TEST=True skips every hyperparameter optimization and uses default parameters.
    - USE_SAVED_PARAMS=True directly loads the parameter block previously exported by
      this program and skips parameter optimization.
    - The program expects these two files in the same directory as this script:
          final_Chinese personal loan.xlsx
          final_default of credit card clients.xlsx
    - The dependent variable must be named exactly: label
    - label=1 means default; label=0 means non-default.

The paper states that IBIE repeatedly identifies misclassified minority samples,
updates their weights, samples seed points with replacement, selects nearby paired
seeds using inverse Euclidean distance, interpolates a new minority sample, retrains
an interpretable sub-model, and accepts only correctly identified generated samples.
The implementation below follows that dynamic-loop logic and deliberately allows the
training-set size to change at every IBIE iteration.

Python version target: Python 3.10+
"""

from __future__ import annotations

import copy
import json
import inspect
import math
import os
import sys
import time
import random
import warnings
import traceback
import subprocess
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional

# -----------------------------------------------------------------------------
# Optional dependency bootstrap. The user can disable automatic installation.
# -----------------------------------------------------------------------------
AUTO_INSTALL_MISSING_PACKAGES = False

REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "imblearn": "imbalanced-learn",
    "scipy": "scipy",
    "optuna": "optuna",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "openpyxl": "openpyxl",
    "matplotlib": "matplotlib",
}


def _install_missing_packages() -> None:
    missing = []
    for module, package in REQUIRED_IMPORTS.items():
        try:
            __import__(module)
        except Exception:
            missing.append(package)
    if missing:
        if not AUTO_INSTALL_MISSING_PACKAGES:
            raise ImportError(
                "Missing packages: " + ", ".join(missing) +
                ". Install them with pip or set AUTO_INSTALL_MISSING_PACKAGES=True."
            )
        print("[Setup] Installing missing packages sequentially:", ", ".join(missing))
        for package in missing:
            print(f"[Setup] pip install {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


_install_missing_packages()

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import check_random_state

from imblearn.under_sampling import RandomUnderSampler, NearMiss, ClusterCentroids, TomekLinks
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN

import optuna
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
print("[Environment] sklearn={}, imbalanced-learn={}, optuna={}, lightgbm={}, xgboost={}".format(
    __import__("sklearn").__version__, __import__("imblearn").__version__, optuna.__version__,
    __import__("lightgbm").__version__, __import__("xgboost").__version__))

# =============================================================================
# PARA AREA
# =============================================================================
# Use English only in this section as requested.

QUICK_TEST = False
USE_SAVED_PARAMS = False
RUN_EXTENDED_EXPERIMENTS = True
RUN_SOTA_EXPERIMENT = True
RUN_EXPLORATION_EXPERIMENT = True
RUN_ABLATION_EXPERIMENT = True
RUN_RUNTIME_EXPERIMENT = True
RUN_INTERPRETABILITY_TRACE = True

# Experimental split settings.
N_SPLITS = 5
TEST_SIZE_IN_OUTER_FOLD = 0.20
VALIDATION_RATIO_WITHIN_REMAINING_TRAIN = 0.20

# seeds
GLOBAL_SEED = random.randint(0, 999999)
SPLIT_SEED = random.randint(0, 999999)
OPTUNA_SEED = random.randint(0, 999999)
RESAMPLER_SEED = random.randint(0, 999999)
IBIE_SEED = random.randint(0, 999999)
ROBUSTNESS_SEED = random.randint(0, 999999)
EXPLORATION_SEED = random.randint(0, 999999)
SOTA_SEED = random.randint(0, 999999)

# Significance settings.
ALPHA_SIGNIFICANCE = 0.05
HOLM_ALPHA = 0.05

# Optimization settings. Increase these for a longer search.
COMPARISON_OPTUNA_TRIALS = 3000
IBIE_OPTUNA_TRIALS = 3000
COMPARISON_OPTUNA_TIMEOUT = None
IBIE_OPTUNA_TIMEOUT = None
OPTUNA_SAMPLER = "TPESampler"

# IBIE defaults from the paper.
IBIE_DEFAULT_ALPHA = 0.10
IBIE_DEFAULT_MULTI = 10
IBIE_WEIGHT_CAP = 10.0
IBIE_WEIGHT_INITIAL = 1.0
IBIE_WEIGHT_NORMALIZE_AFTER_UPDATE = True
IBIE_BASE_MODEL = "DT"
IBIE_BASE_MAX_DEPTH = 8
IBIE_MIN_N = 2
IBIE_MAX_N_GROW = 100000
IBIE_MAX_ITERATIONS = 2000
IBIE_EXPLORE_FRACTION = 0.0
IBIE_DISTANCE_EPSILON = 1e-12

# IBIE optimization ranges.
IBIE_ALPHA_RANGE = (0.01, 1.0)
IBIE_MULTI_RANGE = (1, 30)
IBIE_N_FRACTION_OF_DATA = True
IBIE_N_MIN = 2
IBIE_N_MAX = None

# Robustness grids from the paper.
ROBUST_ALPHA_GRID = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]
ROBUST_MULTI_GRID = [1, 5, 10, 15, 20, 25, 30]
ROBUST_IMBALANCE_RATIOS = [0.05, 0.10, 0.40, 0.60]
EXPLORATION_GRID = [round(x, 2) for x in np.arange(0.01, 0.201, 0.01)]
SOTA_METHODS = ["IBIE", "CTGAN", "VAE"]

# Output paths.
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "IBIE_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_XLSX = OUTPUT_DIR / "IBIE_full_results.xlsx"
OUTPUT_TXT = OUTPUT_DIR / "IBIE_parameter_area.txt"
OUTPUT_LOG = OUTPUT_DIR / "IBIE_run_log.txt"
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "PC": SCRIPT_DIR / "final_Chinese personal loan.xlsx",
    "CC": SCRIPT_DIR / "final_default of credit card clients.xlsx",
}

MODEL_NAMES = [
    "Logistic", "NB", "BP", "k-NN", "DT", "SVM", "AdaBoost", "GBDT",
    "LightGBM", "RF", "XGBoost"
]
BALANCERS = [
    "RU", "NM", "CC", "TL", "RO", "SMOTE", "ADASYN", "BSMOTE",
    "MWSMOTE", "SMOTE-ENN"
]
ALL_METHODS = BALANCERS + ["IBIE"]
METRICS = ["AUC", "Sensitivity", "Specificity", "Precision", "GMean", "FScore"]


# =============================================================================
# GLOBAL STATE
# =============================================================================
GLOBAL_RESULTS: Dict[str, List[Dict[str, Any]]] = {
    "comparison_performance": [],
    "ibie_performance": [],
    "model_params": [],
    "seeds": [],
    "fold_summaries": [],
    "friedman": [],
    "ttest": [],
    "wilcoxon": [],
    "cohen_d": [],
    "robust_submodel": [],
    "robust_alpha_multi": [],
    "robust_imbalance": [],
    "exploration": [],
    "sota": [],
    "ablation": [],
    "runtime": [],
    "trace": [],
}

SAVED_PARAMETER_AREA: Dict[str, Any] = {}


# =============================================================================
# LOGGING
# =============================================================================
class TeeLogger:
    def __init__(self, path: Path):
        self.path = path
        self.fp = open(path, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, message: str) -> None:
        self._stdout.write(message)
        self.fp.write(message)
        self.fp.flush()

    def flush(self) -> None:
        self._stdout.flush()
        self.fp.flush()

    def close(self) -> None:
        self.fp.close()
        sys.stdout = self._stdout


# =============================================================================
# REPRODUCIBILITY UTILITIES
# =============================================================================

def stable_seed(*parts: Any, base: int = 0) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16) % 100000 + int(base)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


# =============================================================================
# DATA PREPROCESSING
# =============================================================================
def load_dataset(name: str, path: Path) -> pd.DataFrame:
    print(f"[Data] Loading {name}: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_excel(path)
    if "label" not in df.columns:
        raise ValueError(f"Dataset {name} must contain a column named 'label'.")
    if df["label"].isna().any():
        raise ValueError(f"Dataset {name} contains missing labels.")
    df = df.copy()
    df["label"] = df["label"].astype(int)
    unique_labels = set(df["label"].dropna().unique().tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Dataset {name} label values must be 0/1. Found: {unique_labels}")
    print(f"[Data] {name}: rows={len(df)}, features={df.shape[1]-1}, default_rate={df['label'].mean():.4f}")
    return df


def make_numeric_matrix(train_df: pd.DataFrame, apply_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Convert arbitrary feature columns to a purely numeric matrix.
    Categorical object columns are factorized using training data only.
    """
    train_x = train_df.drop(columns=["label"]).copy()
    apply_x = apply_df.drop(columns=["label"]).copy()

    for col in train_x.columns:
        if pd.api.types.is_numeric_dtype(train_x[col]):
            train_x[col] = pd.to_numeric(train_x[col], errors="coerce")
            apply_x[col] = pd.to_numeric(apply_x[col], errors="coerce")
        else:
            categories = pd.Index(train_x[col].astype(str).fillna("__MISSING__").unique())
            mapping = {v: i for i, v in enumerate(categories)}
            train_x[col] = train_x[col].astype(str).fillna("__MISSING__").map(mapping).astype(float)
            apply_x[col] = apply_x[col].astype(str).fillna("__MISSING__").map(mapping).fillna(-1).astype(float)
    train_x = train_x.replace([np.inf, -np.inf], np.nan)
    apply_x = apply_x.replace([np.inf, -np.inf], np.nan)
    medians = train_x.median(numeric_only=True)
    train_x = train_x.fillna(medians).fillna(0)
    apply_x = apply_x.fillna(medians).fillna(0)
    return train_x.to_numpy(dtype=float), apply_x.to_numpy(dtype=float), list(train_x.columns)


def split_outer_train_validation(
    X_outer: np.ndarray,
    y_outer: np.ndarray,
    outer_train_index: np.ndarray,
    outer_test_index: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Within each outer fold, use 80% of the remaining 80% as training and 20% as validation.
    Overall proportions are therefore approximately 64% train / 16% validation / 20% test.
    """
    rng = np.random.RandomState(seed)
    tr_indices = np.array(outer_train_index)
    y_tr = y_outer[tr_indices]
    # Stratified split implemented manually to keep one explicit random seed.
    train_idx_local = []
    val_idx_local = []
    for cls in [0, 1]:
        cls_idx = tr_indices[y_tr == cls]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * VALIDATION_RATIO_WITHIN_REMAINING_TRAIN)))
        val_idx_local.extend(cls_idx[:n_val].tolist())
        train_idx_local.extend(cls_idx[n_val:].tolist())
    train_idx = np.array(sorted(train_idx_local))
    val_idx = np.array(sorted(val_idx_local))
    test_idx = np.array(outer_test_index)
    return (
        X_outer[train_idx], y_outer[train_idx],
        X_outer[val_idx], y_outer[val_idx],
        X_outer[test_idx], y_outer[test_idx],
    )


def generate_folds(df: pd.DataFrame, seed: int) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    y = df["label"].to_numpy(dtype=int)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    folds = []
    idx = np.arange(len(df))
    for fold_id, (outer_train, outer_test) in enumerate(skf.split(idx, y), start=1):
        folds.append((outer_train, outer_test, np.array([fold_id])))
    return folds


# =============================================================================
# METRICS
# =============================================================================
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    gmean = math.sqrt(max(sensitivity * specificity, 0.0))
    fscore = f1_score(y_true, y_pred, zero_division=0)
    return {
        "AUC": float(auc),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "GMean": float(gmean),
        "FScore": float(fscore),
    }


# =============================================================================
# MODEL FACTORIES AND SEARCH SPACES
# =============================================================================
def default_model(model_name: str, seed: int):
    if model_name == "Logistic":
        return LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=seed)
    if model_name == "NB":
        return GaussianNB()
    if model_name == "BP":
        return MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam", alpha=1e-4,
                             learning_rate_init=1e-3, max_iter=3000, random_state=seed)
    if model_name == "k-NN":
        return KNeighborsClassifier(n_neighbors=15, weights="distance", p=2, n_jobs=1)
    if model_name == "DT":
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=seed)
    if model_name == "SVM":
        return SVC(C=1.0, gamma="scale", kernel="rbf", probability=True, random_state=seed)
    if model_name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=100, learning_rate=0.05, random_state=seed)
    if model_name == "GBDT":
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3,
                                          min_samples_leaf=5, random_state=seed)
    if model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=100, learning_rate=0.05, num_leaves=31, max_depth=-1,
            subsample=1.0, colsample_bytree=1.0, reg_lambda=0.0,
            random_state=seed, n_jobs=1, verbosity=-1
        )
    if model_name == "RF":
        return RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=1,
                                      max_features="sqrt", random_state=seed, n_jobs=1)
    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5, min_child_weight=1,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss", random_state=seed,
            n_jobs=1, tree_method="hist", verbosity=0
        )
    raise KeyError(model_name)


def suggest_model_params(trial: optuna.Trial, model_name: str, seed: int) -> Dict[str, Any]:
    if model_name == "Logistic":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "solver": trial.suggest_categorical("solver", ["liblinear", "lbfgs"]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "max_iter": trial.suggest_int("max_iter", 500, 3000, step=500),
            "random_state": seed,
        }
    if model_name == "NB":
        return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True)}
    if model_name == "BP":
        units1 = trial.suggest_int("units1", 16, 128, step=16)
        units2 = trial.suggest_categorical("units2", [0, 16, 32, 64])
        hidden = (units1,) if units2 == 0 else (units1, units2)
        return {
            "hidden_layer_sizes": hidden,
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "max_iter": trial.suggest_int("max_iter", 150, 500, step=50),
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
            "random_state": seed,
        }
    if model_name == "k-NN":
        return {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 51, step=2),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2),
            "n_jobs": 1,
        }
    if model_name == "DT":
        return {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", [None, "sqrt", "log2"]),
            "random_state": seed,
        }
    if model_name == "SVM":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-5, 1e0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
            "probability": True,
            "random_state": seed,
        }
    if model_name == "AdaBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
            "random_state": seed,
        }
    if model_name == "GBDT":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": seed,
        }
    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 127),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": seed,
            "n_jobs": 1,
            "verbosity": -1,
        }
    if model_name == "RF":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=25),
            "max_depth": trial.suggest_categorical("max_depth", [None, 5, 8, 12, 16, 24]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "random_state": seed,
            "n_jobs": 1,
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 75, 400, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 9),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": seed,
            "n_jobs": 1,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "verbosity": 0,
        }
    raise KeyError(model_name)


def _constructor_supported_kwargs(estimator_cls, params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter constructor arguments against the installed library version."""
    params = dict(params or {})
    try:
        signature = inspect.signature(estimator_cls.__init__)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            return params
        accepted = {name for name, p in signature.parameters.items()
                    if name != "self" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
        dropped = sorted(set(params) - accepted)
        if dropped:
            print(f"      [Compatibility] {estimator_cls.__name__}: ignored unsupported parameters: {dropped}")
        return {k: v for k, v in params.items() if k in accepted}
    except Exception as exc:
        print(f"      [Compatibility] Signature inspection failed for {estimator_cls.__name__}: {exc}")
        return params

def _build_with_supported_kwargs(estimator_cls, params: Optional[Dict[str, Any]]):
    safe_params = _constructor_supported_kwargs(estimator_cls, params or {})
    return estimator_cls(**safe_params)

def build_model(model_name: str, params: Optional[Dict[str, Any]], seed: int):
    params = dict(params or {})
    factories = {
        "Logistic": LogisticRegression, "NB": GaussianNB, "BP": MLPClassifier,
        "k-NN": KNeighborsClassifier, "DT": DecisionTreeClassifier, "SVM": SVC,
        "AdaBoost": AdaBoostClassifier, "GBDT": GradientBoostingClassifier,
        "LightGBM": LGBMClassifier, "RF": RandomForestClassifier, "XGBoost": XGBClassifier,
    }
    if model_name not in factories:
        raise KeyError(model_name)
    if model_name == "k-NN":
        params["n_neighbors"] = max(1, int(params.get("n_neighbors", 5)))
    return _build_with_supported_kwargs(factories[model_name], params)


def model_predict_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))
    return model.predict(X).astype(float)


# =============================================================================
# RESAMPLING METHODS
# =============================================================================
class MWMOTECompat:
    """
    A sequential MWMOTE-style implementation.
    It identifies difficult minority samples from local majority-neighbor pressure,
    assigns larger synthesis weights to informative minority points, then performs
    weighted minority interpolation until a 1:1 target is obtained.

    This implementation is intentionally self-contained so the script does not depend
    on a third-party package with inconsistent MWMOTE APIs.
    """
    def __init__(self, random_state: int, target_ratio: float = 1.0, k_neighbors: int = 5):
        self.random_state = random_state
        self.target_ratio = target_ratio
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        idx_min = np.flatnonzero(y == 1)
        idx_maj = np.flatnonzero(y == 0)
        if len(idx_min) == 0 or len(idx_maj) == 0:
            return X.copy(), y.copy()
        target_min = int(math.ceil(len(idx_maj) * self.target_ratio))
        if len(idx_min) >= target_min:
            return X.copy(), y.copy()
        need = target_min - len(idx_min)
        # Distance matrix between minority and majority samples.
        min_x = X[idx_min]
        maj_x = X[idx_maj]
        d_mm = pairwise_squared_euclidean(min_x, min_x)
        d_mj = pairwise_squared_euclidean(min_x, maj_x)
        k = min(self.k_neighbors, max(1, len(idx_maj)))
        kth = np.partition(d_mj, kth=k-1, axis=1)[:, k-1]
        difficulty = 1.0 / (kth + 1e-12)
        # Local majority pressure.
        pressure = (d_mj < np.median(d_mj, axis=1, keepdims=True)).mean(axis=1)
        weights = difficulty * (0.5 + pressure)
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
        seeds = rng.choice(len(min_x), size=need, replace=True, p=weights)
        generated = []
        for seed_idx in seeds:
            # Favor close minority partners.
            distances = d_mm[seed_idx].copy()
            distances[seed_idx] = np.inf
            probs = 1.0 / (distances + 1e-12)
            probs[~np.isfinite(probs)] = 0.0
            if probs.sum() <= 0:
                partner_idx = rng.randint(0, len(min_x))
            else:
                probs = probs / probs.sum()
                partner_idx = rng.choice(len(min_x), p=probs)
            beta = rng.uniform(0.0, 1.0)
            generated.append(min_x[seed_idx] + beta * (min_x[partner_idx] - min_x[seed_idx]))
        X_new = np.asarray(generated)
        y_new = np.ones(len(X_new), dtype=int)
        return np.vstack([X, X_new]), np.concatenate([y, y_new])


def pairwise_squared_euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = np.sum(A * A, axis=1, keepdims=True)
    bb = np.sum(B * B, axis=1, keepdims=True).T
    return np.maximum(aa + bb - 2.0 * A.dot(B.T), 0.0)


def resample_data(method: str, X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    print(f"      [Balance] {method}: input={len(y)}, minority={int((y==1).sum())}, majority={int((y==0).sum())}")
    if method == "RU":
        sampler = RandomUnderSampler(sampling_strategy=1.0, random_state=seed)
    elif method == "NM":
        sampler = NearMiss(version=1, sampling_strategy=1.0)
    elif method == "CC":
        sampler = ClusterCentroids(sampling_strategy=1.0, random_state=seed)
    elif method == "TL":
        sampler = TomekLinks()
    elif method == "RO":
        sampler = RandomOverSampler(sampling_strategy=1.0, random_state=seed)
    elif method == "SMOTE":
        k = max(1, min(5, int((y == 1).sum()) - 1))
        sampler = SMOTE(sampling_strategy=1.0, random_state=seed, k_neighbors=k)
    elif method == "ADASYN":
        n_neighbors = max(1, min(5, int((y == 1).sum()) - 1))
        sampler = ADASYN(sampling_strategy=1.0, random_state=seed, n_neighbors=n_neighbors)
    elif method == "BSMOTE":
        k = max(1, min(5, int((y == 1).sum()) - 1))
        sampler = BorderlineSMOTE(sampling_strategy=1.0, random_state=seed, k_neighbors=k)
    elif method == "MWSMOTE":
        sampler = MWMOTECompat(random_state=seed, target_ratio=1.0)
    elif method == "SMOTE-ENN":
        k = max(1, min(5, int((y == 1).sum()) - 1))
        sampler = SMOTEENN(sampling_strategy=1.0, random_state=seed,
                           smote=SMOTE(sampling_strategy=1.0, random_state=seed, k_neighbors=k))
    else:
        raise KeyError(method)
    try:
        Xr, yr = sampler.fit_resample(X, y)
        # ENN can remove an entire class on very small/noisy folds. In that pathological
        # case the downstream binary classifiers and AUC are undefined, so use a
        # deterministic 1:1 SMOTE fallback rather than allowing the experiment to crash.
        if len(np.unique(yr)) < 2:
            print(f"      [Balance] Warning: {method} removed one class. Applying a deterministic SMOTE fallback.")
            k_fb = max(1, min(5, int((y == 1).sum()) - 1))
            fallback = SMOTE(sampling_strategy=1.0, random_state=seed, k_neighbors=k_fb)
            Xr, yr = fallback.fit_resample(X, y)
    except Exception as exc:
        # Some resamplers can become infeasible when a fold has very few minority samples.
        print(f"      [Balance] Warning: {method} failed with {type(exc).__name__}: {exc}. Falling back to RU/RO where possible.")
        if method in {"RU", "NM", "CC", "TL"}:
            fallback = RandomUnderSampler(sampling_strategy=1.0, random_state=seed)
        else:
            fallback = RandomOverSampler(sampling_strategy=1.0, random_state=seed)
        Xr, yr = fallback.fit_resample(X, y)
    print(f"      [Balance] {method}: output={len(yr)}, minority={int((yr==1).sum())}, majority={int((yr==0).sum())}")
    return np.asarray(Xr, dtype=float), np.asarray(yr, dtype=int)


# =============================================================================
# IBIE IMPLEMENTATION
# =============================================================================
@dataclass
class IBIETraceRow:
    dataset: str
    fold: int
    iteration: int
    n_before: int
    minority_before: int
    majority_count: int
    n_used: int
    wrong_minority_count: int
    generated_count: int
    accepted_count: int
    minority_after: int
    weight_max: float
    mean_weight: float
    seed_index_sample: str
    pair_index_sample: str


class IBIEBalancer:
    def __init__(
        self,
        alpha: float,
        n: int,
        multi: int,
        base_model: str = "DT",
        base_max_depth: int = 4,
        weight_cap: float = 1.0,
        weight_initial: float = 1.0,
        normalize_weights_after_update: bool = True,
        explore_fraction: float = 0.0,
        random_state: int = 1,
        max_iterations: int = 500,
        min_n: int = 2,
    ):
        self.alpha = float(alpha)
        self.n = int(max(1, n))
        self.multi = int(max(1, multi))
        self.base_model = base_model
        self.base_max_depth = base_max_depth
        self.weight_cap = float(weight_cap)
        self.weight_initial = float(weight_initial)
        self.normalize_weights_after_update = bool(normalize_weights_after_update)
        self.explore_fraction = float(explore_fraction)
        self.random_state = int(random_state)
        self.max_iterations = int(max_iterations)
        self.min_n = int(max(2, min_n))
        self.trace: List[Dict[str, Any]] = []

    def _make_base_model(self, seed_offset: int = 0):
        seed = self.random_state + seed_offset
        if self.base_model == "Logistic":
            return LogisticRegression(C=1.0, solver="liblinear", max_iter=1500, random_state=seed)
        if self.base_model == "NB":
            return GaussianNB()
        return DecisionTreeClassifier(max_depth=self.base_max_depth, random_state=seed)

    def _generate_from_seed_indices(self, X_min_norm, seed_indices, rng):
        generated = []
        pair_indices = []
        if len(seed_indices) == 0:
            return np.empty((0, X_min_norm.shape[1])), pair_indices
        for s_idx in seed_indices:
            # Use current seed pool only; select partner according to inverse distance.
            local_points = X_min_norm[np.array(seed_indices)]
            s_local = int(np.where(np.array(seed_indices) == s_idx)[0][0])
            d = np.linalg.norm(local_points - X_min_norm[s_idx], axis=1)
            d[s_local] = np.inf
            probs = 1.0 / (d + IBIE_DISTANCE_EPSILON)
            probs[~np.isfinite(probs)] = 0.0
            if probs.sum() <= 0:
                p_local = rng.randint(0, len(seed_indices))
            else:
                probs = probs / probs.sum()
                p_local = rng.choice(len(seed_indices), p=probs)
            partner = int(seed_indices[p_local])
            pair_indices.append(partner)
            for _ in range(self.multi):
                beta = rng.uniform(0.0, 1.0)
                x_new = X_min_norm[s_idx] + beta * (X_min_norm[partner] - X_min_norm[s_idx])
                generated.append(x_new)
        if not generated:
            return np.empty((0, X_min_norm.shape[1])), pair_indices
        return np.asarray(generated, dtype=float), pair_indices

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = "Unknown",
        fold_id: int = 0,
        collect_trace: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        # Min-max normalization exactly for the IBIE feature-space distance calculation.
        scaler = MinMaxScaler()
        X_norm = scaler.fit_transform(X)
        majority_count = int((y == 0).sum())
        minority_mask = y == 1
        X_min_norm = X_norm[minority_mask].copy()
        X_min_raw = X[minority_mask].copy()
        weights = np.full(len(X_min_norm), self.weight_initial, dtype=float)
        n_initial = max(self.min_n, min(self.n, len(X_min_norm)))
        n_current = n_initial
        current_X_norm = X_norm.copy()
        current_y = y.copy()
        current_X_raw = X.copy()
        # Track all minority samples in normalized/raw coordinates.
        minority_norm = X_min_norm.copy()
        minority_raw = X_min_raw.copy()

        if len(minority_raw) >= majority_count:
            return current_X_raw, current_y, []

        for iteration in range(1, self.max_iterations + 1):
            if len(minority_raw) >= majority_count:
                break
            n_before = len(current_y)
            current_X_norm = scaler.transform(current_X_raw)
            # Train the interpretable sub-model on the dynamically growing dataset.
            base = self._make_base_model(iteration)
            base.fit(current_X_norm, current_y)
            pred = base.predict(current_X_norm)
            wrong_positions_global = np.flatnonzero((current_y == 1) & (pred == 0))
            # Map current global minority positions to the dynamic minority arrays.
            minority_global = np.flatnonzero(current_y == 1)
            wrong_set = set(wrong_positions_global.tolist())
            wrong_min_local = [i for i, g in enumerate(minority_global) if g in wrong_set]
            # Step 2: update historical weight of each repeatedly misclassified minority sample.
            if wrong_min_local:
                for li in wrong_min_local:
                    weights[li] = min(self.weight_cap, weights[li] * (1.0 + self.alpha))
                if self.normalize_weights_after_update:
                    # Preserve relative hard-example emphasis while retaining the requested cap of 1.
                    max_w = max(float(weights.max()), 1e-12)
                    weights = np.clip(weights / max_w, 0.0, self.weight_cap)
            else:
                # If no minority sample is misclassified, sample from all minority points.
                wrong_min_local = list(range(len(minority_raw)))

            # Weighted roulette-wheel seed sampling.
            weights_safe = np.nan_to_num(weights, nan=0.0, posinf=self.weight_cap, neginf=0.0)
            if weights_safe.sum() <= 0:
                probs = np.full(len(weights_safe), 1.0 / len(weights_safe))
            else:
                probs = weights_safe / weights_safe.sum()
            n_used = max(self.min_n, min(int(n_current), len(minority_raw)))
            seed_local = rng.choice(len(minority_raw), size=n_used, replace=True, p=probs)

            # Step 4: pair selected seeds by inverse distance and generate multi samples per seed.
            generated_norm, pair_indices = self._generate_from_seed_indices(minority_norm, seed_local.tolist(), rng)
            generated_count = len(generated_norm)
            generated_raw = scaler.inverse_transform(generated_norm) if generated_count else np.empty((0, X.shape[1]))
            generated_raw = np.clip(generated_raw, np.nanmin(X, axis=0), np.nanmax(X, axis=0))

            # Step 5: retrain base model on current + generated samples and validate generated points.
            accepted_count = 0
            selected_trace_idx = []
            if generated_count:
                tmp_X_raw = np.vstack([current_X_raw, generated_raw])
                tmp_y = np.concatenate([current_y, np.ones(generated_count, dtype=int)])
                tmp_norm = scaler.transform(tmp_X_raw)
                validator = self._make_base_model(iteration + 100000)
                validator.fit(tmp_norm, tmp_y)
                generated_pred = validator.predict(tmp_norm[-generated_count:])
                accepted_positions = np.flatnonzero(generated_pred == 1)
                rejected_positions = np.flatnonzero(generated_pred == 0)
                # Controlled exploration mechanism from Section 5.5.
                if self.explore_fraction > 0 and len(rejected_positions) > 0:
                    n_extra = max(1, int(round(len(rejected_positions) * self.explore_fraction)))
                    extra = rng.choice(rejected_positions, size=min(n_extra, len(rejected_positions)), replace=False)
                    accepted_positions = np.unique(np.concatenate([accepted_positions, extra]))
                if len(accepted_positions):
                    remaining = majority_count - len(minority_raw)
                    accepted_positions = accepted_positions[:remaining]
                    accepted_count = len(accepted_positions)
                    if accepted_count > 0:
                        accepted_raw = generated_raw[accepted_positions]
                        current_X_raw = np.vstack([current_X_raw, accepted_raw])
                        current_y = np.concatenate([current_y, np.ones(accepted_count, dtype=int)])
                        accepted_norm = generated_norm[accepted_positions]
                        minority_raw = np.vstack([minority_raw, accepted_raw])
                        minority_norm = np.vstack([minority_norm, accepted_norm])
                        weights = np.concatenate([weights, np.full(accepted_count, self.weight_initial)])
                        selected_trace_idx = accepted_positions.tolist()

            # If no validated sample exists, grow n by 10%, as in the paper.
            if accepted_count == 0:
                new_n = int(round(max(self.min_n, n_current * 1.10)))
                n_current = min(max(new_n, n_current + 1), IBIE_MAX_N_GROW)
            else:
                n_current = n_initial

            if collect_trace:
                self.trace.append(asdict(IBIETraceRow(
                    dataset=dataset_name,
                    fold=fold_id,
                    iteration=iteration,
                    n_before=n_before,
                    minority_before=int(n_before - majority_count) if n_before > majority_count else int((current_y==1).sum() - accepted_count),
                    majority_count=majority_count,
                    n_used=n_used,
                    wrong_minority_count=len(wrong_min_local),
                    generated_count=generated_count,
                    accepted_count=accepted_count,
                    minority_after=int((current_y == 1).sum()),
                    weight_max=float(weights.max()) if len(weights) else 0.0,
                    mean_weight=float(weights.mean()) if len(weights) else 0.0,
                    seed_index_sample=",".join(map(str, seed_local[:20].tolist() if hasattr(seed_local, 'tolist') else seed_local[:20])),
                    pair_index_sample=",".join(map(str, pair_indices[:20])),
                )))

            if len(minority_raw) >= majority_count:
                break

            # Safety condition for pathological data.
            if iteration == self.max_iterations:
                print(f"      [IBIE] Warning: reached max_iterations={self.max_iterations} before exact balance.")
                break

        return current_X_raw, current_y, self.trace


def ibie_default_n_from_data_size(n_samples: int) -> int:
    return max(IBIE_MIN_N, int(round(math.sqrt(n_samples))))


# =============================================================================
# MODEL PIPELINE / TRAINING
# =============================================================================
def fit_model(model_name: str, X_train: np.ndarray, y_train: np.ndarray, params: Optional[Dict[str, Any]], seed: int):
    model = build_model(model_name, params, seed)
    # Tree-based models do not need scaling, but applying one consistent scaling pipeline
    # prevents downstream models from being dominated by feature magnitude.
    if model_name in {"DT", "AdaBoost", "GBDT", "LightGBM", "RF", "XGBoost"}:
        estimator = model
    else:
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ])
    # Final defensive fit-time guard for k-NN after resampling/filtering.
    if model_name == "k-NN":
        try:
            estimator.set_params(model__n_neighbors=max(1, min(int(estimator.get_params().get("model__n_neighbors", 5)), len(y_train))))
        except Exception:
            pass
    estimator.fit(X_train, y_train)
    return estimator


# =============================================================================
# FOLD DATA CACHE
# =============================================================================
@dataclass
class FoldData:
    fold_id: int
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def prepare_fold_data(df: pd.DataFrame, split_seed: int) -> List[FoldData]:
    label = df["label"].to_numpy(dtype=int)
    folds = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=split_seed)
    idx = np.arange(len(df))
    for fold_id, (outer_train, outer_test) in enumerate(skf.split(idx, label), start=1):
        rng = np.random.RandomState(split_seed + fold_id)
        train_idx = []
        val_idx = []
        for cls in [0, 1]:
            cls_idx = outer_train[label[outer_train] == cls].copy()
            rng.shuffle(cls_idx)
            n_val = max(1, int(round(len(cls_idx) * VALIDATION_RATIO_WITHIN_REMAINING_TRAIN)))
            val_idx.extend(cls_idx[:n_val].tolist())
            train_idx.extend(cls_idx[n_val:].tolist())
        train_idx = np.array(sorted(train_idx))
        val_idx = np.array(sorted(val_idx))
        test_idx = np.array(outer_test)

        # All preprocessing statistics are estimated from the current training fold only.
        # Validation and test data are transformed using those training-derived statistics.
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        test_df = df.iloc[test_idx].copy()
        X_train, _, _ = make_numeric_matrix(train_df, train_df)
        _, X_val, _ = make_numeric_matrix(train_df, val_df)
        _, X_test, _ = make_numeric_matrix(train_df, test_df)
        folds.append(FoldData(
            fold_id=fold_id,
            X_train=X_train, y_train=label[train_idx],
            X_val=X_val, y_val=label[val_idx],
            X_test=X_test, y_test=label[test_idx],
        ))
        print(f"[Split] Fold {fold_id}: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
    return folds


# =============================================================================
# OPTUNA SEARCH
# =============================================================================
def make_sampler(seed: int):
    if OPTUNA_SAMPLER == "TPESampler":
        return optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    return optuna.samplers.TPESampler(seed=seed)


def optimize_comparison_model(
    dataset_name: str,
    model_name: str,
    balancer_name: str,
    folds: List[FoldData],
    seed: int,
) -> Dict[str, Any]:
    if QUICK_TEST:
        params = json_safe(default_model(model_name, seed).get_params())
        return {"params": params, "best_value": np.nan, "n_trials": 0, "study_seed": seed, "mode": "quick_test"}

    study_seed = seed
    print(f"    [Optuna] Comparison optimization: dataset={dataset_name}, balancer={balancer_name}, model={model_name}")
    study = optuna.create_study(direction="maximize", sampler=make_sampler(study_seed))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_model_params(trial, model_name, seed + trial.number)
        fold_scores = []
        for fd in folds:
            Xr, yr = resample_data(balancer_name, fd.X_train, fd.y_train, RESAMPLER_SEED + fd.fold_id + trial.number)
            model = fit_model(model_name, Xr, yr, params, seed + fd.fold_id + trial.number)
            p = model_predict_proba(model, fd.X_val)
            fold_scores.append(roc_auc_score(fd.y_val, p))
        score = float(np.mean(fold_scores))
        print(f"      [Optuna] trial={trial.number} mean_validation_AUC={score:.6f}")
        return score

    study.optimize(objective, n_trials=COMPARISON_OPTUNA_TRIALS, timeout=COMPARISON_OPTUNA_TIMEOUT, n_jobs=1, gc_after_trial=True)
    best = study.best_trial
    best_params = dict(best.params)
    # Reconstruct derived BP hidden sizes.
    if model_name == "BP":
        units1 = best_params.pop("units1")
        units2 = best_params.pop("units2")
        best_params["hidden_layer_sizes"] = (units1,) if units2 == 0 else (units1, units2)
        best_params.update({"early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 20})
    if model_name != "NB":
        best_params["random_state"] = seed
    if model_name in {"k-NN", "RF", "LightGBM", "XGBoost"}:
        best_params.setdefault("n_jobs", 1)
    return {
        "params": json_safe(best_params),
        "best_value": float(best.value),
        "n_trials": len(study.trials),
        "study_seed": study_seed,
        "mode": "optuna",
    }


def optimize_ibie_model(
    dataset_name: str,
    model_name: str,
    folds: List[FoldData],
    seed: int,
) -> Dict[str, Any]:
    if USE_SAVED_PARAMS and dataset_name in SAVED_PARAMETER_AREA.get("ibie", {}) and model_name in SAVED_PARAMETER_AREA["ibie"][dataset_name]:
        return SAVED_PARAMETER_AREA["ibie"][dataset_name][model_name]
    if QUICK_TEST:
        return {
            "ibie": {
                "alpha": IBIE_DEFAULT_ALPHA,
                "multi": IBIE_DEFAULT_MULTI,
                "n": ibie_default_n_from_data_size(folds[0].X_train.shape[0]),
                "base_model": IBIE_BASE_MODEL,
                "base_max_depth": IBIE_BASE_MAX_DEPTH,
            },
            "model_params": json_safe(default_model(model_name, seed).get_params()),
            "best_value": np.nan,
            "n_trials": 0,
            "study_seed": seed,
            "mode": "quick_test",
        }

    print(f"    [Optuna] IBIE joint optimization: dataset={dataset_name}, model={model_name}")
    study = optuna.create_study(direction="maximize", sampler=make_sampler(seed))

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("ibie_alpha", IBIE_ALPHA_RANGE[0], IBIE_ALPHA_RANGE[1], log=True)
        multi = trial.suggest_int("ibie_multi", IBIE_MULTI_RANGE[0], IBIE_MULTI_RANGE[1])
        n_default = ibie_default_n_from_data_size(folds[0].X_train.shape[0])
        if IBIE_N_FRACTION_OF_DATA:
            lower = max(IBIE_N_MIN, int(round(n_default * 0.5)))
            upper = int(round(n_default * 2.0))
        else:
            lower = IBIE_N_MIN
            upper = IBIE_N_MAX or max(lower, n_default * 2)
        n = trial.suggest_int("ibie_n", lower, max(lower, upper))
        params = suggest_model_params(trial, model_name, seed + trial.number)
        fold_scores = []
        for fd in folds:
            balancer = IBIEBalancer(
                alpha=alpha,
                n=n,
                multi=multi,
                base_model=IBIE_BASE_MODEL,
                base_max_depth=IBIE_BASE_MAX_DEPTH,
                weight_cap=IBIE_WEIGHT_CAP,
                weight_initial=IBIE_WEIGHT_INITIAL,
                normalize_weights_after_update=IBIE_WEIGHT_NORMALIZE_AFTER_UPDATE,
                explore_fraction=0.0,
                random_state=IBIE_SEED + fd.fold_id + trial.number,
                max_iterations=IBIE_MAX_ITERATIONS,
                min_n=IBIE_MIN_N,
            )
            Xr, yr, _ = balancer.fit_resample(fd.X_train, fd.y_train, dataset_name, fd.fold_id, False)
            model = fit_model(model_name, Xr, yr, params, seed + fd.fold_id + trial.number)
            p = model_predict_proba(model, fd.X_val)
            fold_scores.append(roc_auc_score(fd.y_val, p))
        score = float(np.mean(fold_scores))
        print(f"      [Optuna] trial={trial.number} mean_validation_AUC={score:.6f} alpha={alpha:.4f} n={n} multi={multi}")
        return score

    study.optimize(objective, n_trials=IBIE_OPTUNA_TRIALS, timeout=IBIE_OPTUNA_TIMEOUT, n_jobs=1, gc_after_trial=True)
    best = study.best_trial
    params = dict(best.params)
    ibie_params = {
        "alpha": float(params.pop("ibie_alpha")),
        "multi": int(params.pop("ibie_multi")),
        "n": int(params.pop("ibie_n")),
        "base_model": IBIE_BASE_MODEL,
        "base_max_depth": IBIE_BASE_MAX_DEPTH,
    }
    if model_name == "BP":
        units1 = params.pop("units1")
        units2 = params.pop("units2")
        params["hidden_layer_sizes"] = (units1,) if units2 == 0 else (units1, units2)
        params.update({"early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 20})
    if model_name != "NB":
        params["random_state"] = seed
    if model_name in {"k-NN", "RF", "LightGBM", "XGBoost"}:
        params.setdefault("n_jobs", 1)
    return {
        "ibie": json_safe(ibie_params),
        "model_params": json_safe(params),
        "best_value": float(best.value),
        "n_trials": len(study.trials),
        "study_seed": seed,
        "mode": "optuna",
    }


# =============================================================================
# EVALUATION ROUTINES
# =============================================================================
def evaluate_one_method_model(
    dataset_name: str,
    method_name: str,
    model_name: str,
    folds: List[FoldData],
    params: Dict[str, Any],
    base_seed: int,
) -> List[Dict[str, Any]]:
    rows = []
    for fd in folds:
        print(f"      [Eval] {dataset_name} | {method_name} | {model_name} | fold={fd.fold_id}")
        if method_name == "IBIE":
            ibie_p = params["ibie"]
            balancer = IBIEBalancer(
                alpha=ibie_p["alpha"], n=ibie_p["n"], multi=ibie_p["multi"],
                base_model=ibie_p.get("base_model", IBIE_BASE_MODEL),
                base_max_depth=ibie_p.get("base_max_depth", IBIE_BASE_MAX_DEPTH),
                weight_cap=IBIE_WEIGHT_CAP,
                weight_initial=IBIE_WEIGHT_INITIAL,
                normalize_weights_after_update=IBIE_WEIGHT_NORMALIZE_AFTER_UPDATE,
                explore_fraction=0.0,
                random_state=IBIE_SEED + fd.fold_id,
                max_iterations=IBIE_MAX_ITERATIONS,
                min_n=IBIE_MIN_N,
            )
            Xr, yr, trace = balancer.fit_resample(fd.X_train, fd.y_train, dataset_name, fd.fold_id, RUN_INTERPRETABILITY_TRACE)
            for tr in trace:
                GLOBAL_RESULTS["trace"].append(tr)
        else:
            Xr, yr = resample_data(method_name, fd.X_train, fd.y_train, RESAMPLER_SEED + fd.fold_id)
        model_params = params.get("model_params", params) if method_name == "IBIE" else params
        model = fit_model(model_name, Xr, yr, model_params, base_seed + fd.fold_id)
        p = model_predict_proba(model, fd.X_test)
        metrics = compute_metrics(fd.y_test, p)
        row = {"Dataset": dataset_name, "Model": model_name, "Algorithm": method_name, "Fold": fd.fold_id}
        row.update(metrics)
        rows.append(row)
    return rows


# =============================================================================
# PARAMETER AREA SAVE / LOAD
# =============================================================================
def save_parameter_area(path: Path, comparison_params, ibie_params, seeds) -> None:
    payload = {
        "config": {
            "GLOBAL_SEED": GLOBAL_SEED,
            "SPLIT_SEED": SPLIT_SEED,
            "OPTUNA_SEED": OPTUNA_SEED,
            "RESAMPLER_SEED": RESAMPLER_SEED,
            "IBIE_SEED": IBIE_SEED,
            "ROBUSTNESS_SEED": ROBUSTNESS_SEED,
            "EXPLORATION_SEED": EXPLORATION_SEED,
            "SOTA_SEED": SOTA_SEED,
            "IBIE_WEIGHT_CAP": IBIE_WEIGHT_CAP,
            "IBIE_WEIGHT_INITIAL": IBIE_WEIGHT_INITIAL,
            "IBIE_WEIGHT_NORMALIZE_AFTER_UPDATE": IBIE_WEIGHT_NORMALIZE_AFTER_UPDATE,
            "IBIE_BASE_MODEL": IBIE_BASE_MODEL,
            "IBIE_BASE_MAX_DEPTH": IBIE_BASE_MAX_DEPTH,
        },
        "comparison": comparison_params,
        "ibie": ibie_params,
        "seeds": seeds,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write("# ====================== PARA AREA ======================\n")
        f.write("# Set USE_SAVED_PARAMS=True in the Python file to activate this area.\n")
        f.write("# This file is JSON-compatible Python data for direct copy/paste.\n")
        f.write(json.dumps(json_safe(payload), indent=4, ensure_ascii=True))
        f.write("\n# ==================== END PARA AREA ====================\n")
    print(f"[Output] Parameter area saved to: {path}")


def load_parameter_area(path: Path) -> Dict[str, Any]:
    if not path.exists():
        print(f"[Params] Saved parameter file not found: {path}. Continuing without it.")
        return {}
    text = path.read_text(encoding="utf-8")
    start = text.find("{\n")
    end = text.rfind("\n}")
    if start < 0 or end < 0:
        return {}
    return json.loads(text[start:end+2])


# =============================================================================
# STATISTICAL TESTS
# =============================================================================
def friedman_tests(performance_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # For each dataset/metric, treat each model-fold combination as one block.
    # 11 methods x 55 observations (11 models x 5 folds) matches the paper's design.
    for dataset in sorted(performance_df["Dataset"].unique()):
        sub = performance_df[performance_df["Dataset"] == dataset]
        for metric in METRICS:
            pivot = sub.pivot_table(index=["Model", "Fold"], columns="Algorithm", values=metric, aggfunc="mean")
            pivot = pivot.dropna(axis=0, how="any")
            if pivot.shape[1] < len(ALL_METHODS) or len(pivot) < 2:
                continue
            stat, p = friedmanchisquare(*[pivot[m].to_numpy() for m in ALL_METHODS])
            row = {"Dataset": dataset, "Metric": metric, "Observations": int(len(pivot)), "Friedman_stat": float(stat), "Friedman_p": float(p)}
            rows.append(row)
            conclusion = "PASSED" if p < ALPHA_SIGNIFICANCE else "NOT PASSED"
            level = "1%" if p < 0.01 else ("5%" if p < 0.05 else "not significant")
            print(f"[Stats] Friedman | {dataset} | {metric}: stat={stat:.6g}, p={p:.6g} -> {conclusion} at {level} level")
    return pd.DataFrame(rows)


def paired_tests_vs_ibie(performance_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Average metrics over models within each fold for pairwise algorithm tests.
    grouped = performance_df.groupby(["Dataset", "Algorithm", "Fold"], as_index=False)[METRICS].mean()
    t_rows = []
    w_rows = []
    d_rows = []
    for dataset in sorted(grouped["Dataset"].unique()):
        ds = grouped[grouped["Dataset"] == dataset]
        ibie = ds[ds["Algorithm"] == "IBIE"].sort_values("Fold")
        for algo in BALANCERS:
            comp = ds[ds["Algorithm"] == algo].sort_values("Fold")
            if len(comp) != len(ibie):
                continue
            # Average across six metrics, matching the paper's overall comparison idea.
            ib = ibie[METRICS].mean(axis=1).to_numpy()
            co = comp[METRICS].mean(axis=1).to_numpy()
            t_stat, t_p = ttest_rel(ib, co)
            try:
                w_stat, w_p = wilcoxon(ib, co, zero_method="wilcox", alternative="two-sided", mode="auto")
            except Exception:
                w_stat, w_p = np.nan, np.nan
            diff = ib - co
            d = diff.mean() / (diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan)
            pair = f"IBIE vs {algo}"
            t_rows.append({"Dataset": dataset, "Pair": pair, "Statistic": float(t_stat), "p_value": float(t_p)})
            w_rows.append({"Dataset": dataset, "Pair": pair, "W_Statistic": float(w_stat), "P_Value_Raw": float(w_p)})
            d_rows.append({"Dataset": dataset, "Pair": pair, "Cohen_d": float(d)})
    # Holm correction within each dataset.
    wdf = pd.DataFrame(w_rows)
    if not wdf.empty:
        wdf["P_Value_Holm"] = np.nan
        for ds in wdf["Dataset"].unique():
            mask = wdf["Dataset"] == ds
            pvals = wdf.loc[mask, "P_Value_Raw"].to_numpy()
            order = np.argsort(pvals)
            adjusted = np.empty_like(pvals)
            running = 0.0
            m = len(pvals)
            for rank, idx in enumerate(order):
                adj = (m - rank) * pvals[idx]
                running = max(running, adj)
                adjusted[idx] = min(running, 1.0)
            wdf.loc[mask, "P_Value_Holm"] = adjusted
    tdf = pd.DataFrame(t_rows)
    ddf = pd.DataFrame(d_rows)
    for _, r in tdf.iterrows():
        lev = "1%" if r["p_value"] < 0.01 else ("5%" if r["p_value"] < 0.05 else "not significant")
        print(f"[Stats] Paired t-test | {r['Dataset']} | {r['Pair']}: t={r['Statistic']:.6g}, p={r['p_value']:.6g} -> {'PASSED' if r['p_value'] < ALPHA_SIGNIFICANCE else 'NOT PASSED'} at {lev} level")
    for _, r in wdf.iterrows():
        lev = "1%" if r["P_Value_Holm"] < 0.01 else ("5%" if r["P_Value_Holm"] < 0.05 else "not significant")
        print(f"[Stats] Wilcoxon+Holm | {r['Dataset']} | {r['Pair']}: W={r['W_Statistic']:.6g}, raw_p={r['P_Value_Raw']:.6g}, Holm_p={r['P_Value_Holm']:.6g} -> {'PASSED' if r['P_Value_Holm'] < HOLM_ALPHA else 'NOT PASSED'} at {lev} level")
    for _, r in ddf.iterrows():
        magnitude = abs(r["Cohen_d"])
        label = "large" if magnitude >= 0.8 else ("medium" if magnitude >= 0.5 else ("small" if magnitude >= 0.2 else "negligible"))
        print(f"[Stats] Cohen d | {r['Dataset']} | {r['Pair']}: d={r['Cohen_d']:.4f} ({label})")
    return tdf, wdf, ddf


# =============================================================================
# ROBUSTNESS ANALYSES
# =============================================================================
def aggregate_mean_metrics(rows: List[Dict[str, Any]], group_cols: List[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby(group_cols, as_index=False)[METRICS].mean()


def run_submodel_robustness(dataset_name: str, folds: List[FoldData], ibie_params_by_model: Dict[str, Any], model_params_by_model: Dict[str, Any]) -> pd.DataFrame:
    records = []
    for submodel in ["Logistic", "NB", "DT"]:
        print(f"[Robustness] Sub-model replacement: {dataset_name} | {submodel}")
        for model_name in MODEL_NAMES:
            p = ibie_params_by_model[model_name]
            ibie_p = copy.deepcopy(p["ibie"])
            for fd in folds:
                balancer = IBIEBalancer(
                    alpha=ibie_p["alpha"], n=ibie_p["n"], multi=ibie_p["multi"], base_model=submodel,
                    base_max_depth=ibie_p.get("base_max_depth", IBIE_BASE_MAX_DEPTH),
                    weight_cap=IBIE_WEIGHT_CAP, weight_initial=IBIE_WEIGHT_INITIAL,
                    normalize_weights_after_update=IBIE_WEIGHT_NORMALIZE_AFTER_UPDATE,
                    random_state=ROBUSTNESS_SEED + fd.fold_id, max_iterations=IBIE_MAX_ITERATIONS,
                    min_n=IBIE_MIN_N
                )
                Xr, yr, _ = balancer.fit_resample(fd.X_train, fd.y_train, dataset_name, fd.fold_id, False)
                model = fit_model(model_name, Xr, yr, model_params_by_model[model_name], GLOBAL_SEED + fd.fold_id)
                ptest = model_predict_proba(model, fd.X_test)
                met = compute_metrics(fd.y_test, ptest)
                records.append({"Dataset": dataset_name, "SubModel": submodel, "Model": model_name, **met})
    df = pd.DataFrame(records)
    # Compute standard deviation across the three sub-model configurations.
    out = df.groupby(["Dataset", "Model"], as_index=False)[METRICS].std().fillna(0)
    out = out.rename(columns={m: f"{m}_Std" for m in METRICS})
    print(f"[Robustness] Sub-model average standard deviation: {out[[c for c in out.columns if c.endswith('_Std')]].to_numpy().mean():.6f}")
    return out


def run_alpha_multi_robustness(dataset_name: str, folds: List[FoldData], model_params_by_model: Dict[str, Any]) -> pd.DataFrame:
    records = []
    n_base = ibie_default_n_from_data_size(folds[0].X_train.shape[0])
    # To keep the full grid while controlling runtime, use the paper's fixed n rule.
    for alpha in ROBUST_ALPHA_GRID:
        for multi in ROBUST_MULTI_GRID:
            print(f"[Robustness] Alpha/Multi: {dataset_name} | alpha={alpha} | multi={multi}")
            for model_name in MODEL_NAMES:
                fold_metrics = []
                for fd in folds:
                    ib = IBIEBalancer(
                        alpha=alpha, n=n_base, multi=multi, base_model=IBIE_BASE_MODEL,
                        base_max_depth=IBIE_BASE_MAX_DEPTH, weight_cap=IBIE_WEIGHT_CAP,
                        weight_initial=IBIE_WEIGHT_INITIAL, normalize_weights_after_update=IBIE_WEIGHT_NORMALIZE_AFTER_UPDATE,
                        random_state=ROBUSTNESS_SEED + fd.fold_id, max_iterations=IBIE_MAX_ITERATIONS, min_n=IBIE_MIN_N
                    )
                    Xr, yr, _ = ib.fit_resample(fd.X_train, fd.y_train, dataset_name, fd.fold_id, False)
                    mdl = fit_model(model_name, Xr, yr, model_params_by_model[model_name], GLOBAL_SEED + fd.fold_id)
                    met = compute_metrics(fd.y_test, model_predict_proba(mdl, fd.X_test))
                    fold_metrics.append(met)
                mean_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
                records.append({"Dataset": dataset_name, "Model": model_name, "alpha": alpha, "multi": multi, **mean_metrics})
    df = pd.DataFrame(records)
    # Standard deviation across the alpha/multi grid for every model.
    out = df.groupby(["Dataset", "Model"], as_index=False)[METRICS].std().fillna(0)
    out = out.rename(columns={m: f"{m}_Std" for m in METRICS})
    return out, df


def create_imbalance_dataset(X: np.ndarray, y: np.ndarray, ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx_min = np.flatnonzero(y == 1)
    idx_maj = np.flatnonzero(y == 0)
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    target_min = max(1, int(round(len(idx_maj) * ratio)))
    if target_min < len(idx_min):
        chosen_min = rng.choice(idx_min, size=target_min, replace=False)
        chosen_maj = idx_maj
    else:
        target_maj = max(1, int(round(len(idx_min) / ratio)))
        chosen_min = idx_min
        chosen_maj = rng.choice(idx_maj, size=min(target_maj, len(idx_maj)), replace=False)
    chosen = np.concatenate([chosen_min, chosen_maj])
    rng.shuffle(chosen)
    return X[chosen], y[chosen]


def run_imbalance_robustness(dataset_name: str, df: pd.DataFrame, model_params_by_model: Dict[str, Any]) -> pd.DataFrame:
    X_all, _, _ = make_numeric_matrix(df, df)
    y_all = df["label"].to_numpy(dtype=int)
    records = []
    for ratio in ROBUST_IMBALANCE_RATIOS:
        print(f"[Robustness] Imbalance ratio: {dataset_name} | ratio={ratio}:1")
        Xr0, yr0 = create_imbalance_dataset(X_all, y_all, ratio, ROBUSTNESS_SEED)
        # A single split is used here because the paper describes a ratio sensitivity case-study experiment.
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=ROBUSTNESS_SEED)
        train_idx, test_idx = next(iter(skf.split(Xr0, yr0)))
        Xt, yt = Xr0[train_idx], yr0[train_idx]
        Xte, yte = Xr0[test_idx], yr0[test_idx]
        for model_name in MODEL_NAMES:
            ib = IBIEBalancer(
                alpha=IBIE_DEFAULT_ALPHA, n=ibie_default_n_from_data_size(len(yt)), multi=IBIE_DEFAULT_MULTI,
                base_model=IBIE_BASE_MODEL, base_max_depth=IBIE_BASE_MAX_DEPTH,
                weight_cap=IBIE_WEIGHT_CAP, weight_initial=IBIE_WEIGHT_INITIAL,
                normalize_weights_after_update=IBIE_WEIGHT_NORMALIZE_AFTER_UPDATE,
                random_state=ROBUSTNESS_SEED, max_iterations=IBIE_MAX_ITERATIONS, min_n=IBIE_MIN_N
            )
            Xbal, ybal, _ = ib.fit_resample(Xt, yt, dataset_name, 0, False)
            mdl = fit_model(model_name, Xbal, ybal, model_params_by_model[model_name], GLOBAL_SEED)
            met = compute_metrics(yte, model_predict_proba(mdl, Xte))
            records.append({"Dataset": dataset_name, "Model": model_name, "ImbalanceRatio": ratio, **met})
    return pd.DataFrame(records)


# =============================================================================
# EXPLORATION EXPERIMENT
# =============================================================================
def run_exploration(dataset_name: str, folds: List[FoldData], model_params_by_model: Dict[str, Any]) -> pd.DataFrame:
    records = []
    for theta in [0.0] + EXPLORATION_GRID:
        print(f"[Exploration] {dataset_name} | theta={theta:.2f}")
        for model_name in MODEL_NAMES:
            fold_metric_list = []
            for fd in folds:
                ib = IBIEBalancer(
                    alpha=IBIE_DEFAULT_ALPHA, n=ibie_default_n_from_data_size(len(fd.y_train)), multi=IBIE_DEFAULT_MULTI,
                    base_model=IBIE_BASE_MODEL, base_max_depth=IBIE_BASE_MAX_DEPTH,
                    weight_cap=IBIE_WEIGHT_CAP, weight_initial=IBIE_WEIGHT_INITIAL,
                    normalize_weights_after_update=IBIE_WEIGHT_NORMALIZE_AFTER_UPDATE,
                    explore_fraction=theta, random_state=EXPLORATION_SEED + fd.fold_id,
                    max_iterations=IBIE_MAX_ITERATIONS, min_n=IBIE_MIN_N
                )
                Xbal, ybal, _ = ib.fit_resample(fd.X_train, fd.y_train, dataset_name, fd.fold_id, False)
                mdl = fit_model(model_name, Xbal, ybal, model_params_by_model[model_name], GLOBAL_SEED + fd.fold_id)
                met = compute_metrics(fd.y_test, model_predict_proba(mdl, fd.X_test))
                fold_metric_list.append(met)
            avg = pd.DataFrame(fold_metric_list).mean().to_dict()
            records.append({"Dataset": dataset_name, "Model": model_name, "theta": theta, **avg})
    return pd.DataFrame(records)


# =============================================================================
# SOTA COMPARISON: CTGAN / VAE
# =============================================================================
def try_import_sdv():
    try:
        from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
        from sdv.metadata import SingleTableMetadata
        return CTGANSynthesizer, TVAESynthesizer, SingleTableMetadata
    except Exception as exc:
        return None


def synthesize_with_sdv(X: np.ndarray, y: np.ndarray, method: str, seed: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    modules = try_import_sdv()
    if modules is None:
        raise ImportError("sdv is not installed; CTGAN/VAE experiment requires the sdv package.")
    CTGANSynthesizer, TVAESynthesizer, SingleTableMetadata = modules
    cols = [f"x{i}" for i in range(X.shape[1])] + ["label"]
    data = pd.DataFrame(np.column_stack([X, y]), columns=cols)
    data["label"] = data["label"].astype(str)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    minority = data[data["label"] == "1"].copy()
    # Train generator only on minority class instances.
    if method == "CTGAN":
        synthesizer = CTGANSynthesizer(metadata, epochs=50, verbose=False)
    else:
        synthesizer = TVAESynthesizer(metadata, epochs=50)
    synthesizer.fit(minority)
    synth = synthesizer.sample(num_rows=n_samples)
    synth["label"] = 1
    synth_x = synth[cols[:-1]].to_numpy(dtype=float)
    return np.vstack([X, synth_x]), np.concatenate([y, np.ones(len(synth_x), dtype=int)])


def run_sota_experiment(dataset_name: str, folds: List[FoldData], model_params_by_model: Dict[str, Any]) -> pd.DataFrame:
    records = []
    for fd in folds:
        # Build one balanced training set per method.
        methods = {"IBIE": None, "CTGAN": None, "VAE": None}
        ib = IBIEBalancer(
            alpha=IBIE_DEFAULT_ALPHA, n=ibie_default_n_from_data_size(len(fd.y_train)), multi=IBIE_DEFAULT_MULTI,
            base_model=IBIE_BASE_MODEL, base_max_depth=IBIE_BASE_MAX_DEPTH,
            weight_cap=IBIE_WEIGHT_CAP, weight_initial=IBIE_WEIGHT_INITIAL,
            normalize_weights_after_update=IBIE_WEIGHT_NORMALIZE_AFTER_UPDATE,
            random_state=SOTA_SEED + fd.fold_id, max_iterations=IBIE_MAX_ITERATIONS, min_n=IBIE_MIN_N
        )
        X_ib, y_ib, _ = ib.fit_resample(fd.X_train, fd.y_train, dataset_name, fd.fold_id, False)
        methods["IBIE"] = (X_ib, y_ib)
        for method in ["CTGAN", "VAE"]:
            try:
                majority = int((fd.y_train == 0).sum())
                minority = int((fd.y_train == 1).sum())
                need = max(0, majority - minority)
                methods[method] = synthesize_with_sdv(fd.X_train, fd.y_train, method, SOTA_SEED + fd.fold_id, need)
            except Exception as exc:
                print(f"[SOTA] Warning: {method} unavailable for {dataset_name} fold={fd.fold_id}: {exc}")
                methods[method] = None
        for method, pair in methods.items():
            if pair is None:
                continue
            Xr, yr = pair
            for model_name in MODEL_NAMES:
                mdl = fit_model(model_name, Xr, yr, model_params_by_model[model_name], GLOBAL_SEED + fd.fold_id)
                met = compute_metrics(fd.y_test, model_predict_proba(mdl, fd.X_test))
                records.append({"Dataset": dataset_name, "Model": model_name, "Algorithm": method, "Fold": fd.fold_id, **met})
    return pd.DataFrame(records)


# =============================================================================
# ABLATION EXPERIMENT
# =============================================================================
def ablation_fit_resample(X: np.ndarray, y: np.ndarray, ablation: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    # The ablation design directly corresponds to Section 5.1/5.4 descriptions.
    rng = np.random.RandomState(seed)
    # This function reproduces the corresponding IBIE modification.
    kwargs = dict(
        alpha=IBIE_DEFAULT_ALPHA,
        n=ibie_default_n_from_data_size(len(y)),
        multi=IBIE_DEFAULT_MULTI,
        base_model=IBIE_BASE_MODEL,
        base_max_depth=IBIE_BASE_MAX_DEPTH,
        weight_cap=IBIE_WEIGHT_CAP,
        weight_initial=IBIE_WEIGHT_INITIAL,
        normalize_weights_after_update=IBIE_WEIGHT_NORMALIZE_AFTER_UPDATE,
        random_state=seed,
        max_iterations=IBIE_MAX_ITERATIONS,
        min_n=IBIE_MIN_N,
    )
    if ablation == "no weight":
        kwargs["alpha"] = 0.0
    if ablation == "no validation":
        kwargs["explore_fraction"] = 1.0
    if ablation == "no feedback":
        # Simulate static feedback by setting the maximum iteration to 1 and using a larger generation pool.
        kwargs["max_iterations"] = 1
    ib = IBIEBalancer(**kwargs)
    Xb, yb, _ = ib.fit_resample(X, y, "ablation", 0, False)
    if ablation == "no distance":
        # Replace generated data with equal-probability minority-pair interpolation.
        idx_min = np.flatnonzero(y == 1)
        idx_maj = np.flatnonzero(y == 0)
        need = max(0, len(idx_maj) - len(idx_min))
        if need == 0:
            return X.copy(), y.copy()
        generated = []
        for _ in range(need):
            a, b = rng.choice(idx_min, size=2, replace=True)
            beta = rng.uniform()
            generated.append(X[a] + beta * (X[b] - X[a]))
        return np.vstack([X, np.asarray(generated)]), np.concatenate([y, np.ones(need, dtype=int)])
    return Xb, yb


def run_ablation(dataset_name: str, folds: List[FoldData], model_params_by_model: Dict[str, Any]) -> pd.DataFrame:
    records = []
    ablations = ["no weight", "no validation", "no feedback", "no distance"]
    for abl in ablations:
        print(f"[Ablation] {dataset_name} | {abl}")
        for model_name in MODEL_NAMES:
            fold_metric_list = []
            for fd in folds:
                Xb, yb = ablation_fit_resample(fd.X_train, fd.y_train, abl, ROBUSTNESS_SEED + fd.fold_id)
                mdl = fit_model(model_name, Xb, yb, model_params_by_model[model_name], GLOBAL_SEED + fd.fold_id)
                met = compute_metrics(fd.y_test, model_predict_proba(mdl, fd.X_test))
                fold_metric_list.append(met)
            avg = pd.DataFrame(fold_metric_list).mean().to_dict()
            records.append({"Dataset": dataset_name, "Model": model_name, "Ablation": abl, **avg})
    return pd.DataFrame(records)


# =============================================================================
# RUNTIME EXPERIMENT
# =============================================================================
def run_runtime_experiment(dataset_name: str, folds: List[FoldData]) -> pd.DataFrame:
    methods = BALANCERS + ["IBIE"]
    rows = []
    fd = folds[0]
    for method in methods:
        print(f"[Runtime] {dataset_name} | {method}")
        start = time.perf_counter()
        if method == "IBIE":
            ib = IBIEBalancer(
                alpha=IBIE_DEFAULT_ALPHA, n=ibie_default_n_from_data_size(len(fd.y_train)), multi=IBIE_DEFAULT_MULTI,
                base_model=IBIE_BASE_MODEL, base_max_depth=IBIE_BASE_MAX_DEPTH,
                weight_cap=IBIE_WEIGHT_CAP, weight_initial=IBIE_WEIGHT_INITIAL,
                normalize_weights_after_update=IBIE_WEIGHT_NORMALIZE_AFTER_UPDATE,
                random_state=ROBUSTNESS_SEED, max_iterations=IBIE_MAX_ITERATIONS, min_n=IBIE_MIN_N
            )
            Xr, yr, _ = ib.fit_resample(fd.X_train, fd.y_train, dataset_name, fd.fold_id, False)
        else:
            Xr, yr = resample_data(method, fd.X_train, fd.y_train, RESAMPLER_SEED)
        elapsed = time.perf_counter() - start
        rows.append({"Dataset": dataset_name, "Algorithm": method, "RuntimeSeconds": elapsed,
                     "OutputSamples": len(yr), "OutputMinority": int((yr==1).sum()), "OutputMajority": int((yr==0).sum())})
    return pd.DataFrame(rows)


# =============================================================================
# PLOTS
# =============================================================================
def save_basic_plots(comparison_df: pd.DataFrame, ibie_df: pd.DataFrame) -> None:
    if comparison_df.empty or ibie_df.empty:
        return
    merged = pd.concat([comparison_df, ibie_df], ignore_index=True)
    for dataset in merged["Dataset"].unique():
        ds = merged[merged["Dataset"] == dataset]
        for metric in ["AUC", "Sensitivity", "Specificity", "Precision", "GMean", "FScore"]:
            pivot = ds.groupby("Algorithm")[metric].mean().sort_values(ascending=False)
            plt.figure(figsize=(10, 5))
            pivot.plot(kind="bar")
            plt.title(f"{dataset} mean {metric} by balancing algorithm")
            plt.ylabel(metric)
            plt.tight_layout()
            path = PLOT_DIR / f"{dataset}_{metric}_by_algorithm.png"
            plt.savefig(path, dpi=180)
            plt.close()


def save_heatmap_from_alpha_multi(alpha_df: pd.DataFrame, dataset: str, model: str, metric: str = "AUC") -> None:
    sub = alpha_df[(alpha_df["Dataset"] == dataset) & (alpha_df["Model"] == model)]
    if sub.empty:
        return
    pivot = sub.pivot(index="multi", columns="alpha", values=metric)
    plt.figure(figsize=(10, 6))
    plt.imshow(pivot.to_numpy(), aspect="auto")
    plt.colorbar(label=metric)
    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns], rotation=45)
    plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    plt.xlabel("alpha")
    plt.ylabel("multi")
    plt.title(f"{dataset} {model}: {metric} under alpha/multi grid")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{dataset}_{model}_alpha_multi_{metric}.png", dpi=180)
    plt.close()


# =============================================================================
# EXCEL WRITER
# =============================================================================
def save_excel(output_path: Path) -> None:
    print(f"[Output] Writing Excel workbook: {output_path}")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        sheet_map = {
            "Comparison_Performance": GLOBAL_RESULTS["comparison_performance"],
            "IBIE_Performance": GLOBAL_RESULTS["ibie_performance"],
            "Model_Params": GLOBAL_RESULTS["model_params"],
            "Seeds": GLOBAL_RESULTS["seeds"],
            "Fold_Summaries": GLOBAL_RESULTS["fold_summaries"],
            "Friedman": GLOBAL_RESULTS["friedman"],
            "Paired_t_test": GLOBAL_RESULTS["ttest"],
            "Wilcoxon_Holm": GLOBAL_RESULTS["wilcoxon"],
            "Cohen_d": GLOBAL_RESULTS["cohen_d"],
            "Robust_Submodel": GLOBAL_RESULTS["robust_submodel"],
            "Robust_AlphaMulti": GLOBAL_RESULTS["robust_alpha_multi"],
            "Robust_Imbalance": GLOBAL_RESULTS["robust_imbalance"],
            "Exploration": GLOBAL_RESULTS["exploration"],
            "SOTA_CTGAN_VAE": GLOBAL_RESULTS["sota"],
            "Ablation": GLOBAL_RESULTS["ablation"],
            "Runtime": GLOBAL_RESULTS["runtime"],
            "IBIE_Trace": GLOBAL_RESULTS["trace"],
        }
        for sheet, data in sheet_map.items():
            df = pd.DataFrame(data)
            if df.empty:
                df = pd.DataFrame({"Info": ["No records generated for this sheet."]})
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
        # Add experiment metadata sheet.
        meta = pd.DataFrame([
            ["Quick test", QUICK_TEST],
            ["Use saved params", USE_SAVED_PARAMS],
            ["N splits", N_SPLITS],
            ["Comparison Optuna trials", COMPARISON_OPTUNA_TRIALS],
            ["IBIE Optuna trials", IBIE_OPTUNA_TRIALS],
            ["IBIE weight cap", IBIE_WEIGHT_CAP],
            ["IBIE weight initial", IBIE_WEIGHT_INITIAL],
            ["IBIE base model", IBIE_BASE_MODEL],
            ["IBIE base max depth", IBIE_BASE_MAX_DEPTH],
            ["Significance alpha", ALPHA_SIGNIFICANCE],
        ], columns=["Parameter", "Value"])
        meta.to_excel(writer, sheet_name="Experiment_Metadata", index=False)
    print(f"[Output] Excel saved: {output_path}")


# =============================================================================
# MAIN EXPERIMENT DRIVER
# =============================================================================
def main() -> None:
    seed_everything(GLOBAL_SEED)
    print("=" * 90)
    print("IBIE FULL EXPERIMENTAL REPRODUCTION")
    print("=" * 90)
    print(f"[Config] SCRIPT_DIR={SCRIPT_DIR}")
    print(f"[Config] QUICK_TEST={QUICK_TEST}, USE_SAVED_PARAMS={USE_SAVED_PARAMS}")
    print(f"[Config] RUN_EXTENDED_EXPERIMENTS={RUN_EXTENDED_EXPERIMENTS}")
    print(f"[Config] Split seed={SPLIT_SEED}, Optuna seed={OPTUNA_SEED}, IBIE seed={IBIE_SEED}")

    global SAVED_PARAMETER_AREA
    if USE_SAVED_PARAMS:
        SAVED_PARAMETER_AREA = load_parameter_area(OUTPUT_TXT)
        print("[Params] Saved parameter area loaded.")

    datasets = {}
    fold_cache = {}
    for dataset_name, path in DATASETS.items():
        df = load_dataset(dataset_name, path)
        datasets[dataset_name] = df
        fold_cache[dataset_name] = prepare_fold_data(df, SPLIT_SEED)
        GLOBAL_RESULTS["seeds"].append({
            "Dataset": dataset_name,
            "SplitSeed": SPLIT_SEED,
            "OptunaSeed": OPTUNA_SEED,
            "ResamplerSeed": RESAMPLER_SEED,
            "IBIESeed": IBIE_SEED,
            "RobustnessSeed": ROBUSTNESS_SEED,
            "ExplorationSeed": EXPLORATION_SEED,
            "SOTASeed": SOTA_SEED,
        })

    comparison_best: Dict[str, Dict[str, Dict[str, Any]]] = {d: {} for d in datasets}
    ibie_best: Dict[str, Dict[str, Any]] = {d: {} for d in datasets}
    comparison_model_params: Dict[str, Dict[str, Any]] = {d: {} for d in datasets}
    ibie_model_params: Dict[str, Dict[str, Any]] = {d: {} for d in datasets}

    # -------------------------------------------------------------------------
    # STEP 1: Optimize ALL comparison models before touching IBIE.
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Optimizing all comparison models first.")
    for dataset_name, folds in fold_cache.items():
        comparison_model_params[dataset_name] = {}
        for balancer_name in BALANCERS:
            comparison_best[dataset_name][balancer_name] = {}
            for model_name in MODEL_NAMES:
                key_seed = stable_seed(dataset_name, balancer_name, model_name, base=OPTUNA_SEED)
                if USE_SAVED_PARAMS:
                    saved = SAVED_PARAMETER_AREA.get("comparison", {}).get(dataset_name, {}).get(balancer_name, {}).get(model_name)
                    if saved:
                        result = saved
                    else:
                        result = optimize_comparison_model(dataset_name, model_name, balancer_name, folds, key_seed)
                else:
                    result = optimize_comparison_model(dataset_name, model_name, balancer_name, folds, key_seed)
                comparison_best[dataset_name][balancer_name][model_name] = result
                comparison_model_params[dataset_name].setdefault(model_name, result["params"])
                GLOBAL_RESULTS["model_params"].append({
                    "Dataset": dataset_name,
                    "Stage": "Comparison",
                    "Algorithm": balancer_name,
                    "Model": model_name,
                    "Params": json.dumps(json_safe(result["params"]), ensure_ascii=True),
                    "BestValidationAUC": result.get("best_value", np.nan),
                    "OptimizationSeed": result.get("study_seed", key_seed),
                    "Trials": result.get("n_trials", 0),
                    "Mode": result.get("mode", "unknown"),
                })

    # Evaluate comparison algorithms after all comparison optimization is finished.
    print("\n[STEP 1B] Evaluating all comparison algorithms with optimized model parameters.")
    for dataset_name, folds in fold_cache.items():
        for balancer_name in BALANCERS:
            for model_name in MODEL_NAMES:
                result = comparison_best[dataset_name][balancer_name][model_name]
                rows = evaluate_one_method_model(dataset_name, balancer_name, model_name, folds, result["params"], GLOBAL_SEED)
                GLOBAL_RESULTS["comparison_performance"].extend(rows)

    # -------------------------------------------------------------------------
    # STEP 2: Optimize IBIE jointly with every downstream model.
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Optimizing IBIE and downstream model parameters after all comparisons.")
    for dataset_name, folds in fold_cache.items():
        for model_name in MODEL_NAMES:
            key_seed = stable_seed(dataset_name, model_name, "IBIE", base=OPTUNA_SEED + 500000)
            if USE_SAVED_PARAMS:
                saved = SAVED_PARAMETER_AREA.get("ibie", {}).get(dataset_name, {}).get(model_name)
                if saved:
                    result = saved
                else:
                    result = optimize_ibie_model(dataset_name, model_name, folds, key_seed)
            else:
                result = optimize_ibie_model(dataset_name, model_name, folds, key_seed)
            ibie_best[dataset_name][model_name] = result
            ibie_model_params[dataset_name][model_name] = result["model_params"]
            GLOBAL_RESULTS["model_params"].append({
                "Dataset": dataset_name,
                "Stage": "IBIE",
                "Algorithm": "IBIE",
                "Model": model_name,
                "Params": json.dumps(json_safe(result["model_params"]), ensure_ascii=True),
                "IBIEParams": json.dumps(json_safe(result["ibie"]), ensure_ascii=True),
                "BestValidationAUC": result.get("best_value", np.nan),
                "OptimizationSeed": result.get("study_seed", key_seed),
                "Trials": result.get("n_trials", 0),
                "Mode": result.get("mode", "unknown"),
            })
            print(f"    [IBIE Best] {dataset_name} | {model_name}: {result['ibie']}")

    # Evaluate IBIE performance.
    print("\n[STEP 2B] Evaluating IBIE across all models.")
    for dataset_name, folds in fold_cache.items():
        for model_name in MODEL_NAMES:
            result = ibie_best[dataset_name][model_name]
            rows = evaluate_one_method_model(dataset_name, "IBIE", model_name, folds, result, GLOBAL_SEED)
            GLOBAL_RESULTS["ibie_performance"].extend(rows)

    # -------------------------------------------------------------------------
    # STEP 3: Statistics.
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Statistical tests.")
    full_perf = pd.concat([pd.DataFrame(GLOBAL_RESULTS["comparison_performance"]),
                           pd.DataFrame(GLOBAL_RESULTS["ibie_performance"])], ignore_index=True)
    fried = friedman_tests(full_perf)
    tdf, wdf, ddf = paired_tests_vs_ibie(full_perf)
    GLOBAL_RESULTS["friedman"] = fried.to_dict("records")
    GLOBAL_RESULTS["ttest"] = tdf.to_dict("records")
    GLOBAL_RESULTS["wilcoxon"] = wdf.to_dict("records")
    GLOBAL_RESULTS["cohen_d"] = ddf.to_dict("records")

    # -------------------------------------------------------------------------
    # STEP 4: Extended experiments requested by the paper.
    # -------------------------------------------------------------------------
    if RUN_EXTENDED_EXPERIMENTS:
        print("\n[STEP 4] Extended robustness and additional experiments.")
        for dataset_name, df in datasets.items():
            folds = fold_cache[dataset_name]
            model_params = ibie_model_params[dataset_name]
            saved_ibie_params = ibie_best[dataset_name]

            # 4A. Sub-model replacement.
            sub_df = run_submodel_robustness(dataset_name, folds, saved_ibie_params, model_params)
            GLOBAL_RESULTS["robust_submodel"].extend(sub_df.to_dict("records"))

            # 4B. Alpha x multi.
            alpha_std_df, alpha_raw_df = run_alpha_multi_robustness(dataset_name, folds, model_params)
            GLOBAL_RESULTS["robust_alpha_multi"].extend(alpha_std_df.to_dict("records"))
            for model_name in MODEL_NAMES[:2]:
                save_heatmap_from_alpha_multi(alpha_raw_df, dataset_name, model_name, "AUC")

            # 4C. Imbalance ratio.
            imb_df = run_imbalance_robustness(dataset_name, df, model_params)
            GLOBAL_RESULTS["robust_imbalance"].extend(imb_df.to_dict("records"))

            # 4D. Exploration mechanism.
            if RUN_EXPLORATION_EXPERIMENT:
                exp_df = run_exploration(dataset_name, folds, model_params)
                GLOBAL_RESULTS["exploration"].extend(exp_df.to_dict("records"))

            # 4E. CTGAN/VAE.
            if RUN_SOTA_EXPERIMENT:
                sota_df = run_sota_experiment(dataset_name, folds, model_params)
                GLOBAL_RESULTS["sota"].extend(sota_df.to_dict("records"))

            # 4F. Ablation.
            if RUN_ABLATION_EXPERIMENT:
                abl_df = run_ablation(dataset_name, folds, model_params)
                GLOBAL_RESULTS["ablation"].extend(abl_df.to_dict("records"))

            # 4G. Runtime.
            if RUN_RUNTIME_EXPERIMENT:
                rt_df = run_runtime_experiment(dataset_name, folds)
                GLOBAL_RESULTS["runtime"].extend(rt_df.to_dict("records"))

    # -------------------------------------------------------------------------
    # STEP 5: Parameter TXT and outputs.
    # -------------------------------------------------------------------------
    save_parameter_area(OUTPUT_TXT, comparison_best, ibie_best, GLOBAL_RESULTS["seeds"])
    save_basic_plots(pd.DataFrame(GLOBAL_RESULTS["comparison_performance"]), pd.DataFrame(GLOBAL_RESULTS["ibie_performance"]))
    save_excel(OUTPUT_XLSX)

    # -------------------------------------------------------------------------
    # Final concise screen summary.
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("CORE CONCLUSIONS")
    print("=" * 90)
    perf = full_perf.groupby(["Dataset", "Algorithm"])[METRICS].mean().reset_index()
    for dataset in perf["Dataset"].unique():
        ds = perf[perf["Dataset"] == dataset]
        ib = ds[ds["Algorithm"] == "IBIE"]
        if not ib.empty:
            print(f"[Summary] {dataset} IBIE mean metrics: " + ", ".join([f"{m}={float(ib.iloc[0][m]):.4f}" for m in METRICS]))
    print(f"[Summary] Excel: {OUTPUT_XLSX}")
    print(f"[Summary] Parameter TXT: {OUTPUT_TXT}")
    print(f"[Summary] Plots: {PLOT_DIR}")
    print("[Summary] All processing was sequential; n_jobs=1 was enforced for model libraries where applicable.")
    print("[Summary] Re-run with QUICK_TEST=True to bypass Optuna and exercise the full data flow with default parameters.")


if __name__ == "__main__":
    logger = TeeLogger(OUTPUT_LOG)
    sys.stdout = logger
    try:
        main()
    except Exception as exc:
        print("\n[ERROR] Program terminated with an exception:")
        print(type(exc).__name__, str(exc))
        traceback.print_exc()
        raise
    finally:
        logger.close()
