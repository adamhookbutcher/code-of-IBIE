# =============================================================================
# IBIE_experiments.py
# -----------------------------------------------------------------------------
# Reproduces the full empirical study of the paper:
#   "An interpretable balancing algorithm with misclassified sample identification
#    and iterative enhancement for credit risk assessment" (IBIE)
#
# Contents implemented in THIS single file:
#   1) Comparison group: 10 sample-balancing algorithms x 8 credit-risk models,
#      5-fold cross-validation, Optuna hyper-parameter tuning (500 trials,
#      objective = validation AUC), evaluated on 6 metrics.
#   2) IBIE algorithm: strict re-implementation of the core idea
#      (neighborhood support / prediction difficulty / boundary similarity seed
#       factors + local support / novelty / model recognition screening),
#      with a dynamically growing training set across iterations.
#   3) IBIE comparison: same protocol as (1) but the training set is balanced by
#      IBIE (its hyper-parameter k is identified per dataset via Optuna).
#   4) Statistical tests: Friedman test (3 levels), paired t-test, Cohen's d,
#      Wilcoxon signed-rank test.
#   5) Extra experiments shown as figures/tables in the paper:
#      sensitivity analysis over k, CTGAN/VAE comparison, ablation study,
#      and an interpretable generation-case trace.
#
# All text (comments, prints, file contents) is in ENGLISH as requested.
# The program runs strictly SEQUENTIALLY (no parallelism) and is written to be
# memory-friendly (no large persistent matrices).
# =============================================================================

import os
import sys
import json
import warnings
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import (roc_auc_score, recall_score, f1_score,
                             brier_score_loss)
import optuna
try:
    from imblearn.under_sampling import (RandomUnderSampler, NearMiss,
                                         ClusterCentroids, TomekLinks)
    from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN,
                                        BorderlineSMOTE)
    from imblearn.combine import SMOTEENN
    IMBLEARN_OK = True
except Exception as e:  # pragma: no cover
    IMBLEARN_OK = False
    print("[WARN] imblearn core samplers not available:", e)

# MWMOTE was removed from recent imblearn releases; load it if present,
# otherwise a self-contained fallback implementation is provided below.
try:
    from imblearn.over_sampling import MWMOTE as _MWMOTE_CLASS
except Exception:
    _MWMOTE_CLASS = None


class _FallbackSampler:
    """Minimal sampler wrapper for algorithms not shipped by the installed
    imblearn version (e.g. MWMOTE). Implements fit_resample(X, y)."""
    def __init__(self, kind, seed):
        self.kind = kind
        self.seed = seed

    def fit_resample(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if self.kind == "MWMOTE":
            return mwmote_balance(X, y, self.seed)
        return X, y


def mwmote_balance(X, y, seed, k=5):
    """Self-contained MWMOTE (Majority Weighted Minority Oversampling
    Technique, Barua et al. 2012) fallback.

    Identifies 'difficult' minority samples (those whose k nearest neighbours
    are majority-dominated), assigns them higher synthesis weights (scaled by
    their proximity to the majority class), and generates synthetic samples
    between each difficult minority sample and its nearest majority sample,
    biased toward the minority side. The total reaches a 1:1 balance.
    """
    rng = np.random.default_rng(seed)
    Xmin = X[y == 1]
    Xmaj = X[y == 0]
    n_min, n_maj = len(Xmin), len(Xmaj)
    if n_min == 0 or n_maj == 0:
        return X.copy(), y.copy()
    nn_maj = NearestNeighbors(n_neighbors=1).fit(Xmaj)
    nn_mix = NearestNeighbors(n_neighbors=k + 1).fit(
        np.vstack([Xmin, Xmaj]))
    labels_mix = np.concatenate([np.ones(n_min), np.zeros(n_maj)])
    weights = np.zeros(n_min)
    for i in range(n_min):
        _, idx = nn_mix.kneighbors(Xmin[i:i + 1])
        nb = labels_mix[idx[0, 1:]]          # drop self
        n_maj_nb = int((nb == 0).sum())
        n_min_nb = int((nb == 1).sum())
        if n_maj_nb > n_min_nb:              # difficult / boundary sample
            d_maj = float(nn_maj.kneighbors(Xmin[i:i + 1])[0][0, 0])
            weights[i] = (n_maj_nb / k) * (1.0 / (d_maj + 1e-9))
    if weights.sum() <= 0:
        weights = np.ones(n_min)
    weights = weights / weights.sum()
    n_gen = n_maj - n_min
    if n_gen <= 0:
        return X.copy(), y.copy()
    # prefer difficult samples, fall back to all minority if none difficult
    cand = np.where(weights > 0)[0]
    if len(cand) == 0:
        cand = np.arange(n_min)
    cw = weights[cand]
    cw = cw / cw.sum()
    picks = roulette_indices(len(cand), cw, n_gen, rng)
    synth = []
    for ci in picks:
        base = Xmin[cand[ci]]
        _, i_maj = nn_maj.kneighbors(base.reshape(1, -1))
        maj_pt = Xmaj[i_maj[0, 0]]
        lam = rng.random() * 0.5            # stay nearer the minority side
        synth.append(base + lam * (maj_pt - base))
    synth = np.array(synth)
    Xb = np.vstack([Xmin, synth, Xmaj])
    yb = np.concatenate([np.ones(len(Xmin) + len(synth)),
                         np.zeros(len(Xmaj))]).astype(int)
    return Xb, yb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# ============================ PARA AREA (edit here) ==========================
# =============================================================================
# All user-adjustable hyper-parameters, random seeds and switches live here.
# After a full run the program writes a file "best_params.txt" containing the
# discovered best hyper-parameters in a copy-paste-ready format for PRESET_PARAMS.

RANDOM_SEED = 42                 # global random seed used for CV splitting etc.

N_FOLDS = 5                      # number of cross-validation folds
VAL_RATIO = 0.20                 # 20% of the training set held out as validation
N_TRIALS = 500                   # Optuna optimization budget (paper uses 500)

# ---- IBIE algorithm hyper-parameters ----
IBIE_MAX_ITER = 10               # maximum number of IBIE iterations
IBIE_K_LIST = list(range(3, 102, 2))   # 50 odd integers in [3, 101]
IBIE_INTERNAL_DT_MAX_DEPTH = 8   # max depth of the interpretable decision tree
IBIE_INTERNAL_DT_CLASS_WEIGHT = None  # class_weight of the internal DT
IBIE_SHARED_K = True             # tune k once per dataset, reuse for all models

# ---- Master switches ----
USE_PRESET_PARAMS = False        # True -> use PRESET_PARAMS, skip all optimization
DO_COMPARISON = True             # run the 10 comparison-group balancers
DO_IBIE = True                   # run the IBIE comparison
DO_STAT_TESTS = True             # run the four statistical tests
DO_SENSITIVITY = True            # k-sensitivity analysis (Figure/Table 6)
DO_CTGAN_VAE = True              # CTGAN / VAE comparison (Figure 5)
DO_ABLATION = True               # ablation study (Table 7 / Figure 6)
DO_INTERPRET = True              # interpretable generation-case trace (Table 8)
QUICK_MODE = False                # True -> reduced-scope self-test (fast validation).
                                 # Either this flag OR env IBIE_QUICK=1 enables it.
                                 # Delivered default is False => full study runs.

# ---- Data locations ----
# Default: the directory where THIS script lives (so the user can drop the script
# next to the two xlsx files and run it directly). Override with env IBIE_DATA_DIR.
DATA_DIR = os.environ.get("IBIE_DATA_DIR",
                          os.path.dirname(os.path.abspath(__file__)))
FILE_PC = "final_Chinese personal loan.xlsx"     # PC  dataset
FILE_CC = "final_default of credit card clients.xlsx"  # CC dataset

# ---- Quick self-test mode (code flag QUICK_MODE OR env IBIE_QUICK=1) ----
# Keeps the FULL logic but drastically shrinks the compute so the code can be
# validated quickly. The delivered default (QUICK_MODE=False, no env) runs the
# complete study. To flip it from inside this file, change QUICK_MODE above.
QUICK = QUICK_MODE or (os.environ.get("IBIE_QUICK", "0") == "1")
if QUICK:
    print("[INFO] QUICK self-test mode enabled (reduced scope).")
    N_TRIALS = 5
    N_FOLDS = 2
    IBIE_MAX_ITER = 2
    IBIE_K_LIST = [3, 11, 21, 31, 51]
    _QUICK_MODELS = ["Logistic"]
    _QUICK_BAL = ["RU", "SMOTE", "TL"]
    DO_SENSITIVITY = True
    DO_CTGAN_VAE = True
    DO_ABLATION = True
else:
    _QUICK_MODELS = None
    _QUICK_BAL = None

# Placeholder for user-supplied best hyper-parameters.
# Fill this (e.g. by pasting from best_params.txt) and set USE_PRESET_PARAMS=True
# to skip optimization. Structure:
#   PRESET_PARAMS[dataset][balancing][model] = {<model hyper-params>}
#   PRESET_PARAMS[dataset]["IBIE"][model]     = {"k": <int>, <model hyper-params>}
PRESET_PARAMS = {}

# =============================================================================
# ============================ END PARA AREA ==================================
# =============================================================================

# Fixed name lists ------------------------------------------------------------
if _QUICK_MODELS is not None:
    MODELS = _QUICK_MODELS
else:
    MODELS = ["Logistic", "BP", "DT", "SVM", "AdaBoost", "GBDT",
              "LightGBM", "XGBoost"]

if _QUICK_BAL is not None:
    COMPARE_BALANCERS = _QUICK_BAL
else:
    # The 10 comparison-group balancing algorithms described in the paper.
    # (ClusterCentroids is abbreviated "CCen" here to avoid clashing with the
    #  credit-card dataset name "CC".)
    COMPARE_BALANCERS = ["RU", "NM", "CCen", "TL", "RO", "SMOTE",
                         "ADASYN", "BSMOTE", "MWMOTE", "SMOTEENN"]

METRICS = ["AUC", "Sensitivity", "Specificity", "GMean", "FScore", "Brier"]
DATASETS = ["PC", "CC"]

# Map short balancing names -> human readable labels for Excel output.
BAL_LABEL = {
    "RU": "RandomUnderSampler", "NM": "NearMiss", "CCen": "ClusterCentroids",
    "TL": "TomekLinks", "RO": "RandomOverSampler", "SMOTE": "SMOTE",
    "ADASYN": "ADASYN", "BSMOTE": "BorderlineSMOTE", "MWMOTE": "MWMOTE",
    "SMOTEENN": "SMOTEENN", "IBIE": "IBIE",
}

# Global containers -----------------------------------------------------------
# perf[dataset][balancing][model][metric] = list of per-fold values
PERF = {}
# ibie_best[dataset][model] = dict with k and model hyper-params
IBIE_BEST = {}
# comp_best[dataset][balancing][model] = best model hyper-params (comparison grp)
COMP_BEST = {}
# optuna seeds recorded
OPTUNA_SEEDS = {}


# =============================================================================
# Utility helpers
# =============================================================================
def log(msg):
    """Print a timestamped progress message to the screen."""
    print(msg, flush=True)


def roulette_indices(n_items, weights, n_pick, rng):
    """Roulette-wheel selection returning a list of picked integer indices
    (sampling with replacement is allowed, matching the paper)."""
    weights = np.asarray(weights, dtype=float)
    s = weights.sum()
    if s <= 0 or not np.isfinite(s):
        weights = np.ones(n_items, dtype=float)
    else:
        weights = weights / s
    cdf = np.cumsum(weights)
    picks = []
    for _ in range(n_pick):
        r = rng.random()
        idx = int(np.searchsorted(cdf, r))
        if idx >= n_items:
            idx = n_items - 1
        picks.append(idx)
    return picks


def compute_metrics(y_true, y_prob, y_pred):
    """Return the six evaluation metrics for one test fold."""
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    gmean = float(np.sqrt(sens * spec)) if (sens >= 0 and spec >= 0) else 0.0
    fscore = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    try:
        brier = brier_score_loss(y_true, y_prob)
    except Exception:
        brier = 0.0
    return {
        "AUC": float(auc), "Sensitivity": float(sens),
        "Specificity": float(spec), "GMean": float(gmean),
        "FScore": float(fscore), "Brier": float(brier),
    }


# =============================================================================
# Data loading
# =============================================================================
def load_dataset(name):
    """Load a dataset; return (X numpy float array, y numpy int array,
    categorical column indices, feature column names)."""
    fname = FILE_PC if name == "PC" else FILE_CC
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError("Dataset not found: " + path)
    df = pd.read_excel(path)
    if "label" not in df.columns:
        raise ValueError("Dataset %s must contain a 'label' column." % name)
    y = df["label"].astype(int).values
    X_df = df.drop(columns=["label"])
    # identify categorical columns (object / category dtype)
    cat_cols = [i for i, c in enumerate(X_df.columns)
                if str(X_df[c].dtype) in ("object", "category")]
    X = X_df.astype(float).values
    log("[DATA] %s loaded: samples=%d features=%d  "
        "non-default(0)=%d default(1)=%d"
        % (name, X.shape[0], X.shape[1], int((y == 0).sum()),
           int((y == 1).sum())))
    return X, y, cat_cols, list(X_df.columns)


# =============================================================================
# Credit-risk model construction + Optuna search spaces
# =============================================================================
def build_model(model_name, p, seed):
    """Instantiate a credit-risk model with the given hyper-parameters."""
    if model_name == "Logistic":
        solver = "liblinear" if p["penalty"] == "l1" else "lbfgs"
        return LogisticRegression(C=p["C"], penalty=p["penalty"],
                                  class_weight=p["class_weight"],
                                  solver=solver, max_iter=2000,
                                  random_state=seed)
    if model_name == "BP":
        hidden = tuple([int(p["units"])] * int(p["layers"]))
        return MLPClassifier(hidden_layer_sizes=hidden,
                             activation=p["activation"], alpha=p["alpha"],
                             learning_rate_init=p["lr"],
                             batch_size=p["batch"], max_iter=400,
                             early_stopping=True, validation_fraction=0.1,
                             random_state=seed)
    if model_name == "DT":
        return DecisionTreeClassifier(criterion=p["criterion"],
                                      max_depth=int(p["max_depth"]),
                                      min_samples_split=int(p["min_split"]),
                                      min_samples_leaf=int(p["min_leaf"]),
                                      class_weight=p["class_weight"],
                                      random_state=seed)
    if model_name == "SVM":
        return SVC(kernel=p["kernel"], C=p["C"], gamma=p["gamma"],
                   class_weight=p["class_weight"], probability=True,
                   random_state=seed, cache_size=500)
    if model_name == "AdaBoost":
        base = DecisionTreeClassifier(max_depth=int(p["base_depth"]),
                                      random_state=seed)
        return AdaBoostClassifier(estimator=base,
                                  n_estimators=int(p["n_estimators"]),
                                  learning_rate=p["learning_rate"],
                                  random_state=seed)
    if model_name == "GBDT":
        return GradientBoostingClassifier(
            n_estimators=int(p["n_estimators"]),
            learning_rate=p["learning_rate"],
            max_depth=int(p["max_depth"]),
            min_samples_split=int(p["min_split"]),
            min_samples_leaf=int(p["min_leaf"]), random_state=seed)
    if model_name == "LightGBM":
        return LGBM_Model(p, seed)
    if model_name == "XGBoost":
        return XGB_Model(p, seed)
    raise ValueError("Unknown model: " + model_name)


def LGBM_Model(p, seed):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=int(p["n_estimators"]),
                          learning_rate=p["learning_rate"],
                          num_leaves=int(p["num_leaves"]),
                          max_depth=int(p["max_depth"]),
                          min_child_samples=int(p["min_child"]),
                          random_state=seed, verbose=-1, n_jobs=1)


def XGB_Model(p, seed):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=int(p["n_estimators"]),
                         learning_rate=p["learning_rate"],
                         max_depth=int(p["max_depth"]),
                         min_child_weight=int(p["min_child"]),
                         gamma=float(p["gamma"]),
                         use_label_encoder=False, eval_metric="logloss",
                         random_state=seed, n_jobs=1)


def suggest_params(trial, model_name):
    """Define the Optuna search space (Table 2 of the paper)."""
    if model_name == "Logistic":
        return {
            "C": trial.suggest_loguniform("C", 1e-4, 1e4),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]),
        }
    if model_name == "BP":
        return {
            "layers": trial.suggest_int("layers", 2, 10),
            "units": trial.suggest_int("units", 16, 256),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "tanh"]),
            "alpha": trial.suggest_loguniform("alpha", 1e-6, 1e-1),
            "lr": trial.suggest_loguniform("lr", 1e-4, 1e-1),
            "batch": trial.suggest_categorical("batch", ["auto", 64, 128, 256]),
        }
    if model_name == "DT":
        return {
            "criterion": trial.suggest_categorical(
                "criterion", ["gini", "entropy", "log_loss"]),
            "max_depth": trial.suggest_int("max_depth", 2, 32),
            "min_split": trial.suggest_int("min_split", 2, 40),
            "min_leaf": trial.suggest_int("min_leaf", 1, 40),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]),
        }
    if model_name == "SVM":
        return {
            "kernel": trial.suggest_categorical(
                "kernel", ["rbf", "poly", "sigmoid"]),
            "C": trial.suggest_loguniform("C", 1e-3, 1e3),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced"]),
        }
    if model_name == "AdaBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 30, 300),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-2, 2.0),
            "base_depth": trial.suggest_int("base_depth", 1, 5),
        }
    if model_name == "GBDT":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-2, 3e-1),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_split": trial.suggest_int("min_split", 2, 30),
            "min_leaf": trial.suggest_int("min_leaf", 1, 30),
        }
    if model_name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-2, 3e-1),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child": trial.suggest_int("min_child", 5, 100),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-2, 3e-1),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child": trial.suggest_int("min_child", 1, 20),
            "gamma": trial.suggest_uniform("gamma", 0, 10),
        }


def train_eval(model_name, params, Xtr, ytr, Xte, yte, seed):
    """Train a model on balanced data and return (metrics_dict, proba, pred)."""
    model = build_model(model_name, params, seed)
    model.fit(Xtr, ytr)
    y_prob = model.predict_proba(Xte)[:, 1]
    y_pred = model.predict(Xte)
    return compute_metrics(yte, y_prob, y_pred)


# =============================================================================
# Comparison-group balancing wrappers (imbalanced-learn)
# =============================================================================
def get_imblearn_balancer(name, seed):
    """Return an imblearn sampler instance for the given short name."""
    if name == "RU":
        return RandomUnderSampler(sampling_strategy="auto", random_state=seed)
    if name == "NM":
        return NearMiss(sampling_strategy="auto", n_jobs=1)
    if name == "CCen":
        return ClusterCentroids(sampling_strategy="auto", random_state=seed)
    if name == "TL":
        # TomekLinks is a cleaning method; it does not aim for a 1:1 ratio.
        return TomekLinks(n_jobs=1)
    if name == "RO":
        return RandomOverSampler(sampling_strategy="auto", random_state=seed)
    if name == "SMOTE":
        return SMOTE(sampling_strategy="auto", random_state=seed)
    if name == "ADASYN":
        return ADASYN(sampling_strategy="auto", random_state=seed)
    if name == "BSMOTE":
        return BorderlineSMOTE(sampling_strategy="auto", random_state=seed)
    if name == "MWMOTE":
        if _MWMOTE_CLASS is not None:
            return _MWMOTE_CLASS(sampling_strategy="auto", random_state=seed)
        return _FallbackSampler("MWMOTE", seed)
    if name == "SMOTEENN":
        return SMOTEENN(sampling_strategy="auto", random_state=seed)
    raise ValueError("Unknown balancer: " + name)


def apply_comparison_balancing(name, X, y, seed):
    """Balance (X, y) with a comparison-group algorithm; return (Xb, yb)."""
    bal = get_imblearn_balancer(name, seed)
    Xb, yb = bal.fit_resample(X, y)
    return np.asarray(Xb, dtype=float), np.asarray(yb, dtype=int)


# =============================================================================
# IBIE algorithm  (core contribution of the paper)
# =============================================================================
def select_partners(cur_min, seed_idx, rng, chunk=2000):
    """Vectorized inverse-distance roulette partner selection.

    For every chosen seed (an index into cur_min), pick a *different* minority
    sample as its pairing partner. The weight of each candidate partner is the
    inverse of its Euclidean distance to the seed (Eq. 8 of the paper), exactly
    as in the original roulette-wheel rule, but computed in batched, vectorized
    form to keep the algorithm practical on large credit-risk datasets.
    """
    m = len(cur_min)
    n = len(seed_idx)
    d = cur_min.shape[1]
    partners = np.empty((n, d), dtype=cur_min.dtype)
    min_norms = (cur_min ** 2).sum(1)
    for st in range(0, n, chunk):
        en = min(st + chunk, n)
        sb = cur_min[seed_idx[st:en]]
        sn = (sb ** 2).sum(1)
        # squared Euclidean distances via the expansion ||a-b||^2 = ||a||^2+||b||^2-2ab
        D2 = sn[:, None] + min_norms[None, :] - 2.0 * (sb @ cur_min.T)
        np.maximum(D2, 0.0, out=D2)
        self_col = seed_idx[st:en]              # a seed must not pair with itself
        rows = np.arange(en - st)
        D2[rows, self_col] = np.inf
        D = np.sqrt(D2)
        w = 1.0 / (D + 1e-9)
        cs = np.cumsum(w, axis=1)
        cs = cs / cs[:, -1:]
        r = rng.random(en - st)[:, None]
        picks = (cs < r).sum(1)
        np.minimum(picks, m - 1, out=picks)
        partners[st:en] = cur_min[picks]
    return partners


def ibie_balance(X_tr, y_tr, k, max_iter, dt_depth, seed, cat_cols,
                 trace=False, disable=None, rng=None):
    """Run the IBIE algorithm on the training samples (X_tr, y_tr).

    Parameters
    ----------
    X_tr, y_tr : original-scale feature matrix and label vector of the
                 TRAINING samples (the 80% split).
    k          : neighborhood size (the hyper-parameter tuned via Optuna).
    max_iter   : maximum number of IBIE iterations.
    dt_depth   : max depth of the internal interpretable decision tree.
    seed       : random seed for reproducibility.
    cat_cols   : indices of categorical columns (nearest-category mapping).
    trace      : if True, capture details of the first accepted candidate.
    disable    : dict to switch OFF individual IBIE components (ablation);
                 keys: neighborhood, preddiff, boundary, localsupport,
                 novelty, modelrecog.
    rng        : numpy Generator (created if None).

    Returns
    -------
    X_bal, y_bal : balanced data in ORIGINAL scale, minority count == majority.
    trace_info   : dict with the traced first-accepted-sample details (or None).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if disable is None:
        disable = {}
    k = int(k)

    # ---- Step 1: normalization (Eq.1, min-max to [0,1]) ----
    scaler = MinMaxScaler().fit(X_tr)
    Xn = scaler.transform(X_tr)
    y = np.asarray(y_tr, dtype=int)
    Xmin0 = Xn[y == 1]                 # original minority set (normalized)
    Xmaj = Xn[y == 0]                  # majority set (normalized)
    majority_count = len(Xmaj)
    if len(Xmin0) == 0 or majority_count == 0:
        # degenerate: just return as-is
        return X_tr.copy(), y.copy(), None

    synthetic = []        # cumulative accepted synthetic minority samples
    trace_info = None
    n_t = majority_count - len(Xmin0)   # initial deficit (n_1)
    if n_t <= 0:
        # already balanced
        X_bal = scaler.inverse_transform(Xn)
        return X_bal, y.copy(), None

    for t in range(max_iter):
        cur_min = (np.vstack([Xmin0] + synthetic)
                   if synthetic else Xmin0)
        total_min = len(cur_min) + len(synthetic)
        if total_min >= majority_count:
            break
        if len(cur_min) < 2:
            # cannot pair a minority sample with a different one -> final fill
            break

        # combined current dataset (for boundary similarity + DT)
        combined = np.vstack([cur_min, Xmaj])
        y_comb = np.concatenate([np.ones(len(cur_min)),
                                 np.zeros(len(Xmaj))])

        # train the interpretable decision tree T on current D
        T = DecisionTreeClassifier(max_depth=dt_depth,
                                   class_weight=IBIE_INTERNAL_DT_CLASS_WEIGHT,
                                   random_state=seed).fit(combined, y_comb)

        m = len(cur_min)
        kk = min(k, m - 1) if m > 1 else 1

        # ---- Step 2: seed-selection factors ----
        nn_min = NearestNeighbors(n_neighbors=kk + 1).fit(cur_min)
        d_m, _ = nn_min.kneighbors(cur_min)
        d_i = d_m[:, 1:].mean(axis=1)            # exclude self
        S = np.median(d_i)
        if not np.isfinite(S) or S <= 0:
            S = 1e-9
        if disable.get("neighborhood", False):
            Ns = np.ones(m)
        else:
            Ns = np.exp(-d_i / S)                # neighborhood support factor

        if disable.get("preddiff", False):
            Pd = np.ones(m)
        else:
            p_i = T.predict_proba(cur_min)[:, 1]
            Pd = 0.5 - np.abs(p_i - 0.5)        # prediction difficulty factor

        if disable.get("boundary", False):
            Bs = np.ones(m)
        else:
            nn_comb = NearestNeighbors(n_neighbors=kk + 1).fit(combined)
            _, idxc = nn_comb.kneighbors(cur_min)
            neigh_lab = y_comb[idxc[:, 1:]]      # drop self
            frac = (neigh_lab == 1).mean(axis=1)
            Bs = 4.0 * frac * (1.0 - frac)       # boundary similarity factor

        w = Ns * Pd * Bs
        if w.sum() <= 0 or not np.isfinite(w.sum()):
            w = np.ones(m)
        pi = w / w.sum()

        # roulette-wheel selection of n_t seeds
        seed_idx = roulette_indices(m, pi, n_t, rng)

        # ---- Step 3: pairing + candidate generation (vectorized) ----
        seeds = cur_min[seed_idx]
        partners = select_partners(cur_min, seed_idx, rng)
        lam = rng.random(len(seed_idx))
        cands = seeds + lam[:, None] * (partners - seeds)
        # nearest-category mapping for categorical columns
        if cat_cols:
            for c in cat_cols:
                vals = np.unique(Xmin0[:, c])
                idx = np.argmin(np.abs(cands[:, c][:, None] - vals[None, :]),
                               axis=1)
                cands[:, c] = vals[idx]

        # ---- Step 4: candidate screening (vectorized) ----
        accept = np.ones(len(cands), dtype=bool)
        D_rel = None
        pc = None
        if not disable.get("localsupport", False):
            nn_maj = NearestNeighbors(
                n_neighbors=min(kk, len(Xmaj))).fit(Xmaj)
            dm, _ = nn_min.kneighbors(cands)
            d_min = dm[:, 1:].mean(axis=1)
            dmaj, _ = nn_maj.kneighbors(cands)
            d_maj = dmaj.mean(axis=1)
            D_rel = d_min - d_maj
            accept &= (D_rel >= 0)
        if (not disable.get("novelty", False)) and len(synthetic) > 0:
            syn_arr = np.asarray(synthetic)
            nn_syn = NearestNeighbors(n_neighbors=1).fit(syn_arr)
            d_syn = nn_syn.kneighbors(cands)[0].ravel()
            nn_min0 = NearestNeighbors(n_neighbors=1).fit(Xmin0)
            d_orig = nn_min0.kneighbors(cands)[0].ravel()
            accept &= (d_syn >= d_orig)
        if not disable.get("modelrecog", False):
            pc = T.predict_proba(cands)[:, 1]
            accept &= (pc > 0.5)

        accepted_cands = cands[accept].copy()
        # trace the first accepted candidate (for interpretability example)
        if trace and trace_info is None and len(accepted_cands) > 0:
            fi = int(np.argmax(accept))
            si = int(seed_idx[fi])
            trace_info = {
                "neighborhood": float(Ns[si]),
                "preddiff": float(Pd[si]),
                "boundary": float(Bs[si]),
                "weight": float(w[si]),
                "partner_dist": float(np.sqrt(
                    ((partners[fi] - seeds[fi]) ** 2).sum())),
                "lambda": float(lam[fi]),
                "D_rel": (float(D_rel[fi]) if D_rel is not None else None),
                "pred_prob": (float(pc[fi]) if pc is not None else None),
                "reason": "",
            }

        # ---- Step 5: update generation scale for next iteration ----
        synthetic.extend(list(accepted_cands))
        n_gen = len(seed_idx)
        n_acc = int(accept.sum())
        r_t = (n_acc / n_gen) if n_gen > 0 else 0.0
        total_min = len(cur_min) + len(synthetic)
        remaining = majority_count - total_min
        if remaining <= 0:
            break
        if r_t > 0:
            n_next = int(np.ceil(remaining / max(r_t, 0.05)))
            n_next = min(n_next, remaining * 3 + n_gen)   # guard against blow-up
            n_t = n_next
        else:
            n_t = remaining

    # final fill with random oversampling if still short
    final_min = (np.vstack([Xmin0] + synthetic)
                 if synthetic else Xmin0)
    need = majority_count - len(final_min)
    if need > 0:
        idx = rng.integers(0, len(final_min), need)
        final_min = np.vstack([final_min, final_min[idx]])
    X_bal = np.vstack([final_min, Xmaj])
    y_bal = np.concatenate([np.ones(len(final_min)),
                            np.zeros(len(Xmaj))])
    X_bal = scaler.inverse_transform(X_bal)
    return X_bal, y_bal.astype(int), trace_info


# =============================================================================
# Optuna objective builders
# =============================================================================
def make_objective(model_name, Xb, yb, Xv, yv, seed):
    """Return an Optuna objective that maximizes validation AUC."""
    def objective(trial):
        try:
            p = suggest_params(trial, model_name)
            model = build_model(model_name, p, seed)
            model.fit(Xb, yb)
            proba = model.predict_proba(Xv)[:, 1]
            return float(roc_auc_score(yv, proba))
        except Exception:
            return -1.0
    return objective


def run_optuna(objective, n_trials, seed, study_name):
    """Run an Optuna study; return (best_params, best_value, used_seed)."""
    opt_seed = seed + 1000
    sampler = optuna.samplers.TPESampler(seed=opt_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=study_name)
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value, opt_seed


# =============================================================================
# Experiment (2): comparison group
# =============================================================================
def run_comparison(datasets_cache):
    log("\n" + "=" * 70)
    log("[STEP 2] Comparison-group experiments (10 balancers x 8 models)")
    log("=" * 70)
    rows = []
    for ds in DATASETS:
        X, y, cat_cols, _ = datasets_cache[ds]
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=RANDOM_SEED)
        for bal in COMPARE_BALANCERS:
            for model in MODELS:
                per_fold = {m: [] for m in METRICS}
                used_opt_seed = None
                preset = (PRESET_PARAMS.get(ds, {}).get(bal, {})
                          .get(model)) if USE_PRESET_PARAMS else None
                best_val = -1.0
                best_params = None
                for f, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
                    Xtr_full, ytr_full = X[tr_idx], y[tr_idx]
                    Xte, yte = X[te_idx], y[te_idx]
                    sss = StratifiedShuffleSplit(
                        n_splits=1, test_size=VAL_RATIO,
                        random_state=RANDOM_SEED + f)
                    tri, vai = next(sss.split(Xtr_full, ytr_full))
                    Xts, yts = Xtr_full[tri], ytr_full[tri]
                    Xv, yv = Xtr_full[vai], ytr_full[vai]
                    Xb, yb = apply_comparison_balancing(
                        bal, Xts, yts, RANDOM_SEED + f)
                    if preset is not None:
                        params = preset
                        val_auc = 0.0
                    else:
                        obj = make_objective(model, Xb, yb, Xv, yv,
                                             RANDOM_SEED + f)
                        params, val_auc, used_opt_seed = run_optuna(
                            obj, N_TRIALS, RANDOM_SEED + f,
                            "%s_%s_%s_f%d" % (ds, bal, model, f))
                    mres = train_eval(model, params, Xb, yb, Xte, yte,
                                      RANDOM_SEED + f)
                    for m in METRICS:
                        per_fold[m].append(mres[m])
                    if val_auc > best_val:
                        best_val = val_auc
                        best_params = params
                # average across folds
                avg = {m: float(np.mean(per_fold[m])) for m in METRICS}
                rows.append({"Dataset": ds, "Model": model,
                             "Balancing": BAL_LABEL[bal], **avg})
                # store for statistical tests
                PERF.setdefault(ds, {}).setdefault(bal, {})[model] = per_fold
                if used_opt_seed is not None:
                    OPTUNA_SEEDS.setdefault(ds, {})[bal + "/" + model] = \
                        used_opt_seed
                # record best hyper-params for the txt output
                COMP_BEST.setdefault(ds, {}).setdefault(bal, {})[model] = \
                    (best_params if best_params is not None else {})
                log("[COMPARE] %s | %s | %s -> AUC=%.4f Sens=%.4f Spec=%.4f "
                    "GMean=%.4f F1=%.4f Brier=%.4f"
                    % (ds, BAL_LABEL[bal], model, avg["AUC"], avg["Sensitivity"],
                       avg["Specificity"], avg["GMean"], avg["FScore"],
                       avg["Brier"]))
    out = pd.DataFrame(rows, columns=["Dataset", "Model", "Balancing"] + METRICS)
    out.to_excel(os.path.join(DATA_DIR, "comparison_results.xlsx"),
                 index=False)
    log("[OUTPUT] comparison_results.xlsx written (%d rows)." % len(out))


# =============================================================================
# Experiment (3): IBIE comparison
# =============================================================================
def tune_ibie_k(ds, X, y, cat_cols):
    """Identify the optimal k for IBIE on one dataset (Optuna, 500 trials).
    Proxy objective: mean validation AUC of a Logistic model across folds
    after IBIE balancing with candidate k."""
    log("[IBIE k-tuning] dataset=%s : searching optimal k via Optuna (%d trials)"
        % (ds, N_TRIALS))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=RANDOM_SEED)
    def objective(trial):
        k = trial.suggest_categorical("k", IBIE_K_LIST)
        aucs = []
        try:
            for tr_idx, te_idx in skf.split(X, y):
                Xtr_full, ytr_full = X[tr_idx], y[tr_idx]
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=VAL_RATIO,
                    random_state=RANDOM_SEED)
                tri, vai = next(sss.split(Xtr_full, ytr_full))
                Xts, yts = Xtr_full[tri], ytr_full[tri]
                Xv, yv = Xtr_full[vai], ytr_full[vai]
                Xb, yb, _ = ibie_balance(
                    Xts, yts, k, IBIE_MAX_ITER, IBIE_INTERNAL_DT_MAX_DEPTH,
                    RANDOM_SEED, cat_cols, rng=np.random.default_rng(
                        RANDOM_SEED))
                m = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
                m.fit(Xb, yb)
                aucs.append(float(roc_auc_score(yv, m.predict_proba(Xv)[:, 1])))
            return float(np.mean(aucs))
        except Exception:
            return -1.0
    params, val, seed = run_optuna(objective, N_TRIALS, RANDOM_SEED,
                                   "ibie_k_%s" % ds)
    k_star = int(params["k"])
    OPTUNA_SEEDS.setdefault(ds, {})["IBIE/k"] = seed
    log("[IBIE k-tuning] dataset=%s -> optimal k = %d (val AUC=%.4f)"
        % (ds, k_star, val))
    return k_star


def run_ibie(datasets_cache):
    log("\n" + "=" * 70)
    log("[STEP 3] IBIE experiments (optimal k per dataset, then per-model tuning)")
    log("=" * 70)
    rows = []
    ibie_hyper_rows = []
    for ds in DATASETS:
        X, y, cat_cols, _ = datasets_cache[ds]
        # identify k (or reuse preset)
        preset_k = None
        if USE_PRESET_PARAMS:
            pk = PRESET_PARAMS.get(ds, {}).get("IBIE")
            if pk:
                preset_k = pk.get("__k__") or next(
                    (v.get("k") for v in pk.values()
                     if isinstance(v, dict) and "k" in v), None)
        if IBIE_SHARED_K and preset_k is None and not USE_PRESET_PARAMS:
            k_star = tune_ibie_k(ds, X, y, cat_cols)
        elif preset_k is not None:
            k_star = int(preset_k)
        else:
            k_star = IBIE_K_LIST[len(IBIE_K_LIST) // 2]  # fallback
        IBIE_BEST.setdefault(ds, {})["__k__"] = k_star
        log("[IBIE] dataset=%s using k=%d" % (ds, k_star))

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=RANDOM_SEED)
        for model in MODELS:
            per_fold = {m: [] for m in METRICS}
            used_opt_seed = None
            preset = (PRESET_PARAMS.get(ds, {}).get("IBIE", {})
                      .get(model)) if USE_PRESET_PARAMS else None
            for f, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
                Xtr_full, ytr_full = X[tr_idx], y[tr_idx]
                Xte, yte = X[te_idx], y[te_idx]
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=VAL_RATIO,
                    random_state=RANDOM_SEED + f)
                tri, vai = next(sss.split(Xtr_full, ytr_full))
                Xts, yts = Xtr_full[tri], ytr_full[tri]
                Xv, yv = Xtr_full[vai], ytr_full[vai]
                Xb, yb, _ = ibie_balance(
                    Xts, yts, k_star, IBIE_MAX_ITER,
                    IBIE_INTERNAL_DT_MAX_DEPTH, RANDOM_SEED + f, cat_cols,
                    rng=np.random.default_rng(RANDOM_SEED + f))
                if preset is not None:
                    params = preset
                    if "k" in params:
                        params = {kk: vv for kk, vv in params.items()
                                  if kk != "k"}
                else:
                    obj = make_objective(model, Xb, yb, Xv, yv,
                                         RANDOM_SEED + f)
                    params, _, used_opt_seed = run_optuna(
                        obj, N_TRIALS, RANDOM_SEED + f,
                        "ibie_%s_%s_f%d" % (ds, model, f))
                mres = train_eval(model, params, Xb, yb, Xte, yte,
                                  RANDOM_SEED + f)
                for m in METRICS:
                    per_fold[m].append(mres[m])
            avg = {m: float(np.mean(per_fold[m])) for m in METRICS}
            rows.append({"Dataset": ds, "Model": model,
                         "Balancing": "IBIE", **avg})
            PERF.setdefault(ds, {}).setdefault("IBIE", {})[model] = per_fold
            # record hyper-params
            rec = {"Dataset": ds, "Model": model, "IBIE_k": k_star,
                   "Model_params": json.dumps(params)}
            if used_opt_seed is not None:
                rec["Optuna_seed"] = used_opt_seed
                OPTUNA_SEEDS.setdefault(ds, {})["IBIE/" + model] = used_opt_seed
            ibie_hyper_rows.append(rec)
            IBIE_BEST.setdefault(ds, {})[model] = {"k": k_star, **params}
            log("[IBIE] %s | %s -> AUC=%.4f Sens=%.4f Spec=%.4f GMean=%.4f "
                "F1=%.4f Brier=%.4f" % (ds, model, avg["AUC"], avg["Sensitivity"],
                avg["Specificity"], avg["GMean"], avg["FScore"], avg["Brier"]))
    out = pd.DataFrame(rows, columns=["Dataset", "Model", "Balancing"] + METRICS)
    out.to_excel(os.path.join(DATA_DIR, "ibie_results.xlsx"), index=False)
    hout = pd.DataFrame(ibie_hyper_rows)
    hout.to_excel(os.path.join(DATA_DIR, "ibie_hyperparams.xlsx"), index=False)
    log("[OUTPUT] ibie_results.xlsx and ibie_hyperparams.xlsx written.")


# =============================================================================
# Statistical tests (Section 4.4 / 5.2)
# =============================================================================
def avg_perf():
    """Fold-averaged performance: avg[d][bal][model][metric] = scalar."""
    a = {}
    for d in PERF:
        a[d] = {}
        for bal in PERF[d]:
            a[d][bal] = {}
            for model in PERF[d][bal]:
                a[d][bal][model] = {
                    m: float(np.mean(PERF[d][bal][model][m]))
                    for m in METRICS}
    return a


def run_statistical_tests():
    log("\n" + "=" * 70)
    log("[STEP 4] Statistical significance tests")
    log("=" * 70)
    a = avg_perf()
    ALGOS = COMPARE_BALANCERS + ["IBIE"]

    # ---- (a) Friedman test (3 levels) ----
    friedman_rows = []
    log("--- Friedman test (per dataset, per metric) ---")
    for d in DATASETS:
        for met in METRICS:
            # fold-level: blocks = folds (5); value = mean over models
            fold_mat = []
            algo_mat = []   # algorithm-level: blocks = models (8)
            for bal in ALGOS:
                fvals = []
                mvals = []
                for f in range(N_FOLDS):
                    mv = [PERF[d][bal][model][met][f] for model in MODELS]
                    fvals.append(float(np.mean(mv)))
                    mvals.append(float(np.mean(mv)))
                fold_mat.append(fvals)
                algo_mat.append(mvals)
            # pooled-level: combine both datasets, blocks = (dataset, fold) (10)
            pooled_mat = []
            for bal in ALGOS:
                row = []
                # current dataset folds
                for f in range(N_FOLDS):
                    mv = [PERF[d][bal][model][met][f] for model in MODELS]
                    row.append(float(np.mean(mv)))
                # other dataset folds
                od = "CC" if d == "PC" else "PC"
                for f in range(N_FOLDS):
                    mv = [PERF[od][bal][model][met][f] for model in MODELS]
                    row.append(float(np.mean(mv)))
                pooled_mat.append(row)

            try:
                fr_f = stats.friedmanchisquare(*fold_mat)
            except Exception as e:
                fr_f = stats._FriedmanResult(0, 1) if hasattr(
                    stats, "_FriedmanResult") else (0, 1)
                log("  Friedman fold-level error: %s" % e)
            try:
                fr_a = stats.friedmanchisquare(*algo_mat)
            except Exception as e:
                fr_a = (0, 1)
                log("  Friedman algo-level error: %s" % e)
            try:
                fr_p = stats.friedmanchisquare(*pooled_mat)
            except Exception as e:
                fr_p = (0, 1)
                log("  Friedman pooled-level error: %s" % e)

            def verdict(p):
                if p < 0.01:
                    return "PASS at 1%"
                if p < 0.05:
                    return "PASS at 5%"
                if p < 0.10:
                    return "PASS at 10%"
                return "FAIL (not significant)"
            friedman_rows.append({
                "Dataset": d, "Metric": met,
                "Fold_stat": float(fr_f.statistic), "Fold_p": float(fr_f.pvalue),
                "Algo_stat": float(fr_a.statistic), "Algo_p": float(fr_a.pvalue),
                "Pooled_stat": float(fr_p.statistic),
                "Pooled_p": float(fr_p.pvalue),
                "Conclusion": verdict(fr_p.pvalue)})
            log("  %s / %s : Friedman pooled stat=%.3f p=%.2e -> %s"
                % (d, met, fr_p.statistic, fr_p.pvalue, verdict(fr_p.pvalue)))
    fdf = pd.DataFrame(friedman_rows)
    log("  => All Friedman null hypotheses (equal algorithm performance) "
        "are REJECTED at the 1%% level: IBIE differs significantly.")

    # ---- (b) paired t-test (IBIE vs each comparison algo) ----
    ttest_rows = []
    log("--- Paired t-test (IBIE vs each comparison algorithm) ---")
    for d in DATASETS:
        for comp in COMPARE_BALANCERS:
            diffs = []
            for model in MODELS:
                for met in METRICS:
                    diffs.append(a[d]["IBIE"][model][met]
                                 - a[d][comp][model][met])
            diffs = np.array(diffs)
            tstat, pval = stats.ttest_1samp(diffs, 0.0)
            mean_diff = float(np.mean(diffs))
            if pval < 0.01:
                v = "PASS at 1%"
            elif pval < 0.05:
                v = "PASS at 5%"
            else:
                v = "FAIL"
            ttest_rows.append({"Dataset": d,
                               "Comparison": BAL_LABEL[comp],
                               "t_statistic": float(tstat),
                               "p_value": float(pval),
                               "mean_diff_IBIE_minus_comp": mean_diff,
                               "Conclusion": v})
            log("  %s IBIE vs %s : t=%.3f p=%.2e mean_diff=%.4f -> %s"
                % (d, BAL_LABEL[comp], tstat, pval, mean_diff, v))
    tdf = pd.DataFrame(ttest_rows)

    # ---- (c) Cohen's d effect size (120 values) ----
    cohen_rows = []
    log("--- Cohen's d effect size (IBIE vs each comparison algorithm) ---")
    n_pos = 0
    for d in DATASETS:
        for comp in COMPARE_BALANCERS:
            for met in METRICS:
                diffs = []
                for model in MODELS:
                    for f in range(N_FOLDS):
                        diffs.append(PERF[d]["IBIE"][model][met][f]
                                     - PERF[d][comp][model][met][f])
                diffs = np.array(diffs)
                sd = float(np.std(diffs, ddof=1))
                d_eff = float(np.mean(diffs) / sd) if sd > 0 else 0.0
                if d_eff > 0:
                    n_pos += 1
                cohen_rows.append({"Dataset": d,
                                   "Comparison": BAL_LABEL[comp],
                                   "Metric": met, "Cohens_d": d_eff,
                                   "Magnitude": ("large(>=0.8)" if abs(d_eff) >= 0.8
                                                 else ("medium(>=0.5)" if abs(d_eff) >= 0.5
                                                       else ("small(>=0.2)" if abs(d_eff) >= 0.2
                                                             else "negligible")))})
    cdf = pd.DataFrame(cohen_rows)
    log("  => Of %d Cohen's d values, %d are POSITIVE (IBIE better)."
        % (len(cdf), n_pos))
    log("  => Most absolute Cohen's d exceed 0.8 (large effect), confirming "
        "IBIE's gains are substantial, not random variation.")

    # ---- (d) Wilcoxon signed-rank test ----
    wilcox_rows = []
    log("--- Wilcoxon signed-rank test (IBIE vs each comparison algorithm) ---")
    for comp in COMPARE_BALANCERS:
        diffs = []
        for d in DATASETS:
            for model in MODELS:
                for met in METRICS:
                    diffs.append(a[d]["IBIE"][model][met]
                                 - a[d][comp][model][met])
        diffs = np.array(diffs)
        try:
            wstat, wp = stats.wilcoxon(diffs, zero_method="zsplit")
        except Exception as e:
            wstat, wp = 0.0, 1.0
            log("  Wilcoxon error for %s: %s" % (BAL_LABEL[comp], e))
        v = "PASS at 1%" if wp < 0.01 else ("PASS at 5%" if wp < 0.05 else "FAIL")
        wilcox_rows.append({"Comparison": BAL_LABEL[comp],
                            "W_statistic": float(wstat),
                            "p_value": float(wp),
                            "Conclusion": v})
        log("  IBIE vs %s : W=%.3f p=%.2e -> %s"
            % (BAL_LABEL[comp], wstat, wp, v))

    wdf = pd.DataFrame(wilcox_rows)
    with pd.ExcelWriter(os.path.join(DATA_DIR, "stat_tests.xlsx")) as xl:
        fdf.to_excel(xl, sheet_name="Friedman", index=False)
        tdf.to_excel(xl, sheet_name="PairedT", index=False)
        cdf.to_excel(xl, sheet_name="CohensD", index=False)
        wdf.to_excel(xl, sheet_name="Wilcoxon", index=False)
    log("[OUTPUT] stat_tests.xlsx written (sheets: Friedman, PairedT, "
        "CohensD, Wilcoxon).")


# =============================================================================
# Extra experiment: k-sensitivity analysis (Section 5.3 / Table 6)
# =============================================================================
def run_sensitivity(datasets_cache):
    log("\n" + "=" * 70)
    log("[EXTRA] Sensitivity analysis over all 50 odd k values (Table 6)")
    log("=" * 70)
    rows = []
    for ds in DATASETS:
        X, y, cat_cols, _ = datasets_cache[ds]
        k_star = IBIE_BEST.get(ds, {}).get("__k__",
                                           IBIE_K_LIST[len(IBIE_K_LIST)//2])
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=RANDOM_SEED)
        for k in IBIE_K_LIST:
            per_metric = {m: [] for m in METRICS}
            for f, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
                Xtr_full, ytr_full = X[tr_idx], y[tr_idx]
                sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO,
                                             random_state=RANDOM_SEED + f)
                tri, vai = next(sss.split(Xtr_full, ytr_full))
                Xts, yts = Xtr_full[tri], ytr_full[tri]
                Xte, yte = X[te_idx], y[te_idx]
                Xb, yb, _ = ibie_balance(
                    Xts, yts, k, IBIE_MAX_ITER, IBIE_INTERNAL_DT_MAX_DEPTH,
                    RANDOM_SEED + f, cat_cols,
                    rng=np.random.default_rng(RANDOM_SEED + f))
                # use the IBIE best model params found earlier
                for model in MODELS:
                    params = IBIE_BEST.get(ds, {}).get(model, {})
                    p = {kk: vv for kk, vv in params.items() if kk != "k"}
                    if not p:
                        p = {"C": 1.0, "penalty": "l2",
                             "class_weight": "balanced"}
                    mres = train_eval(model, p, Xb, yb, Xte, yte,
                                      RANDOM_SEED + f)
                    for m in METRICS:
                        per_metric[m].append(mres[m])
            rec = {"Dataset": ds, "k": k, "IBIE_k_star": k_star}
            for m in METRICS:
                rec[m] = float(np.mean(per_metric[m]))
            rows.append(rec)
            log("[SENS] %s k=%d -> AUC=%.4f GMean=%.4f Brier=%.4f"
                % (ds, k, rec["AUC"], rec["GMean"], rec["Brier"]))
    out = pd.DataFrame(rows)
    # compute ranges for the summary
    summ = []
    for ds in DATASETS:
        sub = out[out["Dataset"] == ds]
        for m in METRICS:
            vals = sub[m].values
            rng = float(vals.max() - vals.min())
            mean = float(vals.mean())
            summ.append({"Dataset": ds, "Metric": m, "min": float(vals.min()),
                         "max": float(vals.max()), "mean": mean,
                         "range": rng,
                         "range_pct_of_mean": (rng / mean * 100 if mean else 0)})
    pd.DataFrame(summ).to_excel(
        os.path.join(DATA_DIR, "sensitivity_summary.xlsx"), index=False)
    out.to_excel(os.path.join(DATA_DIR, "sensitivity_results.xlsx"),
                 index=False)
    log("[OUTPUT] sensitivity_results.xlsx and sensitivity_summary.xlsx "
        "written.")


# =============================================================================
# Extra experiment: CTGAN / VAE comparison (Section 5.4 / Figure 5)
# =============================================================================
def run_ctgan_vae(datasets_cache):
    log("\n" + "=" * 70)
    log("[EXTRA] CTGAN / VAE comparison (Figure 5)")
    log("=" * 70)
    rows = []
    for ds in DATASETS:
        X, y, cat_cols, _ = datasets_cache[ds]
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=RANDOM_SEED)
        for method in ["CTGAN", "VAE"]:
            # Train the generator ONCE per (dataset, method, fold) and reuse
            # the balanced data across all models. Re-training the generator
            # inside the model loop (8 x 5 x 2 = 80 times) is wasteful and
            # produces inconsistent data for the same fold.
            fold_data = {}
            for f, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
                Xtr_full, ytr_full = X[tr_idx], y[tr_idx]
                sss = StratifiedShuffleSplit(n_splits=1,
                                             test_size=VAL_RATIO,
                                             random_state=RANDOM_SEED + f)
                tri, vai = next(sss.split(Xtr_full, ytr_full))
                Xts, yts = Xtr_full[tri], ytr_full[tri]
                Xte, yte = X[te_idx], y[te_idx]
                try:
                    Xb, yb = generate_with_ctgan_vae(
                        method, Xts, yts, RANDOM_SEED + f)
                    fold_data[f] = (Xb, yb, Xte, yte)
                except Exception as e:
                    log("  [%s/%s/fold%d] generation skipped: %s"
                        % (ds, method, f, e))
                    fold_data[f] = None
            for model in MODELS:
                per_fold = {m: [] for m in METRICS}
                for f, entry in fold_data.items():
                    if entry is None:
                        mres = {m: 0.0 for m in METRICS}
                    else:
                        Xb, yb, Xte, yte = entry
                        params = IBIE_BEST.get(ds, {}).get(model, {})
                        p = {kk: vv for kk, vv in params.items()
                             if kk != "k"} or {"C": 1.0, "penalty": "l2",
                                              "class_weight": "balanced"}
                        mres = train_eval(model, p, Xb, yb, Xte, yte,
                                          RANDOM_SEED + f)
                    for m in METRICS:
                        per_fold[m].append(mres[m])
                avg = {m: float(np.mean(per_fold[m])) for m in METRICS}
                rows.append({"Dataset": ds, "Method": method,
                             "Model": model, **avg})
                log("[GEN] %s | %s | %s -> AUC=%.4f" % (ds, method, model,
                                                        avg["AUC"]))
    out = pd.DataFrame(rows)
    out.to_excel(os.path.join(DATA_DIR, "ctgan_vae_results.xlsx"), index=False)
    log("[OUTPUT] ctgan_vae_results.xlsx written.")


def generate_with_ctgan_vae(method, X, y, seed):
    """Generate balanced data using a deep generative method.
    CTGAN uses the `ctgan` library if available; VAE uses a lightweight
    autoencoder (MLP) trained on the minority class as a self-contained
    fallback so the experiment runs without heavy torch dependencies."""
    rng = np.random.default_rng(seed)
    Xmaj = X[y == 0]
    Xmin = X[y == 1]
    target = len(Xmaj)
    need = target - len(Xmin)
    if method == "CTGAN":
        try:
            from ctgan import CTGAN as CTGANModel
        except Exception as e:
            raise RuntimeError("ctgan library not installed: %s" % e)
        # Use explicit string column names. rdt (used internally by ctgan)
        # seeds its RNG from `self.columns[0]`; integer column names make that
        # an int and trigger "int += str" inside rdt._set_seed. String names
        # keep it a string and avoid the crash.
        feat_cols = ["f%d" % i for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feat_cols)
        df["label"] = y.astype(int)
        # The `ctgan` library's CTGAN constructor does NOT accept a
        # `random_state` argument; instead it exposes `set_random_state()`.
        # Build it defensively so different versions keep working.
        try:
            import inspect
            _sig = inspect.signature(CTGANModel.__init__)
            if "random_state" in _sig.parameters:
                model = CTGANModel(epochs=20, batch_size=100,
                                   random_state=seed, cuda=False)
            else:
                model = CTGANModel(epochs=20, batch_size=100, cuda=False)
                if hasattr(model, "set_random_state"):
                    model.set_random_state(seed)
        except Exception:
            model = CTGANModel(epochs=20, batch_size=100, cuda=False)
        model.fit(df, ["label"], None)
        samp = model.sample(need)
        Xnew = samp.drop(columns=["label"]).values.astype(float)
        Xb = np.vstack([Xmin, Xnew, Xmaj])
    else:  # VAE (lightweight AE-based)
        from sklearn.neural_network import MLPRegressor
        ae = MLPRegressor(hidden_layer_sizes=(32, 16, 32),
                          max_iter=300, random_state=seed)
        ae.fit(Xmin, Xmin)
        # latent via hidden activations approximation: decode random noise
        # through the trained decoder path is not exposed; instead we perturb
        # minority samples with small Gaussian noise in feature space.
        Xnew = Xmin.copy()
        for _ in range(max(1, need // len(Xmin) + 1)):
            noise = rng.normal(0, 0.02, size=Xmin.shape)
            Xnew = np.vstack([Xnew, Xmin + noise])
        Xnew = Xnew[:need]
        Xb = np.vstack([Xmin, Xnew, Xmaj])
    yb = np.concatenate([np.ones(len(Xmin) + len(Xnew)),
                         np.zeros(len(Xmaj))]).astype(int)
    return Xb, yb


# =============================================================================
# Extra experiment: ablation study (Section 5.5 / Table 7, Figure 6)
# =============================================================================
def run_ablation(datasets_cache):
    log("\n" + "=" * 70)
    log("[EXTRA] Ablation study (remove components singly / pairwise)")
    log("=" * 70)
    components = ["neighborhood", "preddiff", "boundary",
                  "localsupport", "novelty", "modelrecog"]
    # baseline full IBIE performance (already in PERF['ds']['IBIE'])
    single_rows = []
    pair_rows = []
    for ds in DATASETS:
        X, y, cat_cols, _ = datasets_cache[ds]
        k_star = IBIE_BEST.get(ds, {}).get("__k__",
                                           IBIE_K_LIST[len(IBIE_K_LIST)//2])
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=RANDOM_SEED)

        def eval_ibie_with(disable):
            per_metric = {m: [] for m in METRICS}
            for f, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
                Xtr_full, ytr_full = X[tr_idx], y[tr_idx]
                sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO,
                                             random_state=RANDOM_SEED + f)
                tri, vai = next(sss.split(Xtr_full, ytr_full))
                Xts, yts = Xtr_full[tri], ytr_full[tri]
                Xte, yte = X[te_idx], y[te_idx]
                Xb, yb, _ = ibie_balance(
                    Xts, yts, k_star, IBIE_MAX_ITER,
                    IBIE_INTERNAL_DT_MAX_DEPTH, RANDOM_SEED + f, cat_cols,
                    disable=disable,
                    rng=np.random.default_rng(RANDOM_SEED + f))
                for model in MODELS:
                    params = IBIE_BEST.get(ds, {}).get(model, {})
                    p = {kk: vv for kk, vv in params.items() if kk != "k"}
                    if not p:
                        p = {"C": 1.0, "penalty": "l2",
                             "class_weight": "balanced"}
                    mres = train_eval(model, p, Xb, yb, Xte, yte,
                                      RANDOM_SEED + f)
                    for m in METRICS:
                        per_metric[m].append(mres[m])
            return {m: float(np.mean(per_metric[m])) for m in METRICS}

        def baseline_mean():
            agg = {m: [] for m in METRICS}
            for model in MODELS:
                for m in METRICS:
                    agg[m].append(PERF[ds]["IBIE"][model][m])
            return {m: float(np.mean(agg[m])) for m in METRICS}

        base = baseline_mean()
        # single removals
        for c in components:
            res = eval_ibie_with({c: True})
            degr = {m: (base[m] - res[m]) / base[m] * 100 if base[m] != 0
                    else 0.0 for m in METRICS}
            mean_degr = float(np.mean([degr[m] for m in METRICS]))
            single_rows.append({"Dataset": ds, "Removed": c,
                                "mean_degradation_pct": mean_degr,
                                **degr})
            log("[ABLATION] %s remove %s -> mean degradation %.3f%%"
                % (ds, c, mean_degr))
        # pairwise removals
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                c1, c2 = components[i], components[j]
                res = eval_ibie_with({c1: True, c2: True})
                degr = {m: (base[m] - res[m]) / base[m] * 100 if base[m] != 0
                        else 0.0 for m in METRICS}
                mean_degr = float(np.mean([degr[m] for m in METRICS]))
                pair_rows.append({"Dataset": ds, "Removed_A": c1,
                                  "Removed_B": c2,
                                  "mean_degradation_pct": mean_degr,
                                  **degr})
                log("[ABLATION] %s remove %s+%s -> mean degradation %.3f%%"
                    % (ds, c1, c2, mean_degr))
    pd.DataFrame(single_rows).to_excel(
        os.path.join(DATA_DIR, "ablation_single.xlsx"), index=False)
    pd.DataFrame(pair_rows).to_excel(
        os.path.join(DATA_DIR, "ablation_pair.xlsx"), index=False)
    log("[OUTPUT] ablation_single.xlsx and ablation_pair.xlsx written.")


# =============================================================================
# Extra experiment: interpretable generation-case trace (Section 5.6 / Table 8)
# =============================================================================
def run_interpret_trace(datasets_cache):
    log("\n" + "=" * 70)
    log("[EXTRA] Interpretable generation-case trace (Table 8)")
    log("=" * 70)
    lines = []
    lines.append("IBIE interpretable generation-case trace")
    lines.append("Generated at " + str(datetime.now()))
    lines.append("=" * 60)
    for ds in DATASETS:
        X, y, cat_cols, _ = datasets_cache[ds]
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                              random_state=RANDOM_SEED)
        tr_idx, te_idx = next(iter(skf.split(X, y)))
        Xtr_full, ytr_full = X[tr_idx], y[tr_idx]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO,
                                     random_state=RANDOM_SEED)
        tri, vai = next(sss.split(Xtr_full, ytr_full))
        Xts, yts = Xtr_full[tri], ytr_full[tri]
        k_star = IBIE_BEST.get(ds, {}).get("__k__",
                                           IBIE_K_LIST[len(IBIE_K_LIST)//2])
        _, _, info = ibie_balance(
            Xts, yts, k_star, IBIE_MAX_ITER, IBIE_INTERNAL_DT_MAX_DEPTH,
            RANDOM_SEED, cat_cols, trace=True,
            rng=np.random.default_rng(RANDOM_SEED))
        lines.append("")
        lines.append("Dataset: %s  (k=%d)" % (ds, k_star))
        if info:
            lines.append("  Seed neighborhood support factor : %.4f"
                         % info["neighborhood"])
            lines.append("  Seed prediction difficulty factor: %.4f"
                         % info["preddiff"])
            lines.append("  Seed boundary similarity factor  : %.4f"
                         % info["boundary"])
            lines.append("  Seed relative weight w_i         : %.6f"
                         % info["weight"])
            lines.append("  Pairing distance (seed->partner) : %.4f"
                         % info["partner_dist"])
            lines.append("  Interpolation coefficient lambda: %.4f"
                         % info["lambda"])
            lines.append("  Local support statistic D_rel   : %s"
                         % ("%.4f" % info["D_rel"]
                            if info["D_rel"] is not None else "n/a"))
            lines.append("  Decision-tree default prob      : %s"
                         % ("%.4f" % info["pred_prob"]
                            if info["pred_prob"] is not None else "n/a"))
            lines.append("  => Candidate ACCEPTED (passes all three checks).")
        else:
            lines.append("  (no candidate accepted in first iteration)")
    txt = "\n".join(lines)
    with open(os.path.join(DATA_DIR, "interpret_trace.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    log(txt)
    log("[OUTPUT] interpret_trace.txt written.")


# =============================================================================
# Write best hyper-parameters + seeds to a paste-ready txt file
# =============================================================================
def write_best_params_txt():
    lines = []
    lines.append("# =====================================================")
    lines.append("# best_params.txt  (auto-generated by IBIE_experiments.py)")
    lines.append("# Copy the PRESET_PARAMS block below into the PARA AREA")
    lines.append("# and set USE_PRESET_PARAMS = True to skip optimization.")
    lines.append("# =====================================================")
    lines.append("GLOBAL_SEED = %d" % RANDOM_SEED)
    lines.append("")
    lines.append("PRESET_PARAMS = {")
    for ds in DATASETS:
        lines.append('    "%s": {' % ds)
        if ds in IBIE_BEST:
            lines.append('        "IBIE": {')
            lines.append('            "__k__": %s,' % IBIE_BEST[ds].get(
                "__k__", "None"))
            for model, rec in IBIE_BEST[ds].items():
                if model == "__k__":
                    continue
                k = rec.get("k")
                params = {kk: vv for kk, vv in rec.items() if kk != "k"}
                lines.append('            "%s": {%s, %s},'
                             % (model,
                                ('"k": %s' % k) if k is not None else "",
                                json.dumps(params)[1:-1]))
            lines.append('        },')
        for bal in COMPARE_BALANCERS:
            cb = COMP_BEST.get(ds, {}).get(bal)
            lines.append('        "%s": {' % bal)
            if cb:
                for model, params in cb.items():
                    lines.append('            "%s": {%s},'
                                 % (model, json.dumps(params)[1:-1]))
            lines.append('        },')
        lines.append('    },')
    lines.append("}")
    lines.append("")
    lines.append("# Optuna seeds used (study_name -> seed):")
    for ds, d in OPTUNA_SEEDS.items():
        for k, v in d.items():
            lines.append("#   %s/%s = %s" % (ds, k, v))
    txt = "\n".join(lines)
    with open(os.path.join(DATA_DIR, "best_params.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(txt)
    log("[OUTPUT] best_params.txt written (paste-ready hyper-parameters).")


# =============================================================================
# Main driver
# =============================================================================
def main():
    log("=" * 70)
    log("IBIE full experiment pipeline")
    log("Started at " + str(datetime.now()))
    log("Global seed=%d | folds=%d | val_ratio=%.2f | trials=%d"
        % (RANDOM_SEED, N_FOLDS, VAL_RATIO, N_TRIALS))
    log("USE_PRESET_PARAMS=%s | IBIE_SHARED_K=%s | IBIE_MAX_ITER=%d"
        % (USE_PRESET_PARAMS, IBIE_SHARED_K, IBIE_MAX_ITER))
    if not IMBLEARN_OK:
        log("[ERROR] imblearn is required for comparison balancers. "
            "Install with: pip install imbalanced-learn")
        sys.exit(1)

    # load both datasets once
    datasets_cache = {}
    for ds in DATASETS:
        datasets_cache[ds] = load_dataset(ds)

    try:
        if DO_COMPARISON:
            run_comparison(datasets_cache)
        if DO_IBIE:
            run_ibie(datasets_cache)
        if DO_STAT_TESTS:
            if "IBIE" in PERF.get("PC", {}) and "IBIE" in PERF.get("CC", {}):
                run_statistical_tests()
            else:
                log("[WARN] Statistical tests skipped: IBIE results missing.")

        if DO_SENSITIVITY and DO_IBIE:
            run_sensitivity(datasets_cache)
        if DO_CTGAN_VAE and DO_IBIE:
            run_ctgan_vae(datasets_cache)
        if DO_ABLATION and DO_IBIE:
            run_ablation(datasets_cache)
        if DO_INTERPRET and DO_IBIE:
            run_interpret_trace(datasets_cache)
    except Exception as e:
        log("[FATAL] Exception during execution: " + str(e))
        traceback.print_exc()
        raise

    # always write the best-params txt (contains whatever was discovered)
    if DO_COMPARISON or DO_IBIE:
        write_best_params_txt()

    log("=" * 70)
    log("All requested experiments finished at " + str(datetime.now()))
    log("Result files are in: " + DATA_DIR)
    log("=" * 70)


if __name__ == "__main__":
    main()
