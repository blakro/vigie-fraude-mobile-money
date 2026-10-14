"""Entraînement du modèle supervisé et de la référence non supervisée.

- Modèle principal : HistGradientBoosting (poids de classes équilibrés) + calibration sigmoïde,
  pour que la probabilité sortie soit interprétable.
- Référence : Isolation Forest (sans étiquettes), pour mesurer ce qu'apportent les étiquettes.
- Découpage TEMPOREL : entraînement sur les 40 premiers jours, test sur les 20 suivants.
- Seuil d'alerte FIXE sur la probabilité calibrée (`FraudConfig.ml_review_proba`, 20 %). Un seuil
  optimisé sur une période de validation s'est révélé instable : les fraudes récentes n'y sont pas
  encore signalées, si bien que la validation « voit » de fausses fausses alertes.
- La période de VALIDATION (10 jours avant la coupure) sert à mesurer cet effet : la précision
  calculée avec les seules fraudes déjà signalées sous-estime la vraie précision.
- Étiquettes RÉALISTES : à la date de coupure, on ne connaît que les fraudes déjà SIGNALÉES
  (plainte d'un client ou audit interne). Les fraudes signalées plus tard, ou jamais, sont
  vues comme normales pendant l'entraînement, comme dans une vraie équipe fraude.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from vigie.config import SIM_START, TRAIN_DAYS, FraudConfig
from vigie.features.store import FeatureStore

FEATURES = FeatureStore.FEATURE_NAMES


def cutoff() -> pd.Timestamp:
    return pd.Timestamp(SIM_START + timedelta(days=TRAIN_DAYS))


def temporal_split(tx: pd.DataFrame):
    cut = cutoff()
    train = (tx.timestamp < cut).to_numpy()
    return train, ~train, cut


def known_labels(tx: pd.DataFrame, cases: pd.DataFrame, at: pd.Timestamp) -> np.ndarray:
    """Étiquette connue à la date `at` : fraude ET cas signalé avant cette date."""
    reported = tx.fraud_case_id.map(cases.set_index("case_id").reported_at)
    return ((tx.is_fraud == 1) & reported.notna() & (reported < at)).to_numpy().astype(int)


VALIDATION_DAYS = 10


def make_model(seed: int = 42) -> CalibratedClassifierCV:
    base = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
        class_weight="balanced", random_state=seed)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def train_models(tx: pd.DataFrame, feats: pd.DataFrame, cases: pd.DataFrame,
                 out_dir: str | Path = "models", seed: int = 42, threshold: float | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X = feats[FEATURES].to_numpy(dtype=float)
    y_true = tx.is_fraud.to_numpy()
    tr, te, cut = temporal_split(tx)
    y_known = known_labels(tx, cases, cut)

    threshold = FraudConfig().ml_review_proba if threshold is None else threshold

    # 1) validation (10 derniers jours avant la coupure) : précision mesurée avec les fraudes déjà
    #    signalées, comparée à la vraie précision (connue ici grâce à la simulation)
    val_start = cut - pd.Timedelta(days=VALIDATION_DAYS)
    fit = (tx.timestamp < val_start).to_numpy()
    val = tr & ~fit
    probe = make_model(seed).fit(X[fit], y_known[fit])
    val_flag = probe.predict_proba(X[val])[:, 1] >= threshold
    val_precision_known = float(y_known[val][val_flag].mean()) if val_flag.any() else 0.0
    val_precision_true = float(y_true[val][val_flag].mean()) if val_flag.any() else 0.0

    # 2) modèle final sur toute la période d'entraînement
    model = make_model(seed).fit(X[tr], y_known[tr])
    p_te = model.predict_proba(X[te])[:, 1]

    iso = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(X[tr])
    iso_te = -iso.score_samples(X[te])

    # Importance par permutation sur un échantillon du test (toutes les fraudes + 20 000 normales)
    rng = np.random.default_rng(seed)
    te_idx = np.where(te)[0]
    fr = te_idx[y_true[te_idx] == 1]
    nf = rng.choice(te_idx[y_true[te_idx] == 0], min(20000, int((y_true[te_idx] == 0).sum())), replace=False)
    sample = np.concatenate([fr, nf])
    imp = permutation_importance(model, X[sample], y_true[sample], scoring="average_precision",
                                 n_repeats=3, random_state=seed, n_jobs=-1)
    importance = sorted(zip(FEATURES, imp.importances_mean), key=lambda x: -x[1])

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_period_end": str(cut),
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "n_fraud_train_true": int(y_true[tr].sum()), "n_fraud_train_known": int(y_known[tr].sum()),
        "label_coverage_train": float(y_known[tr].sum() / max(1, y_true[tr].sum())),
        "ml_threshold": round(threshold, 4), "validation_start": str(val_start),
        "validation_precision_known_labels": round(val_precision_known, 4),
        "validation_precision_true": round(val_precision_true, 4),
        "fraud_rate_train": float(y_true[tr].mean()), "fraud_rate_test": float(y_true[te].mean()),
        "hgb_pr_auc": float(average_precision_score(y_true[te], p_te)),
        "hgb_roc_auc": float(roc_auc_score(y_true[te], p_te)),
        "iforest_pr_auc": float(average_precision_score(y_true[te], iso_te)),
        "iforest_roc_auc": float(roc_auc_score(y_true[te], iso_te)),
        "feature_importance": [{"feature": f, "importance": round(float(v), 4)}
                               for f, v in importance[:15]],
    }
    joblib.dump({"model": model, "feature_names": FEATURES, "threshold": threshold, "metrics": metrics},
                out_dir / "fraud_model.joblib")
    joblib.dump({"model": iso, "feature_names": FEATURES}, out_dir / "iforest.joblib")
    return metrics
