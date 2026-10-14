"""Moteur de scoring : feature store + contrôles + modèle ML + décision."""
from __future__ import annotations

import time
from dataclasses import replace
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from vigie.config import FraudConfig
from vigie.engine.aggregator import Decision, decide
from vigie.engine.checks import run_checks
from vigie.features.store import FeatureStore, to_seconds


class FraudEngine:
    def __init__(self, cfg: FraudConfig | None = None, store: FeatureStore | None = None,
                 model=None, feature_names: list[str] | None = None):
        self.cfg = cfg or FraudConfig()
        self.store = store or FeatureStore(self.cfg)
        self.model = model
        self.feature_names = feature_names or FeatureStore.FEATURE_NAMES
        self.stats = Counter()
        self.latency_ms_total = 0.0

    @classmethod
    def from_artifacts(cls, model_path: str | Path, cfg: FraudConfig | None = None,
                       store: FeatureStore | None = None) -> "FraudEngine":
        bundle = joblib.load(model_path)
        cfg = cfg or FraudConfig()
        if "threshold" in bundle:  # seuil choisi en validation à l'entraînement
            cfg = replace(cfg, ml_review_proba=float(bundle["threshold"]))
        return cls(cfg, store, bundle["model"], bundle["feature_names"])

    def _ml_proba(self, features: dict) -> float | None:
        if self.model is None:
            return None
        x = np.array([[features[n] for n in self.feature_names]], dtype=float)
        return float(self.model.predict_proba(x)[0, 1])

    def score(self, tx: dict, update: bool = True) -> tuple[dict, Decision]:
        t0 = time.perf_counter()
        features = self.store.compute(tx)
        signals = run_checks(features, tx["txn_type"], self.cfg)
        decision = decide(signals, self._ml_proba(features), self.cfg)
        if update:
            self.store.update(tx)
        self.stats[decision.decision] += 1
        self.stats["total"] += 1
        self.latency_ms_total += (time.perf_counter() - t0) * 1000
        return features, decision

    def apply_event(self, ev: dict):
        self.store.apply_event(ev)


def replay(transactions: pd.DataFrame, events: pd.DataFrame, store: FeatureStore,
           engine: FraudEngine | None = None, progress: bool = False) -> pd.DataFrame:
    """Rejoue transactions et événements dans l'ordre chronologique.

    Sans moteur : renvoie seulement les variables (pour l'entraînement).
    Avec moteur : renvoie aussi les scores, décisions et raisons. Le modèle ML est appliqué
    en un seul lot à la fin (il ne modifie pas l'état), ce qui donne exactement le même
    résultat que `FraudEngine.score` transaction par transaction, en beaucoup plus rapide.
    """
    tx_records = transactions.to_dict("records")
    ev_records = events.to_dict("records") if events is not None else []
    ev_ts = [to_seconds(e["timestamp"]) for e in ev_records]
    j = 0
    feats, signals = [], []
    n = len(tx_records)
    for i, tx in enumerate(tx_records):
        t = to_seconds(tx["timestamp"])
        while j < len(ev_records) and ev_ts[j] <= t:
            store.apply_event(ev_records[j])
            j += 1
        f = store.process(tx)
        feats.append(f)
        if engine is not None:
            signals.append(run_checks(f, tx["txn_type"], engine.cfg))
        if progress and i and i % 50000 == 0:
            print(f"  … {i:,}/{n:,} transactions".replace(",", " "))
    fdf = pd.DataFrame(feats, index=transactions.index)
    if engine is None:
        return fdf

    names = engine.feature_names
    probas = [None] * n
    if engine.model is not None:
        probas = engine.model.predict_proba(fdf[names].to_numpy(dtype=float))[:, 1]
    rows = []
    for sig, p in zip(signals, probas):
        d = decide(sig, None if p is None else float(p), engine.cfg).to_dict()
        d["signals"] = "|".join(d["signals"])
        d["reasons"] = " ; ".join(d["reasons"])
        rows.append(d)
    return fdf[names].join(pd.DataFrame(rows, index=transactions.index))
