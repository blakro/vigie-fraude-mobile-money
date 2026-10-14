import numpy as np

from vigie.config import FraudConfig
from vigie.engine.engine import FraudEngine, replay
from vigie.features.store import FeatureStore, to_seconds


class DummyModel:
    """Modèle factice déterministe : sert à vérifier que le scoring par lot = scoring unitaire."""
    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(X[:, 1] - 10)))  # dépend de log_amount
        return np.column_stack([1 - p, p])


def test_batch_replay_matches_online_scoring(small):
    """Pas d'écart entre l'entraînement (rejeu par lot) et la production (opération par opération)."""
    cfg = FraudConfig()
    tx, ev = small.transactions.head(4000), small.events
    batch_engine = FraudEngine(cfg, FeatureStore(cfg, small.customers, small.merchants), DummyModel())
    batch = replay(tx, ev, batch_engine.store, batch_engine)

    online = FraudEngine(cfg, FeatureStore(cfg, small.customers, small.merchants), DummyModel())
    events = ev.to_dict("records")
    j = 0
    for i, row in enumerate(tx.to_dict("records")):
        while j < len(events) and to_seconds(events[j]["timestamp"]) <= to_seconds(row["timestamp"]):
            online.apply_event(events[j])
            j += 1
        _, d = online.score(row)
        assert d.decision == batch.decision.iat[i]
        assert abs(d.final_score - batch.final_score.iat[i]) < 0.11


def test_rules_catch_most_takeovers(small):
    store = FeatureStore(customers=small.customers, merchants=small.merchants)
    scored = replay(small.transactions, small.events, store, FraudEngine(store=store))
    ato = small.transactions.fraud_scenario == "PRISE_CONTROLE_COMPTE"
    assert (scored.decision[ato] != "ACCEPTER").mean() > 0.7
    normal = small.transactions.is_fraud == 0
    assert (scored.decision[normal] != "ACCEPTER").mean() < 0.01


def test_every_alert_has_a_reason(small):
    """Un analyste doit toujours savoir pourquoi une opération est en revue ou bloquée."""
    cfg = FraudConfig()
    cfg.ml_review_proba = 0.2
    engine = FraudEngine(cfg, FeatureStore(cfg, small.customers, small.merchants), DummyModel())
    scored = replay(small.transactions, small.events, engine.store, engine)
    alerts = scored[scored.decision != "ACCEPTER"]
    assert len(alerts) > 0
    assert (alerts.reasons.str.len() > 0).all()
