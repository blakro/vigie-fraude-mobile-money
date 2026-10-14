"""Pipeline complet en une commande :

    python run_pipeline.py

1. simule 60 jours d'activité Doxal Money (opérateur fictif)   -> data/*.parquet
2. calcule les variables (feature store, ordre chronologique)    -> data/features.parquet
3. entraîne le modèle sur les fraudes signalées avant la coupure -> models/*.joblib
4. score toutes les opérations (règles + ML)                     -> data/scored_transactions.parquet
5. évalue sur les 20 derniers jours                              -> reports/metrics.json, reports/model_card.md
6. exporte un échantillon lisible                                -> data/echantillon_transactions.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from vigie.config import FraudConfig
from vigie.data.simulator import simulate
from vigie.engine.engine import FraudEngine, replay
from vigie.features.store import FeatureStore
from vigie.ml.evaluate import evaluate
from vigie.ml.train import temporal_split, train_models
from vigie.reporting import write_model_card

ROOT = Path(__file__).resolve().parent
TABLES = ("transactions", "customers", "merchants", "agents", "events", "cases")


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def main(seed: int = 42, n_customers: int = 4000, root: Path = ROOT):
    t0 = time.time()
    data_dir, model_dir, rep_dir = root / "data", root / "models", root / "reports"
    for d in (data_dir, model_dir, rep_dir):
        d.mkdir(exist_ok=True)
    cfg = FraudConfig()

    print("1/6 Simulation de l'activité…")
    d = simulate(seed=seed, n_customers=n_customers)
    for name in TABLES:
        getattr(d, name).to_parquet(data_dir / f"{name}.parquet", index=False)
    tx = d.transactions
    print(f"    {fmt(len(tx))} opérations, {fmt(len(d.customers))} clients, "
          f"taux de fraude {tx.is_fraud.mean():.2%}, {len(d.cases)} cas de fraude")

    print("2/6 Calcul des variables (rejeu chronologique)…")
    feats = replay(tx, d.events, FeatureStore(cfg, d.customers, d.merchants), progress=True)
    feats.to_parquet(data_dir / "features.parquet", index=False)

    print("3/6 Entraînement du modèle (fraudes signalées avant la coupure)…")
    ml = train_models(tx, feats, d.cases, model_dir, seed=seed)
    print(f"    étiquettes connues : {ml['label_coverage_train']:.0%} des fraudes d'entraînement | "
          f"seuil du modèle : {ml['ml_threshold']:.0%}")
    print(f"    PR-AUC test : boosting {ml['hgb_pr_auc']:.3f}, isolation forest {ml['iforest_pr_auc']:.3f}")

    print("4/6 Scoring complet (règles + ML)…")
    engine = FraudEngine.from_artifacts(model_dir / "fraud_model.joblib", cfg,
                                        FeatureStore(cfg, d.customers, d.merchants))
    scores = replay(tx, d.events, engine.store, engine, progress=True)
    scored = tx.join(scores[["rule_score", "ml_proba", "final_score", "decision", "signals", "reasons"]])
    scored.to_parquet(data_dir / "scored_transactions.parquet", index=False)

    print("5/6 Évaluation sur la période de test…")
    _, test_mask, _ = temporal_split(tx)
    metrics = {"ml": ml, "evaluation_test": evaluate(scored, test_mask, d.cases, engine.cfg)}
    (rep_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
                                          encoding="utf-8")
    write_model_card(metrics, rep_dir / "model_card.md")
    ev = metrics["evaluation_test"]
    for k in ("regles", "ml", "combine"):
        print(f"    {k:8s} précision {ev[k]['precision']:.1%}  rappel {ev[k]['recall']:.1%}  "
              f"fausses alertes/jour {ev[k]['fausses_alertes_par_jour']}")
    c = ev["cas"]
    print(f"    cas repérés {c['detectes']}/{c['n']}, préjudice alerté "
          f"{c['prejudice_alerte'] / max(1, c['prejudice_total']):.0%}")

    print("6/6 Export d'un échantillon lisible…")
    sample = pd.concat([scored[scored.is_fraud == 1].sample(min(300, int(scored.is_fraud.sum())), random_state=seed),
                        scored[scored.is_fraud == 0].sample(1700, random_state=seed)])
    sample.sort_values("timestamp").to_csv(data_dir / "echantillon_transactions.csv", index=False)
    print(f"Terminé en {time.time() - t0:.0f} s.")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Vigie : détection de fraude mobile money sur données synthétiques")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clients", type=int, default=4000, help="nombre de clients simulés")
    a = ap.parse_args()
    main(a.seed, a.clients)
