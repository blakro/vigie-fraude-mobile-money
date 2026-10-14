"""API de scoring temps réel (FastAPI).

Lancer :  uvicorn api.app:app --reload
Docs   :  http://127.0.0.1:8000/docs

Le système central de l'opérateur appelle /score AVANT d'exécuter une opération
(il transmet le solde du compte), puis /events pour les événements de sécurité.

Variables d'environnement :
    VIGIE_MODEL        chemin du modèle (défaut : models/fraud_model.joblib)
    VIGIE_WARM_START   "1" pour recharger l'historique simulé de data/ au démarrage
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from vigie.config import FEE_RATE_TRANSFER, OPERATOR_NAME, FraudConfig
from vigie.engine.engine import FraudEngine
from vigie.features.store import FeatureStore

ROOT = Path(__file__).resolve().parent.parent

TxnType = Literal["DEPOT", "RETRAIT", "TRANSFERT", "PAIEMENT_MARCHAND", "FACTURE", "CREDIT", "EPARGNE",
                  "VIREMENT_BANCAIRE"]


class Transaction(BaseModel):
    txn_id: str | None = None
    timestamp: datetime
    txn_type: TxnType
    account_id: str
    account_type: Literal["CLIENT", "MARCHAND"] = "CLIENT"
    counterparty_id: str
    counterparty_type: Literal["USER", "AGENT", "MERCHANT", "BILLER", "TELCO", "EPARGNE", "BANQUE"]
    amount: float = Field(gt=0, description="Montant en FCFA")
    fee: float | None = Field(None, description="Frais ; calculés automatiquement (1 % sur un transfert) si absents")
    balance_before: float | None = Field(None, description="Solde du compte avant l'opération (FCFA)")
    device_id: str | None = None
    region: str | None = None
    status: Literal["REUSSIE", "ECHOUEE"] = Field(
        "REUSSIE", description="Statut final, s'il est déjà connu ; sert seulement à mettre l'historique à jour")

    model_config = {"json_schema_extra": {"example": {
        "txn_id": "DX260830.000142", "timestamp": "2026-08-30T02:14:00", "txn_type": "TRANSFERT",
        "account_id": "C4N7QX2A", "counterparty_id": "CZ8RKM3T", "counterparty_type": "USER",
        "amount": 150000, "balance_before": 152300, "device_id": "9f3c2a71be04", "region": "Dakar"}}}


class SecurityEvent(BaseModel):
    account_id: str
    timestamp: datetime
    event_type: Literal["REINIT_PIN", "CONNEXION_NOUVEL_APPAREIL"]
    device_id: str | None = None


def build_engine() -> FraudEngine:
    cfg = FraudConfig()
    data = ROOT / "data"

    def load(name):
        p = data / f"{name}.parquet"
        return pd.read_parquet(p) if p.exists() else None

    store = FeatureStore(cfg, load("customers"), load("merchants"))
    model_path = Path(os.getenv("VIGIE_MODEL", ROOT / "models" / "fraud_model.joblib"))
    engine = FraudEngine.from_artifacts(model_path, cfg, store) if model_path.exists() else FraudEngine(cfg, store)
    if os.getenv("VIGIE_WARM_START") == "1" and (data / "transactions.parquet").exists():
        from vigie.engine.engine import replay
        replay(load("transactions"), load("events"), store)
    return engine


def create_app(engine: FraudEngine | None = None) -> FastAPI:
    state = {"engine": engine}

    @asynccontextmanager
    async def lifespan(_app):
        if state["engine"] is None:
            state["engine"] = build_engine()
        yield

    app = FastAPI(title=f"Vigie — scoring de fraude ({OPERATOR_NAME}, opérateur fictif)",
                  description="Démonstration sur données synthétiques.", version="2.0.0", lifespan=lifespan)

    def eng() -> FraudEngine:
        if state["engine"] is None:
            state["engine"] = build_engine()
        return state["engine"]

    @app.get("/health")
    def health():
        e = eng()
        return {"status": "ok", "modele_ml": e.model is not None, "seuil_ml": e.cfg.ml_review_proba,
                "comptes_en_memoire": len(e.store.accounts)}

    @app.post("/events")
    def post_event(ev: SecurityEvent):
        eng().apply_event(ev.model_dump())
        return {"status": "enregistré"}

    @app.post("/score")
    def score(tx: Transaction):
        payload = tx.model_dump()
        if payload["fee"] is None:
            payload["fee"] = round(tx.amount * FEE_RATE_TRANSFER) if tx.txn_type == "TRANSFERT" else 0
        features, decision = eng().score(payload)
        out = {"txn_id": tx.txn_id, **decision.to_dict()}
        out["variables_cles"] = {k: features[k] for k in (
            "n_tx_1h", "share_of_balance", "amount_z", "is_new_device", "hours_since_pin_reset",
            "is_new_counterparty", "receiver_n_senders_24h", "receiver_age_days", "account_age_days")}
        return out

    @app.get("/metrics")
    def metrics():
        e = eng()
        total = e.stats.get("total", 0)
        return {"operations_scorees": total,
                "decisions": {k: v for k, v in e.stats.items() if k != "total"},
                "latence_moyenne_ms": round(e.latency_ms_total / total, 3) if total else None}

    return app


app = create_app()
