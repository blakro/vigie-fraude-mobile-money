"""Évaluation sur la période de test : règles seules, ML seul, moteur combiné.

Trois niveaux de lecture, comme dans une équipe fraude :
1. les OPÉRATIONS (précision, rappel, fausses alertes par jour) ;
2. les CAS de fraude (part des cas repérés, délai avant la première alerte) ;
3. le PRÉJUDICE en FCFA (part du préjudice porté par des opérations alertées ou bloquées).

Le moteur est évalué en "mode observation" : il note les opérations sans les bloquer.
En production, un blocage arrêterait la suite de l'attaque.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from vigie.config import FraudConfig


def _confusion(y, flag):
    tp = int(((flag == 1) & (y == 1)).sum())
    fp = int(((flag == 1) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "alert_rate": round(float(flag.mean()), 5)}


def evaluate(scored: pd.DataFrame, test_mask: np.ndarray, cases: pd.DataFrame,
             cfg: FraudConfig | None = None) -> dict:
    cfg = cfg or FraudConfig()
    s = scored[test_mask].copy()
    y = s.is_fraud.to_numpy()
    n_days = max(1, s.timestamp.dt.normalize().nunique())
    flags = {
        "regles": (s.rule_score >= cfg.review_threshold).astype(int).to_numpy(),
        "ml": (s.ml_proba >= cfg.ml_review_proba).astype(int).to_numpy(),
        "combine": (s.decision != "ACCEPTER").astype(int).to_numpy(),
    }
    out = {}
    for name, flag in flags.items():
        out[name] = _confusion(y, flag)
        out[name]["fausses_alertes_par_jour"] = round(out[name]["fp"] / n_days, 1)
        out[name]["alertes_par_jour"] = round(float(flag.sum()) / n_days, 1)
    out["combine"]["pr_auc_score_final"] = round(float(average_precision_score(y, s.final_score)), 4)
    out["regles"]["pr_auc_score_regles"] = round(float(average_precision_score(y, s.rule_score)), 4)

    # ---- par scénario (opérations)
    per_scenario = []
    for scen, g in s[s.is_fraud == 1].groupby("fraud_scenario"):
        row = {"scenario": scen, "n": int(len(g))}
        for name, flag in flags.items():
            row[f"rappel_{name}"] = round(float(pd.Series(flag, index=s.index).loc[g.index].mean()), 4)
        per_scenario.append(row)
    out["par_scenario"] = per_scenario

    # ---- par cas de fraude et préjudice
    s["alerte"] = flags["combine"]
    s["bloque"] = (s.decision == "BLOQUER").astype(int)
    fraud = s[s.is_fraud == 1].sort_values("timestamp")
    scenario_of = dict(zip(cases.case_id, cases.scenario))
    case_rows = []
    for cid, g in fraud.groupby("fraud_case_id"):
        alerted = g[g.alerte == 1]
        first = alerted.timestamp.min() if len(alerted) else pd.NaT
        n_before = int((g.timestamp < first).sum()) if len(alerted) else int(len(g))
        case_rows.append({
            "case_id": cid, "scenario": scenario_of[cid],
            "detecte": int(len(alerted) > 0), "operations_avant_alerte": n_before,
            "minutes_avant_alerte": round((first - g.timestamp.min()).total_seconds() / 60, 1)
            if len(alerted) else None,
            "prejudice": int(g.loss_fcfa.sum()), "prejudice_alerte": int(g.loss_fcfa[g.alerte == 1].sum()),
            "prejudice_bloque": int(g.loss_fcfa[g.bloque == 1].sum()),
        })
    cases = pd.DataFrame(case_rows)
    out["cas"] = {
        "n": int(len(cases)), "detectes": int(cases.detecte.sum()),
        "taux_detection": round(float(cases.detecte.mean()), 4) if len(cases) else 0.0,
        "operations_avant_alerte_mediane": float(cases[cases.detecte == 1].operations_avant_alerte.median())
        if cases.detecte.any() else None,
        "prejudice_total": int(cases.prejudice.sum()),
        "prejudice_alerte": int(cases.prejudice_alerte.sum()),
        "prejudice_bloque": int(cases.prejudice_bloque.sum()),
    }
    by_scen = []
    for scen, g in cases.groupby("scenario"):
        by_scen.append({"scenario": scen, "cas": int(len(g)), "detectes": int(g.detecte.sum()),
                        "prejudice": int(g.prejudice.sum()), "prejudice_alerte": int(g.prejudice_alerte.sum()),
                        "prejudice_bloque": int(g.prejudice_bloque.sum())})
    out["cas_par_scenario"] = by_scen

    normal_alerts = s[(s.is_fraud == 0) & (s.alerte == 1)]
    out["fausses_alertes_par_type"] = normal_alerts.txn_type.value_counts().to_dict()
    out["decisions"] = s.decision.value_counts().to_dict()
    return out
