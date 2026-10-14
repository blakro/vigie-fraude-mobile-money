"""Tableau de bord Streamlit de Vigie.

Lancer :  streamlit run dashboard/app.py
Si les données n'existent pas encore, le pipeline est lancé automatiquement (environ 2 min).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vigie.config import OPERATOR_NAME, TXN_TYPES, FraudConfig  # noqa: E402
from vigie.engine.aggregator import decide  # noqa: E402
from vigie.engine.checks import run_checks  # noqa: E402
from vigie.features.store import NO_DEPOSIT_MIN, FeatureStore  # noqa: E402
from vigie.reporting import SCENARIO_LABELS  # noqa: E402

st.set_page_config(page_title="Vigie — fraude mobile money", layout="wide")
COLORS = {"ACCEPTER": "#9aa5b1", "REVUE": "#d08b00", "BLOQUER": "#c8433a"}
SERIES = {"regles": "#2a78d6", "ml": "#eb6834", "combine": "#1baf7a"}


def fmt(x) -> str:
    return f"{int(x):,}".replace(",", " ")


def pct(x) -> str:
    return f"{x:.1%}".replace(".", ",").replace("%", " %")


@st.cache_resource(show_spinner=False)
def ensure_data() -> bool:
    if not (ROOT / "data" / "scored_transactions.parquet").exists():
        from run_pipeline import main
        main(root=ROOT)
    return True


@st.cache_data(show_spinner=False)
def load():
    scored = pd.read_parquet(ROOT / "data" / "scored_transactions.parquet")
    cases = pd.read_parquet(ROOT / "data" / "cases.parquet")
    customers = pd.read_parquet(ROOT / "data" / "customers.parquet")
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    return scored, cases, customers, metrics


@st.cache_resource(show_spinner=False)
def load_model():
    path = ROOT / "models" / "fraud_model.joblib"
    return joblib.load(path) if path.exists() else None


with st.spinner("Première ouverture : simulation des données et entraînement du modèle (environ 2 min)…"):
    ensure_data()
scored, cases, customers, metrics = load()
scored["scenario"] = scored.fraud_scenario.map(SCENARIO_LABELS)
ev, ml = metrics["evaluation_test"], metrics["ml"]
cut = pd.Timestamp(ml["train_period_end"])

st.title("Vigie — détection de fraude mobile money")
st.caption(f"Opérateur fictif **{OPERATOR_NAME}** (Sénégal). Toutes les données sont **synthétiques**.")

with st.sidebar:
    st.header("Filtres")
    periode = st.radio("Période", ["Test (20 derniers jours)", "Tout"], index=0)
    data = scored[scored.timestamp >= cut] if periode.startswith("Test") else scored
    types = st.multiselect("Types d'opération", TXN_TYPES, default=TXN_TYPES)
    data = data[data.txn_type.isin(types)]

tabs = st.tabs(["Vue d'ensemble", "Performance", "Cas et préjudice", "File d'alertes", "Réseau de mules",
                "Simulateur"])

# ------------------------------------------------------------------ vue d'ensemble
with tabs[0]:
    alerts = data[data.decision != "ACCEPTER"]
    n_days = max(1, data.timestamp.dt.normalize().nunique())
    c = st.columns(5)
    c[0].metric("Opérations", fmt(len(data)))
    c[1].metric("Volume réussi", fmt(data.loc[data.status == "REUSSIE", "amount"].sum()) + " FCFA")
    c[2].metric("Alertes par jour", fmt(len(alerts) / n_days))
    c[3].metric("Opérations échouées", pct((data.status == "ECHOUEE").mean()))
    c[4].metric("Fraude simulée", pct(data.is_fraud.mean()))

    daily = data.assign(jour=data.timestamp.dt.date, alerte=(data.decision != "ACCEPTER")) \
        .groupby("jour").agg(operations=("txn_id", "size"), alertes=("alerte", "sum"),
                             fraudes=("is_fraud", "sum")).reset_index()
    st.plotly_chart(px.line(daily, x="jour", y="operations", title="Opérations par jour")
                    .update_layout(height=300, margin=dict(t=40)), width="stretch")
    long = daily.melt(id_vars="jour", value_vars=["alertes", "fraudes"], var_name="série", value_name="n")
    st.plotly_chart(px.bar(long, x="jour", y="n", color="série", barmode="group",
                           color_discrete_map={"alertes": "#c8433a", "fraudes": "#7b8597"},
                           title="Alertes et fraudes simulées par jour").update_layout(height=300, margin=dict(t=40)),
                    width="stretch")
    col1, col2 = st.columns(2)
    by_type = alerts.groupby(["txn_type", "decision"]).size().reset_index(name="n")
    col1.plotly_chart(px.bar(by_type, x="txn_type", y="n", color="decision", color_discrete_map=COLORS,
                             title="Alertes par type d'opération"), width="stretch")
    sig = alerts.signals.str.split("|").explode()
    sig = sig[sig.astype(bool)].value_counts().reset_index()
    sig.columns = ["signal", "n"]
    col2.plotly_chart(px.bar(sig.iloc[::-1], x="n", y="signal", orientation="h", title="Signaux déclenchés"),
                      width="stretch")

# ------------------------------------------------------------------ performance
with tabs[1]:
    st.subheader("Règles seules, ML seul, moteur combiné (période de test)")
    st.write(f"Le modèle envoie en revue dès que sa probabilité calibrée atteint **{pct(ml['ml_threshold'])}**. "
             f"À la date de coupure, seules **{pct(ml['label_coverage_train'])}** des fraudes d'entraînement "
             "étaient signalées : le modèle apprend avec des étiquettes incomplètes.")
    perf = pd.DataFrame([{
        "Approche": lbl, "Précision": pct(ev[k]["precision"]), "Rappel": pct(ev[k]["recall"]),
        "Alertes / jour": ev[k]["alertes_par_jour"], "Fausses alertes / jour": ev[k]["fausses_alertes_par_jour"],
        "Fraudes repérées": ev[k]["tp"], "Fraudes manquées": ev[k]["fn"], "Fausses alertes": ev[k]["fp"],
    } for k, lbl in (("regles", "Règles seules"), ("ml", "ML seul"), ("combine", "Moteur combiné"))])
    st.dataframe(perf, hide_index=True, width="stretch")
    ps = pd.DataFrame(ev["par_scenario"])
    ps["scenario"] = ps.scenario.map(SCENARIO_LABELS)
    long = ps.melt(id_vars=["scenario", "n"], value_vars=["rappel_regles", "rappel_ml", "rappel_combine"],
                   var_name="approche", value_name="rappel")
    long["approche"] = long.approche.str.replace("rappel_", "")
    st.plotly_chart(px.bar(long, x="rappel", y="scenario", color="approche", barmode="group", orientation="h",
                           color_discrete_map=SERIES, range_x=[0, 1], title="Rappel par scénario"),
                    width="stretch")
    fi = pd.DataFrame(ml["feature_importance"])
    st.plotly_chart(px.bar(fi.iloc[::-1], x="importance", y="feature", orientation="h",
                           title="Importance des variables (permutation, PR-AUC)"), width="stretch")
    st.info("Les scores sont optimistes : les fraudes ont été simulées par l'auteur du détecteur.")

# ------------------------------------------------------------------ cas et préjudice
with tabs[2]:
    cs = ev["cas"]
    c = st.columns(4)
    c[0].metric("Cas de fraude en test", cs["n"])
    c[1].metric("Cas avec au moins une alerte", f"{cs['detectes']}/{cs['n']}")
    c[2].metric("Préjudice alerté", pct(cs["prejudice_alerte"] / max(1, cs["prejudice_total"])))
    c[3].metric("Préjudice bloqué", pct(cs["prejudice_bloque"] / max(1, cs["prejudice_total"])))
    cps = pd.DataFrame(ev["cas_par_scenario"])
    cps["scenario"] = cps.scenario.map(SCENARIO_LABELS)
    cps["non alerté"] = cps.prejudice - cps.prejudice_alerte
    cps["alerté, non bloqué"] = cps.prejudice_alerte - cps.prejudice_bloque
    cps["bloqué"] = cps.prejudice_bloque
    long = cps.melt(id_vars="scenario", value_vars=["bloqué", "alerté, non bloqué", "non alerté"],
                    var_name="statut", value_name="FCFA")
    st.plotly_chart(px.bar(long, x="FCFA", y="scenario", color="statut", orientation="h",
                           color_discrete_map={"bloqué": "#c8433a", "alerté, non bloqué": "#d08b00",
                                               "non alerté": "#c9ced6"},
                           title="Préjudice par scénario (période de test)"), width="stretch")
    st.caption("Préjudice : argent perdu par les victimes (ou commissions indûment versées pour les cycles "
               "d'agent). Les retraits de mules ne sont pas recomptés.")
    test_cases = cases[cases.end >= cut].copy()
    test_cases["scenario"] = test_cases.scenario.map(SCENARIO_LABELS)
    st.dataframe(test_cases[["case_id", "scenario", "start", "end", "reported_at", "n_tx", "loss_fcfa"]]
                 .rename(columns={"case_id": "Cas", "scenario": "Scénario", "start": "Début", "end": "Fin",
                                  "reported_at": "Signalé le", "n_tx": "Opérations", "loss_fcfa": "Préjudice (FCFA)"}),
                 hide_index=True, width="stretch")

# ------------------------------------------------------------------ alertes
with tabs[3]:
    dec_filter = st.multiselect("Décision", ["BLOQUER", "REVUE"], default=["BLOQUER", "REVUE"])
    q = data[data.decision.isin(dec_filter)].sort_values("final_score", ascending=False)
    cols = {"txn_id": "Référence", "timestamp": "Date", "txn_type": "Type", "account_id": "Compte",
            "counterparty_id": "Contrepartie", "amount": "Montant", "balance_before": "Solde avant",
            "status": "Statut", "final_score": "Score final", "rule_score": "Score règles",
            "ml_proba": "Probabilité ML", "decision": "Décision", "reasons": "Raisons",
            "scenario": "Vérité simulée"}
    view = q[list(cols)].rename(columns=cols)
    st.dataframe(view, hide_index=True, width="stretch", height=480)
    st.download_button("Télécharger (CSV)", view.to_csv(index=False, sep=";").encode("utf-8-sig"),
                       "alertes.csv", "text/csv")

# ------------------------------------------------------------------ réseau
with tabs[4]:
    st.write("Flux autour d'un compte mule : expéditeurs → mule → agents de retrait.")
    mules = scored[scored.fraud_scenario == "MULE_RETRAIT"].account_id.value_counts()
    acc = st.selectbox("Compte mule", mules.index.tolist())
    win = scored[((scored.counterparty_id == acc) & (scored.txn_type == "TRANSFERT") & (scored.is_fraud == 1))
                 | ((scored.account_id == acc) & (scored.fraud_scenario == "MULE_RETRAIT"))]
    G = nx.DiGraph()
    for r in win.itertuples():
        G.add_edge(r.account_id, r.counterparty_id)
    pos = nx.spring_layout(G, seed=1, k=0.9)
    fig = go.Figure()
    ex, ey = [], []
    for u, v in G.edges():
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(color="#bbb", width=1), hoverinfo="none", showlegend=False)
    role = {n: "mule" if n == acc else ("agent" if n.startswith("A") else "expéditeur") for n in G}
    for r, color in (("expéditeur", "#2a78d6"), ("mule", "#c8433a"), ("agent", "#d08b00")):
        ns = [n for n in G if role[n] == r]
        fig.add_scatter(x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers+text", text=ns,
                        textposition="top center", name=r, marker=dict(size=22 if r == "mule" else 13, color=color))
    fig.update_layout(height=460, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(win[["timestamp", "txn_type", "account_id", "counterparty_id", "amount", "status", "decision",
                      "reasons"]].rename(columns={"timestamp": "Date", "txn_type": "Type", "account_id": "De",
                                                  "counterparty_id": "Vers", "amount": "Montant", "status": "Statut",
                                                  "decision": "Décision", "reasons": "Raisons"}),
                 hide_index=True, width="stretch")

# ------------------------------------------------------------------ simulateur
with tabs[5]:
    st.write("Modifiez le contexte d'une opération et voyez comment le moteur décide.")
    st.caption("Le simulateur ne renseigne qu'une partie des 45 variables : la probabilité du modèle y est "
               "indicative. Le score règles, lui, est exactement celui du moteur.")
    cfg = FraudConfig()
    bundle = load_model()
    if bundle and "threshold" in bundle:
        cfg.ml_review_proba = float(bundle["threshold"])
    c1, c2, c3 = st.columns(3)
    t_type = c1.selectbox("Type", TXN_TYPES, index=2)
    amount = c1.number_input("Montant (FCFA)", 100, 3_000_000, 25000, step=500)
    balance = c1.number_input("Solde avant l'opération (FCFA)", 0, 10_000_000, 60000, step=500)
    hour = c1.slider("Heure", 0, 23, 14)
    age = c1.number_input("Âge du compte (jours)", 0, 3000, 400)
    new_dev = c2.checkbox("Appareil jamais vu sur ce compte")
    dev_h = c2.number_input("Heures depuis connexion sur nouvel appareil (999 = jamais)", 0.0, 999.0, 999.0)
    pin_h = c2.number_input("Heures depuis réinitialisation du PIN (999 = jamais)", 0.0, 999.0, 999.0)
    new_cp = c2.checkbox("Premier transfert vers ce bénéficiaire")
    dep_min = c2.number_input("Minutes depuis le dernier dépôt", 0, NO_DEPOSIT_MIN, NO_DEPOSIT_MIN)
    z = c3.slider("Écart aux habitudes du compte (z)", -2.0, 8.0, 0.0, 0.5)
    n1h = c3.slider("Opérations dans la dernière heure", 1, 30, 1)
    failed = c3.slider("Tentatives échouées dans l'heure", 0, 10, 0)
    rcv = c3.slider("Expéditeurs distincts du bénéficiaire (24 h)", 0, 20, 1)
    rcv_age = c3.number_input("Âge du compte bénéficiaire (jours)", 0, 3000, 400)

    fee = round(amount * 0.01) if t_type == "TRANSFERT" else 0
    out = t_type not in ("DEPOT",)
    f = {n: 0 for n in FeatureStore.FEATURE_NAMES}
    f.update({"amount": amount, "log_amount": float(np.log1p(amount)), "fee": fee, "hour": hour,
              "is_night": int(cfg.night_start_hour <= hour < cfg.night_end_hour), "account_age_days": age,
              "kyc_level": 2, "balance_before": balance,
              "share_of_balance": min(5.0, (amount + fee) / balance) if (out and balance > 0) else (5.0 if out else 0),
              "n_tx_1h": n1h, "n_tx_24h": max(n1h, 3), "out_amount_24h": amount if out else 0, "n_failed_1h": failed,
              "amount_z": z, "history_n_type": 20, "is_new_device": int(new_dev), "hours_since_device_event": dev_h,
              "hours_since_pin_reset": pin_h, "device_n_accounts": 1, "is_new_counterparty": int(new_cp),
              "n_new_cp_24h": int(new_cp), "minutes_since_deposit": dep_min, "receiver_n_senders_24h": rcv,
              "receiver_age_days": rcv_age, "is_round_amount": int(amount >= 10000 and amount % 5000 == 0),
              f"type_{t_type}": 1})
    p = None
    if bundle:
        p = float(bundle["model"].predict_proba(np.array([[f[n] for n in bundle["feature_names"]]], dtype=float))[0, 1])
    d = decide(run_checks(f, t_type, cfg), p, cfg)
    k = st.columns(4)
    k[0].metric("Décision", d.decision)
    k[1].metric("Score final", f"{d.final_score:.0f}/100")
    k[2].metric("Score règles", f"{d.rule_score:.0f}/100")
    k[3].metric("Probabilité ML", "—" if p is None else pct(p))
    for r in d.reasons or ["Aucun signal de risque"]:
        st.write("•", r)
