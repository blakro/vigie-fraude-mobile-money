"""Tests de cohérence des données simulées : le journal doit pouvoir passer pour un vrai."""
import re

import numpy as np
import pandas as pd

from vigie.config import AGENT_COMMISSION_RATE, FEE_RATE_MERCHANT, FEE_RATE_TRANSFER, LIMITS
from vigie.data.simulator import simulate

SCENARIOS = {"PRISE_CONTROLE_COMPTE", "ARNAQUE_INGENIERIE_SOCIALE", "MULE_RETRAIT", "AGENT_CYCLES",
             "FAUX_MARCHAND", "RAFALE_CREDIT"}


def test_schema_order_and_ids(small):
    tx = small.transactions
    expected = {"txn_id", "timestamp", "txn_type", "account_id", "account_type", "counterparty_id",
                "counterparty_type", "amount", "fee", "merchant_fee", "agent_commission", "balance_before",
                "balance_after", "status", "failure_reason", "device_id", "region", "is_fraud",
                "fraud_scenario", "fraud_case_id", "loss_fcfa"}
    assert expected <= set(tx.columns)
    assert tx.timestamp.is_monotonic_increasing
    assert tx.txn_id.is_unique
    assert tx.txn_id.str.match(r"^DX\d{6}\.\d{6}$").all()
    assert (tx.amount > 0).all()
    assert SCENARIOS <= set(tx.fraud_scenario)
    assert 0.001 < tx.is_fraud.mean() < 0.05


def test_ledger_is_consistent(small):
    """Rejouer le journal depuis les soldes d'ouverture redonne exactement chaque solde."""
    bal = dict(zip(small.customers.account_id, small.customers.opening_balance))
    bal.update(dict(zip(small.merchants.merchant_id, small.merchants.opening_balance)))
    for r in small.transactions.itertuples():
        assert bal[r.account_id] == r.balance_before, r.txn_id
        if r.status == "REUSSIE":
            if r.txn_type == "DEPOT":
                bal[r.account_id] += r.amount
            else:
                bal[r.account_id] -= r.amount + r.fee
            if r.txn_type == "TRANSFERT":
                bal[r.counterparty_id] += r.amount
            elif r.txn_type == "PAIEMENT_MARCHAND":
                bal[r.counterparty_id] += r.amount - r.merchant_fee
        assert bal[r.account_id] == r.balance_after, r.txn_id
        assert r.balance_after >= 0


def test_failed_operations_do_not_move_money(small):
    failed = small.transactions[small.transactions.status == "ECHOUEE"]
    assert len(failed) > 0
    assert (failed.balance_before == failed.balance_after).all()
    assert failed.failure_reason.notna().all()
    assert (failed.fee == 0).all()


def test_limits_respected(small):
    tx = small.transactions[small.transactions.status == "REUSSIE"]
    kyc = dict(zip(small.customers.account_id, small.customers.kyc_level))
    level = tx.account_id.map(kyc).fillna("MARCHAND")
    caps = level.map(lambda k: LIMITS[k if k == "MARCHAND" else int(k)]["balance_max"])
    tmax = level.map(lambda k: LIMITS[k if k == "MARCHAND" else int(k)]["txn_max"])
    assert (tx.balance_after <= caps).all()
    assert (tx.amount <= tmax).all()


def test_fees_and_commissions(small):
    tx = small.transactions[small.transactions.status == "REUSSIE"]
    tr = tx[tx.txn_type == "TRANSFERT"]
    assert np.allclose(tr.fee, (tr.amount * FEE_RATE_TRANSFER).round())
    assert (tx[tx.txn_type != "TRANSFERT"].fee == 0).all()
    pm = tx[tx.txn_type == "PAIEMENT_MARCHAND"]
    assert np.allclose(pm.merchant_fee, (pm.amount * FEE_RATE_MERCHANT).round())
    ag = tx[tx.txn_type.isin(["DEPOT", "RETRAIT"])]
    assert np.allclose(ag.agent_commission, (ag.amount * AGENT_COMMISSION_RATE).round())


def test_agent_operations_during_opening_hours(small):
    ag = small.transactions[small.transactions.counterparty_type == "AGENT"]
    assert ag.timestamp.dt.hour.between(7, 21).all()


def test_identifiers_do_not_leak_labels(small):
    """Les identifiants sont opaques : rien ne distingue un fraudeur par son numéro de compte
    ou d'appareil."""
    tx = small.transactions
    assert tx.device_id.dropna().map(lambda d: bool(re.fullmatch(r"[0-9a-f]{12}", d))).all()
    assert tx.account_id[tx.account_type == "CLIENT"].str.fullmatch(r"C[A-Z2-9]{7}").all()


def test_cases_and_reporting(small):
    cases = small.cases
    assert set(cases.scenario) == SCENARIOS - {"MULE_RETRAIT"}
    rep = cases.dropna(subset=["reported_at"])
    assert (rep.reported_at > rep.end).all()
    assert cases.reported_at.isna().any() or len(cases) < 10  # certaines fraudes ne sont jamais signalées
    fraud = small.transactions[small.transactions.is_fraud == 1]
    assert fraud.fraud_case_id.notna().all()
    assert set(fraud.fraud_case_id) <= set(cases.case_id)


def test_deterministic():
    a = simulate(seed=3, n_customers=150, n_agents=20, n_merchants=30, days=20).transactions
    b = simulate(seed=3, n_customers=150, n_agents=20, n_merchants=30, days=20).transactions
    pd.testing.assert_frame_equal(a, b)


def test_reproducible_across_processes():
    """Même graine => mêmes données, quel que soit l'ordre interne des ensembles Python."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = ("import hashlib; from vigie.data.simulator import simulate; "
            "t = simulate(seed=5, n_customers=150, n_agents=20, n_merchants=30, days=20).transactions; "
            "print(hashlib.md5(t.to_csv(index=False).encode()).hexdigest())")
    root = Path(__file__).resolve().parent.parent
    outs = {subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}).stdout for seed in ("1", "2")}
    assert len(outs) == 1
