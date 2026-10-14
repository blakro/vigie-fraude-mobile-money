from datetime import datetime, timedelta

from vigie.features.store import NO_DEPOSIT_MIN, NO_EVENT_H, FeatureStore

T0 = datetime(2026, 8, 1, 10, 0)


def tx(minutes=0, **kw):
    base = {"timestamp": T0 + timedelta(minutes=minutes), "txn_type": "TRANSFERT",
            "account_id": "C1", "counterparty_id": "C2", "counterparty_type": "USER",
            "amount": 5000, "fee": 50, "device_id": "d1", "region": "Dakar",
            "balance_before": 50000, "status": "REUSSIE"}
    base.update(kw)
    return base


def test_compute_does_not_modify_state():
    st = FeatureStore()
    st.process(tx())
    a = st.compute(tx(minutes=1))
    assert a == st.compute(tx(minutes=1))
    assert st.accounts["C1"].n_tx == 1


def test_velocity_window_expires():
    st = FeatureStore()
    for i in range(5):
        st.process(tx(minutes=i))
    assert st.compute(tx(minutes=6))["n_tx_1h"] == 6
    assert st.compute(tx(minutes=200))["n_tx_1h"] == 1


def test_new_device_and_counterparty():
    st = FeatureStore()
    st.process(tx())
    f = st.compute(tx(minutes=5, device_id="d_new", counterparty_id="C9"))
    assert f["is_new_device"] == 1 and f["is_new_counterparty"] == 1
    f = st.compute(tx(minutes=5))
    assert f["is_new_device"] == 0 and f["is_new_counterparty"] == 0


def test_first_operation_is_not_flagged_as_new():
    f = FeatureStore().compute(tx())
    assert f["is_new_device"] == 0 and f["is_new_counterparty"] == 0


def test_share_of_balance():
    st = FeatureStore()
    f = st.compute(tx(amount=99000, fee=990, balance_before=100000))
    assert abs(f["share_of_balance"] - 0.9999) < 1e-3
    assert st.compute(tx(txn_type="DEPOT", counterparty_id="A1", balance_before=0))["share_of_balance"] == 0
    assert st.compute(tx(balance_before=None))["share_of_balance"] == -1


def test_failed_attempts_are_counted_but_not_learned():
    st = FeatureStore()
    st.process(tx(amount=500000, status="ECHOUEE"))
    st.process(tx(minutes=1, amount=400000, status="ECHOUEE", counterparty_id="C7"))
    f = st.compute(tx(minutes=2))
    assert f["n_failed_1h"] == 2
    assert "C7" not in st.accounts["C1"].counterparties  # une tentative échouée n'est pas une relation
    assert st.accounts["C1"].amount_stats["TRANSFERT"].n == 0


def test_pin_reset_and_device_event_hours():
    st = FeatureStore()
    assert st.compute(tx())["hours_since_pin_reset"] == NO_EVENT_H
    st.apply_event({"account_id": "C1", "timestamp": T0, "event_type": "REINIT_PIN", "device_id": "x"})
    st.apply_event({"account_id": "C1", "timestamp": T0, "event_type": "CONNEXION_NOUVEL_APPAREIL", "device_id": "x"})
    f = st.compute(tx(minutes=30, device_id="x"))
    assert abs(f["hours_since_pin_reset"] - 0.5) < 1e-6
    assert abs(f["hours_since_device_event"] - 0.5) < 1e-6


def test_minutes_since_deposit_and_unusual_region():
    st = FeatureStore()
    assert st.compute(tx())["minutes_since_deposit"] == NO_DEPOSIT_MIN
    for i in range(5):
        st.process(tx(minutes=i))
    st.process(tx(minutes=10, txn_type="DEPOT", counterparty_id="A1", counterparty_type="AGENT"))
    f = st.compute(tx(minutes=25, region="Kolda"))
    assert abs(f["minutes_since_deposit"] - 15) < 1e-6
    assert f["is_unusual_region"] == 1
    assert st.compute(tx(minutes=25))["is_unusual_region"] == 0


def test_mule_inbound_and_cashout_ratio():
    st = FeatureStore()
    st.created["MULE"] = (T0 - timedelta(days=3)).timestamp()
    for i, sender in enumerate(["S1", "S2", "S3", "S4"]):
        st.process(tx(minutes=i, account_id=sender, counterparty_id="MULE", amount=30000))
    f = st.compute(tx(minutes=10, account_id="S5", counterparty_id="MULE", amount=30000))
    assert f["receiver_n_senders_24h"] == 5 and f["receiver_age_days"] < 4
    f = st.compute(tx(minutes=60, txn_type="RETRAIT", account_id="MULE", counterparty_id="A1",
                      counterparty_type="AGENT", amount=108000, fee=0))
    assert f["inbound_n_senders_24h"] == 4
    assert abs(f["cashout_ratio"] - 0.9) < 1e-6


def test_merchant_payments_count_as_inbound():
    st = FeatureStore()
    for i in range(3):
        st.process(tx(minutes=i, account_id=f"P{i}", txn_type="PAIEMENT_MARCHAND", counterparty_id="M1",
                      counterparty_type="MERCHANT", amount=40000, fee=0))
    f = st.compute(tx(minutes=5, account_id="M1", account_type="MARCHAND", txn_type="RETRAIT",
                      counterparty_id="A1", counterparty_type="AGENT", amount=100000, fee=0))
    assert f["inbound_n_senders_24h"] == 3 and f["is_merchant_account"] == 1
    assert abs(f["cashout_ratio"] - 100000 / 120000) < 1e-6


def test_agent_roundtrips():
    st = FeatureStore()
    for i, client in enumerate(["C1", "C2", "C3"]):
        st.process(tx(minutes=10 * i, account_id=client, txn_type="DEPOT", counterparty_id="A1",
                      counterparty_type="AGENT", amount=20000, fee=0))
        st.process(tx(minutes=10 * i + 5, account_id=client, txn_type="RETRAIT", counterparty_id="A1",
                      counterparty_type="AGENT", amount=20000, fee=0))
    f = st.compute(tx(minutes=40, account_id="C4", txn_type="DEPOT", counterparty_id="A1",
                      counterparty_type="AGENT", amount=25000, fee=0))
    assert f["agent_roundtrips_24h"] == 3
    f = st.compute(tx(minutes=8, account_id="C1", txn_type="RETRAIT", counterparty_id="A1",
                      counterparty_type="AGENT", amount=20000, fee=0))
    assert f["agent_pair_roundtrip"] == 1
