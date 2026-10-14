from vigie.config import FraudConfig
from vigie.engine.aggregator import ACCEPTER, BLOQUER, REVUE, decide, rule_score
from vigie.engine.checks import Signal, run_checks
from vigie.features.store import NO_DEPOSIT_MIN, NO_EVENT_H, FeatureStore

CFG = FraudConfig()


def base(**kw):
    f = {name: 0 for name in FeatureStore.FEATURE_NAMES}
    f.update({"amount": 5000, "hour": 12, "account_age_days": 400, "n_tx_1h": 1, "n_tx_24h": 1,
              "hours_since_device_event": NO_EVENT_H, "hours_since_pin_reset": NO_EVENT_H,
              "device_n_accounts": 1, "receiver_age_days": 400, "minutes_since_deposit": NO_DEPOSIT_MIN,
              "balance_before": 50000, "share_of_balance": 0.1, "kyc_level": 2})
    f.update(kw)
    return f


def names(f, t="TRANSFERT"):
    return {s.name for s in run_checks(f, t, CFG)}


def test_normal_operation_has_no_signal():
    assert names(base()) == set()


def test_account_takeover_is_blocked():
    f = base(amount=150000, share_of_balance=0.99, is_new_device=1, hours_since_device_event=0.1,
             hours_since_pin_reset=0.2, is_new_counterparty=1, n_new_cp_24h=1)
    assert {"nouvel_appareil", "reinit_pin", "nouveau_beneficiaire", "vidage_compte"} <= names(f)
    assert decide(run_checks(f, "TRANSFERT", CFG), None, CFG).decision == BLOQUER


def test_scam_pattern_deposit_then_send_to_young_account():
    f = base(amount=60000, is_new_counterparty=1, minutes_since_deposit=12, receiver_age_days=6)
    assert {"depot_puis_envoi", "beneficiaire_recent", "nouveau_beneficiaire"} <= names(f)
    assert decide(run_checks(f, "TRANSFERT", CFG), None, CFG).decision in (REVUE, BLOQUER)


def test_mule_collection_and_cashout():
    assert "mule_collecte" in names(base(amount=40000, receiver_n_senders_24h=6, receiver_age_days=5))
    f = base(amount=200000, account_age_days=7, inbound_amount_24h=220000, cashout_ratio=0.9,
             inbound_n_senders_24h=6)
    assert "mule_retrait" in names(f, "RETRAIT")
    f.update(account_age_days=900, inbound_n_senders_24h=1)
    assert "mule_retrait" not in names(f, "RETRAIT")


def test_established_merchant_payout_is_not_a_mule():
    f = base(amount=300000, account_age_days=800, is_merchant_account=1, inbound_amount_24h=100000,
             cashout_ratio=3.0, inbound_n_senders_24h=12)
    assert "mule_retrait" not in names(f, "VIREMENT_BANCAIRE")


def test_fake_merchant_burst():
    f = base(amount=60000, receiver_age_days=5, receiver_n_senders_24h=12, merchant_category_z=4.0)
    assert {"marchand_afflux", "marchand_atypique"} <= names(f, "PAIEMENT_MARCHAND")


def test_agent_cycles_by_pair_or_by_agent():
    assert "cycles_agent" in names(base(agent_pair_n_24h=7, agent_pair_roundtrip=1), "DEPOT")
    assert "cycles_agent" in names(base(agent_roundtrips_24h=4), "RETRAIT")
    assert "cycles_agent" not in names(base(agent_pair_n_24h=2, agent_roundtrips_24h=1), "DEPOT")


def test_airtime_burst_and_failures():
    assert "rafale_credit" in names(base(n_credit_30min=12), "CREDIT")
    assert "rafale_credit" not in names(base(n_credit_30min=3), "CREDIT")
    assert "echecs_repetes" in names(base(n_failed_1h=3))


def test_night_only_for_meaningful_amounts():
    assert "nuit" not in names(base(is_night=1, hour=2, amount=500), "CREDIT")
    assert "nuit" in names(base(is_night=1, hour=2, amount=50000))


def test_combo_bonus_and_cap():
    s = [Signal("nouvel_appareil", 1.0, ""), Signal("reinit_pin", 1.0, "")]
    assert rule_score(s, CFG) == 30 + 25 + 20
    assert rule_score([Signal(n, 1.0, "") for n in CFG.weights], CFG) == 100


def test_decision_policy():
    weak = [Signal("nouveau_beneficiaire", 0.7, "")]
    assert decide(weak, 0.01, CFG).decision == ACCEPTER
    # le modèle seul suffit pour la revue, pas pour bloquer
    assert decide([], 0.95, CFG).decision == REVUE
    # les règles seules suffisent aussi pour la revue, même si le modèle hésite
    assert decide([Signal("mule_retrait", 1.0, "")], 0.01, CFG).decision == REVUE
    # règles et modèle d'accord -> blocage
    strong = [Signal("mule_retrait", 1.0, ""), Signal("velocite", 1.0, "")]
    assert decide(strong, 0.9, CFG).decision == BLOQUER
    # règle très forte, non diluée par le modèle
    ato = [Signal("nouvel_appareil", 1.0, ""), Signal("reinit_pin", 1.0, ""), Signal("nouveau_beneficiaire", 1.0, "")]
    assert decide(ato, 0.05, CFG).decision == BLOQUER


def test_reasons_are_human_readable():
    d = decide([Signal("velocite", 0.6, "12 opérations en 60 min")], 0.8, CFG)
    assert d.reasons[0] == "12 opérations en 60 min"
    assert "modèle" in d.reasons[-1]
