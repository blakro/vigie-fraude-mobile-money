"""Contrôles de fraude explicables.

Chaque contrôle lit les variables calculées par le feature store et renvoie `None`
(rien à signaler) ou un `Signal` : une intensité entre 0 et 1 et une raison lisible par
un analyste.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from vigie.config import CASHOUT_TYPES, FraudConfig


@dataclass(frozen=True)
class Signal:
    name: str
    strength: float
    reason: str


def _d(x: float) -> str:
    """Nombre décimal au format français (virgule)."""
    return f"{x:.1f}".replace(".", ",")


def _fcfa(x: float) -> str:
    return f"{int(x):,}".replace(",", " ") + " FCFA"


Check = Callable[[dict, str, FraudConfig], "Signal | None"]


def check_velocity(f, t, cfg):
    if f["n_tx_1h"] > cfg.velocity_max_tx:
        s = min(1.0, 0.5 + (f["n_tx_1h"] - cfg.velocity_max_tx) / cfg.velocity_max_tx)
        return Signal("velocite", s, f"{int(f['n_tx_1h'])} opérations en {cfg.velocity_window_min} min")
    return None


def check_new_device(f, t, cfg):
    recent_event = f["hours_since_device_event"] <= cfg.device_risk_window_h
    if not (f["is_new_device"] or recent_event):
        return None
    s = 1.0 if (f["is_new_device"] and f["hours_since_device_event"] <= 1) else 0.6
    txt = "appareil jamais vu sur ce compte" if f["is_new_device"] else \
        f"connexion sur un nouvel appareil il y a {_d(f['hours_since_device_event'])} h"
    if f["device_n_accounts"] >= 3:
        s = 1.0
        txt += f" ; appareil utilisé par {int(f['device_n_accounts'])} comptes"
    return Signal("nouvel_appareil", s, txt)


def check_pin_reset(f, t, cfg):
    if f["hours_since_pin_reset"] <= cfg.pin_reset_window_h and t in ("TRANSFERT", "RETRAIT", "CREDIT"):
        return Signal("reinit_pin", 1.0, f"PIN réinitialisé il y a {_d(f['hours_since_pin_reset'])} h")
    return None


def check_night(f, t, cfg):
    if f["is_night"] and f["amount"] >= 10000:
        return Signal("nuit", 1.0, f"opération de {_fcfa(f['amount'])} à {int(f['hour'])} h")
    return None


def check_amount_anomaly(f, t, cfg):
    if f["amount_z"] >= cfg.amount_z_threshold and f["amount"] >= 10000:
        s = min(1.0, 0.5 + (f["amount_z"] - cfg.amount_z_threshold) / 6)
        return Signal("montant_anormal", s,
                      f"montant très supérieur aux habitudes du compte (z = {_d(f['amount_z'])})")
    return None


def check_new_beneficiary(f, t, cfg):
    if t == "TRANSFERT" and f["is_new_counterparty"] and f["amount"] >= 10000:
        n = int(f["n_new_cp_24h"])
        return Signal("nouveau_beneficiaire", 1.0 if n >= 3 else 0.7, "premier transfert vers ce bénéficiaire"
                      + (f" ({n} nouveaux bénéficiaires en 24 h)" if n >= 2 else ""))
    return None


def check_drain(f, t, cfg):
    if t == "TRANSFERT" and f["share_of_balance"] >= cfg.drain_share and f["amount"] >= cfg.drain_min_amount:
        share = f["share_of_balance"]
        txt = ("montant supérieur au solde disponible" if share > 1
               else f"l'opération vide {share:.0%} du solde".replace("%", " %"))
        return Signal("vidage_compte", 1.0 if share >= 0.97 else 0.7, txt)
    return None


def check_failed_attempts(f, t, cfg):
    if f["n_failed_1h"] >= cfg.failed_max:
        s = min(1.0, 0.5 + 0.25 * (f["n_failed_1h"] - cfg.failed_max))
        return Signal("echecs_repetes", s, f"{int(f['n_failed_1h'])} tentatives échouées dans l'heure")
    return None


def check_unusual_region(f, t, cfg):
    if f["is_unusual_region"] and f["amount"] >= 20000 and t in ("TRANSFERT", "RETRAIT"):
        return Signal("region_inhabituelle", 0.6, "opération dans une région jamais vue pour ce compte")
    return None


def check_deposit_then_send(f, t, cfg):
    if (t == "TRANSFERT" and f["is_new_counterparty"] and f["amount"] >= 20000
            and f["minutes_since_deposit"] <= cfg.deposit_then_send_min):
        return Signal("depot_puis_envoi", 0.8,
                      f"envoi vers un inconnu {int(f['minutes_since_deposit'])} min après un dépôt")
    return None


def check_young_beneficiary(f, t, cfg):
    if (t == "TRANSFERT" and f["is_new_counterparty"] and f["amount"] >= 20000
            and f["receiver_age_days"] < cfg.young_account_days):
        return Signal("beneficiaire_recent", 0.8,
                      f"premier envoi vers un compte ouvert il y a {f['receiver_age_days']:.0f} j")
    return None


def check_mule_inbound(f, t, cfg):
    if t == "TRANSFERT" and f["receiver_n_senders_24h"] >= cfg.mule_min_distinct_senders:
        young = f["receiver_age_days"] < cfg.young_account_days
        return Signal("mule_collecte", 1.0 if young else 0.5,
                      f"le bénéficiaire a reçu de {int(f['receiver_n_senders_24h'])} expéditeurs en 24 h"
                      + (f" (compte ouvert il y a {f['receiver_age_days']:.0f} j)" if young else ""))
    return None


def check_mule_cashout(f, t, cfg):
    """Sortie rapide de l'argent reçu dans les 24 h (le retrait peut être fractionné :
    on alerte dès 30 % si le compte est récent)."""
    if t not in CASHOUT_TYPES or f["inbound_amount_24h"] < 50000 or f["cashout_ratio"] < 0.3:
        return None
    young = f["account_age_days"] < cfg.young_account_days
    full = f["cashout_ratio"] >= cfg.mule_cashout_ratio
    many = f["inbound_n_senders_24h"] >= 3
    if young:
        s = 1.0 if full else 0.9
    elif full and many and not f["is_merchant_account"]:
        s = 0.6
    else:
        return None
    n = int(f["inbound_n_senders_24h"])
    who = (f"{n} payeurs, marchand" if f["is_merchant_account"] else f"{n} expéditeurs, compte")
    return Signal("mule_retrait", s,
                  f"sortie de {min(f['cashout_ratio'], 1):.0%} des fonds reçus en 24 h ".replace("%", " %")
                  + f"({who} ouvert il y a {f['account_age_days']:.0f} j)")


def check_agent_cycles(f, t, cfg):
    """Gonflement de commissions : dépôts/retraits répétés avec un même client, ou cycles
    (dépôt puis retrait du même montant) qui se multiplient chez un même agent."""
    if t not in ("DEPOT", "RETRAIT"):
        return None
    by_pair = f["agent_pair_n_24h"] > cfg.agent_pair_max_tx
    by_agent = f["agent_roundtrips_24h"] >= cfg.agent_roundtrips_max
    if not (by_pair or by_agent):
        return None
    s = 0.0
    parts = []
    if by_pair:
        s = 0.5 + (f["agent_pair_n_24h"] - cfg.agent_pair_max_tx) / 8
        parts.append(f"{int(f['agent_pair_n_24h'])} dépôts/retraits avec le même agent en 24 h")
    if by_agent:
        s = max(s, 0.5 + 0.1 * (f["agent_roundtrips_24h"] - cfg.agent_roundtrips_max))
        parts.append(f"{int(f['agent_roundtrips_24h'])} cycles dépôt puis retrait du même montant "
                     "chez cet agent en 24 h")
    if f["agent_pair_roundtrip"]:
        s += 0.2
        parts.append("dépôt et retrait du même montant à moins d'une heure d'intervalle")
    return Signal("cycles_agent", min(1.0, s), " ; ".join(parts))


def check_merchant_atypical(f, t, cfg):
    if t != "PAIEMENT_MARCHAND":
        return None
    z = max(f["merchant_category_z"], f["merchant_amount_z"])
    if z >= cfg.merchant_amount_z_threshold:
        return Signal("marchand_atypique", 1.0 if f["is_round_amount"] else 0.6,
                      f"panier atypique pour ce type de marchand (z = {_d(z)})"
                      + (" ; montant rond" if f["is_round_amount"] else ""))
    return None


def check_merchant_burst(f, t, cfg):
    if (t == "PAIEMENT_MARCHAND" and f["receiver_age_days"] < cfg.young_account_days
            and f["receiver_n_senders_24h"] >= cfg.merchant_burst_min_payers):
        return Signal("marchand_afflux", 1.0,
                      f"marchand ouvert il y a {f['receiver_age_days']:.0f} j : "
                      f"{int(f['receiver_n_senders_24h'])} payeurs différents en 24 h")
    return None


def check_airtime_burst(f, t, cfg):
    if t == "CREDIT" and f["n_credit_30min"] > cfg.airtime_max_tx:
        return Signal("rafale_credit", 1.0,
                      f"{int(f['n_credit_30min'])} achats de crédit en {cfg.airtime_window_min} min")
    return None


ALL_CHECKS: list[Check] = [
    check_velocity, check_new_device, check_pin_reset, check_night, check_amount_anomaly,
    check_new_beneficiary, check_drain, check_failed_attempts, check_unusual_region,
    check_deposit_then_send, check_young_beneficiary, check_mule_inbound, check_mule_cashout, check_agent_cycles,
    check_merchant_atypical, check_merchant_burst, check_airtime_burst,
]


def run_checks(features: dict, txn_type: str, cfg: FraudConfig) -> list[Signal]:
    out = []
    for check in ALL_CHECKS:
        sig = check(features, txn_type, cfg)
        if sig is not None:
            out.append(sig)
    return out
