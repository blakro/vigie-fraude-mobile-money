"""Feature store en ligne (en mémoire) : une seule implémentation pour l'entraînement,
le scoring par lot et l'API temps réel.

Principe : les transactions sont traitées dans l'ordre chronologique. Pour chaque transaction,
`compute()` calcule les variables avec l'historique PASSÉ uniquement (pas de fuite
d'information) sans rien y ajouter, puis `update()` ajoute la transaction à l'historique.
L'entraînement et l'API utilisent le même code : les variables vues en production sont
exactement celles vues à l'entraînement.

Le moteur score une opération AVANT son exécution (son statut final n'est pas encore connu) ;
le statut sert seulement à mettre l'historique à jour ensuite.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd

from vigie.config import CASHOUT_TYPES, OUTFLOW_TYPES, TXN_TYPES, FraudConfig

H = 3600.0
DAY = 86400.0
NO_EVENT_H = 999.0           # plafond quand aucun événement n'a eu lieu
NO_DEPOSIT_MIN = 7 * 24 * 60  # plafond : pas de dépôt depuis 7 jours ou plus


def to_seconds(ts) -> float:
    if isinstance(ts, (int, float)):
        return float(ts)
    return pd.Timestamp(ts).value / 1e9


@dataclass
class Welford:
    """Moyenne et écart-type en ligne (sur le log du montant)."""
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, x: float):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0

    def z(self, x: float, min_n: int) -> float:
        if self.n < min_n:
            return 0.0
        return (x - self.mean) / max(self.std, 0.25)


@dataclass
class AccountState:
    recent: deque = field(default_factory=deque)          # (ts, type, montant, réussie) sur 24 h
    failed: deque = field(default_factory=deque)          # ts des tentatives échouées
    credit_recent: deque = field(default_factory=deque)   # ts des achats de crédit
    new_cp_recent: deque = field(default_factory=deque)   # ts des nouveaux bénéficiaires
    inbound: deque = field(default_factory=deque)         # (ts, expéditeur, montant) reçus sur 24 h
    devices: set = field(default_factory=set)
    counterparties: set = field(default_factory=set)
    regions: set = field(default_factory=set)
    amount_stats: dict = field(default_factory=lambda: defaultdict(Welford))
    n_tx: int = 0
    last_deposit: float | None = None
    last_pin_reset: float | None = None
    last_device_event: float | None = None


_EMPTY_STATS = Welford()  # statistiques vides, pour lire un historique qui n'existe pas encore


def _prune(dq: deque, now: float, window: float, key=lambda x: x[0]):
    while dq and now - key(dq[0]) > window:
        dq.popleft()


class FeatureStore:
    FEATURE_NAMES = [
        "amount", "log_amount", "fee", "hour", "is_night", "is_weekend",
        "account_age_days", "kyc_level", "is_merchant_account",
        "balance_before", "share_of_balance",
        "n_tx_1h", "n_tx_24h", "out_amount_24h", "n_failed_1h", "n_credit_30min",
        "amount_z", "history_n_type",
        "is_new_device", "hours_since_device_event", "hours_since_pin_reset", "device_n_accounts",
        "is_new_counterparty", "n_new_cp_24h", "minutes_since_deposit", "is_unusual_region",
        "inbound_n_senders_24h", "inbound_amount_24h", "cashout_ratio",
        "receiver_n_senders_24h", "receiver_age_days",
        "agent_pair_n_24h", "agent_pair_roundtrip", "agent_roundtrips_24h",
        "merchant_category_z", "merchant_amount_z", "is_round_amount",
    ] + [f"type_{t}" for t in TXN_TYPES]

    def __init__(self, cfg: FraudConfig | None = None, customers: pd.DataFrame | None = None,
                 merchants: pd.DataFrame | None = None):
        self.cfg = cfg or FraudConfig()
        self.accounts: dict[str, AccountState] = defaultdict(AccountState)
        self.created: dict[str, float] = {}
        self.kyc: dict[str, int] = {}
        self.merchant_category: dict[str, str] = {}
        self.device_accounts: dict[str, set] = defaultdict(set)
        self.agent_pairs: dict[tuple, deque] = defaultdict(deque)  # (agent, compte) -> (ts, type, montant)
        self.agent_roundtrips: dict[str, deque] = defaultdict(deque)  # agent -> ts des allers-retours
        self.merchant_stats: dict[str, Welford] = defaultdict(Welford)
        self.category_stats: dict[str, Welford] = defaultdict(Welford)
        if customers is not None:
            self.created.update({a: to_seconds(c) for a, c in zip(customers.account_id, customers.created_at)})
            self.kyc.update(dict(zip(customers.account_id, customers.kyc_level.astype(int))))
        if merchants is not None:
            self.created.update({m: to_seconds(c) for m, c in zip(merchants.merchant_id, merchants.created_at)})
            self.merchant_category.update(dict(zip(merchants.merchant_id, merchants.category)))

    # ----------------------------------------------------------- événements
    def apply_event(self, ev: dict):
        """Événements de sécurité : REINIT_PIN, CONNEXION_NOUVEL_APPAREIL."""
        t = to_seconds(ev["timestamp"])
        st = self.accounts[ev["account_id"]]
        if ev["event_type"] == "REINIT_PIN":
            st.last_pin_reset = t
        elif ev["event_type"] == "CONNEXION_NOUVEL_APPAREIL":
            st.last_device_event = t
        if ev.get("device_id"):
            self.device_accounts[ev["device_id"]].add(ev["account_id"])

    # ------------------------------------------------------------- calcul
    def _age_days(self, acc_id: str, now: float) -> float:
        c = self.created.get(acc_id)
        return max(0.0, (now - c) / DAY) if c is not None else 365.0

    def compute(self, tx: dict) -> dict:
        cfg = self.cfg
        now = to_seconds(tx["timestamp"])
        acc_id, cp_id = tx["account_id"], tx["counterparty_id"]
        t_type, amount = tx["txn_type"], float(tx["amount"])
        fee = float(tx.get("fee") or 0)
        st = self.accounts[acc_id]

        _prune(st.recent, now, DAY)
        _prune(st.failed, now, cfg.failed_window_min * 60, key=lambda x: x)
        _prune(st.credit_recent, now, cfg.airtime_window_min * 60, key=lambda x: x)
        _prune(st.new_cp_recent, now, DAY, key=lambda x: x)
        _prune(st.inbound, now, cfg.mule_window_h * H)

        hour = int((now % DAY) // H)  # heure de Dakar = UTC
        weekday = int((now // DAY + 3) % 7)  # 0 = lundi (le 1er janvier 1970 était un jeudi)
        log_amount = math.log1p(amount)
        is_out = t_type in OUTFLOW_TYPES
        is_merchant = tx.get("account_type") == "MARCHAND" or acc_id in self.merchant_category

        balance = tx.get("balance_before")
        if balance is None or (isinstance(balance, float) and math.isnan(balance)):
            balance, share = -1.0, -1.0
        else:
            balance = float(balance)
            share = min(5.0, (amount + fee) / balance) if (is_out and balance > 0) else (5.0 if is_out else 0.0)

        n_1h = sum(1 for r in st.recent if now - r[0] <= cfg.velocity_window_min * 60)
        out_24h = sum(r[2] for r in st.recent if r[1] in OUTFLOW_TYPES and r[3])

        stats = st.amount_stats.get(t_type, _EMPTY_STATS)
        dev = tx.get("device_id")
        is_new_device = int(bool(dev) and st.n_tx > 0 and dev not in st.devices)
        dev_accounts = len(self.device_accounts.get(dev, set()) | {acc_id}) if dev else 1

        tracks_cp = t_type in ("TRANSFERT", "PAIEMENT_MARCHAND", "DEPOT", "RETRAIT")
        is_new_cp = int(tracks_cp and st.n_tx > 0 and cp_id not in st.counterparties)

        region = tx.get("region")
        unusual_region = int(bool(region) and len(st.regions) > 0 and st.n_tx >= 5 and region not in st.regions)

        in_senders = len({s for _, s, _ in st.inbound})
        in_amount = sum(a for _, _, a in st.inbound)
        cashout_ratio = 0.0
        if t_type in CASHOUT_TYPES and in_amount > 0:
            prior = sum(r[2] for r in st.recent if r[1] in CASHOUT_TYPES and r[3]
                        and now - r[0] <= cfg.mule_window_h * H)
            cashout_ratio = (prior + amount) / in_amount

        recv_senders, recv_age = 0, 365.0
        if t_type in ("TRANSFERT", "PAIEMENT_MARCHAND"):
            rs = self.accounts.get(cp_id)
            senders = {s for ts_, s, _ in rs.inbound if now - ts_ <= cfg.mule_window_h * H} if rs else set()
            recv_senders = len(senders | {acc_id})
            recv_age = self._age_days(cp_id, now)

        pair_n, roundtrip, agent_rt = 0, 0, 0
        if t_type in ("DEPOT", "RETRAIT"):
            dq = [r for r in self.agent_pairs.get((cp_id, acc_id), ()) if now - r[0] <= cfg.agent_pair_window_h * H]
            pair_n = len(dq) + 1
            roundtrip = int(self._is_roundtrip(dq, t_type, amount, now))
            cycles = [x for x in self.agent_roundtrips.get(cp_id, ()) if now - x <= DAY]
            agent_rt = len(cycles) + int(self._closes_cycle(dq, t_type, amount, now))

        cat_z = m_z = 0.0
        if t_type == "PAIEMENT_MARCHAND":
            cat = self.merchant_category.get(cp_id, "inconnu")
            cat_z = self.category_stats.get(cat, _EMPTY_STATS).z(log_amount, cfg.merchant_min_history)
            m_z = self.merchant_stats.get(cp_id, _EMPTY_STATS).z(log_amount, cfg.merchant_min_history)

        def since_h(t):
            return min(NO_EVENT_H, (now - t) / H) if t is not None and t <= now else NO_EVENT_H

        dep_min = NO_DEPOSIT_MIN
        if st.last_deposit is not None:
            dep_min = min(NO_DEPOSIT_MIN, (now - st.last_deposit) / 60)

        f = {
            "amount": amount, "log_amount": log_amount, "fee": fee, "hour": hour,
            "is_night": int(cfg.night_start_hour <= hour < cfg.night_end_hour),
            "is_weekend": int(weekday >= 5),
            "account_age_days": self._age_days(acc_id, now),
            "kyc_level": 3 if is_merchant else self.kyc.get(acc_id, 1),
            "is_merchant_account": int(is_merchant),
            "balance_before": balance, "share_of_balance": share,
            "n_tx_1h": n_1h + 1, "n_tx_24h": len(st.recent) + 1,
            "out_amount_24h": out_24h + (amount if is_out else 0),
            "n_failed_1h": len(st.failed),
            "n_credit_30min": len(st.credit_recent) + (1 if t_type == "CREDIT" else 0),
            "amount_z": stats.z(log_amount, cfg.min_history_for_z), "history_n_type": stats.n,
            "is_new_device": is_new_device,
            "hours_since_device_event": since_h(st.last_device_event),
            "hours_since_pin_reset": since_h(st.last_pin_reset),
            "device_n_accounts": dev_accounts,
            "is_new_counterparty": is_new_cp,
            "n_new_cp_24h": len(st.new_cp_recent) + is_new_cp,
            "minutes_since_deposit": dep_min,
            "is_unusual_region": unusual_region,
            "inbound_n_senders_24h": in_senders, "inbound_amount_24h": in_amount,
            "cashout_ratio": cashout_ratio,
            "receiver_n_senders_24h": recv_senders, "receiver_age_days": recv_age,
            "agent_pair_n_24h": pair_n, "agent_pair_roundtrip": roundtrip, "agent_roundtrips_24h": agent_rt,
            "merchant_category_z": cat_z, "merchant_amount_z": m_z,
            "is_round_amount": int(amount >= 10000 and amount % 5000 == 0),
        }
        for t in TXN_TYPES:
            f[f"type_{t}"] = int(t_type == t)
        return f

    # ------------------------------------------------------------ mise à jour
    def update(self, tx: dict):
        now = to_seconds(tx["timestamp"])
        acc_id, cp_id = tx["account_id"], tx["counterparty_id"]
        t_type, amount = tx["txn_type"], float(tx["amount"])
        ok = (tx.get("status") or "REUSSIE") == "REUSSIE"
        st = self.accounts[acc_id]
        had_history = st.n_tx > 0

        st.recent.append((now, t_type, amount, ok))
        st.n_tx += 1
        dev = tx.get("device_id")
        if dev:
            st.devices.add(dev)
            self.device_accounts[dev].add(acc_id)
        if tx.get("region"):
            st.regions.add(tx["region"])
        if not ok:
            st.failed.append(now)
            return

        if t_type == "CREDIT":
            st.credit_recent.append(now)
        if t_type == "DEPOT":
            st.last_deposit = now
        if t_type in ("TRANSFERT", "PAIEMENT_MARCHAND", "DEPOT", "RETRAIT"):
            if had_history and cp_id not in st.counterparties:
                st.new_cp_recent.append(now)
            st.counterparties.add(cp_id)
        st.amount_stats[t_type].add(math.log1p(amount))

        if t_type == "TRANSFERT":
            self.accounts[cp_id].inbound.append((now, acc_id, amount))
        elif t_type == "PAIEMENT_MARCHAND":
            self.accounts[cp_id].inbound.append((now, acc_id, amount))
            self.merchant_stats[cp_id].add(math.log1p(amount))
            self.category_stats[self.merchant_category.get(cp_id, "inconnu")].add(math.log1p(amount))
        elif t_type in ("DEPOT", "RETRAIT"):
            dq = self.agent_pairs[(cp_id, acc_id)]
            _prune(dq, now, self.cfg.agent_pair_window_h * H)
            if self._closes_cycle(dq, t_type, amount, now):
                cycles = self.agent_roundtrips[cp_id]
                _prune(cycles, now, DAY, key=lambda x: x)
                cycles.append(now)
            dq.append((now, t_type, amount))

    @staticmethod
    def _is_roundtrip(dq, t_type: str, amount: float, now: float) -> bool:
        """Dépôt et retrait (dans un sens ou l'autre) du même montant, même client, même agent, < 1 h."""
        opposite = "RETRAIT" if t_type == "DEPOT" else "DEPOT"
        return any(r[1] == opposite and r[2] == amount and now - r[0] <= H for r in dq)

    @staticmethod
    def _closes_cycle(dq, t_type: str, amount: float, now: float) -> bool:
        """Un RETRAIT qui suit un DÉPÔT du même montant (même client, même agent, < 1 h) ferme un cycle.
        Chaque cycle n'est compté qu'une fois, sur son retrait."""
        return t_type == "RETRAIT" and any(r[1] == "DEPOT" and r[2] == amount and now - r[0] <= H for r in dq)

    def process(self, tx: dict) -> dict:
        f = self.compute(tx)
        self.update(tx)
        return f
