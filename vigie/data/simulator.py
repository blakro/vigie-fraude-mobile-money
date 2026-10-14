"""Simulateur de transactions synthétiques pour Doxal Money (opérateur FICTIF, Sénégal).

Aucune donnée réelle n'est utilisée. Le simulateur produit un journal de transactions
cohérent :
- chaque compte a un SOLDE, qui ne devient jamais négatif ;
- chaque opération respecte les PLAFONDS fictifs du niveau KYC du compte
  (sinon elle est enregistrée comme ÉCHOUÉE, avec son motif) ;
- un client sans solde suffisant va d'abord déposer chez un agent, réduit son montant,
  ou voit son opération échouer ;
- les comportements dépendent du SEGMENT du client (urbain, commerçant, étudiant,
  famille rurale) et du CALENDRIER (jour de la semaine, période de paie, Grand Magal de Touba) ;
- six scénarios de fraude sont injectés, chacun avec un identifiant de cas, un préjudice et
  une date de signalement (certaines fraudes ne sont jamais signalées).

Fonctionnement : simulation à événements discrets. Toutes les opérations prévues sont rangées
dans une file triée par date (heapq) puis exécutées dans l'ordre chronologique ; certaines
exécutions programment de nouvelles opérations (retrait d'une mule, étape suivante d'une
attaque...).
"""
from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from vigie.config import (AGENT_COMMISSION_RATE, FEE_RATE_MERCHANT, FEE_RATE_TRANSFER, LIMITS,
                          MAGAL_DATE, SIM_DAYS, SIM_START, TRAIN_DAYS, TXN_PREFIX)

H = 3600.0
DAY = 86400.0
ALNUM = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
HEX = "0123456789abcdef"

REGIONS = ["Dakar", "Thiès", "Diourbel", "Saint-Louis", "Louga", "Kaolack", "Fatick", "Kaffrine",
           "Ziguinchor", "Kolda", "Sédhiou", "Tambacounda", "Kédougou", "Matam"]
# Répartition approximative des clients par région selon le segment (hypothèse de simulation)
REGION_WEIGHTS = {
    "urbain":         [40, 14, 10, 7, 4, 7, 3, 2, 5, 2, 1, 3, 1, 1],
    "commercant":     [38, 12, 18, 6, 4, 9, 2, 2, 3, 2, 1, 2, 0.5, 0.5],
    "etudiant":       [45, 12, 4, 14, 2, 4, 2, 1, 8, 2, 1, 3, 1, 1],
    "famille_rurale": [0, 12, 12, 8, 9, 10, 8, 6, 6, 8, 5, 8, 3, 5],
}

SEGMENTS = {
    # part de la clientèle, opérations/jour, montant typique (FCFA), part en KYC complet
    "urbain": {"share": .50, "rate": .75, "scale": 7000, "kyc2": .70, "mix": {
        "TRANSFERT": .30, "PAIEMENT_MARCHAND": .22, "DEPOT": .14, "RETRAIT": .09,
        "CREDIT": .13, "FACTURE": .09, "EPARGNE": .03}},
    "commercant": {"share": .12, "rate": 2.0, "scale": 20000, "kyc2": .90, "mix": {
        "TRANSFERT": .30, "PAIEMENT_MARCHAND": .07, "DEPOT": .28, "RETRAIT": .25,
        "CREDIT": .06, "FACTURE": .03, "EPARGNE": .01}},
    "etudiant": {"share": .10, "rate": .55, "scale": 2500, "kyc2": .50, "mix": {
        "TRANSFERT": .18, "PAIEMENT_MARCHAND": .27, "DEPOT": .07, "RETRAIT": .18,
        "CREDIT": .28, "FACTURE": .02}},
    "famille_rurale": {"share": .28, "rate": .35, "scale": 6000, "kyc2": .40, "mix": {
        "TRANSFERT": .10, "PAIEMENT_MARCHAND": .10, "DEPOT": .05, "RETRAIT": .47,
        "CREDIT": .18, "FACTURE": .10}},
}

# catégorie : (panier médian en FCFA, dispersion log-normale, part des marchands)
MERCHANT_CATEGORIES = {
    "boutique": (2500, 0.7, .30), "restaurant": (4000, 0.5, .18), "pharmacie": (6000, 0.6, .10),
    "transport": (1500, 0.5, .12), "station_service": (10000, 0.5, .08),
    "supermarche": (12000, 0.6, .12), "e_commerce": (15000, 0.7, .10),
}
BILLERS = ["SENELEC", "WOYOFAL", "SEN_EAU", "CANAL_PLUS"]
TELCOS = ["ORANGE", "YAS", "EXPRESSO"]
BANKS = ["BANQUE_A", "BANQUE_B", "BANQUE_C", "BANQUE_D"]
MSISDN_PREFIXES = ["77", "78", "76", "70", "75"]

# Profil horaire de base (24 valeurs, normalisées plus bas)
_BASE_HOURS = np.array([
    0.3, 0.2, 0.15, 0.15, 0.2, 0.5, 1.5, 3.0, 4.5, 5.0, 5.5, 6.0,
    6.5, 6.0, 5.0, 5.0, 5.5, 6.0, 6.5, 7.0, 6.5, 5.0, 3.0, 1.2])


def _profile(mult: dict[range, float]) -> np.ndarray:
    p = _BASE_HOURS.copy()
    for hours, m in mult.items():
        p[list(hours)] *= m
    return p / p.sum()


HOUR_PROFILES = {
    "urbain": _profile({}),
    "commercant": _profile({range(6, 10): 1.5, range(21, 24): 0.6}),
    "etudiant": _profile({range(6, 10): 0.6, range(18, 24): 1.6}),
    "famille_rurale": _profile({range(8, 19): 1.2, range(0, 6): 0.5, range(21, 24): 0.5}),
}
DOW_MULT = [1.0, 0.97, 0.97, 1.0, 1.06, 1.10, 0.85]  # lundi -> dimanche

# Délai de signalement par scénario : (probabilité d'être signalé, délai médian en heures)
REPORTING = {
    "PRISE_CONTROLE_COMPTE": (0.90, 20),
    "ARNAQUE_INGENIERIE_SOCIALE": (0.80, 48),
    "FAUX_MARCHAND": (0.70, 96),
    "RAFALE_CREDIT": (0.80, 24),
    "AGENT_CYCLES": (0.85, None),  # audit interne des commissions : 7 à 25 jours
}


@dataclass
class Account:
    id: str
    kind: str                 # CLIENT ou MARCHAND
    region: str
    kyc: object               # 1, 2 ou "MARCHAND"
    created: float            # secondes (epoch)
    balance: int
    device: str
    segment: str = "marchand"
    scale: float = 5000.0
    rate: float = 0.0
    contacts: list = field(default_factory=list)
    family: list = field(default_factory=list)
    fav_agents: list = field(default_factory=list)
    fav_merchants: list = field(default_factory=list)
    category: str | None = None
    last_ts: float = float("-inf")
    is_mule: bool = False
    is_fake_merchant: bool = False
    pending_cashout: bool = False
    last_case: str | None = None
    victim: bool = False
    opening: int = 0


@dataclass
class SimulatedData:
    transactions: pd.DataFrame
    customers: pd.DataFrame
    merchants: pd.DataFrame
    agents: pd.DataFrame
    events: pd.DataFrame
    cases: pd.DataFrame


class DoxalSimulator:
    def __init__(self, seed: int = 42, n_customers: int = 4000, n_agents: int = 180,
                 n_merchants: int = 250, days: int = SIM_DAYS):
        self.rng = np.random.default_rng(seed)
        self.n_customers, self.n_agents, self.n_merchants, self.days = n_customers, n_agents, n_merchants, days
        self.t_start = pd.Timestamp(SIM_START).value / 1e9
        self.t_end = self.t_start + days * DAY
        self.heap: list = []
        self._seq = 0
        self.rows: list[dict] = []
        self.events: list[dict] = []
        self.cases: dict[str, dict] = {}
        self._ids: set[str] = set()
        self._msisdns: set[str] = set()
        self.acc: dict[str, Account] = {}
        self.agent_region: dict[str, str] = {}

    # ================================================================ outils
    def _new_id(self, prefix: str, n: int = 7, alphabet: str = ALNUM) -> str:
        while True:
            i = prefix + "".join(self.rng.choice(list(alphabet), n))
            if i not in self._ids:
                self._ids.add(i)
                return i

    def _device(self) -> str:
        return self._new_id("", 12, HEX)

    def _msisdn(self) -> str:
        while True:
            m = "+221" + self.rng.choice(MSISDN_PREFIXES) + "".join(self.rng.choice(list("0123456789"), 7))
            if m not in self._msisdns:
                self._msisdns.add(m)
                return m

    def _push(self, ts: float, item: dict):
        if ts < self.t_end:
            self._seq += 1
            heapq.heappush(self.heap, (ts, self._seq, item))

    def _at(self, day: int, hour: int) -> float:
        return self.t_start + day * DAY + hour * H + int(self.rng.integers(3600))

    def _hour(self, segment: str) -> int:
        return int(self.rng.choice(24, p=HOUR_PROFILES.get(segment, HOUR_PROFILES["urbain"])))

    @staticmethod
    def _round(x: float, step: int) -> int:
        return int(max(step, round(x / step) * step))

    @staticmethod
    def _floor(x: float, step: int) -> int:
        return int(x // step * step)

    @staticmethod
    def _ceil(x: float, step: int) -> int:
        return int(-(-x // step) * step)

    def _human_round(self, x: float, step: int) -> int:
        """Au-delà de 10 000 FCFA, 40 % des montants sont ronds (10 000, 25 000...)."""
        if x >= 10000 and self.rng.random() < 0.4:
            return self._round(x, 5000)
        return self._round(x, step)

    def _hour_of(self, ts: float) -> float:
        return ((ts - self.t_start) % DAY) / H

    def _agent_open(self, ts: float) -> bool:
        """Les points agents sont ouverts de 7 h à 22 h."""
        return 7 <= self._hour_of(ts) < 22

    def _next_open(self, ts: float) -> float:
        h = self._hour_of(ts)
        if 7 <= h < 22:
            return ts
        wait = (31 - h) if h >= 22 else (7 - h)
        return ts + wait * H + self.rng.uniform(0, 1.5) * H

    def _date(self, day: int):
        return (SIM_START + timedelta(days=day)).date()

    def _pick(self, seq):
        return seq[int(self.rng.integers(len(seq)))]

    # ============================================================ référentiels
    def _build_reference(self):
        rng = self.rng
        segs = list(SEGMENTS)
        seg_of = rng.choice(segs, self.n_customers, p=[SEGMENTS[s]["share"] for s in segs])
        for seg in seg_of:
            p = SEGMENTS[seg]
            w = np.array(REGION_WEIGHTS[seg], dtype=float)
            region = str(rng.choice(REGIONS, p=w / w.sum()))
            old = rng.random() < 0.85
            created = self.t_start + (-int(rng.integers(30, 1500)) if old
                                      else int(rng.integers(0, self.days - 5))) * DAY + int(rng.integers(8, 20)) * H
            kyc = 2 if rng.random() < p["kyc2"] else 1
            scale = float(np.clip(rng.lognormal(np.log(p["scale"]), 0.6), 800, 150000))
            rate = float(np.clip(rng.lognormal(np.log(p["rate"]), 0.5), 0.02, 8))
            if rng.random() < 0.10:  # comptes quasi dormants
                rate *= 0.05
            balance = 0
            if old:
                balance = min(self._round(rng.lognormal(np.log(scale * 1.5), 0.9), 5),
                              int(LIMITS[kyc]["balance_max"] * 0.8))
            a = Account(self._new_id("C"), "CLIENT", region, kyc, created, balance, self._device(),
                        segment=str(seg), scale=scale, rate=rate, opening=balance)
            self.acc[a.id] = a
        self.customer_ids = list(self.acc)
        self.base_customers = list(self.customer_ids)

        counts = pd.Series([self.acc[c].region for c in self.customer_ids]).value_counts()
        w = np.array([counts.get(r, 0) + 1 for r in REGIONS], dtype=float)
        self.agents = []
        for _ in range(self.n_agents):
            aid = self._new_id("A", 6, "0123456789")
            reg = str(rng.choice(REGIONS, p=w / w.sum()))
            self.agent_region[aid] = reg
            self.agents.append({"agent_id": aid, "region": reg,
                                "created_at": self.t_start - int(rng.integers(60, 2000)) * DAY})
        self.agents_by_region = {}
        for a in self.agents:
            self.agents_by_region.setdefault(a["region"], []).append(a["agent_id"])

        cats = list(MERCHANT_CATEGORIES)
        cat_p = np.array([MERCHANT_CATEGORIES[c][2] for c in cats])
        wm = np.array(REGION_WEIGHTS["urbain"], dtype=float) + 1
        self.merchant_ids = []
        for _ in range(self.n_merchants):
            opening = self._round(rng.lognormal(np.log(40000), 0.8), 5)
            m = Account(self._new_id("M", 6, "0123456789"), "MARCHAND",
                        str(rng.choice(REGIONS, p=wm / wm.sum())), "MARCHAND",
                        self.t_start - int(rng.integers(60, 1500)) * DAY, opening, self._device(),
                        category=str(rng.choice(cats, p=cat_p / cat_p.sum())), opening=opening)
            self.acc[m.id] = m
            self.merchant_ids.append(m.id)
        self.merchants_by_region = {}
        for mid in self.merchant_ids:
            self.merchants_by_region.setdefault(self.acc[mid].region, []).append(mid)

        # Réseau social : contacts, famille, agents et marchands habituels
        by_seg = {s: [c for c in self.customer_ids if self.acc[c].segment == s] for s in SEGMENTS}
        senders = by_seg["urbain"] + by_seg["commercant"]
        for c in by_seg["etudiant"]:  # les étudiants reçoivent de 1 ou 2 parents
            for parent in rng.choice(senders, int(rng.integers(1, 3)), replace=False):
                self.acc[parent].family.append(c)
        for c in senders:
            k = int(rng.choice([0, 1, 2, 3], p=[.25, .4, .25, .1]))
            if k and by_seg["famille_rurale"]:
                self.acc[c].family += list(rng.choice(by_seg["famille_rurale"], k, replace=False))
        pools = {
            "urbain": by_seg["urbain"] + by_seg["commercant"] + by_seg["etudiant"],
            "commercant": by_seg["commercant"] * 3 + by_seg["urbain"],
            "etudiant": by_seg["etudiant"] * 2 + by_seg["urbain"],
            "famille_rurale": by_seg["famille_rurale"],
        }
        n_contacts = {"urbain": (4, 13), "commercant": (5, 16), "etudiant": (3, 11), "famille_rurale": (2, 7)}
        for c in self.customer_ids:
            a = self.acc[c]
            lo, hi = n_contacts[a.segment]
            pool = pools[a.segment]
            # dict.fromkeys dédoublonne en gardant l'ordre (un set rendrait la simulation non reproductible)
            a.contacts = [x for x in dict.fromkeys(rng.choice(pool, int(rng.integers(lo, hi)))) if x != c]
            a.fav_agents = [self._agent_in(a.region) for _ in range(2)]
            local = self.merchants_by_region.get(a.region) or self.merchant_ids
            a.fav_merchants = list(rng.choice(local, min(len(local), int(rng.integers(3, 6))), replace=False))
        for c in senders:  # les proches renvoient parfois de l'argent
            for f in self.acc[c].family:
                if self.acc[f].segment == "famille_rurale":
                    self.acc[f].contacts.append(c)

    def _agent_in(self, region: str) -> str:
        pool = self.agents_by_region.get(region) or [a["agent_id"] for a in self.agents]
        return self._pick(pool)

    def _merchant_in(self, region: str) -> str:
        pool = [m for m in self.merchants_by_region.get(region, []) if not self.acc[m].is_fake_merchant] \
            or [m for m in self.merchant_ids if not self.acc[m].is_fake_merchant]
        return self._pick(pool)

    # =================================================== intentions normales
    def _normal_amount(self, a: Account, t: str, cp: str) -> int:
        rng, s = self.rng, a.scale
        if t == "TRANSFERT":
            x = s * rng.lognormal(0, 0.8)
            if rng.random() < 0.01:  # gros transfert légitime (loyer, cérémonie, famille)
                x *= rng.uniform(5, 12)
            return self._human_round(x, 500 if rng.random() < 0.6 else 50)
        if t == "DEPOT":
            return self._human_round(s * 2 * rng.lognormal(0, 0.7), 500)
        if t == "RETRAIT":
            return max(1000, self._human_round(s * 1.8 * rng.lognormal(0, 0.7), 500))
        if t == "PAIEMENT_MARCHAND":
            base, sig, _ = MERCHANT_CATEGORIES[self.acc[cp].category]
            return self._round(base * rng.lognormal(0, sig), 25)
        if t == "FACTURE":
            if cp == "SENELEC":
                return self._round(rng.uniform(8000, 40000), 5)
            if cp == "WOYOFAL":
                return self._round(rng.uniform(2000, 20000), 1000)
            if cp == "SEN_EAU":
                return self._round(rng.uniform(5000, 25000), 5)
            return int(rng.choice([5000, 10000, 15000, 20000]))
        if t == "CREDIT":
            return int(rng.choice([200, 500, 1000, 2000, 5000], p=[.2, .35, .25, .13, .07]))
        return self._round(s * rng.lognormal(0, 0.6), 1000)  # EPARGNE

    def _plan_normal(self):
        rng = self.rng
        magal_day = (MAGAL_DATE - SIM_START.date()).days
        travel_days = set(range(magal_day - 1, magal_day + 2)) if 0 <= magal_day < self.days else set()
        pilgrims = {c for c in self.base_customers
                    if self.acc[c].region != "Diourbel" and self.acc[c].segment != "famille_rurale"
                    and rng.random() < 0.20}
        for c in self.base_customers:
            a = self.acc[c]
            mix = SEGMENTS[a.segment]["mix"]
            types, probs = list(mix), np.array(list(mix.values()))
            probs = probs / probs.sum()
            # changement de téléphone légitime : ~4 % des clients, dont 35 % avec réinitialisation du PIN
            if rng.random() < 0.04:
                self._push(self._at(int(rng.integers(1, self.days - 1)), int(rng.integers(8, 21))),
                           {"kind": "device_change", "acc": c, "pin": rng.random() < 0.35})
            for day in range(self.days):
                d = self._date(day)
                m = DOW_MULT[d.weekday()]
                if a.segment in ("urbain", "commercant") and (d.day >= 26 or d.day <= 4):
                    m *= 1.12  # période de paie
                if a.region == "Diourbel" and day in travel_days:
                    m *= 1.6   # afflux autour du Grand Magal de Touba
                for _ in range(rng.poisson(a.rate * m)):
                    t = types[int(rng.choice(len(types), p=probs))]
                    hour = self._hour(a.segment)
                    if t in ("DEPOT", "RETRAIT") and not 7 <= hour <= 21:
                        hour = int(rng.integers(8, 21))  # opération chez un agent : heures d'ouverture
                    ts = self._at(day, hour)
                    if ts <= a.created:
                        continue
                    loc = "Diourbel" if (c in pilgrims and day in travel_days) else a.region
                    cp, cp_type = self._normal_counterparty(a, t, loc)
                    if cp is None:
                        continue
                    self._push(ts, {"kind": "op", "type": t, "acc": c, "cp": cp, "cp_type": cp_type,
                                    "amount": self._normal_amount(a, t, cp), "policy": "flex", "loc": loc})
            # envois à la famille en période de paie (fin/début de mois)
            for f in a.family:
                for lo, hi in ((0, 4), (25, 35), (55, 59)):
                    if hi < self.days and rng.random() < 0.75:
                        ts = self._at(int(rng.integers(lo, hi + 1)), int(rng.integers(8, 21)))
                        if ts > a.created and ts > self.acc[f].created:
                            amt = self._human_round(a.scale * rng.uniform(1.2, 3.5), 1000)
                            self._push(ts, {"kind": "op", "type": "TRANSFERT", "acc": c, "cp": f,
                                            "cp_type": "USER", "amount": amt, "policy": "flex"})
            # envoi aux proches de Touba avant le Magal
            touba = [x for x in a.family + a.contacts if self.acc[x].region == "Diourbel"]
            if travel_days and touba and a.segment in ("urbain", "commercant") and rng.random() < 0.45:
                ts = self._at(int(rng.integers(magal_day - 4, magal_day)), int(rng.integers(8, 22)))
                if ts > a.created:
                    self._push(ts, {"kind": "op", "type": "TRANSFERT", "acc": c, "cp": self._pick(touba),
                                    "cp_type": "USER", "policy": "flex",
                                    "amount": self._human_round(a.scale * rng.uniform(1.5, 5), 1000)})
        for mid in self.merchant_ids:  # reversements des marchands
            self._push(self._at(int(rng.integers(0, 4)), int(rng.integers(9, 19))), {"kind": "payout", "acc": mid})

    def _normal_counterparty(self, a: Account, t: str, loc: str):
        rng = self.rng
        if t == "TRANSFERT":
            r = rng.random()
            if a.family and r < 0.12:
                return self._pick(a.family), "USER"
            if a.contacts and r < 0.90:
                return self._pick(a.contacts), "USER"
            cp = self._pick(self.customer_ids)
            return (cp, "USER") if cp != a.id else (None, None)
        if t in ("DEPOT", "RETRAIT"):
            if loc == a.region and rng.random() < 0.85:
                return self._pick(a.fav_agents), "AGENT"
            return self._agent_in(loc), "AGENT"
        if t == "PAIEMENT_MARCHAND":
            if loc == a.region and a.fav_merchants and rng.random() < 0.7:
                return self._pick(a.fav_merchants), "MERCHANT"
            return self._merchant_in(loc), "MERCHANT"
        if t == "FACTURE":
            return BILLERS[int(rng.choice(4, p=[.35, .3, .2, .15]))], "BILLER"
        if t == "CREDIT":
            return TELCOS[int(rng.choice(3, p=[.6, .3, .1]))], "TELCO"
        return f"EP{a.id[1:]}", "EPARGNE"

    # ================================================================ exécution
    def _region_of(self, a: Account, t: str, cp: str, loc: str | None) -> str:
        if t in ("DEPOT", "RETRAIT"):
            return self.agent_region.get(cp, a.region)
        if t == "PAIEMENT_MARCHAND":
            return self.acc[cp].region
        if t == "VIREMENT_BANCAIRE":
            return a.region
        return loc or a.region

    def _record(self, ts, t, a: Account, cp, cp_type, amount, fee, bal_before, status, reason,
                device, region, scenario=None, case=None, merchant_fee=0, commission=0, loss=0):
        self.rows.append({
            "ts": ts, "seq": len(self.rows), "txn_type": t, "account_id": a.id, "account_type": a.kind,
            "counterparty_id": cp, "counterparty_type": cp_type, "amount": int(amount),
            "fee": int(fee), "merchant_fee": int(merchant_fee), "agent_commission": int(commission),
            "balance_before": int(bal_before), "balance_after": int(a.balance), "status": status,
            "failure_reason": reason, "device_id": device, "region": region,
            "is_fraud": int(scenario is not None), "fraud_scenario": scenario or "NORMAL",
            "fraud_case_id": case, "loss_fcfa": int(loss),
        })
        a.last_ts = max(a.last_ts, ts)
        if case:
            c = self.cases[case]
            c["start"] = min(c["start"], ts)
            c["end"] = max(c["end"], ts)
            c["n_tx"] += 1
            c["loss"] += int(loss)

    def _execute(self, ts: float, it: dict) -> str:
        """Exécute une opération. Renvoie le statut ("REUSSIE", "ECHOUEE" ou "IGNOREE")."""
        a = self.acc[it["acc"]]
        ts = max(ts, a.last_ts + 1)  # les opérations d'un même compte restent dans l'ordre
        t, cp, cp_type = it["type"], it["cp"], it["cp_type"]
        if ts >= self.t_end:
            return "IGNOREE"  # hors de la période simulée : ne doit modifier aucun solde
        if cp_type == "AGENT" and not self._agent_open(ts):
            return "IGNOREE"  # point agent fermé
        if t in ("TRANSFERT", "PAIEMENT_MARCHAND") and self.acc[cp].created > ts:
            return "IGNOREE"  # le bénéficiaire n'a pas encore de compte
        amount = int(it["amount"])
        policy = it.get("policy", "flex")
        scen, case = it.get("scenario"), it.get("case")
        lim = LIMITS[a.kyc]
        device = it.get("device") or a.device
        region = self._region_of(a, t, cp, it.get("loc"))

        def fail(reason, amt):
            self._record(ts, t, a, cp, cp_type, amt, 0, a.balance, "ECHOUEE", reason, device, region, scen, case)
            return "ECHOUEE"

        if t == "DEPOT":
            room = min(lim["txn_max"], lim["balance_max"] - a.balance)
            if amount > room:
                if policy == "exact":
                    return fail("PLAFOND_SOLDE", amount)
                amount = self._floor(room, 500)
            if amount < 500:
                return "IGNOREE"
            before = a.balance
            a.balance += amount
            com = round(amount * AGENT_COMMISSION_RATE)
            self._record(ts, t, a, cp, cp_type, amount, 0, before, "REUSSIE", None, device, region, scen, case,
                         commission=com, loss=com if scen == "AGENT_CYCLES" else 0)
            return "REUSSIE"

        rate = FEE_RATE_TRANSFER if t == "TRANSFERT" else 0.0
        if amount > lim["txn_max"]:
            if policy == "exact":
                return fail("PLAFOND_OPERATION", amount)
            amount = lim["txn_max"]
        fee = round(amount * rate)
        if a.balance < amount + fee:
            if policy == "exact":
                return fail("SOLDE_INSUFFISANT", amount)
            if policy == "drain" or t in ("RETRAIT", "VIREMENT_BANCAIRE"):
                amount = self._floor(a.balance / (1 + rate), 500 if t != "CREDIT" else 100)
                if amount < (1000 if t in ("RETRAIT", "VIREMENT_BANCAIRE") else 200):
                    return "IGNOREE"
            else:  # client normal : dépôt d'abord, montant réduit, ou échec
                r = self.rng.random()
                t_dep = max(a.last_ts + 1, ts - self.rng.uniform(180, 1200), self.t_start)
                if r < 0.55 and not self._agent_open(t_dep):
                    r = 0.6  # pas d'agent ouvert : le client réduit son montant
                if r < 0.55 and t != "EPARGNE" and a.kind == "CLIENT":
                    need = amount + round(amount * rate) - a.balance
                    dep = max(self._ceil(need, 1000), self._round(need * self.rng.uniform(1.0, 1.5), 1000))
                    if a.balance + dep > lim["balance_max"] or dep > lim["txn_max"]:
                        return fail("SOLDE_INSUFFISANT", amount)
                    agent = self._pick(a.fav_agents) if it.get("loc", a.region) == a.region \
                        else self._agent_in(it["loc"])
                    self._execute(t_dep, {"kind": "op", "type": "DEPOT", "acc": a.id, "cp": agent,
                                          "cp_type": "AGENT", "amount": dep, "policy": "exact"})
                    if a.balance < amount + round(amount * rate):
                        return fail("SOLDE_INSUFFISANT", amount)
                elif r < 0.75:
                    amount = self._floor(a.balance / (1 + rate), 25)
                    if amount < 200:
                        return "IGNOREE"
                else:
                    return fail("SOLDE_INSUFFISANT", amount)
            fee = round(amount * rate)

        recv, credit, mfee = None, 0, 0
        if t in ("TRANSFERT", "PAIEMENT_MARCHAND"):
            recv = self.acc[cp]
            mfee = round(amount * FEE_RATE_MERCHANT) if t == "PAIEMENT_MARCHAND" else 0
            credit = amount - mfee
            if recv.balance + credit > LIMITS[recv.kyc]["balance_max"]:
                return fail("PLAFOND_SOLDE_BENEFICIAIRE", amount)

        before = a.balance
        a.balance -= amount + fee
        com = round(amount * AGENT_COMMISSION_RATE) if t == "RETRAIT" else 0
        loss = 0
        if scen in ("PRISE_CONTROLE_COMPTE", "ARNAQUE_INGENIERIE_SOCIALE"):
            loss = amount + fee
        elif scen in ("RAFALE_CREDIT",) or (scen == "FAUX_MARCHAND" and t == "PAIEMENT_MARCHAND"):
            loss = amount
        elif scen == "AGENT_CYCLES":
            loss = com
        self._record(ts, t, a, cp, cp_type, amount, fee, before, "REUSSIE", None, device, region, scen, case,
                     merchant_fee=mfee, commission=com, loss=loss)
        if recv is not None:
            recv.balance += credit
            recv.last_ts = max(recv.last_ts, ts)
            self._on_credit(recv, ts, case if scen else None)
        return "REUSSIE"

    def _on_credit(self, recv: Account, ts: float, case: str | None):
        """Une mule ou un faux marchand vide son compte peu après avoir été crédité."""
        if not (recv.is_mule or recv.is_fake_merchant) or case is None:
            return
        recv.last_case = case
        cap = LIMITS[recv.kyc]["balance_max"]
        urgent = recv.balance > 0.6 * cap
        threshold = 150_000 if recv.is_fake_merchant else 0
        if not recv.pending_cashout and (urgent or recv.balance >= threshold):
            recv.pending_cashout = True
            delay = self.rng.uniform(10, 40) if urgent else self.rng.uniform(20, 240)
            self._push(self._next_open(ts + delay * 60), {"kind": "cashout", "acc": recv.id})

    def _cashout(self, ts: float, it: dict):
        a = self.acc[it["acc"]]
        a.pending_cashout = False
        scen = "FAUX_MARCHAND" if a.is_fake_merchant else "MULE_RETRAIT"
        lim = LIMITS[a.kyc]["txn_max"]
        amount = self._floor(min(a.balance * self.rng.uniform(0.85, 1.0), lim,
                                 self.rng.uniform(100_000, 150_000)), 500)
        if amount < 5000:
            return
        if not self._agent_open(ts):
            a.pending_cashout = True
            self._push(self._next_open(ts), it)
            return
        if a.is_fake_merchant and self.rng.random() < 0.3:
            t, cp, cp_type = "VIREMENT_BANCAIRE", self._pick(BANKS), "BANQUE"
        else:  # retrait chez un agent de sa zone (souvent un agent différent à chaque fois)
            t, cp, cp_type = "RETRAIT", self._agent_in(a.region), "AGENT"
        self._execute(ts, {"kind": "op", "type": t, "acc": a.id, "cp": cp, "cp_type": cp_type,
                           "amount": amount, "policy": "exact", "scenario": scen, "case": a.last_case})
        if a.balance >= 20_000:  # reste à sortir : nouveau retrait chez un autre agent
            a.pending_cashout = True
            self._push(self._next_open(ts + self.rng.uniform(5, 25) * 60), {"kind": "cashout", "acc": a.id})

    def _payout(self, ts: float, it: dict):
        """Reversement régulier d'un marchand normal vers sa banque ou en espèces."""
        m = self.acc[it["acc"]]
        if m.is_fake_merchant:
            return
        if m.balance >= 20_000:
            amount = self._floor(m.balance * self.rng.uniform(0.7, 0.95), 500)
            if self.rng.random() < 0.6:
                t, cp, cp_type = "VIREMENT_BANCAIRE", self._pick(BANKS), "BANQUE"
            else:
                t, cp, cp_type = "RETRAIT", self._agent_in(m.region), "AGENT"
            self._execute(ts, {"kind": "op", "type": t, "acc": m.id, "cp": cp, "cp_type": cp_type,
                               "amount": amount, "policy": "exact"})
        day = int((ts - self.t_start) // DAY) + int(self.rng.integers(2, 7))
        self._push(self._at(day, int(self.rng.integers(9, 19))), {"kind": "payout", "acc": m.id})

    def _device_change(self, ts: float, it: dict):
        a = self.acc[it["acc"]]
        new = self._device()
        if it.get("pin"):
            self.events.append({"account_id": a.id, "ts": ts - 180, "event_type": "REINIT_PIN", "device_id": new})
        self.events.append({"account_id": a.id, "ts": ts, "event_type": "CONNEXION_NOUVEL_APPAREIL",
                            "device_id": new})
        a.device = new
        a.last_ts = max(a.last_ts, ts)

    # ============================================================ fraudes
    def _new_case(self, scenario: str) -> str:
        cid = f"CAS-{len(self.cases) + 1:03d}"
        self.cases[cid] = {"case_id": cid, "scenario": scenario, "start": float("inf"),
                           "end": float("-inf"), "n_tx": 0, "loss": 0}
        return cid

    def _victim(self, ts: float, min_balance: int = 0, tries: int = 300) -> Account | None:
        for _ in range(tries):
            a = self.acc[self._pick(self.customer_ids)]
            if (not a.is_mule and not a.victim and a.created < ts - 60 * DAY
                    and a.balance >= min_balance and a.segment != "famille_rurale"):
                a.victim = True
                return a
        return None

    def _plan_mule(self, first_use: float, rented: bool) -> str:
        """Mule : compte LOUÉ (client existant recruté) ou NOUVEAU compte ouvert en KYC allégé."""
        rng = self.rng
        if rented:
            for _ in range(300):
                a = self.acc[self._pick(self.customer_ids)]
                if not a.is_mule and not a.victim and a.created < first_use - 30 * DAY:
                    a.is_mule = True
                    return a.id
        created = first_use - rng.uniform(1, 15) * DAY
        w = np.array(REGION_WEIGHTS["urbain"], dtype=float)
        a = Account(self._new_id("C"), "CLIENT", str(rng.choice(REGIONS, p=w / w.sum())), 1, created, 0,
                    self._device(), segment="urbain", scale=3000, rate=0.0, is_mule=True)
        self.acc[a.id] = a
        self.customer_ids.append(a.id)
        a.fav_agents = [self._agent_in(a.region)]
        # petite activité de couverture : un dépôt puis un achat de crédit (opérations normales)
        t1 = self._next_open(created + rng.uniform(2, 24) * H)  # dépôt pendant les heures d'ouverture
        if t1 >= self.t_start:
            self._push(t1, {"kind": "op", "type": "DEPOT", "acc": a.id, "cp": a.fav_agents[0], "cp_type": "AGENT",
                            "amount": int(rng.choice([2000, 3000, 5000])), "policy": "flex"})
            self._push(t1 + rng.uniform(1, 3) * H, {"kind": "op", "type": "CREDIT", "acc": a.id,
                                                    "cp": self._pick(TELCOS), "cp_type": "TELCO",
                                                    "amount": int(rng.choice([500, 1000])), "policy": "flex"})
        return a.id

    def _plan_frauds(self):
        rng = self.rng
        k = self.n_customers / 4000

        def n(x):
            return max(2, int(round(x * k)))

        # Un cas sur trois tombe dans les 20 derniers jours (période de test), les autres avant :
        # chaque scénario est ainsi présent à l'entraînement ET au test.
        def case_day(i, length=1):
            if i % 3 == 2:
                lo, hi = TRAIN_DAYS + 1, self.days - length
            else:
                lo, hi = 2, TRAIN_DAYS - 1 - length
            return int(rng.integers(lo, max(lo + 1, hi)))

        def when(i, night_share=0.0):
            hour = int(rng.choice([1, 2, 3, 4])) if rng.random() < night_share else int(rng.integers(8, 22))
            return self._at(case_day(i), hour)

        # 1. Prise de contrôle de compte -> mules (pool réutilisé)
        ato_times = sorted(when(i, 0.4) for i in range(n(36)))
        pool: list[tuple[float, str]] = []
        for t0 in ato_times:
            usable = [m for c, m in pool if c < t0 - DAY]
            if not usable or rng.random() < 0.35:
                mid = self._plan_mule(t0, rented=rng.random() < 0.4)
                pool.append((self.acc[mid].created, mid))
                usable.append(mid)
            mules = list(rng.choice(usable, min(len(usable), int(rng.integers(1, 3))), replace=False))
            self._push(t0, {"kind": "ato", "case": self._new_case("PRISE_CONTROLE_COMPTE"), "mules": mules})

        # 2. Arnaque "faux service client" : plusieurs victimes -> une mule
        for i in range(n(15)):
            t0 = when(i)
            mule = self._plan_mule(t0, rented=rng.random() < 0.5)
            case = self._new_case("ARNAQUE_INGENIERIE_SOCIALE")
            hours = rng.uniform(6, 30)
            for _ in range(int(rng.integers(4, 9))):
                ts = t0 + rng.uniform(0, hours) * H
                hour = ((ts - self.t_start) % DAY) / H
                if hour < 8 or hour > 22:
                    ts += (9 - hour if hour < 8 else 33 - hour) * H
                self._push(ts, {"kind": "scam", "case": case, "mule": mule})

        # 3. Cycles dépôt/retrait d'un agent avec ses complices
        for i, agent in enumerate(rng.choice([a["agent_id"] for a in self.agents], n(8), replace=False)):
            reg = self.agent_region[agent]
            local = [c for c in self.customer_ids if self.acc[c].region == reg and not self.acc[c].is_mule
                     and self.acc[c].created < self.t_start] or self.customer_ids
            complices = list(rng.choice(local, min(len(local), int(rng.integers(2, 5))), replace=False))
            case = self._new_case("AGENT_CYCLES")
            length = int(rng.integers(3, 8))
            start = case_day(i, length)
            for day in range(start, start + length):
                for _ in range(int(rng.integers(2, 7))):
                    c = self._pick(complices)
                    amt = int(rng.choice([10000, 20000, 25000, 50000]))
                    t = self._at(day, int(rng.integers(8, 20)))
                    base = {"kind": "op", "acc": c, "cp": agent, "cp_type": "AGENT", "amount": amt,
                            "policy": "exact", "scenario": "AGENT_CYCLES", "case": case}
                    self._push(t, {**base, "type": "DEPOT"})
                    self._push(t + rng.uniform(3, 40) * 60, {**base, "type": "RETRAIT"})

        # 4. Faux marchand : compte marchand récent, fausses ventes en ligne, puis retrait des fonds
        for i in range(n(9)):
            t0 = self._at(case_day(i, 3), int(rng.integers(8, 12)))
            m = Account(self._new_id("M", 6, "0123456789"), "MARCHAND",
                        "Dakar" if rng.random() < 0.7 else "Thiès", "MARCHAND",
                        t0 - rng.uniform(3, 20) * DAY, 0, self._device(),
                        category="boutique" if rng.random() < 0.6 else "e_commerce", is_fake_merchant=True)
            self.acc[m.id] = m
            self.merchant_ids.append(m.id)
            self.merchants_by_region.setdefault(m.region, []).append(m.id)
            case = self._new_case("FAUX_MARCHAND")
            span = rng.uniform(1, 2) * DAY
            for _ in range(int(rng.integers(12, 31))):
                ts = t0 + rng.uniform(0, span)
                hour = ((ts - self.t_start) % DAY) / H
                if hour < 8:
                    ts += (8 - hour) * H + rng.uniform(0, 3) * H
                self._push(ts, {"kind": "fake_purchase", "case": case, "merchant": m.id})
            self._push(self._next_open(t0 + span + rng.uniform(1, 4) * H), {"kind": "cashout", "acc": m.id})

        # 5. Rafale d'achats de crédit sur un compte compromis
        for i in range(n(20)):
            self._push(when(i, 0.2), {"kind": "airtime", "case": self._new_case("RAFALE_CREDIT")})

        self.attacker_devices = [self._device() for _ in range(10)]

    def _ato(self, ts: float, it: dict):
        rng = self.rng
        v = self._victim(ts, min_balance=5000)
        if v is None:
            return
        dev = self._pick(self.attacker_devices)
        loc = "Dakar" if rng.random() < 0.6 else ("Thiès" if rng.random() < 0.5 else self._pick(REGIONS))
        if rng.random() < 0.8:
            self.events.append({"account_id": v.id, "ts": ts, "event_type": "REINIT_PIN", "device_id": dev})
        self.events.append({"account_id": v.id, "ts": ts + rng.uniform(30, 120),
                            "event_type": "CONNEXION_NOUVEL_APPAREIL", "device_id": dev})
        base = {"kind": "ato_step", "acc": v.id, "device": dev, "loc": loc, "mules": it["mules"],
                "case": it["case"]}
        t = ts + rng.uniform(2, 6) * 60
        if rng.random() < 0.35:  # transfert test de petit montant
            self._push(t, {**base, "test": True, "left": int(rng.integers(2, 6))})
        else:
            self._push(t, {**base, "test": False, "left": int(rng.integers(2, 6))})

    def _ato_step(self, ts: float, it: dict):
        rng = self.rng
        v = self.acc[it["acc"]]
        common = {"kind": "op", "acc": v.id, "device": it["device"], "loc": it["loc"],
                  "scenario": "PRISE_CONTROLE_COMPTE", "case": it["case"]}
        if v.balance < 2000:
            if v.balance >= 500 and rng.random() < 0.3:  # reste converti en crédit téléphonique
                self._execute(ts, {**common, "type": "CREDIT", "cp": self._pick(TELCOS), "cp_type": "TELCO",
                                   "amount": self._floor(v.balance, 100), "policy": "exact"})
            return
        mule = self._pick(it["mules"])
        if it["test"]:
            amount = int(rng.choice([500, 1000, 2000]))
        else:
            amount = self._floor(v.balance / (1 + FEE_RATE_TRANSFER) * rng.uniform(0.85, 1.0), 500)
            amount = min(amount, int(rng.choice([LIMITS[v.kyc]["txn_max"], 10 ** 9], p=[.5, .5])))
        status = self._execute(ts, {**common, "type": "TRANSFERT", "cp": mule, "cp_type": "USER",
                                    "amount": amount, "policy": "exact"})
        left = it["left"] - (0 if it["test"] else 1)
        if status == "ECHOUEE":
            left -= 1  # le fraudeur réessaie avec un montant plus petit ou une autre mule
        if left > 0:
            self._push(ts + rng.uniform(2, 10) * 60, {**it, "test": False, "left": left})

    def _scam(self, ts: float, it: dict):
        rng = self.rng
        v = self._victim(ts)
        if v is None:
            return
        amount = self._round(min(max(15000, v.scale * rng.uniform(4, 12)), 150000),
                             5000 if rng.random() < 0.5 else 1000)
        fee = round(amount * FEE_RATE_TRANSFER)
        if v.balance < amount + fee:
            t_dep = max(v.last_ts + 1, ts - rng.uniform(10, 40) * 60, self.t_start)
            if rng.random() < 0.85 and self._agent_open(t_dep):  # la victime va déposer chez un agent
                dep = self._ceil(amount + fee - v.balance, 1000)
                self._execute(t_dep, {"kind": "op", "type": "DEPOT", "acc": v.id, "cp": self._pick(v.fav_agents),
                                      "cp_type": "AGENT", "amount": dep, "policy": "exact"})
            elif v.balance >= 10000 + fee:
                amount = self._floor(v.balance / (1 + FEE_RATE_TRANSFER), 1000)
            else:
                return
        status = self._execute(ts, {"kind": "op", "type": "TRANSFERT", "acc": v.id, "cp": it["mule"],
                                    "cp_type": "USER", "amount": amount, "policy": "exact",
                                    "scenario": "ARNAQUE_INGENIERIE_SOCIALE", "case": it["case"]})
        if status == "ECHOUEE" and rng.random() < 0.5:  # l'escroc demande d'envoyer moins
            self._push(ts + rng.uniform(5, 15) * 60, {"kind": "op", "type": "TRANSFERT", "acc": v.id,
                                                      "cp": it["mule"], "cp_type": "USER",
                                                      "amount": self._round(amount / 2, 1000), "policy": "exact",
                                                      "scenario": "ARNAQUE_INGENIERIE_SOCIALE", "case": it["case"]})

    def _fake_purchase(self, ts: float, it: dict):
        rng = self.rng
        for _ in range(50):
            p = self.acc[self._pick(self.customer_ids)]
            if not p.is_mule and p.created < ts:
                break
        else:
            return
        amount = self._round(rng.uniform(20000, 120000), 500)
        if p.balance < amount:
            t_dep = max(p.last_ts + 1, ts - rng.uniform(10, 60) * 60, self.t_start)
            if rng.random() > 0.9 or p.balance + amount > LIMITS[p.kyc]["balance_max"] \
                    or not self._agent_open(t_dep):
                return
            self._execute(t_dep,
                          {"kind": "op", "type": "DEPOT", "acc": p.id, "cp": self._pick(p.fav_agents),
                           "cp_type": "AGENT", "amount": self._ceil(amount - p.balance, 1000), "policy": "exact"})
        self._execute(ts, {"kind": "op", "type": "PAIEMENT_MARCHAND", "acc": p.id, "cp": it["merchant"],
                           "cp_type": "MERCHANT", "amount": amount, "policy": "exact",
                           "scenario": "FAUX_MARCHAND", "case": it["case"]})

    def _airtime(self, ts: float, it: dict):
        rng = self.rng
        v = self._victim(ts, min_balance=10000)
        if v is None:
            return
        dev = v.device
        if rng.random() < 0.6:
            dev = self._pick(self.attacker_devices)
            if rng.random() < 0.4:
                self.events.append({"account_id": v.id, "ts": ts - 240, "event_type": "REINIT_PIN", "device_id": dev})
            self.events.append({"account_id": v.id, "ts": ts - 60, "event_type": "CONNEXION_NOUVEL_APPAREIL",
                                "device_id": dev})
        self._push(ts, {"kind": "airtime_step", "acc": v.id, "device": dev, "case": it["case"],
                        "left": int(rng.integers(8, 21))})

    def _airtime_step(self, ts: float, it: dict):
        v = self.acc[it["acc"]]
        options = [x for x in (1000, 2000, 5000) if x <= v.balance]
        if not options or it["left"] <= 0:
            return
        self._execute(ts, {"kind": "op", "type": "CREDIT", "acc": v.id, "cp": self._pick(TELCOS),
                           "cp_type": "TELCO", "amount": int(self.rng.choice(options)), "policy": "exact",
                           "device": it["device"], "scenario": "RAFALE_CREDIT", "case": it["case"]})
        self._push(ts + self.rng.uniform(20, 90), {**it, "left": it["left"] - 1})

    # ============================================================ assemblage
    def run(self) -> SimulatedData:
        self._build_reference()
        self._plan_frauds()
        self._plan_normal()
        handlers = {"op": self._execute, "cashout": self._cashout, "payout": self._payout,
                    "device_change": self._device_change, "ato": self._ato, "ato_step": self._ato_step,
                    "scam": self._scam, "fake_purchase": self._fake_purchase, "airtime": self._airtime,
                    "airtime_step": self._airtime_step}
        while self.heap:
            ts, _, it = heapq.heappop(self.heap)
            if ts < self.t_start:
                continue
            if "acc" in it and it["kind"] != "payout":
                ts = max(ts, self.acc[it["acc"]].last_ts + 1)
                if ts >= self.t_end:
                    continue
            handlers[it["kind"]](ts, it)
        return self._assemble()

    def _assemble(self) -> SimulatedData:
        rng = self.rng
        tx = pd.DataFrame(self.rows)
        tx = tx[(tx.ts >= self.t_start) & (tx.ts < self.t_end)]
        tx = tx.sort_values(["ts", "seq"], kind="stable").reset_index(drop=True)
        tx.insert(0, "timestamp", pd.to_datetime(tx.ts, unit="s").dt.floor("s"))
        day_key = tx.timestamp.dt.strftime("%y%m%d")
        tx.insert(0, "txn_id", TXN_PREFIX + day_key + "." + (tx.groupby(day_key).cumcount() + 1)
                  .astype(str).str.zfill(6))
        tx = tx.drop(columns=["ts", "seq"])

        # Cas de fraude : date de signalement (certains ne sont jamais signalés)
        cases = []
        for c in self.cases.values():
            if c["n_tx"] == 0:
                continue
            p_report, median_h = REPORTING.get(c["scenario"], (0.8, 48))
            reported = rng.random() < p_report
            delay_h = rng.uniform(7, 25) * 24 if median_h is None else rng.lognormal(np.log(median_h), 0.7)
            cases.append({"case_id": c["case_id"], "scenario": c["scenario"],
                          "start": pd.to_datetime(c["start"], unit="s").floor("s"),
                          "end": pd.to_datetime(c["end"], unit="s").floor("s"),
                          "reported_at": pd.to_datetime(c["end"] + delay_h * H, unit="s").floor("s")
                          if reported else pd.NaT,
                          "n_tx": c["n_tx"], "loss_fcfa": c["loss"]})
        cases = pd.DataFrame(cases)

        customers = pd.DataFrame([{
            # numéro pseudonymisé : un numéro aléatoire pourrait appartenir à une vraie personne
            "account_id": a.id, "msisdn_hash": hashlib.sha256(self._msisdn().encode()).hexdigest()[:16],
            "segment": a.segment, "region": a.region,
            "kyc_level": a.kyc, "created_at": pd.to_datetime(a.created, unit="s").floor("s"),
            "opening_balance": a.opening,
        } for a in self.acc.values() if a.kind == "CLIENT"])
        merchants = pd.DataFrame([{
            "merchant_id": a.id, "category": a.category, "region": a.region,
            "created_at": pd.to_datetime(a.created, unit="s").floor("s"), "opening_balance": a.opening,
        } for a in self.acc.values() if a.kind == "MARCHAND"])
        agents = pd.DataFrame(self.agents)
        agents["created_at"] = pd.to_datetime(agents.created_at, unit="s").dt.floor("s")
        ev = pd.DataFrame(self.events)
        ev = ev[(ev.ts >= self.t_start) & (ev.ts < self.t_end)].copy()
        ev.insert(1, "timestamp", pd.to_datetime(ev.ts, unit="s").dt.floor("s"))
        ev = ev.drop(columns=["ts"]).sort_values("timestamp", kind="stable").reset_index(drop=True)
        return SimulatedData(tx, customers, merchants, agents, ev, cases)


def simulate(seed: int = 42, **kwargs) -> SimulatedData:
    return DoxalSimulator(seed=seed, **kwargs).run()
