"""Configuration centrale de Vigie.

Doxal Money est un opérateur de mobile money FICTIF. Sa grille tarifaire, ses plafonds et ses
commissions ci-dessous sont des hypothèses de simulation, plausibles pour un portefeuille
mobile en Afrique de l'Ouest, mais ils ne décrivent aucun opérateur réel.

Tous les seuils du moteur de détection sont aussi réunis ici, pour pouvoir les ajuster
sans toucher au code des contrôles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

OPERATOR_NAME = "Doxal Money"   # opérateur fictif
TXN_PREFIX = "DX"               # préfixe des références de transaction

# ---------------------------------------------------------------------------
# Grille tarifaire fictive
# ---------------------------------------------------------------------------
FEE_RATE_TRANSFER = 0.01        # 1 % payé par l'expéditeur d'un transfert
FEE_RATE_MERCHANT = 0.01        # 1 % retenu sur l'encaissement du marchand
AGENT_COMMISSION_RATE = 0.004   # 0,4 % versé à l'agent sur chaque dépôt ou retrait
# Dépôts, retraits, factures, crédit et paiements marchands : gratuits pour le client.

# ---------------------------------------------------------------------------
# Plafonds fictifs par niveau de vérification d'identité (KYC)
#   1 = KYC allégé (numéro + pièce déclarée), 2 = KYC complet, MARCHAND = compte marchand
# ---------------------------------------------------------------------------
LIMITS = {
    1: {"balance_max": 200_000, "txn_max": 100_000},
    2: {"balance_max": 2_000_000, "txn_max": 1_000_000},
    "MARCHAND": {"balance_max": 10_000_000, "txn_max": 3_000_000},
}

TXN_TYPES = [
    "DEPOT",              # cash-in chez un agent
    "RETRAIT",            # cash-out chez un agent
    "TRANSFERT",          # P2P entre deux portefeuilles
    "PAIEMENT_MARCHAND",  # paiement par QR code chez un marchand
    "FACTURE",            # Senelec, Woyofal, Sen'Eau, Canal+
    "CREDIT",             # achat de crédit téléphonique
    "EPARGNE",            # versement vers le compte épargne du client
    "VIREMENT_BANCAIRE",  # portefeuille -> compte bancaire (surtout les marchands)
]
OUTFLOW_TYPES = {"RETRAIT", "TRANSFERT", "PAIEMENT_MARCHAND", "FACTURE", "CREDIT", "EPARGNE",
                 "VIREMENT_BANCAIRE"}
CASHOUT_TYPES = {"RETRAIT", "VIREMENT_BANCAIRE"}

SIM_START = datetime(2026, 7, 1)
SIM_DAYS = 60
TRAIN_DAYS = 40   # découpage temporel : 40 jours d'entraînement, 20 jours de test

# Grand Magal de Touba : célébré le dimanche 2 août 2026 (source : Le Soleil, Dakaractu)
MAGAL_DATE = date(2026, 8, 2)


@dataclass
class FraudConfig:
    """Seuils des contrôles et pondération de la décision."""

    # Vélocité
    velocity_window_min: int = 60
    velocity_max_tx: int = 8

    # Nouvel appareil / réinitialisation du PIN (l'équivalent du "SIM swap" pour une application)
    device_risk_window_h: float = 24.0
    pin_reset_window_h: float = 24.0

    # Heures de nuit (heure de Dakar = UTC)
    night_start_hour: int = 0
    night_end_hour: int = 5

    # Montant anormal par rapport à l'historique du compte (z-score sur le log du montant)
    amount_z_threshold: float = 3.0
    min_history_for_z: int = 5

    # Vidage du portefeuille
    drain_share: float = 0.9
    drain_min_amount: int = 20_000

    # Tentatives échouées répétées
    failed_window_min: int = 60
    failed_max: int = 2

    # Dépôt suivi d'un envoi vers un inconnu (schéma typique d'arnaque guidée par téléphone)
    deposit_then_send_min: int = 30

    # Comptes mules
    mule_window_h: float = 24.0
    mule_min_distinct_senders: int = 5
    mule_cashout_ratio: float = 0.7
    young_account_days: int = 30

    # Cycles agent (dépôts/retraits répétés avec les mêmes clients)
    agent_pair_window_h: float = 24.0
    agent_pair_max_tx: int = 4
    agent_roundtrips_max: int = 3

    # Marchands
    merchant_amount_z_threshold: float = 3.0
    merchant_min_history: int = 10
    merchant_burst_min_payers: int = 8

    # Rafale de crédit téléphonique
    airtime_window_min: int = 30
    airtime_max_tx: int = 6

    # Poids de chaque contrôle dans le score règles (0-100)
    weights: dict = field(default_factory=lambda: {
        "velocite": 20,
        "nouvel_appareil": 30,
        "reinit_pin": 25,
        "nuit": 10,
        "montant_anormal": 25,
        "nouveau_beneficiaire": 10,
        "vidage_compte": 20,
        "echecs_repetes": 15,
        "region_inhabituelle": 10,
        "depot_puis_envoi": 15,
        "beneficiaire_recent": 15,
        "mule_collecte": 35,
        "mule_retrait": 45,
        "cycles_agent": 45,
        "marchand_atypique": 30,
        "marchand_afflux": 40,
        "rafale_credit": 45,
    })

    # Bonus d'interaction : certaines combinaisons sont plus graves que la somme des signaux
    combo_bonus: dict = field(default_factory=lambda: {
        ("nouvel_appareil", "reinit_pin"): 20,
        ("nouvel_appareil", "nouveau_beneficiaire"): 10,
        ("nouvel_appareil", "vidage_compte"): 15,
        ("montant_anormal", "nouveau_beneficiaire"): 15,
        ("depot_puis_envoi", "beneficiaire_recent"): 15,
        ("marchand_afflux", "marchand_atypique"): 10,
    })

    # Mélange règles / ML et seuils de décision
    ml_weight: float = 0.5
    # Le modèle est calibré : sa sortie est une probabilité. Il envoie seul en revue dès que le risque
    # atteint 20 % (une chance sur cinq d'être une fraude, plus de 30 fois le taux de fraude moyen).
    ml_review_proba: float = 0.20
    rule_override_score: float = 85.0
    review_threshold: float = 40.0
    block_threshold: float = 70.0
