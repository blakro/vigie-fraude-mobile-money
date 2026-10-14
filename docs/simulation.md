# Simulation : hypothèses et dictionnaire de données

**Doxal Money** est un opérateur de mobile money **fictif** au Sénégal. Tous les paramètres ci-dessous
sont des **hypothèses de simulation**, plausibles pour un portefeuille mobile en Afrique de l'Ouest, et
réunis dans `vigie/config.py` et `vigie/data/simulator.py`. Ils ne décrivent aucun opérateur réel.

## Fonctionnement

Simulation **à événements discrets** : toutes les opérations prévues sont rangées dans une file triée
par date, puis exécutées dans l'ordre. Chaque exécution vérifie le **solde** et les **plafonds** du compte.
Certaines exécutions en programment d'autres (une mule qui retire après avoir été créditée, l'étape
suivante d'une attaque…).

Un test (`tests/test_simulator.py`) rejoue tout le journal depuis les soldes d'ouverture et vérifie
que chaque solde avant/après est exact : **le grand livre est cohérent**.

## Paramètres de l'opérateur fictif

| Paramètre | Valeur |
|---|---|
| Frais de transfert | 1 % payé par l'expéditeur |
| Dépôt, retrait, facture, crédit, paiement marchand | gratuits pour le client |
| Frais marchand | 1 % retenu sur l'encaissement |
| Commission agent | 0,4 % de chaque dépôt ou retrait |
| Plafonds KYC allégé (niveau 1) | solde 200 000 FCFA, opération 100 000 FCFA |
| Plafonds KYC complet (niveau 2) | solde 2 000 000 FCFA, opération 1 000 000 FCFA |
| Plafonds compte marchand | solde 10 000 000 FCFA, opération 3 000 000 FCFA |
| Heures d'ouverture des agents | 7 h – 22 h |

## Population

| Segment | Part | Opérations / jour | Montant typique | KYC complet | Usage dominant |
|---|---|---|---|---|---|
| Urbain | 50 % | 0,75 | 7 000 FCFA | 70 % | transferts, paiements, factures |
| Commerçant | 12 % | 2,0 | 20 000 FCFA | 90 % | dépôts et retraits, fournisseurs |
| Étudiant | 10 % | 0,55 | 2 500 FCFA | 50 % | crédit, paiements, argent des parents |
| Famille rurale | 28 % | 0,35 | 6 000 FCFA | 40 % | réception de transferts, retraits |

- 14 régions du Sénégal, avec une répartition approximative par segment (les familles rurales vivent
  hors de Dakar).
- 85 % des comptes existent avant la période, 15 % sont ouverts pendant ; 10 % des comptes sont
  presque dormants.
- Chaque client a des contacts, deux agents et quelques marchands habituels. Les urbains et les
  commerçants envoient aussi de l'argent à des proches (familles rurales, étudiants).

## Calendrier

- **Jour de la semaine** : samedi plus actif, dimanche plus calme.
- **Période de paie** (du 26 au 4) : activité en hausse et envois aux proches.
- **Grand Magal de Touba**, le **2 août 2026** : envois vers les proches de Touba les jours précédents,
  activité x1,6 dans la région de Diourbel du 1er au 3 août, et 20 % des clients urbains, commerçants et
  étudiants des autres régions y font leurs opérations ces jours-là.

## Étiquettes et signalement

Une fraude n'est connue que lorsqu'elle est **signalée** :

| Scénario | Part signalée | Délai typique |
|---|---|---|
| Prise de contrôle de compte | 90 % | environ 1 jour |
| Arnaque « faux service client » | 80 % | environ 2 jours |
| Faux marchand | 70 % | environ 4 jours |
| Rafale de crédit | 80 % | environ 1 jour |
| Cycles d'agent | 85 % | 7 à 25 jours (audit interne) |

Le modèle est entraîné avec les seules fraudes **signalées avant la date de coupure**. L'évaluation sur
la période de test utilise la vérité terrain de la simulation.

## Dictionnaire de données

### `transactions`
| Colonne | Description |
|---|---|
| `txn_id` | Référence de l'opération (`DX` + date + numéro du jour) |
| `timestamp` | Date et heure (heure de Dakar = UTC) |
| `txn_type` | `DEPOT`, `RETRAIT`, `TRANSFERT`, `PAIEMENT_MARCHAND`, `FACTURE`, `CREDIT`, `EPARGNE`, `VIREMENT_BANCAIRE` |
| `account_id`, `account_type` | Compte qui initie l'opération (`CLIENT` ou `MARCHAND`), identifiant pseudonymisé |
| `counterparty_id`, `counterparty_type` | Contrepartie : `USER`, `AGENT`, `MERCHANT`, `BILLER`, `TELCO`, `EPARGNE`, `BANQUE` |
| `amount`, `fee` | Montant et frais payés par le client (FCFA) |
| `merchant_fee`, `agent_commission` | Frais retenu au marchand, commission versée à l'agent |
| `balance_before`, `balance_after` | Solde du compte avant et après |
| `status`, `failure_reason` | `REUSSIE` ou `ECHOUEE` (`SOLDE_INSUFFISANT`, `PLAFOND_OPERATION`, `PLAFOND_SOLDE`, `PLAFOND_SOLDE_BENEFICIAIRE`) |
| `device_id`, `region` | Appareil utilisé (identifiant opaque), région de l'opération |
| `is_fraud`, `fraud_scenario`, `fraud_case_id`, `loss_fcfa` | **Vérité terrain** de la simulation, inconnue du moteur |

### Autres tables
- `customers` : segment, région, niveau KYC, date d'ouverture, solde d'ouverture, numéro pseudonymisé (`msisdn_hash`).
- `merchants` : catégorie, région, date d'ouverture, solde d'ouverture.
- `agents` : région, date d'ouverture.
- `events` : `REINIT_PIN` et `CONNEXION_NOUVEL_APPAREIL`, avec l'appareil concerné.
- `cases` : un cas de fraude par ligne, avec début, fin, date de signalement et préjudice.
