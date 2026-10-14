# Vigie — détection de fraude mobile money

Moteur de détection de fraude **explicable** pour un portefeuille mobile money au Sénégal :
17 contrôles métier, un modèle de gradient boosting et une décision combinée
(**ACCEPTER**, **REVUE** ou **BLOQUER**), avec une API de scoring, un tableau de bord et 39 tests.

Le moteur tourne sur 60 jours de transactions **synthétiques** d'un opérateur **fictif**,
**Doxal Money**.

> ⚠️ **Projet de démonstration.** Doxal Money n'existe pas : ses clients, sa grille tarifaire et ses
> plafonds sont simulés. Aucune donnée réelle n'est utilisée.

---

## Résultats en bref

Période de test : les 20 derniers jours, jamais vus pendant l'entraînement.

| Approche | Précision | Rappel | Alertes / jour | Fausses alertes / jour |
|---|---|---|---|---|
| Règles seules | 58,5 % | 55,2 % | 15,8 | 6,5 |
| ML seul | 86,5 % | 80,3 % | 15,6 | 2,1 |
| **Moteur combiné** | **63,9 %** | **88,4 %** | **23,1** | **8,3** |

- **28 cas de fraude sur 28** déclenchent au moins une alerte.
- **79 % du préjudice** (4,7 M FCFA sur 6,0 M) passe par des opérations alertées. Les opérations
  bloquées en représentent 21 %.
- **Règles et modèle se complètent** : le modèle apporte le plus de rappel (cycles d'agent 95 %, faux
  marchands 92 %, retraits de mules 71 %). Les règles rattrapent une partie de ce qu'il manque : sur les
  rafales de crédit, 95 % en combiné contre 73 % pour le modèle seul ; sur les arnaques, 37 % contre 15 %.
- **Point faible** : l'arnaque « faux service client » (37 % des opérations repérées). La victime
  utilise son propre téléphone, et les premiers envois partent avant que la mule soit repérée.
- **Stabilité** : sur trois graines aléatoires (42, 7 et 13), le moteur combiné garde 64 à 66 % de
  précision, 76 à 88 % de rappel et 7 à 8 fausses alertes par jour.

Détail par scénario : [reports/model_card.md](reports/model_card.md).

> Ces chiffres sont **optimistes** : les fraudes ont été simulées par l'auteur du détecteur. Ils
> montrent que la chaîne fonctionne de bout en bout, pas la performance attendue sur des données réelles.

## Pourquoi ce projet

Il s'inspire du projet [M-Pesa Fraud Anomaly Detection](https://github.com/Victor-Kipruto-Rop/mpesa_safaricom-fraud_anomaly_detection),
**adapté à un portefeuille par application** et volontairement allégé : pas de Kafka, Flink ni dbt.
Tout tourne en local, en Python, en une commande.

| Portefeuille USSD lié à la SIM | Portefeuille par application | Adaptation |
|---|---|---|
| Le **SIM swap** est le signal clé | Le compte vit dans une **application** | Suivi des **nouveaux appareils** et des **réinitialisations du PIN** |
| Dépôts et retraits payants | Dépôts et retraits **gratuits**, agents **commissionnés** | Contrôle des **cycles dépôt/retrait** d'agent |
| Paiement marchand classique | Paiement **QR**, gratuit pour le client | Contrôle des **faux marchands** |

## Ce que la simulation rend réaliste

- **Soldes et plafonds** : chaque compte a un solde qui ne devient jamais négatif, et des plafonds selon
  son niveau KYC. Un test rejoue tout le journal et vérifie chaque solde.
- **Opérations échouées** : 5 % des opérations échouent (solde insuffisant, plafond atteint). Un client à
  court d'argent va d'abord déposer chez un agent, réduit son montant ou abandonne.
- **Segments de clients** : urbains, commerçants, étudiants, familles rurales, avec leurs montants, leurs
  horaires et leurs usages.
- **Calendrier** : week-end, période de paie, et le **Grand Magal de Touba** (2 août 2026) avec un pic
  d'activité dans la région de Diourbel.
- **Agents ouverts de 7 h à 22 h** : les mules retirent dans leur zone, chez plusieurs agents.
- **Fraudes crédibles** : mules louées ou récentes, appareils de fraudeurs partagés entre comptes,
  tentatives qui échouent sur les plafonds, victimes qui déposent avant d'envoyer.
- **Signalement tardif** : une fraude n'est connue qu'après une plainte ou un audit. À la date de
  coupure, seules **85 %** des fraudes d'entraînement étaient signalées.
- **Identifiants opaques** et numéros de téléphone pseudonymisés.

Hypothèses et dictionnaire de données : [docs/simulation.md](docs/simulation.md).
Les six scénarios de fraude : [docs/typologie_fraudes.md](docs/typologie_fraudes.md).

## Comment le moteur décide

```
Opération (avant exécution) ou événement de sécurité (PIN, nouvel appareil)
        │
        ▼
┌───────────────────────┐   historique passé uniquement : fenêtres 30 min / 1 h / 24 h,
│ Feature store         │   solde, appareils, bénéficiaires, flux entrants,
│ en mémoire (45 var.)  │   paires agent-client, cycles dépôt-retrait chez l'agent
└──────────┬────────────┘
           ▼
┌───────────────────────┐      ┌───────────────────────┐
│ 17 contrôles métier   │      │ Gradient boosting     │
│ explicables           │      │ calibré               │
└──────────┬────────────┘      └──────────┬────────────┘
           └───────────────┬──────────────┘
                           ▼
            ┌──────────────────────────────┐
            │ Décision + raisons lisibles   │
            └──────────────────────────────┘
```

- **REVUE** dès que les règles **ou** le modèle le demandent.
- **BLOQUER** seulement si règles **et** modèle convergent, ou si une règle très forte se déclenche.
- Le modèle est **calibré** : sa sortie est une probabilité. Il envoie seul en revue dès que le risque
  atteint **20 %**, soit une chance sur cinq d'être une fraude. Ce seuil est fixé à l'avance : un seuil
  optimisé sur une période de validation s'est révélé instable, car les fraudes récentes n'y sont pas
  encore signalées.
- Chaque décision liste ses raisons, par exemple : *« appareil jamais vu sur ce compte ; PIN réinitialisé
  il y a 0,1 h ; l'opération vide 99 % du solde »*.

## Démarrage rapide (Windows)

Dans l'invite de commandes (cmd), depuis le dossier du projet :

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
```

- `python run_pipeline.py` simule, entraîne, score et évalue (environ 2 min).
- `streamlit run dashboard\app.py` ouvre le tableau de bord.
- `uvicorn api.app:app --reload` lance l'API sur http://127.0.0.1:8000/docs.
- `python -m pytest -q` lance les 39 tests.

Dans PowerShell, l'activation se fait avec `.venv\Scripts\Activate.ps1`. Si Windows refuse d'exécuter le
script, utiliser l'invite de commandes. Sous Linux ou macOS : `source .venv/bin/activate`.
Options : `python run_pipeline.py --clients 8000 --seed 7`.

## API

Le système central appelle `/score` **avant** d'exécuter une opération, en transmettant le solde du
compte. Exemple de prise de contrôle de compte :

```bash
URL=http://127.0.0.1:8000
H="Content-Type: application/json"

# 1. Opération normale depuis le téléphone habituel  -> ACCEPTER
curl -X POST $URL/score -H "$H" -d '{"timestamp":"2026-08-30T01:00:00","txn_type":"TRANSFERT",
  "account_id":"CHPWYXVR","counterparty_id":"CB2HT9KD","counterparty_type":"USER",
  "amount":5000,"balance_before":152300,"device_id":"5e33124b8a19","region":"Dakar"}'

# 2. Réinitialisation du PIN puis connexion sur un autre téléphone
curl -X POST $URL/events -H "$H" -d '{"account_id":"CHPWYXVR","timestamp":"2026-08-30T02:10:00",
  "event_type":"REINIT_PIN","device_id":"9f3c2a71be04"}'
curl -X POST $URL/events -H "$H" -d '{"account_id":"CHPWYXVR","timestamp":"2026-08-30T02:11:00",
  "event_type":"CONNEXION_NOUVEL_APPAREIL","device_id":"9f3c2a71be04"}'

# 3. Transfert de presque tout le solde vers un inconnu, à 2 h du matin  -> BLOQUER
curl -X POST $URL/score -H "$H" -d '{"timestamp":"2026-08-30T02:14:00","txn_type":"TRANSFERT",
  "account_id":"CHPWYXVR","counterparty_id":"CZ8RKM3T","counterparty_type":"USER",
  "amount":145000,"balance_before":147250,"device_id":"9f3c2a71be04","region":"Dakar"}'
```

Réponse du dernier appel (extrait) :

```json
{
  "decision": "BLOQUER", "final_score": 100.0, "rule_score": 100.0, "ml_proba": 0.401,
  "signals": ["nouvel_appareil", "reinit_pin", "nuit", "nouveau_beneficiaire", "vidage_compte"],
  "reasons": ["appareil jamais vu sur ce compte", "PIN réinitialisé il y a 0,1 h",
              "opération de 145 000 FCFA à 2 h", "l'opération vide 99 % du solde",
              "premier transfert vers ce bénéficiaire",
              "le modèle estime le risque de fraude à 40 % (seuil d'alerte : 20 %)"]
}
```

Autres routes : `GET /health` et `GET /metrics` (décisions et latence moyenne).
Avec `VIGIE_WARM_START=1`, l'API recharge l'historique simulé au démarrage.

## Tableau de bord

Six onglets : vue d'ensemble, performance (règles / ML / combiné), **cas et préjudice**, file
d'alertes avec export CSV, **réseau d'une mule** et **simulateur** de décision.

Pour le publier gratuitement : sur [Streamlit Community Cloud](https://streamlit.io/cloud), connecter
ce dépôt GitHub et choisir `dashboard/app.py`. À la première ouverture, l'application lance le pipeline
toute seule (quelques minutes), puis garde les résultats en cache.

## Structure

```
vigie/
  config.py             opérateur fictif (frais, plafonds), seuils et poids des contrôles
  data/simulator.py     simulation à événements discrets : clients, soldes, calendrier, 6 fraudes
  features/store.py     feature store en ligne (45 variables)
  engine/checks.py      17 contrôles explicables
  engine/aggregator.py  score, bonus d'interaction, décision
  engine/engine.py      moteur + rejeu chronologique
  ml/train.py           gradient boosting calibré, effet du signalement tardif, Isolation Forest
  ml/evaluate.py        opérations, cas de fraude et préjudice
  reporting.py          fiche modèle
api/app.py              API FastAPI
dashboard/app.py        tableau de bord Streamlit
tests/                  39 tests (pytest), dont la cohérence du grand livre
docs/                   hypothèses de simulation et typologie des fraudes
run_pipeline.py         pipeline complet
```

Les fichiers `.parquet` et les modèles sont régénérés par `run_pipeline.py`. Un échantillon lisible
est versionné : [data/echantillon_transactions.csv](data/echantillon_transactions.csv).

## Choix techniques

- **Un seul feature store** pour l'entraînement, le scoring par lot et l'API : pas d'écart entre les
  variables d'entraînement et celles de production. Un test le vérifie.
- **Pas de fuite d'information** : chaque variable n'utilise que le passé.
- **Découpage temporel** : 40 jours d'entraînement, 20 jours de test.
- **Étiquettes incomplètes** : le modèle n'apprend qu'avec les fraudes déjà signalées.
- **Calibration** du modèle : sa probabilité est interprétable.
- **Évaluation métier** : en plus de la précision et du rappel, part des cas repérés et part du
  préjudice alerté.
- **Reproductible** : même graine, mêmes données, d'une exécution à l'autre (test dédié). Les scores du
  modèle peuvent varier un peu selon la version de scikit-learn.

## Limites et pistes

- Données simulées : à valider sur des données réelles anonymisées.
- Évaluation en **mode observation** : en production, un blocage arrêterait la suite d'une attaque.
- Pas de signal téléphonique (appel reçu juste avant un envoi), pourtant le meilleur indice contre
  l'ingénierie sociale.
- Les fraudes récentes ne sont pas encore signalées : mesurée sur des données récentes, la précision d'un
  modèle paraît plus basse qu'elle ne l'est (sur l'une des graines, 33 % mesurés contre 88 % réels). Il
  faut en tenir compte pour juger un modèle en production.
- Avec 28 cas de fraude en test, les chiffres bougent de plusieurs points d'une graine à l'autre.
- Pistes : analyse de graphe pour relier les mules, seuils par segment, retour des analystes pour
  réentraîner le modèle.

## Licence

MIT — Mohamed Assane.
