# Fiche modèle — Vigie (Doxal Money, opérateur fictif)

Générée le 2026-10-14. Toutes les données sont **synthétiques**.

## Données
- Entraînement : 129 748 opérations jusqu'au 2026-08-10 (taux de fraude 0,6 %).
- Étiquettes disponibles à la date de coupure : 665 opérations frauduleuses signalées sur 781 (85,1 %). Les autres fraudes, signalées plus tard ou jamais, comptent comme normales à l'entraînement.
- Seuil d'alerte du modèle : probabilité calibrée ≥ 20 %, fixé à l'avance (une chance sur cinq d'être une fraude).
- Sur la validation (10 jours avant la coupure), la précision du modèle mesurée avec les seules fraudes déjà signalées est de 58,5 %, proche de la réalité (58,5 %) : les fraudes de cette période étaient déjà signalées.
- Test : 69 804 opérations sur les 20 jours suivants (taux de fraude 0,5 %), évaluées avec la vérité terrain de la simulation.

## Modèles

| Modèle | PR-AUC | ROC-AUC |
|---|---|---|
| Gradient boosting calibré | 0,889 | 0,996 |
| Isolation Forest (sans étiquettes) | 0,350 | 0,916 |

## Opérations (période de test)

| Approche | Précision | Rappel | Alertes / jour | Fausses alertes / jour |
|---|---|---|---|---|
| Règles seules | 58,5 % | 55,2 % | 15,8 | 6,5 |
| ML seul (p ≥ 0,20) | 86,5 % | 80,3 % | 15,6 | 2,1 |
| Moteur combiné | 63,9 % | 88,4 % | 23,1 | 8,3 |

## Cas de fraude et préjudice (moteur combiné, période de test)

- Cas repérés : **28 sur 28** (100,0 %).
- Dans au moins la moitié des cas, la toute première opération frauduleuse est déjà alertée.
- Préjudice porté par des opérations alertées : 4 703 195 FCFA sur 5 968 495 (78,8 %), dont 1 250 445 FCFA sur des opérations bloquées.

| Scénario | Opérations | Rappel règles | Rappel ML | Rappel combiné | Cas repérés | Préjudice alerté |
|---|---|---|---|---|---|---|
| Cycles dépôt/retrait d'agent | 78 | 32,0 % | 94,9 % | 96,2 % | 2/2 | 96,5 % |
| Arnaque « faux service client » | 27 | 33,3 % | 14,8 % | 37,0 % | 5/5 | 41,3 % |
| Faux marchand (vente fictive) | 92 | 76,1 % | 92,4 % | 93,5 % | 3/3 | 90,7 % |
| Retrait par compte mule | 31 | 12,9 % | 71,0 % | 71,0 % | — | — |
| Prise de contrôle de compte | 25 | 76,0 % | 96,0 % | 100,0 % | 12/12 | 100,0 % |
| Rafale d'achats de crédit | 82 | 70,7 % | 73,2 % | 95,1 % | 6/6 | 94,0 % |

Les retraits de mules sont rattachés au cas qui les a alimentés ; leur préjudice est déjà compté chez la victime.

## Variables les plus utiles (importance par permutation, PR-AUC)

- `agent_roundtrips_24h` : 0,1934
- `hours_since_device_event` : 0,1218
- `amount` : 0,0577
- `history_n_type` : 0,0509
- `inbound_amount_24h` : 0,0467
- `share_of_balance` : 0,0431
- `is_round_amount` : 0,0416
- `receiver_age_days` : 0,0368
- `n_tx_1h` : 0,0201
- `receiver_n_senders_24h` : 0,0173

## Limites
- Les scénarios de fraude ont été simulés par l'auteur du détecteur : les scores sont **optimistes** et ne préjugent pas des performances sur des données réelles.
- Le moteur est évalué en mode observation : en production, un blocage arrêterait la suite d'une attaque et changerait les chiffres.
- Les premières opérations d'une rafale ou d'un cycle d'agent passent sous les seuils : la détection arrive après quelques opérations.
- Aucun signal téléphonique (appel reçu juste avant un envoi) : c'est pourtant le meilleur indice contre l'ingénierie sociale.
- Doxal Money est un opérateur fictif ; sa grille tarifaire et ses plafonds sont des hypothèses.
