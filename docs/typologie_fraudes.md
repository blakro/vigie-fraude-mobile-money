# Typologie des fraudes simulées

Chaque scénario est simulé dans `vigie/data/simulator.py` et détecté par un ou plusieurs contrôles de
`vigie/engine/checks.py`. Les scénarios s'inspirent de schémas connus du mobile money en Afrique de
l'Ouest ; ils ne décrivent aucun cas réel.

Chaque fraude a un **identifiant de cas** (`fraud_case_id`), un **préjudice** en FCFA (`loss_fcfa`) et,
le plus souvent, une **date de signalement** (table `cases`). Les tentatives échouées (solde insuffisant,
plafond dépassé) restent étiquetées comme fraudes : ce sont des tentatives.

## Pourquoi adapter la logique M-Pesa ?

| Portefeuille USSD lié à la SIM (type M-Pesa) | Portefeuille par application (Doxal Money) | Conséquence pour la détection |
|---|---|---|
| Le « SIM swap » est le signal clé de prise de contrôle | Le compte vit dans une **application** | On suit les **nouveaux appareils** et les **réinitialisations du PIN** |
| Frais de transfert par paliers | **1 %** sur les transferts | Le fractionnement ne fait pas économiser de frais : inutile de le surveiller |
| Dépôts et retraits payants | Dépôts et retraits **gratuits**, agent **commissionné** | Un agent peut simuler des dépôts/retraits pour toucher des commissions |
| Paiement marchand | Paiement **QR**, gratuit pour le client | Un compte marchand peut servir à encaisser de fausses ventes |

## Les six scénarios

### 1. Prise de contrôle de compte — `PRISE_CONTROLE_COMPTE`
- **Schéma** : le fraudeur obtient l'accès (code soutiré par téléphone, téléphone volé), **réinitialise
  le PIN** et se connecte depuis **son appareil**. Il fait parfois un petit **transfert test**, puis
  **vide le portefeuille** vers une ou deux mules en quelques minutes. Les appareils des fraudeurs servent
  sur plusieurs comptes. Une partie des tentatives échoue sur les plafonds.
- **Préjudice** : montant transféré + frais.
- **Signaux** : `nouvel_appareil`, `reinit_pin`, `vidage_compte`, `nouveau_beneficiaire`, `nuit`,
  `echecs_repetes`, `region_inhabituelle`.

### 2. Arnaque « faux service client » — `ARNAQUE_INGENIERIE_SOCIALE`
- **Schéma** : la victime est convaincue par téléphone d'envoyer de l'argent. Elle utilise **son propre
  téléphone** : aucun signal technique. Si son solde ne suffit pas, elle **va d'abord déposer** chez un
  agent. De 4 à 8 victimes envoient à la **même mule** en 6 à 30 heures.
- **Préjudice** : montant transféré + frais.
- **Signaux** : `depot_puis_envoi`, `beneficiaire_recent`, `nouveau_beneficiaire`, `montant_anormal`,
  `mule_collecte` (5 expéditeurs ou plus en 24 h).
- **C'est le scénario le plus difficile** : les premiers envois arrivent avant que la mule soit repérée.

### 3. Retrait par compte mule — `MULE_RETRAIT`
- **Schéma** : la mule retire l'argent reçu chez des agents de sa zone, en plusieurs fois (les agents
  ont peu de liquidités, et le plafond d'un compte en KYC allégé l'oblige à vider son solde souvent).
  Près d'une mule sur deux est un **compte loué** (client existant, avec un historique normal) ; l'autre est un
  **compte récent** ouvert en KYC allégé, avec une petite activité de couverture.
- **Préjudice** : 0 (déjà compté chez la victime). Le retrait est rattaché au cas d'origine.
- **Signal** : `mule_retrait` — sortie d'au moins 30 % des fonds reçus en 24 h sur un compte récent,
  ou d'au moins 70 % reçus de 3 expéditeurs ou plus.

### 4. Cycles dépôt/retrait d'agent — `AGENT_CYCLES`
- **Schéma** : un agent enchaîne avec 2 à 4 complices des **dépôts puis retraits du même montant**
  (10 000 à 50 000 FCFA) pour toucher deux fois la commission.
- **Préjudice** : les commissions versées à tort (0,4 % par opération).
- **Signal** : `cycles_agent` — plus de 4 opérations entre l'agent et un même client en 24 h, ou au moins
  3 cycles chez cet agent en 24 h (un cycle = un dépôt puis un retrait du même montant, par le même client,
  à moins d'une heure d'intervalle).
- **Signalement** : par l'audit interne des commissions, 7 à 25 jours plus tard.

### 5. Faux marchand (vente fictive) — `FAUX_MARCHAND`
- **Schéma** : un compte marchand ouvert depuis quelques jours (déclaré « boutique » ou « e-commerce »)
  encaisse en 1 à 2 jours 12 à 30 paiements de 20 000 à 120 000 FCFA, venus d'acheteurs de tout le pays
  (fausse vente sur les réseaux sociaux). Il retire l'argent dès que son solde dépasse 150 000 FCFA.
- **Préjudice** : montant payé par les acheteurs.
- **Signaux** : `marchand_afflux` (marchand récent, 8 payeurs différents ou plus en 24 h),
  `marchand_atypique` (panier anormal pour la catégorie), puis `mule_retrait` sur les retraits.

### 6. Rafale d'achats de crédit — `RAFALE_CREDIT`
- **Schéma** : un compte compromis (souvent depuis un nouvel appareil) achète du crédit téléphonique
  toutes les 20 à 90 secondes (8 à 20 achats au plus), jusqu'à épuisement du solde, pour le revendre.
- **Préjudice** : montant du crédit acheté.
- **Signal** : `rafale_credit` — plus de 6 achats de crédit en 30 minutes.

## Comportements normaux qui ressemblent à de la fraude

Pour éviter un problème trop facile, la simulation contient aussi du bruit légitime :

- **4 % des clients changent de téléphone** pendant la période, dont 35 % avec réinitialisation du PIN ;
- les **familles rurales** reçoivent de plusieurs proches en fin de mois et **retirent presque tout** ;
- les **commerçants** font beaucoup de dépôts et retraits avec le même agent ;
- **1 % des transferts** sont de gros montants légitimes (loyer, cérémonie) ;
- **40 % des transferts, dépôts et retraits** au-delà de 10 000 FCFA ont un montant rond ;
- autour du **Grand Magal de Touba** (2 août 2026), des pèlerins font leurs opérations dans la région de Diourbel ;
- environ **5 % des opérations échouent** (solde insuffisant, plafond atteint).
