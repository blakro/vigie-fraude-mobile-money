"""Fiche modèle (Markdown) générée à chaque exécution du pipeline."""
from __future__ import annotations

from pathlib import Path

from vigie.config import OPERATOR_NAME

SCENARIO_LABELS = {
    "PRISE_CONTROLE_COMPTE": "Prise de contrôle de compte",
    "ARNAQUE_INGENIERIE_SOCIALE": "Arnaque « faux service client »",
    "MULE_RETRAIT": "Retrait par compte mule",
    "AGENT_CYCLES": "Cycles dépôt/retrait d'agent",
    "FAUX_MARCHAND": "Faux marchand (vente fictive)",
    "RAFALE_CREDIT": "Rafale d'achats de crédit",
    "NORMAL": "Normal",
}


def pct(x: float) -> str:
    return f"{x:.1%}".replace(".", ",").replace("%", " %")


def num(x) -> str:
    return f"{int(x):,}".replace(",", " ")


def dec(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def delay_sentence(median_ops) -> str:
    """Phrase lisible sur le délai de détection (médiane des opérations passées avant la 1re alerte)."""
    if median_ops is None:
        return "Aucun cas n'a déclenché d'alerte."
    if median_ops == 0:
        return "Dans au moins la moitié des cas, la toute première opération frauduleuse est déjà alertée."
    return f"La première alerte arrive après {dec(median_ops, 0)} opérations frauduleuses (médiane)."


def validation_sentence(known: float, true: float) -> str:
    """Compare la précision mesurée avec les fraudes déjà signalées à la vraie précision."""
    start = (f"Sur la validation (10 jours avant la coupure), la précision du modèle mesurée avec les seules "
             f"fraudes déjà signalées est de {pct(known)}")
    if true - known >= 0.05:
        return (f"{start}, contre {pct(true)} en réalité : une partie des alertes justes porte sur des fraudes "
                "pas encore signalées.")
    return f"{start}, proche de la réalité ({pct(true)}) : les fraudes de cette période étaient déjà signalées."


def write_model_card(metrics: dict, path: Path):
    ml, ev = metrics["ml"], metrics["evaluation_test"]
    c = ev["cas"]
    lines = [
        f"# Fiche modèle — Vigie ({OPERATOR_NAME}, opérateur fictif)",
        "",
        f"Générée le {ml['trained_at'][:10]}. Toutes les données sont **synthétiques**.",
        "",
        "## Données",
        f"- Entraînement : {num(ml['n_train'])} opérations jusqu'au {ml['train_period_end'][:10]} "
        f"(taux de fraude {pct(ml['fraud_rate_train'])}).",
        f"- Étiquettes disponibles à la date de coupure : {num(ml['n_fraud_train_known'])} opérations "
        f"frauduleuses signalées sur {num(ml['n_fraud_train_true'])} ({pct(ml['label_coverage_train'])}). "
        "Les autres fraudes, signalées plus tard ou jamais, comptent comme normales à l'entraînement.",
        f"- Seuil d'alerte du modèle : probabilité calibrée ≥ {pct(ml['ml_threshold']).replace(',0 ', ' ')}, "
        "fixé à l'avance (une chance sur cinq d'être une fraude).",
        "- " + validation_sentence(ml['validation_precision_known_labels'], ml['validation_precision_true']),
        f"- Test : {num(ml['n_test'])} opérations sur les 20 jours suivants (taux de fraude "
        f"{pct(ml['fraud_rate_test'])}), évaluées avec la vérité terrain de la simulation.",
        "",
        "## Modèles",
        "",
        "| Modèle | PR-AUC | ROC-AUC |",
        "|---|---|---|",
        f"| Gradient boosting calibré | {dec(ml['hgb_pr_auc'])} | {dec(ml['hgb_roc_auc'])} |",
        f"| Isolation Forest (sans étiquettes) | {dec(ml['iforest_pr_auc'])} | {dec(ml['iforest_roc_auc'])} |",
        "",
        "## Opérations (période de test)",
        "",
        "| Approche | Précision | Rappel | Alertes / jour | Fausses alertes / jour |",
        "|---|---|---|---|---|",
    ]
    thr = dec(ml.get("ml_threshold", 0.5), 2)
    for k, label in (("regles", "Règles seules"), ("ml", f"ML seul (p ≥ {thr})"), ("combine", "Moteur combiné")):
        e = ev[k]
        lines.append(f"| {label} | {pct(e['precision'])} | {pct(e['recall'])} | "
                     f"{dec(e['alertes_par_jour'], 1)} | {dec(e['fausses_alertes_par_jour'], 1)} |")
    lines += [
        "",
        "## Cas de fraude et préjudice (moteur combiné, période de test)",
        "",
        f"- Cas repérés : **{c['detectes']} sur {c['n']}** ({pct(c['taux_detection'])}).",
        "- " + delay_sentence(c.get("operations_avant_alerte_mediane")),
        f"- Préjudice porté par des opérations alertées : {num(c['prejudice_alerte'])} FCFA sur "
        f"{num(c['prejudice_total'])} ({pct(c['prejudice_alerte'] / max(1, c['prejudice_total']))}), "
        f"dont {num(c['prejudice_bloque'])} FCFA sur des opérations bloquées.",
        "",
        "| Scénario | Opérations | Rappel règles | Rappel ML | Rappel combiné | Cas repérés | Préjudice alerté |",
        "|---|---|---|---|---|---|---|",
    ]
    cases = {r["scenario"]: r for r in ev["cas_par_scenario"]}
    for r in ev["par_scenario"]:
        cs = cases.get(r["scenario"])
        cas_txt = f"{cs['detectes']}/{cs['cas']}" if cs else "—"
        prej = pct(cs["prejudice_alerte"] / cs["prejudice"]) if cs and cs["prejudice"] else "—"
        lines.append(f"| {SCENARIO_LABELS.get(r['scenario'], r['scenario'])} | {r['n']} | "
                     f"{pct(r['rappel_regles'])} | {pct(r['rappel_ml'])} | {pct(r['rappel_combine'])} | "
                     f"{cas_txt} | {prej} |")
    lines += ["", "Les retraits de mules sont rattachés au cas qui les a alimentés ; leur préjudice est "
              "déjà compté chez la victime.", "",
              "## Variables les plus utiles (importance par permutation, PR-AUC)", ""]
    for fi in ml["feature_importance"][:10]:
        lines.append(f"- `{fi['feature']}` : {dec(fi['importance'], 4)}")
    lines += [
        "",
        "## Limites",
        "- Les scénarios de fraude ont été simulés par l'auteur du détecteur : les scores sont "
        "**optimistes** et ne préjugent pas des performances sur des données réelles.",
        "- Le moteur est évalué en mode observation : en production, un blocage arrêterait la suite "
        "d'une attaque et changerait les chiffres.",
        "- Les premières opérations d'une rafale ou d'un cycle d'agent passent sous les seuils : "
        "la détection arrive après quelques opérations.",
        "- Aucun signal téléphonique (appel reçu juste avant un envoi) : c'est pourtant le meilleur "
        "indice contre l'ingénierie sociale.",
        f"- {OPERATOR_NAME} est un opérateur fictif ; sa grille tarifaire et ses plafonds sont des hypothèses.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
