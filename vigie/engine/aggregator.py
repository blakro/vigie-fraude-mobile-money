"""Agrégation des signaux en un score de risque 0-100 et une décision."""
from __future__ import annotations

from dataclasses import dataclass, field

from vigie.config import FraudConfig
from vigie.engine.checks import Signal

ACCEPTER, REVUE, BLOQUER = "ACCEPTER", "REVUE", "BLOQUER"


@dataclass
class Decision:
    rule_score: float
    ml_proba: float | None
    final_score: float
    decision: str
    signals: list[Signal] = field(default_factory=list)
    ml_threshold: float = 0.5  # seuil à partir duquel le modèle envoie seul en revue

    @property
    def reasons(self) -> list[str]:
        """Raisons lisibles, des plus fortes aux plus faibles. Toute alerte en a au moins une."""
        out = [s.reason for s in sorted(self.signals, key=lambda s: -s.strength)]
        if self.ml_proba is not None and self.ml_proba >= self.ml_threshold:
            out.append(f"le modèle estime le risque de fraude à {self.ml_proba:.0%} "
                       f"(seuil d'alerte : {self.ml_threshold:.0%})".replace("%", " %"))
        return out

    def to_dict(self) -> dict:
        return {
            "rule_score": round(self.rule_score, 1),
            "ml_proba": None if self.ml_proba is None else round(self.ml_proba, 4),
            "final_score": round(self.final_score, 1),
            "decision": self.decision,
            "signals": [s.name for s in self.signals],
            "reasons": self.reasons,
        }


def rule_score(signals: list[Signal], cfg: FraudConfig) -> float:
    names = {s.name for s in signals}
    score = sum(cfg.weights.get(s.name, 0) * s.strength for s in signals)
    for combo, bonus in cfg.combo_bonus.items():
        if set(combo) <= names:
            score += bonus
    return min(100.0, score)


def decide(signals: list[Signal], ml_proba: float | None, cfg: FraudConfig) -> Decision:
    rs = rule_score(signals, cfg)
    if ml_proba is None:
        final = rs
    else:
        final = (1 - cfg.ml_weight) * rs + cfg.ml_weight * 100 * ml_proba
        # Politique de décision :
        #  - REVUE dès que les règles OU le modèle le demandent ;
        #  - BLOQUER seulement si règles ET modèle convergent (mélange >= seuil de blocage),
        #    ou si une règle très forte se déclenche (prise de contrôle évidente, par exemple).
        if ml_proba >= cfg.ml_review_proba or rs >= cfg.review_threshold:
            final = max(final, cfg.review_threshold)
        if rs >= cfg.rule_override_score:
            final = max(final, rs)
    if final >= cfg.block_threshold:
        d = BLOQUER
    elif final >= cfg.review_threshold:
        d = REVUE
    else:
        d = ACCEPTER
    return Decision(rs, ml_proba, final, d, signals, cfg.ml_review_proba)
