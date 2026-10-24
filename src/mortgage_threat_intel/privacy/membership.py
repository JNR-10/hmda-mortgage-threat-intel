"""Confidence-score black-box membership inference (J-05)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.metrics import roc_auc_score


class ProbaModel(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class MIAResult:
    attack: str
    evaluation_prior: str
    n_members_eval: int
    n_non_members_eval: int
    auc: float
    advantage: float
    precision_at_balanced: float
    auc_ci_low: float
    auc_ci_high: float
    member_nonmember_disjoint: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def confidence_scores(proba: np.ndarray) -> np.ndarray:
    """Maximum predicted class probability (confidence-score attack)."""
    proba = np.asarray(proba)
    if proba.ndim != 2:
        raise ValueError("predict_proba must return shape (n, n_classes)")
    return proba.max(axis=1)


def _balanced_eval_sets(
    member_scores: np.ndarray,
    non_member_scores: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = min(len(member_scores), len(non_member_scores))
    if n == 0:
        raise ValueError("Need at least one member and one non-member for MIA eval")
    m_idx = rng.choice(len(member_scores), size=n, replace=False)
    nm_idx = rng.choice(len(non_member_scores), size=n, replace=False)
    return member_scores[m_idx], non_member_scores[nm_idx]


def attack_metrics(
    member_scores: np.ndarray,
    non_member_scores: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int = 200,
) -> tuple[float, float, float, float, float]:
    """Return AUC, advantage, precision@balanced, AUC CI low/high under 1:1 prior."""
    m_s, nm_s = _balanced_eval_sets(member_scores, non_member_scores, seed)
    y = np.concatenate([np.ones(len(m_s)), np.zeros(len(nm_s))])
    s = np.concatenate([m_s, nm_s])

    # Higher confidence -> more likely member (classic shadow-free attack)
    auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")

    # Threshold sweep for max advantage = TPR - FPR
    thresholds = np.unique(s)
    best_adv = 0.0
    best_prec = 0.0
    for t in thresholds:
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        adv = tpr - fpr
        prec = tp / max(tp + fp, 1)
        if adv >= best_adv:
            best_adv = float(adv)
            best_prec = float(prec)

    rng = np.random.default_rng(seed + 1)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boot.append(float(roc_auc_score(y[idx], s[idx])))
    if boot:
        lo = float(np.percentile(boot, 2.5))
        hi = float(np.percentile(boot, 97.5))
    else:
        lo, hi = float("nan"), float("nan")
    return auc, best_adv, best_prec, lo, hi


def run_confidence_mia(
    model: ProbaModel,
    X_members: np.ndarray,
    X_non_members: np.ndarray,
    *,
    member_ids: frozenset[str] | set[str] | None = None,
    non_member_ids: frozenset[str] | set[str] | None = None,
    seed: int = 42,
    n_bootstrap: int = 200,
    evaluation_prior: str = "balanced_1_to_1",
) -> MIAResult:
    if member_ids is not None and non_member_ids is not None:
        disjoint = set(member_ids).isdisjoint(set(non_member_ids))
        if not disjoint:
            raise RuntimeError("MIA member and non-member id sets overlap")
    else:
        disjoint = True

    member_scores = confidence_scores(model.predict_proba(np.asarray(X_members)))
    non_member_scores = confidence_scores(model.predict_proba(np.asarray(X_non_members)))
    auc, adv, prec, lo, hi = attack_metrics(
        member_scores, non_member_scores, seed=seed, n_bootstrap=n_bootstrap
    )
    n_eval = min(len(member_scores), len(non_member_scores))
    return MIAResult(
        attack="confidence_score",
        evaluation_prior=evaluation_prior,
        n_members_eval=n_eval,
        n_non_members_eval=n_eval,
        auc=auc,
        advantage=adv,
        precision_at_balanced=prec,
        auc_ci_low=lo,
        auc_ci_high=hi,
        member_nonmember_disjoint=disjoint,
        notes=(
            "Balanced 1:1 evaluation prior is an analysis assumption, "
            "not an estimate of real-world attacker prevalence."
        ),
    )
