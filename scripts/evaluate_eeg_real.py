#!/usr/bin/env python3
"""
Repeated, stratified subject-level evaluation of the real EEG model.

The single GroupKFold number reported by ``train_eeg_real.py`` is one draw on
65 subjects. This script quantifies how much to trust it:

* repeated stratified group k-fold over several random subject partitions
* per-fold and per-repeat spread (not just one pooled number)
* ROC-AUC, PR-AUC, sensitivity, specificity, precision, Brier score
* bootstrap 95% CIs over subjects
* a simple logistic-regression baseline on subject-mean band power
* sex/age confound checks (sex-only and age-only baselines, within-sex AUC)

Usage:
    PYTHONPATH=. python scripts/evaluate_eeg_real.py --repeats 5 --out eval.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neurodegenerai.src.data.eeg_openneuro import build_dataset
from neurodegenerai.src.models.eeg_real import _fit_model, _subject_scores

PARTICIPANTS = Path("neurodegenerai/data/openneuro/ds004504_participants.tsv")
CI_KEYS = ("auc", "ap", "acc", "sens", "spec", "precision")


def load_data(cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (or build and cache) epoch features, labels, and subject ids."""
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["y"], z["groups"]
    ds = build_dataset(max_per_class=40)
    if ds is None:
        raise SystemExit("EEG dataset unavailable")
    np.savez(cache, X=ds["X"], y=ds["y"], groups=ds["groups"])
    return ds["X"], ds["y"], ds["groups"]


def metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "acc": float((pred == y).mean()),
        "sens": tp / max(tp + fn, 1),
        "spec": tn / max(tn + fp, 1),
        "precision": tp / max(tp + fp, 1),
        "brier": float(brier_score_loss(y, p)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def bootstrap_ci(
    y: np.ndarray, p: np.ndarray, n: int = 2000, seed: int = 0
) -> dict[str, list[float]]:
    """Percentile 95% CIs from resampling subjects with replacement."""
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in CI_KEYS}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        m = metrics(y[idx], p[idx])
        for k in CI_KEYS:
            draws[k].append(m[k])
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        for k, v in draws.items()
    }


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    subjects: list[str],
    repeats: int,
    epochs: int,
    seed: int,
) -> dict[str, np.ndarray | list[float]]:
    """Repeated stratified group k-fold; returns out-of-fold subject scores."""
    index = {s: i for i, s in enumerate(subjects)}
    subj_y = np.array([int(y[groups == s][0]) for s in subjects])
    feats = np.stack([X[groups == s].mean(axis=0).ravel() for s in subjects])

    cnn = np.full((repeats, len(subjects)), np.nan)
    lr = np.full_like(cnn, np.nan)
    fold_auc: list[float] = []

    for r in range(repeats):
        rs = seed + r
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=rs)
        for f, (tr, te) in enumerate(splitter.split(X, y, groups)):
            t0 = time.time()
            model, mean, std = _fit_model(X[tr], y[tr], epochs, rs)
            true, prob = _subject_scores(model, X[te], y[te], groups[te], mean, std)
            te_ids = [index[s] for s in sorted(set(groups[te].tolist()))]
            tr_ids = [index[s] for s in sorted(set(groups[tr].tolist()))]
            assert list(subj_y[te_ids]) == true
            cnn[r, te_ids] = prob
            if len(set(true)) == 2:
                fold_auc.append(float(roc_auc_score(true, prob)))

            clf = make_pipeline(
                StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)
            )
            clf.fit(feats[tr_ids], subj_y[tr_ids])
            lr[r, te_ids] = clf.predict_proba(feats[te_ids])[:, 1]
            print(
                f"repeat {r + 1}/{repeats} fold {f + 1}/5 "
                f"test_subjects={len(te_ids)} fold_auc={fold_auc[-1]:.3f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    return {"cnn": cnn, "lr": lr, "fold_auc": fold_auc, "subj_y": subj_y}


def spread(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
    }


def confound_report(
    subj_y: np.ndarray, score: np.ndarray, subjects: list[str]
) -> dict[str, object]:
    info = pd.read_csv(PARTICIPANTS, sep="\t").set_index("participant_id")
    male = np.array([info.loc[s, "Gender"] == "M" for s in subjects])
    age = np.array([float(info.loc[s, "Age"]) for s in subjects])

    def within(mask: np.ndarray) -> dict[str, object]:
        if len(set(subj_y[mask].tolist())) < 2:
            return {"n": int(mask.sum()), "auc": None}
        ci = bootstrap_ci(subj_y[mask], score[mask], n=1000)["auc"]
        return {
            "n": int(mask.sum()),
            "n_ad": int(subj_y[mask].sum()),
            "auc": float(roc_auc_score(subj_y[mask], score[mask])),
            "auc_ci95": ci,
        }

    return {
        "sex_only_auc_female_predicts_ad": float(roc_auc_score(subj_y, ~male)),
        "age_only_auc_older_predicts_ad": float(roc_auc_score(subj_y, age)),
        "within_male": within(male),
        "within_female": within(~male),
        "mean_score_by_group_sex": {
            f"{'AD' if c else 'control'}_{'male' if m else 'female'}": float(
                score[(subj_y == c) & (male == m)].mean()
            )
            for c in (1, 0)
            for m in (True, False)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--cache", type=Path, default=Path("eeg_features_cache.npz"))
    ap.add_argument("--out", type=Path, default=Path("eeg_eval.json"))
    args = ap.parse_args()

    X, y, groups = load_data(args.cache)
    subjects = sorted(set(groups.tolist()))
    print(f"subjects={len(subjects)} epochs={len(X)}", flush=True)

    cv = cross_validate(X, y, groups, subjects, args.repeats, args.epochs, args.seed)
    subj_y = cv["subj_y"]
    report: dict[str, object] = {"n_subjects": len(subjects), "repeats": args.repeats}

    for name in ("cnn", "lr"):
        mat = cv[name]
        per_repeat = [metrics(subj_y, mat[r]) for r in range(args.repeats)]
        avg_score = mat.mean(axis=0)  # out-of-fold score averaged over repeats
        report[name] = {
            "per_repeat_auc": spread([m["auc"] for m in per_repeat]),
            "per_repeat_ap": spread([m["ap"] for m in per_repeat]),
            "per_repeat_acc": spread([m["acc"] for m in per_repeat]),
            "per_repeat_sens": spread([m["sens"] for m in per_repeat]),
            "per_repeat_spec": spread([m["spec"] for m in per_repeat]),
            "averaged_oof": metrics(subj_y, avg_score),
            "averaged_oof_ci95": bootstrap_ci(subj_y, avg_score),
        }
        if name == "cnn":
            report["confounds"] = confound_report(subj_y, avg_score, subjects)

    report["cnn_per_fold_auc"] = spread(cv["fold_auc"])
    report["prevalence_ad"] = float(subj_y.mean())
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
