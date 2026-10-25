"""Подбор гиперпараметров бустинга по OOF на train.

python tune.py [--n 40]

Метрика — сумма ошибок среди N самых уверенных документов, N = 1500..1900 (по исправленной разметке:
опечатки разметки не предсказуемы и одинаково мешают всем конфигурациям). Каждая конфигурация считается
на двух разбиениях на фолды, чтобы не выбрать случайно удачную.
"""
import argparse
import os
import random

os.environ.setdefault("OMP_NUM_THREADS", "1")  # параллелим по конфигурациям, а не внутри бустинга

import numpy as np
from joblib import Parallel, delayed

from pp.common import load_jsonl
from pp.model import ConfidenceModel
from pp.text import load_reference
from train import HERE, N_FOLDS, doc_ok, fix_label_typos, load_labels, make_folds, oof_features

SIZES = [1500, 1600, 1700, 1800, 1900]
GRID = dict(learning_rate=[0.03, 0.05, 0.1], max_iter=[100, 300, 600], max_leaf_nodes=[7, 15, 31],
            min_samples_leaf=[10, 20, 50], l2_regularization=[0.0, 1.0, 10.0])
DEFAULT = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0)


def oof_scores(recs, labels, folds, params):
    score = np.zeros(len(recs))
    for k in range(N_FOLDS):
        tr = [r for r in recs if folds[r["doc_id"]] != k]
        idx = [i for i, r in enumerate(recs) if folds[r["doc_id"]] == k]
        score[idx], _ = ConfidenceModel(**params).fit(tr, labels).doc_score([recs[i] for i in idx])
    return score


def evaluate(params, splits):
    res = []
    for recs, folds, labels, ok in splits:
        s = oof_scores(recs, labels, folds, params)
        order = np.argsort(-s, kind="stable")
        res.append([int((~ok[order[:n]]).sum()) for n in SIZES] + [int((s >= 0.995).sum()), int(((s >= 0.995) & ~ok).sum())])
    return params, np.mean(res, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="сколько случайных конфигураций из сетки")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()

    preds = load_jsonl(os.path.join(HERE, "data", "train_predictions.jsonl"))
    raw = load_labels(os.path.join(HERE, "data", "train_labels.jsonl"))
    ref = load_reference(os.path.join(HERE, "reference", "subdivisions.csv"))
    splits = []
    for seed in args.seeds:
        folds = make_folds(raw, seed)
        recs = oof_features(preds, raw, ref, folds)
        labels, _ = fix_label_typos(recs, raw)
        splits.append((recs, folds, labels, np.array([doc_ok(r, labels[r["doc_id"]]) for r in recs])))

    keys = sorted(GRID)
    full = [dict(zip(keys, vals)) for vals in np.array(np.meshgrid(*[GRID[k] for k in keys])).T.reshape(-1, len(keys))]
    full = [{k: (int(v) if k in ("max_iter", "max_leaf_nodes", "min_samples_leaf") else float(v)) for k, v in p.items()}
            for p in full]
    configs = [DEFAULT] + random.Random(0).sample(full, args.n)

    results = Parallel(n_jobs=-1)(delayed(evaluate)(p, splits) for p in configs)
    results.sort(key=lambda t: t[1][:len(SIZES)].sum())
    print("ошибок среди топ-N (среднее по разбиениям), исправленная разметка:")
    print(f"  {'сумма':>6}" + "".join(f"{n:>6}" for n in SIZES) + f"  {'принято@0.995':>14} {'ош.':>4}  параметры")
    for p, m in results[:15] + [r for r in results if r[0] == DEFAULT][:1]:
        tag = "  <- текущие" if p == DEFAULT else ""
        print(f"  {m[:len(SIZES)].sum():>6.1f}" + "".join(f"{x:>6.1f}" for x in m[:len(SIZES)])
              + f"  {m[-2]:>14.0f} {m[-1]:>4.1f}  {p}{tag}")


if __name__ == "__main__":
    main()
