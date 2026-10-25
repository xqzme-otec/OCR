"""Сравнение моделей уверенности на одном и том же OOF-протоколе: бустинг, логрегрессия, нейросеть (MLP).

python compare_models.py
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pp.common import load_jsonl
from pp.model import ConfidenceModel
from pp.text import load_reference
from train import HERE, N_FOLDS, doc_ok, fix_label_typos, load_labels, make_folds, oof_features
import os

MODELS = {
    "boosting": lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
        l2_regularization=1.0, random_state=0),
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
    # лучший из проверенных вариантов: с early stopping и слабой регуляризацией сеть заметно хуже
    "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(32,), alpha=1e-2, max_iter=300, random_state=0)),
}


class AnyModel(ConfidenceModel):
    """ConfidenceModel с произвольным классификатором вместо бустинга."""

    def __init__(self, factory):
        super().__init__()
        self.factory = factory

    def fit(self, recs, labels):
        from pp.model import FIELDS, _matrix, gate, targets
        recs = [r for r in recs if not gate(r)]
        for f in FIELDS:
            X, self.keys[f] = _matrix(recs, f)
            y = targets(recs, labels, f)
            self.models[f] = float(y[0]) if y.min() == y.max() else self.factory().fit(X, y)
        return self


def oof(recs, labels, folds, factory):
    score = np.zeros(len(recs))
    for k in range(N_FOLDS):
        tr = [r for r in recs if folds[r["doc_id"]] != k]
        idx = [i for i, r in enumerate(recs) if folds[r["doc_id"]] == k]
        score[idx], _ = AnyModel(factory).fit(tr, labels).doc_score([recs[i] for i in idx])
    return score


def main():
    preds = load_jsonl(os.path.join(HERE, "data", "train_predictions.jsonl"))
    raw = load_labels(os.path.join(HERE, "data", "train_labels.jsonl"))
    folds = make_folds(raw)
    recs = oof_features(preds, raw, load_reference(os.path.join(HERE, "reference", "subdivisions.csv")), folds)
    labels, _ = fix_label_typos(recs, raw)  # обучение — на разметке без явных опечаток
    ok = np.array([doc_ok(r, raw[r["doc_id"]]) for r in recs])  # качество — по исходной разметке

    sizes = [1000, 1200, 1400, 1600, 1700, 1800, 1900, 2000]
    print("ошибок среди N самых уверенных документов (OOF, train 3000):")
    print(f"  {'модель':<10}" + "".join(f"{n:>7}" for n in sizes) + "   | порог 0.995: принято / ошибок")
    for name, factory in MODELS.items():
        s = oof(recs, labels, folds, factory)
        order = np.argsort(-s, kind="stable")
        errs = [int((~ok[order[:n]]).sum()) for n in sizes]
        acc = s >= 0.995
        print(f"  {name:<10}" + "".join(f"{e:>7}" for e in errs)
              + f"   | {acc.sum():>5} / {(acc & ~ok).sum()}")


if __name__ == "__main__":
    main()
