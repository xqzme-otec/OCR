"""Модель уверенности: P(поле верно) по признакам, P(документ верен) = произведение по полям."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .common import FIELDS, NAME_FIELDS, norm
from .resolve import SRC_CODES, resolve

def featurize(docs, vocab_for):
    """docs -> список записей {doc_id, source, fields, x: {поле: вектор}, validity}.

    vocab_for(doc_id) -> словарь для исправления текстов (на train — без разметки своего фолда).
    """
    out = []
    for d in docs:
        fields, feats, doc_feats = resolve(d, vocab_for(d["doc_id"]))
        status = doc_feats.pop("validity")
        x = {}
        for f in FIELDS:
            row = dict(feats[f])
            row.update(doc_feats)
            x[f] = row
        out.append({"doc_id": d["doc_id"], "source": d.get("source"), "fields": fields,
                    "x": x, "validity": status})
    return out


# Жёсткие правила (на train эти источники значений дают ошибки в 5-10% случаев):
# поля, продублированные в MRZ, принимаем только если MRZ их подтверждает.
CHECKED = {"Birth_date", "Issue_date", "Issue_organisation_code", "Licence_number"}
_OK_CHECKED = {SRC_CODES["agree"], SRC_CODES["mrz"]}
_OK_NAMES = {SRC_CODES["agree"], SRC_CODES["agree_trunc"]}
NAME_MARGIN = 0.05


def gate(rec):
    """None, если документ проходит жёсткие правила, иначе причина отказа."""
    if rec["validity"] != "valid":
        return "validity:" + rec["validity"]
    x = rec["x"]
    for f in FIELDS:
        if rec["fields"][f] is None or (rec["fields"][f] == "" and f != "Middle_name_ru"):
            return "empty:" + f
    for f in CHECKED:
        if x[f]["src"] not in _OK_CHECKED:
            return "mrz_mismatch:" + f
    for f in NAME_FIELDS:
        # ФИО: совпадение с MRZ или одно расхождение с явным перевесом уверенности одной из сторон
        if x[f]["src"] not in _OK_NAMES and not (
                x[f]["name_status"] == 1 and x[f]["name_n_ops"] == 1 and x[f]["name_margin"] >= NAME_MARGIN):
            return "name_mismatch:" + f
    if x["Sex_ru"]["votes_against"]:
        return "sex_conflict"
    # высота рамки поля больше, чем у прочитанных строк: строка потеряна (на train 72 из 72 ошибочны)
    for f in ("Birth_place_ru", "Issue_organization_ru"):
        if x[f]["missing_lines"] > 0:
            return "missing_line:" + f
    return None


def _matrix(recs, f, keys=None):
    if keys is None:
        keys = sorted(recs[0]["x"][f])
    X = np.array([[float(r["x"][f].get(k, 0.0)) for k in keys] for r in recs])
    return X, keys


def targets(recs, labels, f):
    return np.array([int(r["fields"][f] is not None and norm(r["fields"][f]) == norm(labels[r["doc_id"]][f]))
                     for r in recs])


class ConfidenceModel:
    """По модели на поле. Если на поле в обучении нет ни одной ошибки — константа."""

    def __init__(self, **params):
        self.params = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
                           l2_regularization=1.0, random_state=0)
        self.params.update(params)
        self.models, self.keys = {}, {}

    def fit(self, recs, labels):
        # учимся только на документах, прошедших жёсткие правила: решение нужно только для них
        recs = [r for r in recs if not gate(r)]
        for f in FIELDS:
            X, self.keys[f] = _matrix(recs, f)
            y = targets(recs, labels, f)
            if y.min() == y.max():
                self.models[f] = float(y[0])
            else:
                self.models[f] = HistGradientBoostingClassifier(**self.params).fit(X, y)
        return self

    def field_proba(self, recs):
        out = {}
        for f in FIELDS:
            mdl = self.models[f]
            if isinstance(mdl, float):
                out[f] = np.full(len(recs), mdl)
            else:
                X, _ = _matrix(recs, f, self.keys[f])
                out[f] = mdl.predict_proba(X)[:, 1]
        return out

    def doc_score(self, recs):
        """P(документ верен); 0, если документ не прошёл жёсткие правила (gate)."""
        fp = self.field_proba(recs)
        score = np.ones(len(recs))
        for f in FIELDS:
            score *= fp[f]
        for i, r in enumerate(recs):
            if gate(r):
                score[i] = 0.0
        return score, fp
