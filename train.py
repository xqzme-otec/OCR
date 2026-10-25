"""Обучение модели уверенности и оценка по кросс-валидации на train.

python train.py   ->  models/model.pkl + отчёт о качестве в консоль
"""
import argparse
import os
import pickle
import random

import numpy as np

from pp.common import FIELDS, NAME_FIELDS, load_jsonl, norm, write_jsonl
from pp.model import ConfidenceModel, featurize, gate
from pp.resolve import SRC_CODES
from pp.text import build_vocab, distance, load_reference
from pp.validity import validity

HERE = os.path.dirname(os.path.abspath(__file__))
N_FOLDS = 5
# Порог P(документ верен). Выбран по OOF: выше него доля ошибок ровная (~1%, в основном шум разметки
# ФИО, который не виден во входных данных), ниже — быстро растёт: 3% в [0.99, 0.995), 6% в [0.98, 0.99).
THRESHOLD = 0.995
BINS = [1.01, 0.9995, 0.999, 0.998, 0.995, 0.99, 0.98, 0.95, 0.9, 0.5, 0.0]


def doc_ok(rec, label):
    """Документ принят верно: все поля совпали и паспорт действителен (по разметке)."""
    if validity(label["Birth_date"], label["Issue_date"]) != "valid":
        return False
    return all(rec["fields"][f] is not None and norm(rec["fields"][f]) == norm(label[f]) for f in FIELDS)


def load_labels(path):
    return {d["doc_id"]: d["fields"] for d in load_jsonl(path)}


TYPO_MIN_CONF = 0.99


def fix_label_typos(recs, labels):
    """Явные опечатки разметки ФИО -> разметка для обучения.

    Визуальная зона и MRZ совпали, уверенность OCR по всем буквам >= 0.99, а разметка отличается на один
    символ (ЕВГЕНИЙ / EVGENIQ при разметке ЕВТЕНИЙ). MRZ генерируется из истинного значения: опечатки
    самого документа (ММИХАЙЛОВИЧ) есть и в MRZ. Качество и прогноз считаются по исходной разметке.
    """
    fixed = {d: dict(f) for d, f in labels.items()}
    n = 0
    for r in recs:
        for f in NAME_FIELDS:
            v, lab, x = r["fields"][f], labels[r["doc_id"]][f], r["x"][f]
            if (v and x["src"] == SRC_CODES["agree"] and x["min_conf"] >= TYPO_MIN_CONF
                    and distance(norm(v), norm(lab)) == 1):
                fixed[r["doc_id"]][f] = v
                n += 1
    return fixed, n


def make_folds(ids, seed=0):
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    return {d: i % N_FOLDS for i, d in enumerate(ids)}


def oof_features(preds, labels, ref, folds):
    """Признаки train так, как они выглядели бы на test: словарь без разметки своего фолда."""
    vocabs = {k: build_vocab(ref, [labels[d] for d in labels if folds[d] != k]) for k in range(N_FOLDS)}
    return featurize(preds, lambda doc_id: vocabs[folds[doc_id]])


def oof_scores(recs, labels, folds):
    score = np.zeros(len(recs))
    for k in range(N_FOLDS):
        tr = [r for r in recs if folds[r["doc_id"]] != k]
        idx = [i for i, r in enumerate(recs) if folds[r["doc_id"]] == k]
        s, _ = ConfidenceModel().fit(tr, labels).doc_score([recs[i] for i in idx])
        score[idx] = s
    return score


def report(score, ok, src, thr):
    print(f"\nдоля ошибок по интервалам P(документ верен), OOF:")
    for hi, lo in zip(BINS, BINS[1:]):
        sel = (score >= lo) & (score < hi)
        if sel.sum():
            print(f"  [{lo:<6}, {min(hi, 1):<6})  документов {sel.sum():>5}  ошибок {(sel & ~ok).sum():>4}"
                  f"  ({(sel & ~ok).sum() / sel.sum():.1%})")
    acc = score >= thr
    print(f"\nпорог {thr}: принято {acc.sum()} из {len(acc)} ({acc.mean():.1%}), "
          f"из них с ошибкой {(acc & ~ok).sum()} ({(acc & ~ok).sum() / max(acc.sum(), 1):.2%})")
    for s in sorted(set(src)):
        m = src == s
        print(f"  {s:<10} принято {(acc & m).sum():>5} из {m.sum():>5} ({(acc & m).sum() / m.sum():.1%}),"
              f" ошибок {(acc & m & ~ok).sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=os.path.join(HERE, "data", "train_predictions.jsonl"))
    ap.add_argument("--labels", default=os.path.join(HERE, "data", "train_labels.jsonl"))
    ap.add_argument("--ref", default=os.path.join(HERE, "reference", "subdivisions.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "models", "model.pkl"))
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--oof-answers", help="записать OOF-ответы по train (для evaluate.py)")
    args = ap.parse_args()

    preds = load_jsonl(args.preds)
    raw = load_labels(args.labels)  # как у проверяющего: по ней качество и прогноз
    ref = load_reference(args.ref)
    folds = make_folds(raw)

    recs = oof_features(preds, raw, ref, folds)
    labels, n_fixed = fix_label_typos(recs, raw)  # для обучения
    print(f"исправлено опечаток разметки ФИО: {n_fixed}")
    ok = np.array([doc_ok(r, raw[r["doc_id"]]) for r in recs])
    ok_fixed = np.array([doc_ok(r, labels[r["doc_id"]]) for r in recs])
    src = np.array([r["source"] for r in recs])
    n_gate = sum(not gate(r) for r in recs)
    print(f"документов {len(recs)}; прошли жёсткие правила {n_gate}; "
          f"верны и действительны после исправлений {ok.sum()}")

    score = oof_scores(recs, labels, folds)
    report(score, ok, src, args.threshold)
    acc = score >= args.threshold
    print(f"  по исправленной разметке ошибок {(acc & ~ok_fixed).sum()}; "
          f"остальные {(acc & ~ok).sum() - (acc & ~ok_fixed).sum()} — опечатки разметки")

    if args.oof_answers:
        write_jsonl(args.oof_answers, [
            {"doc_id": r["doc_id"], "decision": "accept" if s >= args.threshold else "reject",
             "fields": r["fields"] if s >= args.threshold else {}} for r, s in zip(recs, score)])

    model = ConfidenceModel().fit(recs, labels)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as fh:
        # только деревья и порог: словарь строится при запуске из data/train_labels.jsonl, данных в модели нет
        pickle.dump({"model": model, "threshold": args.threshold}, fh)
    print(f"\nмодель сохранена в {args.out}")


if __name__ == "__main__":
    main()
