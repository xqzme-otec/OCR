"""Постобработка OCR паспортов: python postprocess.py data/test_predictions.jsonl answers.jsonl

Для каждого документа: собрать поля (визуальная зона + MRZ + словарь), проверить жёсткие правила
и действительность, оценить P(документ верен) моделью и принять, если она не ниже порога.
Модель берётся из models/model.pkl (python train.py); если её нет — обучается на train заново.
Словарь для исправления текстов: справочник подразделений + тексты разметки data/train_labels.jsonl.
"""
import argparse
import collections
import os
import pickle
import sys

from pp.common import load_jsonl, write_jsonl
from pp.model import featurize, gate
from pp.text import build_vocab, load_reference

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(path):
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception as e:  # нет файла или несовместимая версия sklearn
        print(f"не удалось загрузить {path} ({e}); обучаю заново на train", file=sys.stderr)
        import train
        sys.argv = ["train.py", "--out", path]
        train.main()
        with open(path, "rb") as fh:
            return pickle.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("answers")
    ap.add_argument("--model", default=os.path.join(HERE, "models", "model.pkl"))
    ap.add_argument("--ref", default=os.path.join(HERE, "reference", "subdivisions.csv"))
    ap.add_argument("--train-labels", default=os.path.join(HERE, "data", "train_labels.jsonl"))
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    bundle = load_model(args.model)
    model, thr = bundle["model"], args.threshold or bundle["threshold"]
    if os.path.exists(args.train_labels):
        rows = [d["fields"] for d in load_jsonl(args.train_labels)]
    else:
        print(f"нет {args.train_labels}: словарь только из справочника, принятых будет меньше", file=sys.stderr)
        rows = []
    vocab = build_vocab(load_reference(args.ref), rows)

    docs = load_jsonl(args.predictions)
    recs = featurize(docs, lambda _: vocab)
    score, _ = model.doc_score(recs)

    out, reasons = [], collections.Counter()
    for r, s in zip(recs, score):
        g = gate(r)
        reasons[g.split(":")[0] if g else ("accept" if s >= thr else "low_score")] += 1
        if s >= thr:
            out.append({"doc_id": r["doc_id"], "decision": "accept", "fields": r["fields"]})
        else:
            out.append({"doc_id": r["doc_id"], "decision": "reject"})
    write_jsonl(args.answers, out)
    n_acc = sum(o["decision"] == "accept" for o in out)
    print(f"документов {len(out)}, принято {n_acc} ({n_acc / len(out):.1%}); порог {thr}")
    print("причины:", dict(reasons.most_common()))


if __name__ == "__main__":
    main()
