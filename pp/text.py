"""Исправление длинных текстовых полей (место рождения, кем выдан) по словарю слов.

Словарь: слова из справочника подразделений + слова из разметки train. Слово OCR, которого нет
в словаре, заменяем на словарное, если оно единственное на расстоянии 1 правки (для длинных
слов — 2). Исправление считаем надёжным, только если оно однозначно.
"""
import collections
import csv
import re

TOKEN_RE = re.compile(r"[А-ЯЁA-Z0-9]+")


def tokens(s):
    return TOKEN_RE.findall(s.upper())


def load_reference(path):
    by_code = collections.defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by_code[r["code"]].append(r["name"])
    return dict(by_code)


def _deletes(w, depth):
    """Все варианты слова с удалёнными не более чем depth символами (SymSpell)."""
    out, frontier = {w}, {w}
    for _ in range(depth):
        frontier = {x[:i] + x[i + 1:] for x in frontier for i in range(len(x))}
        out |= frontier
    return out


def distance(a, b):
    """Расстояние Дамерау-Левенштейна (с перестановкой соседних символов)."""
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[-1]


MAX_DIST = 2


def max_dist(w):
    return 2 if len(w) >= 8 else 1


class Vocab:
    def __init__(self, texts, ref_codes=()):
        self.ref_codes = set(ref_codes)
        self.freq = collections.Counter(t for s in texts for t in tokens(s))
        # первые/последние слова полных названий: обрезанный текст начинается/кончается иначе
        self.first = collections.Counter(tokens(s)[0] for s in texts if tokens(s))
        self.last = collections.Counter(tokens(s)[-1] for s in texts if tokens(s))
        self.index = collections.defaultdict(set)
        for w in self.freq:
            for d in _deletes(w, MAX_DIST):
                self.index[d].add(w)
        self._cache = {}

    def __contains__(self, w):
        return w in self.freq

    def candidates(self, w):
        """Словарные слова на минимальном расстоянии <= max_dist(w), по убыванию частоты."""
        if w in self._cache:
            return self._cache[w]
        md = max_dist(w)
        found = set()
        for d in _deletes(w, md):
            found |= self.index.get(d, set())
        best, cands = md + 1, []
        for c in found:
            dist = distance(w, c)
            if dist < best:
                best, cands = dist, [c]
            elif dist == best:
                cands.append(c)
        res = (sorted(cands, key=lambda c: -self.freq[c]), best if cands else 0)
        self._cache[w] = res
        return res


# Правила, отсечённые по ошибкам на train (CV):
MIN_FIX_LEN = 4      # короткие слова (ПА -> ПО/НА) исправляются неоднозначно
FREQ_RATIO = 10      # частый кандидат должен быть в 10+ раз частотнее следующего


def _risky_fix(w, cand):
    """Правка, которая обычно означает потерянную пунктуацию, а не опечатку:
    ГМОСКВЕ -> МОСКВЕ (на деле Г.МОСКВЕ), ОБЛЮ -> ОБЛ (на деле ОБЛ.)."""
    return cand == w[1:] or cand == w[:-1]


def correct(text, vocab, char_conf=None, conf_thr=1.01, min_fix_len=None, freq_ratio="default",
            check_risky=True):
    """Исправить текст по словарю. Возвращает (текст, статистика для признаков).

    Исправляем только слова, в которых минимальная уверенность OCR ниже conf_thr: редкие
    топонимы, которых нет в словаре, иначе «исправляются» в похожие частые слова.
    """
    cc = char_conf if char_conf and len(char_conf) == len(text) else [0.0] * len(text)
    stats = {"n_tok": 0, "n_oov": 0, "n_fixed": 0, "n_ambig": 0, "n_unfixable": 0, "n_oov_kept": 0,
             "max_dist": 0, "min_tok_freq": 10 ** 6, "fixed_min_conf": 1.0, "n_kept_low": 0,
             "fixed_short": 0, "n_risky": 0}
    min_fix_len = MIN_FIX_LEN if min_fix_len is None else min_fix_len
    freq_ratio = FREQ_RATIO if freq_ratio == "default" else freq_ratio
    kept = [True] * len(text)  # символы, которые остались как прочитал OCR
    out = []
    pos = 0
    for m in TOKEN_RE.finditer(text):
        w = m.group()
        out.append(text[pos:m.start()])
        pos = m.end()
        stats["n_tok"] += 1
        tok_conf = min(cc[m.start():m.end()])
        if w.isdigit() or len(w) == 1 or w in vocab:
            if w in vocab:
                stats["min_tok_freq"] = min(stats["min_tok_freq"], vocab.freq[w])
            stats["n_kept_low"] += tok_conf < 0.99
            out.append(w)
            continue
        stats["n_oov"] += 1
        cands, dist = ([], 0) if tok_conf >= conf_thr or len(w) < min_fix_len else vocab.candidates(w)
        unique = len(cands) == 1 or (len(cands) > 1 and freq_ratio is not None
                                     and vocab.freq[cands[0]] >= freq_ratio * vocab.freq[cands[1]])
        if unique and check_risky and _risky_fix(w, cands[0]):
            stats["n_risky"] += 1
            unique = False
        if unique:
            out.append(cands[0])
            stats["n_fixed"] += 1
            stats["max_dist"] = max(stats["max_dist"], dist)
            stats["min_tok_freq"] = min(stats["min_tok_freq"], vocab.freq[cands[0]])
            stats["fixed_min_conf"] = min(stats["fixed_min_conf"], tok_conf)
            stats["fixed_short"] += len(w) <= 3
            kept[m.start():m.end()] = [False] * len(w)
        else:
            stats["n_ambig" if cands else "n_unfixable"] += 1
            stats["n_oov_kept"] += 1
            stats["n_kept_low"] += tok_conf < 0.99
            out.append(w)
    out.append(text[pos:])
    if stats["min_tok_freq"] == 10 ** 6:
        stats["min_tok_freq"] = 0
    kc = [c for c, k in zip(cc, kept) if k]
    stats["kept_min_conf"] = min(kc) if kc else 1.0
    pc = [c for ch, c in zip(text, cc) if not ch.isalnum() and ch != " "]
    stats["punct_min_conf"] = min(pc) if pc else 1.0
    res = "".join(out)
    toks = tokens(res)
    stats["first_freq"] = vocab.first[toks[0]] if toks else 0
    stats["last_freq"] = vocab.last[toks[-1]] if toks else 0
    return res, stats


def build_vocab(reference, label_rows=()):
    """Словарь из справочника и (если есть) разметки train: label_rows — список dict полей."""
    texts = [n for names in reference.values() for n in names]
    for r in label_rows:
        texts += [r["Birth_place_ru"], r["Issue_organization_ru"]]
    return Vocab(texts, reference.keys())
