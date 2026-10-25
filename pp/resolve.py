"""Сбор итоговых значений полей документа и признаков для модели уверенности.

Для каждого поля выбираем значение из визуальной зоны (viz), MRZ или словаря и записываем,
откуда оно взято и насколько согласуются источники. Модель потом оценивает P(поле верно).
"""
import re

from . import mrz as mrzlib
from . import names
from . import text as textlib
from . import viz
from .common import FIELDS, NAME_FIELDS
from .validity import validity

SOURCES = ["web", "mobile", "mobile_v2"]
TEXT_CONF_THR = 0.95  # подобрано на train (CV): ниже — исправляем слово по словарю

MALE_END = ("ВИЧ", "ОГЛЫ", "УУЛУ", "ОГЛУ", "УГЛИ")
FEMALE_END = ("ВНА", "ЧНА", "КЫЗЫ", "ГЫЗЫ", "КИЗИ")


def _conf_stats(cc, prefix):
    if not cc:
        return {f"{prefix}min_conf": 0.0, f"{prefix}mean_conf": 0.0, f"{prefix}n_low99": 0,
                f"{prefix}n_low90": 0, f"{prefix}second_min_conf": 0.0}
    s = sorted(cc)
    return {f"{prefix}min_conf": s[0], f"{prefix}mean_conf": sum(cc) / len(cc),
            f"{prefix}n_low99": sum(c < 0.99 for c in cc), f"{prefix}n_low90": sum(c < 0.9 for c in cc),
            f"{prefix}second_min_conf": s[1] if len(s) > 1 else s[0]}


def _viz_feats(field):
    """Признаки визуальной зоны, общие для всех полей."""
    t = viz.text(field)
    f = {"viz_present": int(field is not None), "len": len(t),
         "n_lines": len(field["lines"]) if field else 0,
         "field_conf": field["conf"] if field else 0.0,
         "double_letter": int(bool(re.search(r"([А-ЯЁ])\1", t))),
         "odd_chars": len(re.findall(r"[^А-ЯЁ0-9 .\-()№,/\"]", t))}
    f.update(_conf_stats(viz.char_conf(field), ""))
    if field and field["lines"]:
        # сколько строк должно быть по высоте рамки поля (строка ~31px): потерянные строки
        f["missing_lines"] = round(field["box"][3] / 31) - len(field["lines"])
    else:
        f["missing_lines"] = 0
    return f


def _choose(viz_val, mrz_val, mrz_ok):
    """Выбор для полей, продублированных в MRZ с контрольной цифрой."""
    if viz_val and viz_val == mrz_val:
        return viz_val, "agree"
    if mrz_ok and mrz_val:
        return mrz_val, "mrz"
    if viz_val:
        return viz_val, "viz"
    return None, "none"


def _no_patronymic(l1_text, last, first):
    """В строке 1 MRZ после фамилии и имени из viz только заполнитель (допускаем один шумовой символ).

    Разбор по '<' здесь не годится: при 'ROGAVA<LHABIRA<TULIEVNA' (потерян один '<') отчество не находится.
    """
    if last is None or first is None:
        return False
    tail = l1_text[5 + len(mrzlib.translit(last)) + 2 + len(mrzlib.translit(first)):]
    return sum(c != "<" for c in tail) <= 1


SRC_CODES = {"agree": 0, "mrz": 1, "viz": 2, "none": 3, "vocab": 4, "agree_trunc": 5, "vote": 6}


def resolve(doc, vocab):
    """-> (fields: dict значений, feats: dict поле -> dict признаков, doc_feats)."""
    v = doc["viz"]
    m = mrzlib.parse(doc.get("mrz"))
    q = doc.get("quality") or {}
    doc_feats = {
        "src_" + s: int(doc.get("source") == s) for s in SOURCES}
    doc_feats.update({
        "sharpness": q.get("sharpness", 0.0), "doc_px": q.get("doc_px", 0.0), "glare": q.get("glare", 0.0),
        "mrz_present": int(m["present"]), "mrz_lines": m.get("n_lines", 0),
        "mrz_l2_len_ok": int(m.get("l2_len") == mrzlib.LINE_LEN),
        "mrz_l1_len_ok": int(m.get("l1_len") == mrzlib.LINE_LEN),
        "mrz_composite_ok": int(bool(m.get("ok_composite"))),
        "mrz_l1_min_conf": m.get("l1_min_conf", 0.0), "mrz_l2_min_conf": m.get("l2_min_conf", 0.0),
        "n_viz_missing": sum(v.get(f) is None for f in FIELDS),
    })

    fields, feats = {}, {}

    # --- поля с контрольной цифрой в MRZ ---
    numeric = {
        "Birth_date": (viz.norm_date, "birth_date", m.get("ok_birth"), m.get("birth_conf", 0.0)),
        "Issue_date": (viz.norm_date, "issue_date", m.get("ok_opt"), m.get("opt_conf", 0.0)),
        "Issue_organisation_code": (viz.norm_code, "code", m.get("ok_opt"), m.get("opt_conf", 0.0)),
        "Licence_number": (viz.norm_number, "series_number",
                           bool(m.get("ok_number")) and bool(m.get("ok_opt")),
                           min(m.get("number_conf", 0.0), m.get("opt_conf", 0.0))),
    }
    for f, (normf, key, ok, mconf) in numeric.items():
        vv = normf(viz.text(v.get(f))) if v.get(f) else None
        mv = m.get(key)
        val, src = _choose(vv, mv, ok)
        fields[f] = val
        ff = _viz_feats(v.get(f))
        ff.update({"src": SRC_CODES[src], "mrz_ok": int(bool(ok)), "mrz_conf": mconf,
                   "viz_parsed": int(vv is not None), "mrz_parsed": int(mv is not None),
                   "agree": int(src == "agree")})
        feats[f] = ff

    # --- ФИО: сверка с транслитерацией в строке 1 MRZ ---
    vn = [viz.text(v[f]) if v.get(f) is not None else None for f in NAME_FIELDS]
    vc = [viz.char_conf(v[f]) if v.get(f) is not None else None for f in NAME_FIELDS]
    cmp = names.compare(vn, vc, m["l1_text"], m["l1_conf"]) if "l1_text" in m else [None] * 3
    for k, f in enumerate(NAME_FIELDS):
        ff = _viz_feats(v.get(f))
        c = cmp[k] or {"status": "none", "margin": 0.0, "n_ops": 0}
        val, src = vn[k], "viz"
        if c["status"] == "agree":
            src = "agree"
        elif c["status"] == "trunc":
            src = "agree_trunc"
        elif (f == "Middle_name_ru" and vn[k] is None and "l1_text" in m
              and _no_patronymic(m["l1_text"], vn[0], vn[1])):
            # отчества нет ни в визуальной зоне, ни в MRZ
            val, src = "", "agree"
        elif c["status"] == "diff":
            # одно расхождение: верим источнику с большей уверенностью в этом месте
            if c["margin"] < 0 and c["value_mrz"]:
                val, src = c["value_mrz"], "mrz"
        ff.update({"src": SRC_CODES[src], "agree": int(src in ("agree", "agree_trunc")),
                   "name_margin": abs(c["margin"]), "name_n_ops": c["n_ops"],
                   "name_status": {"agree": 0, "trunc": 0, "diff": 1, "bad": 2, "none": 3}[c["status"]]})
        fields[f] = val
        feats[f] = ff

    # --- пол: viz, MRZ и окончание отчества ---
    sv = viz.norm_sex(viz.text(v.get("Sex_ru"))) if v.get("Sex_ru") else None
    sm = m.get("sex")
    mid = (fields.get("Middle_name_ru") or "").replace("-", " ").strip()
    sp = ("МУЖ." if mid.endswith(MALE_END) else "ЖЕН." if mid.endswith(FEMALE_END) else None)
    votes = [x for x in (sv, sm, sp) if x]
    if votes:
        best = max(set(votes), key=votes.count)
        n_for, n_against = votes.count(best), len(votes) - votes.count(best)
    else:
        best, n_for, n_against = None, 0, 0
    fields["Sex_ru"] = best
    ff = _viz_feats(v.get("Sex_ru"))
    ff.update({"src": SRC_CODES["vote"], "votes_for": n_for, "votes_against": n_against,
               "viz_parsed": int(sv is not None), "mrz_sex_conf": m.get("sex_conf", 0.0),
               "patronymic_sex": int(sp is not None), "agree": int(n_for >= 2 and n_against == 0)})
    feats["Sex_ru"] = ff

    # --- длинные тексты: исправление по словарю ---
    for f in ("Birth_place_ru", "Issue_organization_ru"):
        field = v.get(f)
        ff = _viz_feats(field)
        if field is None:
            fields[f] = None
            ff.update(textlib.correct("", vocab)[1])
        else:
            t, st = textlib.correct(viz.text(field), vocab, viz.char_conf(field), TEXT_CONF_THR)
            fields[f] = t
            ff.update(st)
        # «С.ГУЖОВО» и «С. ГУЖОВО» встречаются в разметке оба: пробел после точки OCR мог потерять
        ff["dot_nospace"] = len(re.findall(r"\.[А-ЯЁ]", fields[f] or ""))
        ff["src"] = SRC_CODES["vocab" if ff.get("n_fixed") else "viz"]
        feats[f] = ff

    # согласованность кода подразделения со справочником: код существует?
    code = fields.get("Issue_organisation_code")
    doc_feats["code_in_ref"] = int(code in vocab.ref_codes) if code else 0
    status = validity(fields.get("Birth_date"), fields.get("Issue_date"))
    doc_feats["validity"] = status
    return fields, feats, doc_feats
