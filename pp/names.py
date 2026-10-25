"""Сверка ФИО визуальной зоны со строкой 1 MRZ через выравнивание всей строки.

Строим ожидаемую строку 1 из транслитерации визуальной зоны и выравниваем с прочитанной MRZ.
Так ошибки MRZ вроде '<' -> 'K' в заполнителе не сдвигают границы полей.
"""
import difflib

from . import mrz as mrzlib


def _span_conf(conf, i1, i2):
    """Мин. уверенность на отрезке [i1, i2); для пустого отрезка — по соседям (вставка/пропуск)."""
    if i2 > i1:
        return min(conf[i1:i2])
    near = conf[max(i1 - 1, 0):i1 + 1]
    return min(near) if near else 0.0


def compare(viz_names, viz_confs, l1_text, l1_conf):
    """Для каждого из трёх полей: dict(status, value_mrz, margin, n_ops).

    status: 'agree' | 'trunc' (совпало, но MRZ обрезана) | 'diff' | 'bad' (нет транслитерации).
    value_mrz: значение поля в кириллице с исправлениями из MRZ (None, если не получить).
    margin: уверенность viz минус уверенность MRZ в месте расхождения (>0 — верим viz).
    """
    trs = [mrzlib.translit(n) if n is not None else None for n in viz_names]
    out = [None, None, None]
    for k in range(3):
        if trs[k] is None:
            out[k] = {"status": "bad", "value_mrz": None, "margin": 0.0, "n_ops": 0}
    trs = [t or "" for t in trs]
    exp = "PNRUS" + trs[0] + "<<" + trs[1] + "<" + trs[2]
    exp = exp[:mrzlib.LINE_LEN].ljust(mrzlib.LINE_LEN, "<")
    # уверенность символов ожидаемой строки = уверенность viz (разделители и заполнитель = 1.0)
    econf = [1.0] * 5
    for k in range(3):
        cc = viz_confs[k] if viz_confs[k] and len(viz_confs[k]) == len(trs[k]) else [0.0] * len(trs[k])
        econf += cc + ([1.0, 1.0] if k == 0 else [1.0] if k == 1 else [])
    econf = (econf + [1.0] * mrzlib.LINE_LEN)[:mrzlib.LINE_LEN]
    starts = [5, 5 + len(trs[0]) + 2, 5 + len(trs[0]) + 2 + len(trs[1]) + 1]
    ends = [starts[k] + len(trs[k]) for k in range(3)]

    ops = [o for o in difflib.SequenceMatcher(None, exp, l1_text, autojunk=False).get_opcodes()
           if o[0] != "equal"]
    for k in range(3):
        if out[k] is not None:
            continue
        s, e = starts[k], min(ends[k], mrzlib.LINE_LEN)
        mine = []
        for tag, i1, i2, j1, j2 in ops:
            # операция задевает поле (вставка внутри поля — тоже; на краю поля — если вставлена буква:
            # MRZ 'FUDOROVI3' при viz 'УДОРОВИЧ' — viz потеряла первую букву)
            if (i1 < e and i2 > s) or (i1 == i2 and s < i1 < e) or (
                    i1 == i2 and i1 in (s, e) and l1_text[j1:j2].strip("<")):
                mine.append((tag, i1, i2, j1, j2))
        truncated = ends[k] > mrzlib.LINE_LEN
        if not mine:
            out[k] = {"status": "trunc" if truncated else "agree", "value_mrz": None,
                      "margin": 0.0, "n_ops": 0}
            continue
        # значение по MRZ: в кириллице viz заменяем расходящиеся участки обратной транслитерацией
        value = None
        if not truncated and viz_names[k] is not None:
            cyr = list(viz_names[k])
            for tag, i1, i2, j1, j2 in sorted(mine, key=lambda o: -o[1]):
                if i1 < s or i2 > e:
                    cyr = None
                    break
                rep = [mrzlib.LAT2CYR.get(ch, " " if ch == "<" else None) for ch in l1_text[j1:j2]]
                if None in rep:
                    cyr = None
                    break
                cyr[i1 - s:i2 - s] = rep
            value = "".join(cyr).strip() if cyr is not None else None
        vconf = min(_span_conf(econf, i1, i2) for _, i1, i2, _, _ in mine)
        mconf = min(_span_conf(l1_conf, j1, j2) for _, _, _, j1, j2 in mine)
        out[k] = {"status": "diff", "value_mrz": value, "margin": vconf - mconf,
                  "n_ops": len(mine), "viz_conf": vconf, "mrz_conf": mconf}
    return out
