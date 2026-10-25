"""Разбор MRZ внутреннего паспорта РФ.

Строка 1: PNRUS<ФАМИЛИЯ><<ИМЯ><ОТЧЕСТВО><<<...  (44 символа, кириллица по таблице МВД)
Строка 2: [серия 3][номер 6][к]RUS[ддмм гг рождения 6][к][пол]<<<<<<<
          [посл. цифра серии][дата выдачи ггммдд][код подразделения 6]<[к][к]
"""
from .common import parse_date, REF_DATE

CYR2LAT = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "2", "Ж": "J", "З": "Z",
    "И": "I", "Й": "Q", "К": "K", "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R",
    "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "H", "Ц": "C", "Ч": "3", "Ш": "4", "Щ": "W",
    "Ъ": "X", "Ы": "Y", "Ь": "9", "Э": "6", "Ю": "7", "Я": "8", " ": "<", "-": "<",
}
LAT2CYR = {v: k for k, v in CYR2LAT.items() if v != "<"}

# буквы, которые OCR ставит вместо цифр в числовых позициях MRZ
DIGIT_FIX = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6",
             "B": "8", "T": "7"}

LINE_LEN = 44


def translit(s):
    """Кириллица -> MRZ. None, если встретился символ не из таблицы."""
    out = []
    for ch in s:
        if ch not in CYR2LAT:
            return None
        out.append(CYR2LAT[ch])
    return "".join(out)


def _val(c):
    if c.isdigit():
        return int(c)
    if c == "<":
        return 0
    return ord(c) - 55


def check_digit(s):
    return str(sum(_val(c) * w for c, w in zip(s, [7, 3, 1] * 20)) % 10)


def _digits(s):
    return "".join(DIGIT_FIX.get(c, c) for c in s)


def _yymmdd(s, kind):
    """'ггммдд' -> 'дд.мм.гггг'. Век однозначен: рождённые после 2012 ещё не имеют паспорта
    (14 лет на REF_DATE), а паспорта нового образца выдаются с 1997 года."""
    if not s.isdigit():
        return None
    yy, mm, dd = s[0:2], s[2:4], s[4:6]
    if kind == "birth":
        century = "20" if int(yy) <= REF_DATE.year % 100 - 14 else "19"
    else:
        century = "19" if int(yy) >= 97 else "20"
    out = f"{dd}.{mm}.{century}{yy}"
    return out if parse_date(out) else None


def parse(mrz):
    """Возвращает dict со значениями из MRZ и признаками их надёжности.

    Каждое числовое поле подтверждено своей контрольной цифрой (ok_*). Строка 2 разбирается
    с двух концов: левая часть (номер, дата рождения, пол) от начала, правая (серия, дата
    выдачи, код) от конца, поэтому вставка/пропуск символа в середине ломает только композитную
    контрольную цифру.
    """
    res = {"present": False}
    if not mrz or not mrz.get("lines"):
        return res
    lines = mrz["lines"]
    res["present"] = True
    res["n_lines"] = len(lines)
    # строку с ФИО узнаём по префиксу, а не по номеру: детектор мог потерять одну из строк
    l1 = next((l for l in lines if l["text"].startswith("PN")), None)
    l2 = next((l for l in lines if l is not l1), None)

    if l2 is not None:
        t, cc = l2["text"], l2["char_conf"]
        res["l2_len"] = len(t)
        res["l2_min_conf"] = min(cc) if cc else 0.0
        if len(t) >= 30:
            num, ser_d = _digits(t[0:9]), None
            res["ok_number"] = check_digit(num) == t[9] and num.isdigit()
            res["number_conf"] = min(cc[0:10])
            b = _digits(t[13:19])
            res["ok_birth"] = check_digit(b) == t[19] and b.isdigit()
            res["birth_conf"] = min(cc[13:20])
            res["birth_date"] = _yymmdd(b, "birth")
            res["sex"] = {"M": "МУЖ.", "F": "ЖЕН."}.get(t[20])
            res["sex_conf"] = cc[20]
            opt = _digits(t[-16:-3]) + t[-3]
            res["ok_opt"] = check_digit(opt) == t[-2] and opt[:13].isdigit()
            res["opt_conf"] = min(cc[-16:-1])
            ser_d = opt[0]
            res["series_number"] = f"{num[0:2]} {num[2]}{ser_d} {num[3:9]}"
            res["issue_date"] = _yymmdd(opt[1:7], "issue")
            res["code"] = f"{opt[7:10]}-{opt[10:13]}"
            res["ok_composite"] = (len(t) == LINE_LEN and check_digit(
                _digits(t[0:10]) + _digits(t[13:20]) + _digits(t[21:43])) == t[43])

    if l1 is not None:
        t, cc = l1["text"], l1["char_conf"]
        res["l1_len"] = len(t)
        res["l1_text"] = t
        res["l1_conf"] = cc
        res["l1_min_conf"] = min(cc) if cc else 0.0
    return res


def names_from_l1(text):
    """Разбить имя из строки 1 на (фамилия, имя, отчество) в MRZ-алфавите; обрезку учитывается
    признаком full (дошла ли строка до конца без заполнителей)."""
    body = text[5:]
    full = not body.endswith("<")
    body = body.rstrip("<")
    if "<<" in body:
        last, rest = body.split("<<", 1)
    else:
        last, rest = body, ""
    first, _, middle = rest.partition("<")
    return last, first, middle, full


def back_translit(s, template=None):
    """MRZ -> кириллица. '<' -> пробел или дефис: берём разделитель из шаблона (визуальная зона)."""
    out = []
    for i, ch in enumerate(s):
        if ch == "<":
            sep = " "
            if template and i < len(template) and template[i] in " -":
                sep = template[i]
            out.append(sep)
        elif ch in LAT2CYR:
            out.append(LAT2CYR[ch])
        else:
            return None
    return "".join(out)

