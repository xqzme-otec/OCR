"""Чтение и нормализация полей визуальной зоны."""
import re

# цифры, которые OCR путает с буквами (кириллица и латиница)
_DIGIT_FIX = str.maketrans({"О": "0", "O": "0", "D": "0", "З": "3", "Б": "6", "б": "6", "I": "1",
                            "l": "1", "|": "1", "S": "5", "B": "8", "В": "8", "G": "6", "Z": "2"})


def text(field):
    """Строки поля через пробел ('' если поле не найдено)."""
    if not field:
        return ""
    return " ".join(l["text"] for l in field["lines"])


def char_conf(field):
    """Уверенности по символам всех строк подряд (пробел между строками = 1.0)."""
    out = []
    if not field:
        return out
    for k, l in enumerate(field["lines"]):
        if k:
            out.append(1.0)
        out.extend(l["char_conf"])
    return out


def digits(s):
    return re.sub(r"\D", "", s.translate(_DIGIT_FIX))


def norm_date(s):
    d = digits(s)
    return f"{d[0:2]}.{d[2:4]}.{d[4:8]}" if len(d) == 8 else None


def norm_number(s):
    d = digits(s)
    return f"{d[0:2]} {d[2:4]} {d[4:10]}" if len(d) == 10 else None


def norm_code(s):
    d = digits(s)
    return f"{d[0:3]}-{d[3:6]}" if len(d) == 6 else None


def norm_sex(s):
    s = s.upper()
    if re.fullmatch(r"\s*М[УY]Ж\.?\s*", s):
        return "МУЖ."
    if re.fullmatch(r"\s*ЖЕН\.?\s*", s):
        return "ЖЕН."
    return None
