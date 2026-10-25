"""Общие константы и утилиты."""
import datetime as dt
import json

FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
          "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
          "Issue_organization_ru"]
NAME_FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru"]

# дата, на которую паспорт должен быть действителен
REF_DATE = dt.date(2026, 10, 1)


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm(s):
    """Нормализация как в evaluate.py."""
    return " ".join(str(s or "").split()).upper()


def parse_date(s):
    """'дд.мм.гггг' -> date или None."""
    try:
        return dt.datetime.strptime(s, "%d.%m.%Y").date()
    except (TypeError, ValueError):
        return None


def add_years(d, years):
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 февраля
        return d.replace(year=d.year + years, day=28)
