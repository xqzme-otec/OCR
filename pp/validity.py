"""Действительность паспорта на REF_DATE."""
import datetime as dt

from .common import add_years, parse_date, REF_DATE


def validity(birth, issue):
    """Статус действительности паспорта на REF_DATE: 'valid', 'expired', 'grace', 'bad'.

    Паспорт выдаётся с 14 лет и подлежит замене в 20 и 45. Паспорт, подлежащий замене,
    действует до получения нового, но не дольше 90 дней после дня рождения ('grace'):
    из данных нельзя понять, получен ли новый, поэтому такие считаем рискованными.
    """
    b, i = parse_date(birth), parse_date(issue)
    if not b or not i:
        return "bad"
    if i > REF_DATE or i < add_years(b, 14) or i < dt.date(1997, 10, 1):
        return "bad"
    if i >= add_years(b, 45):
        return "valid"
    limit = add_years(b, 20) if i < add_years(b, 20) else add_years(b, 45)
    if REF_DATE < limit:
        return "valid"
    if REF_DATE < limit + dt.timedelta(days=90):
        return "grace"
    return "expired"
