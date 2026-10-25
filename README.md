# Постобработка OCR паспорта РФ

Решение тестового задания из [TASK.md](TASK.md). Отчёт с результатами и прогнозом: [REPORT.md](REPORT.md).

```
pip install -r requirements.txt
python postprocess.py data/test_predictions.jsonl answers.jsonl
```

- `postprocess.py` — решение accept/reject по каждому документу (CPU, без интернета, ~3 с).
- `train.py` — обучение модели уверенности (`models/model.pkl`) и оценка по кросс-валидации на train.
- `tune.py` — подбор гиперпараметров бустинга, `compare_models.py` — сравнение с логрегрессией и нейросетью.
- `answers.jsonl` — ответы для test.
- `pp/` — модули: `mrz.py` (разбор MRZ, контрольные цифры, транслитерация), `names.py` (сверка ФИО с MRZ),
  `viz.py` (нормализация полей визуальной зоны), `text.py` (исправление текстов по словарю),
  `validity.py` (действительность паспорта на 01.10.2026), `resolve.py` (итоговые значения и признаки),
  `model.py` (жёсткие правила и модель).
