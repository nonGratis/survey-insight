"""07_form_type_classifier.py — keyword-based form-type classifier.

Парсимо `data/Form Catalog.tsv`, класифікуємо кожну форму у одну з
категорій за keyword-rules на title + description + short_name.

Категорії (мотивація — testable гіпотеза користувача):
- `event_registration` — реєстрації на лекції/заходи/концерти/кіно/тренінги.
  Очікувана динаміка: експонентний burst з agitation waves.
- `event_feedback` — фідбек ПІСЛЯ заходу. Очікувано короткий burst після event.
- `survey` — опитування (ДА, СП, оцінювання, дослідження).
  Очікувано: один потужний logarithmic flow + менші follow-ups.
- `recruitment` — набір/перехід/відбір до Студради/відділів.
- `service` — адмін-сервіси (M365, поселення, довідка, додаткові бали, стипендії).
- `volunteer_donor` — донорство крові, волонтерські виїзди.
- `political` — підписи кандидатів, вибори, реєстрація делегатів.
- `creative_submission` — збір фото/дизайнів/творчих робіт + мерч-передзамовлення.
- `holiday` — Таємний Санта/Миколайчик/новорічне.
- `other` — решта.

Output: `research/reports/figures/07_form_types.csv` з колонками
(form_id, form_title, form_type).

Запуск:
    .venv/Scripts/python.exe research/benchmarks/07_form_type_classifier.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Keyword rules: priority order (first match wins, evaluated by category).
# Прихований принцип — починаємо від найспецифічніших (politicalmail
# мінює service), завершуємо найзагальнішим (survey).

POLITICAL_PATTERNS = [
    r"\bпідпис\w* (за|для|на) (реєстраці|кандидат)",
    r"\bдля реєстрації .* (кандидат|голов)",
    r"\bвиборч\w* комісі",
    r"реєстраці\w* (делегат|кандидат)",
    r"\bна посаду голов",
]

VOLUNTEER_DONOR_PATTERNS = [
    r"\bдонор\w*\b",
    r"\bдонац[іі]\w*\b",
    r"\bкров[іиь]\b",
    r"\bволонтер\w*\b",
    r"\bгероїк[ау]\b",
    r"\bлекц[іі]\w* про донорство",  # лекція про донорство — типу volunteer-info
    r"\bрозкажи про проблему",  # community-reporting
]

HOLIDAY_PATTERNS = [
    r"\bтаємн\w* (санта|миколай)",
    r"\bноворічн\w*\b",
    r"\bріздвян\w*\b",
    r"\b8 березня\b",
    r"\bдень закоханих\b",
    r"\bдень святого\b",
    r"\bналуйчик\b",
]

SERVICE_PATTERNS = [
    r"\bmicrosoft\s*365\b",
    r"\bms\s*365\b",
    r"\bm365\b",
    r"\bпошт[аи]?\b.*\b(edu|kpi)",
    r"\bпоселенн[яю]\b",
    r"\bдовідк[аи] про навчанн",
    r"\bстипенді\w*\b",
    r"\bліценз\w*\b",
    r"\bcoursera\b",
    r"\bдодатков\w* бал\w*\b",
    r"\bпільг\w*\b",
    r"\bвступ\w* до студрад",  # admission to studrada is closer to service in some cases — but treat as recruitment
]

RECRUITMENT_PATTERNS = [
    r"\bнабір до\b",
    r"\bперехід (до|в|у)\b",
    r"\bвідбір\b",
    r"\bвступ до студ\w*\b",
    r"\bвступ в\b",
    r"\btryouts?\b",
    r"\bsupergerogi|супергерої\b",
    r"\bhr department\b",
]

EVENT_FEEDBACK_PATTERNS = [
    r"\bfeedback\b",
    r"\bфідбек\b",
    r"\bвідгук\w*\b",
    r"\bвраженн[ьія]\b",
    r"\bзвіт по\b",
    r"\bоцінка робот\w*\b",
    r"\bформа зворотн\w*\b",
    r"\bform.*зворотн\w*\b",
]

EVENT_REGISTRATION_PATTERNS = [
    r"\bреєстраці[яії]\b",
    r"\bзапрошуємо на лекці\w*\b",
    r"\bлекці[яії]\b",
    r"\bкіновечір\b",
    r"\bпоказ фільм\w*\b",
    r"\bконцерт\w*\b",
    r"\bтренінг\b",
    r"\bвечір (стендапу|поезії|залюблених)\b",
    r"\bквартирник\b",
    r"\bвечірк\w*\b",
    r"\bkpi party\b",
    r"\bchill day\b",
    r"\bday f\b",
    r"\bday\s*\w*\s*event\b",
    r"\bфотосушк\w*\b",
    r"\bфотомайстерн\w*\b",
    r"\bтурнір\b",
    r"\bстрітбол\b",
    r"\bфутбольн\w* турнір\b",
    r"\bвечір (різдвян|новорічн)",
    r"\bзахід\b",
    r"\bподі[ьія]\b",
    r"\bна перегляд фільму\b",
    r"\bна квест\b",
    r"\bна вечір\b",
    r"\bна донацію\b",  # donation drive registration ≈ event
    r"\bна подію\b",
    r"\bтімбілдинг\b",
    r"\bvelocity\b",
    r"\bвиставк\w*\b",
    r"\bмайстер[\s-]*клас\b",
    r"\bлекці.*про\b",
    r"\bна закритий захід\b",
    r"\bподкаст\b",
    r"\bkpi talks\b",
    r"\bна зустріч\b",
    r"\bпрезентаці\w*\b",
    r"\bна спринт\b",
    r"\bпершокурсн\w* спринт\b",
    r"\bна шп\b",
    r"\bна шаф\b",
    r"\bна chill\b",
    r"\bтренінг(у|и)?\b",
    r"\bна chill day\b",
    r"\bна квест\b",
]

CREATIVE_PATTERNS = [
    r"\bзбір фото\b",
    r"\bзбір дизайн\w*\b",
    r"\bпередзамовленн\w* мерч\w*\b",
    r"\bмакет\w*\b",
    r"\bстворі?и? дизайн\b",
    r"\bопитування про брендов\w* одяг\b",
    r"\bзбір історій\b",
    r"\bбарахолк\w*\b",
    r"\bна (барахолку|chill day)\b",
    r"\bпродавц\w*\b",
    r"\bвиконавц\w*\b",  # співаки/виконавці на квартирник — semi-event
]

SURVEY_PATTERNS = [
    r"^\s*да\b",  # ДА = Департамент аналітики
    r"^\s*сп\b",  # СП = СтудПарламент
    r"\bопитуванн[яії]\b",
    r"\bоцінюванн\w*\b",
    r"\bдослідженн\w*\b",
    r"\bвяо\b",
    r"\bпропозиції до\b",
    r"\bгромадськ\w* обговоренн\w*\b",
    r"\bобираємо формат\b",
    r"\bоцінка\b",
    r"\bsurvey\b",
    r"\bаналітик\w*\b",
    r"\bзаповнен\w* комірк\w*\b",
    r"\bфінансова аналітик\w*\b",
    r"\bзбираємо думк\w*\b",
    r"\bрозкажи про проблему\b",
    r"\bзбираємо запитанн\w*\b",
    r"\bзалученн\w* до",  # involvement
    r"\bтипові запитанн\w*\b",
]

CATEGORY_PATTERNS = [
    ("political", POLITICAL_PATTERNS),
    ("volunteer_donor", VOLUNTEER_DONOR_PATTERNS),
    ("holiday", HOLIDAY_PATTERNS),
    ("service", SERVICE_PATTERNS),
    ("event_feedback", EVENT_FEEDBACK_PATTERNS),
    ("recruitment", RECRUITMENT_PATTERNS),
    ("creative_submission", CREATIVE_PATTERNS),
    ("event_registration", EVENT_REGISTRATION_PATTERNS),
    ("survey", SURVEY_PATTERNS),
]


def _classify_one(title: str, short_name: str, description: str) -> tuple[str, str]:
    """Повертає (category, matched_pattern). other якщо нічого не співпало."""
    text = " ".join([title or "", short_name or "", description or ""]).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    for cat, patterns in CATEGORY_PATTERNS:
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                return cat, pat
    return "other", ""


def main(catalog_path: Path, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(catalog_path, sep="\t", dtype=str).fillna("")
    df = df[["form_id", "form_title", "short_name", "description"]].copy()
    df = df[df["form_id"].str.strip() != ""].reset_index(drop=True)

    results = []
    for _, row in df.iterrows():
        cat, pat = _classify_one(row["form_title"], row["short_name"], row["description"])
        results.append(
            {
                "form_id": row["form_id"],
                "form_title": row["form_title"],
                "form_type": cat,
                "matched_pattern": pat,
            }
        )
    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)
    print(f"Classified {len(out)} forms -> {output_csv}")
    print("\nDistribution:")
    print(out["form_type"].value_counts().to_string())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    p.add_argument("--catalog", type=Path, default=repo_root / "data" / "Form Catalog.tsv")
    p.add_argument(
        "--output",
        type=Path,
        default=repo_root / "research" / "reports" / "figures" / "07_form_types.csv",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.catalog, args.output)
