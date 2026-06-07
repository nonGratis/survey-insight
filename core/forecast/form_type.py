"""Keyword-based form-type classifier (runtime).

Класифікує форму в одну з категорій за keyword-rules на title (+опційно
short_name, description). Категорії визначають per-type пороги детектора
хвиль та priors у wave_estimator/wave_detector.

Це core-версія логіки з research/benchmarks/07_form_type_classifier.py —
єдине джерело правди; research-скрипт імпортує звідси. У проді доступний
лише title форми (з Forms API), тож short_name/description опційні.

Категорії (мотивація — гіпотеза користувача про природу динаміки):
- event_registration — реєстрації на лекції/заходи (exponential burst + waves)
- event_feedback — фідбек після заходу (короткий burst)
- survey — опитування (один потужний logarithmic flow + follow-ups)
- recruitment — набір/відбір до Студради/відділів
- service — адмін-сервіси (M365, поселення, довідки, бали)
- volunteer_donor — донорство, волонтерські виїзди
- political — підписи, вибори, делегати
- creative_submission — збір фото/дизайнів/мерч
- holiday — Таємний Санта/Миколай/новорічне
- other — решта
"""

from __future__ import annotations

import re

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
    r"\bлекц[іі]\w* про донорство",
    r"\bрозкажи про проблему",
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
    r"\bвступ\w* до студрад",
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
    r"\bна донацію\b",
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
    r"\bвиконавц\w*\b",
]

SURVEY_PATTERNS = [
    r"^\s*да\b",
    r"^\s*сп\b",
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
    r"\bзалученн\w* до",
    r"\bтипові запитанн\w*\b",
]

# Priority order: найспецифічніші → найзагальніші (survey останній).
CATEGORY_PATTERNS: list[tuple[str, list[str]]] = [
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

# Усі категорії (для ручного вибору в UI). "other" — fallback.
FORM_TYPES: list[str] = [cat for cat, _ in CATEGORY_PATTERNS] + ["other"]


def classify_form_type(title: str, short_name: str = "", description: str = "") -> str:
    """Класифікувати форму за keyword-rules. Повертає категорію (або 'other').

    Args:
        title: назва форми (основний сигнал; у проді часто єдиний доступний).
        short_name: коротка назва (опційно).
        description: опис форми (опційно).

    Returns:
        Одна з категорій CATEGORY_PATTERNS або 'other'.
    """
    text = " ".join([title or "", short_name or "", description or ""]).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    for cat, patterns in CATEGORY_PATTERNS:
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                return cat
    return "other"
