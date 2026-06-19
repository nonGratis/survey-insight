"""Per-question quality analysis — design linter (pre-data) + response stats (post-data).

Дві сутності життєвого циклу анкети:

- `analyze_form_design(form)` — ДО публікації, лише зі структури форми (без
  відповідей). Лінтер якості питань: довжина, можливе подвійне (та/або),
  к-сть опцій, тип. Допомагає виявити проблеми ще на етапі проектування.

- `analyze_responses(form, responses)` — ПІСЛЯ збору. Розподіл відповідей,
  % пропуску (item non-response), статистика відкритих питань.

Чисті функції (без I/O) — парсять структуру Forms API і список відповідей.
Дані: `forms.forms.get` (структура) + `forms.responses.list` (відповіді з
`answers`). Sheet не потрібен.

Свідомо НЕ реалізовано (потребує стандартизованих Likert-матриць / paradata,
яких у Google Forms немає): straightlining, Cronbach α, час відповіді.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Any

# Пороги лінтера (з методології опитувань; названі, не «магічні»).
LONG_QUESTION_CHARS = 120  # > ~120 символів = висока когнітивна складність
MAX_OPTIONS = 11  # оптимум шкали 5-11 (SQP); > 11 ускладнює вибір
DOUBLE_BARRELED_MARKERS = (" та ", " або ", " і/або ")  # склейка двох об'єктів

# Режими сортування розподілу (спільні для екрана та PDF-звіту).
SORT_BY_COUNT = "За величиною"
SORT_ALPHA = "Алфавіт"
SORT_FORM_ORDER = "Порядок у формі"
SORT_MODES = (SORT_BY_COUNT, SORT_ALPHA, SORT_FORM_ORDER)

# Тип питання Forms API → людська назва.
_QTYPE_LABEL = {
    "radio": "один вибір",
    "checkbox": "кілька виборів",
    "dropdown": "випадайка",
    "scale": "шкала",
    "rating": "рейтинг",
    "text": "відкрите",
    "paragraph": "відкрите (абзац)",
    "date": "дата",
    "time": "час",
    "file": "файл",
    "other": "інше",
}


@dataclass
class QuestionDesign:
    """Дизайн-профіль одного питання (зі структури, без відповідей)."""

    question_id: str
    title: str
    qtype: str  # ключ із _QTYPE_LABEL
    n_options: int | None  # None для text/date/...
    has_other: bool
    required: bool
    char_len: int
    word_count: int
    flags: list[str] = field(default_factory=list)

    @property
    def qtype_label(self) -> str:
        return _QTYPE_LABEL.get(self.qtype, self.qtype)


@dataclass
class QuestionResponseStats:
    """Статистика відповідей на одне питання (post-data)."""

    question_id: str
    n_answered: int
    n_total: int
    distribution: dict[str, int]  # value → count (для choice/scale); {} для text
    is_text: bool
    text_median_len: float | None = None

    @property
    def non_response_pct(self) -> float:
        if self.n_total <= 0:
            return 0.0
        return round((self.n_total - self.n_answered) / self.n_total * 100, 1)


def _question_type(question: dict[str, Any]) -> str:
    """Визначити тип питання за ключами Forms API."""
    if "choiceQuestion" in question:
        t = question["choiceQuestion"].get("type", "")
        return {"RADIO": "radio", "CHECKBOX": "checkbox", "DROP_DOWN": "dropdown"}.get(t, "radio")
    if "scaleQuestion" in question:
        return "scale"
    if "ratingQuestion" in question:
        return "rating"
    if "textQuestion" in question:
        return "paragraph" if question["textQuestion"].get("paragraph") else "text"
    if "dateQuestion" in question:
        return "date"
    if "timeQuestion" in question:
        return "time"
    if "fileUploadQuestion" in question:
        return "file"
    return "other"


def _iter_questions(form: dict[str, Any]):
    """Yield (question_id, title, question_dict) для кожного питання форми.

    Обробляє одиночні (`questionItem`) та групові/матричні (`questionGroupItem`)
    елементи; не-питання (секції, текст, зображення) пропускає.
    """
    for item in form.get("items", []):
        title = item.get("title", "")
        if "questionItem" in item:
            q = item["questionItem"].get("question", {})
            qid = q.get("questionId", "")
            if qid:
                yield qid, title, q
        elif "questionGroupItem" in item:
            for sub in item["questionGroupItem"].get("questions", []):
                qid = sub.get("questionId", "")
                row = sub.get("rowQuestion", {}).get("title", "")
                if qid:
                    yield qid, f"{title} — {row}".strip(" —"), sub


def _design_flags(title: str, qtype: str, n_options: int | None) -> list[str]:
    flags: list[str] = []
    if len(title) > LONG_QUESTION_CHARS:
        flags.append("задовге")
    low = f" {title.lower()} "
    if any(m in low for m in DOUBLE_BARRELED_MARKERS):
        flags.append("можливо подвійне (та/або)")
    if n_options is not None and n_options > MAX_OPTIONS:
        flags.append(f"забагато опцій ({n_options})")
    return flags


def analyze_form_design(form: dict[str, Any]) -> list[QuestionDesign]:
    """Лінтер дизайну: профіль + прапори якості для кожного питання (без даних)."""
    out: list[QuestionDesign] = []
    for qid, title, q in _iter_questions(form):
        qtype = _question_type(q)
        n_options: int | None = None
        has_other = False
        if "choiceQuestion" in q:
            opts = q["choiceQuestion"].get("options", [])
            n_options = len(opts)
            has_other = any(o.get("isOther") for o in opts)
        elif "scaleQuestion" in q:
            sc = q["scaleQuestion"]
            n_options = int(sc.get("high", 0)) - int(sc.get("low", 0)) + 1
        out.append(
            QuestionDesign(
                question_id=qid,
                title=title,
                qtype=qtype,
                n_options=n_options,
                has_other=has_other,
                required=bool(q.get("required", False)),
                char_len=len(title),
                word_count=len(title.split()),
                flags=_design_flags(title, qtype, n_options),
            )
        )
    return out


def _answer_values(response: dict[str, Any], qid: str) -> list[str]:
    ans = response.get("answers", {}).get(qid)
    if not ans:
        return []
    return [a.get("value", "") for a in ans.get("textAnswers", {}).get("answers", [])]


def analyze_responses(
    form: dict[str, Any], responses: list[dict[str, Any]]
) -> dict[str, QuestionResponseStats]:
    """Розподіл + % пропуску + статистика відкритих по кожному питанню."""
    n_total = len(responses)
    stats: dict[str, QuestionResponseStats] = {}
    for qid, _title, q in _iter_questions(form):
        qtype = _question_type(q)
        is_text = qtype in ("text", "paragraph")
        distribution: dict[str, int] = {}
        n_answered = 0
        text_lens: list[int] = []
        for r in responses:
            vals = _answer_values(r, qid)
            if not vals:
                continue
            n_answered += 1
            if is_text:
                text_lens.append(len(" ".join(vals)))
            else:
                for v in vals:  # checkbox → кілька значень
                    distribution[v] = distribution.get(v, 0) + 1
        stats[qid] = QuestionResponseStats(
            question_id=qid,
            n_answered=n_answered,
            n_total=n_total,
            distribution=distribution,
            is_text=is_text,
            text_median_len=float(median(text_lens)) if text_lens else None,
        )
    return stats


def normalize_label(value: str) -> str:
    """Нормалізувати мітку для зіставлення: стиснути пробіли + casefold."""
    return " ".join(str(value).split()).casefold()


def canonicalize_distribution(
    distribution: Mapping[str, int],
    allowed_options: Sequence[str] = (),
) -> dict[str, int]:
    """Згорнути однакові за змістом мітки перед показом у графіках.

    Google Forms може повертати free-text варіанти з різним регістром або
    пробілами ("Не знаю" / "не знаю"). Для coded options повертаємо офіційну
    мітку з форми, для інших значень — найчастіший очищений варіант.
    """
    allowed = {normalize_label(option): option for option in allowed_options}
    grouped: dict[str, dict[str, Any]] = {}
    for raw_value, count in distribution.items():
        normalized = normalize_label(raw_value)
        if not normalized:
            continue
        display = allowed.get(normalized, " ".join(str(raw_value).split()))
        bucket = grouped.setdefault(
            normalized,
            {"total": 0, "display": display, "display_count": 0},
        )
        bucket["total"] += count
        if count > bucket["display_count"]:
            bucket["display"] = display
            bucket["display_count"] = count
    return {str(bucket["display"]): int(bucket["total"]) for bucket in grouped.values()}


def anonymize_distribution(
    distribution: Mapping[str, int],
    allowed_options: Sequence[str],
    anonymized_label: str,
) -> dict[str, int]:
    """Згорнути значення поза кодованими варіантами у спільну анонімну мітку.

    Захищає персональні дані у вільних («Інше») відповідях: усе, чого немає
    серед офіційних варіантів питання, об'єднується під `anonymized_label`.
    """
    allowed = {normalize_label(option): option for option in allowed_options}
    output: dict[str, int] = {}
    for raw_value, count in distribution.items():
        canonical = allowed.get(normalize_label(raw_value))
        key = canonical if canonical is not None else anonymized_label
        output[key] = output.get(key, 0) + count
    return output


def sort_distribution(
    distribution: Mapping[str, int],
    sort_mode: str,
    form_options: Sequence[str] = (),
    keep_label_last: bool = False,
    label_last_value: str = "",
) -> list[tuple[str, int]]:
    """Посортувати розподіл за величиною / алфавітом / порядком у формі.

    `keep_label_last` тримає `label_last_value` (напр., «Інше*») в кінці.
    """

    def _last(value: str) -> int:
        return 1 if keep_label_last and value == label_last_value else 0

    items = list(distribution.items())
    if sort_mode == SORT_ALPHA:
        return sorted(items, key=lambda kv: (_last(kv[0]), normalize_label(kv[0])))
    if sort_mode == SORT_FORM_ORDER:
        order = {normalize_label(o): i for i, o in enumerate(form_options)}
        return sorted(
            items,
            key=lambda kv: (
                _last(kv[0]),
                order.get(normalize_label(kv[0]), len(order)),
                normalize_label(kv[0]),
            ),
        )
    return sorted(items, key=lambda kv: (_last(kv[0]), -kv[1], normalize_label(kv[0])))


def question_options(form: dict[str, Any]) -> dict[str, list[str]]:
    """Кодовані варіанти кожного choice-питання: {question_id: [значення...]}.

    Потрібно для анонімізації (відрізнити офіційні варіанти від вільних) і
    для сортування «у порядку форми». Не-choice питання відсутні у словнику.
    """
    out: dict[str, list[str]] = {}
    for qid, _title, q in _iter_questions(form):
        if "choiceQuestion" in q:
            out[qid] = [o.get("value", "") for o in q["choiceQuestion"].get("options", [])]
    return out
