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

from dataclasses import dataclass, field
from statistics import median
from typing import Any

# Пороги лінтера (з методології опитувань; названі, не «магічні»).
LONG_QUESTION_CHARS = 120  # > ~120 символів = висока когнітивна складність
MAX_OPTIONS = 11  # оптимум шкали 5-11 (SQP); > 11 ускладнює вибір
DOUBLE_BARRELED_MARKERS = (" та ", " або ", " і/або ")  # склейка двох об'єктів

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
