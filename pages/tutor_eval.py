"""Dashboard: Tutor Evaluation Pipeline — results, markup view, pipeline reference."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Оценка тьютора", page_icon="🎓", layout="wide")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TUTOR_EVAL_DIR = Path(__file__).resolve().parent.parent / "tutor_eval"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tutor_eval"
RESULTS_DIR = DATA_DIR / "results"
LOGS_DIR = DATA_DIR / "logs"

CRITERIA = [
    "expectations", "transparency", "learning_goal", "adaptivity",
    "simplicity", "encourages_thinking", "error_handling", "friendly", "adequacy",
]
CRITERIA_RU = {
    "expectations": "Ожидания",
    "transparency": "Прозрачность",
    "learning_goal": "Обр. цель",
    "adaptivity": "Адаптивность",
    "simplicity": "Простота",
    "encourages_thinking": "Даёт думать",
    "error_handling": "Работа с ошибками",
    "friendly": "Френдли",
    "adequacy": "Адекватность",
}

MAX_SCORE = 3


# ---------------------------------------------------------------------------
# Score conversion
# ---------------------------------------------------------------------------

SCALE_OPTIONS = {"0-3": "raw", "%": "percent", "0-1": "normalized"}


def _to_display_score(score: float | int, scale: str) -> float:
    if scale == "percent":
        return round(score / MAX_SCORE * 100, 1)
    if scale == "normalized":
        return round(score / MAX_SCORE, 3)
    return round(score, 2)


def _score_label(scale: str) -> str:
    if scale == "percent":
        return "%"
    if scale == "normalized":
        return "0-1"
    return f"0-{MAX_SCORE}"


def _score_range(scale: str) -> tuple[float, float]:
    if scale == "percent":
        return (0, 100)
    if scale == "normalized":
        return (0, 1)
    return (0, MAX_SCORE)


def _score_axis_range(scale: str) -> list[float]:
    if scale == "percent":
        return [0, 110]
    if scale == "normalized":
        return [0, 1.15]
    return [0, MAX_SCORE + 0.5]


def _round_for_scale(value: float, scale: str) -> float:
    if scale == "percent":
        return round(value, 1)
    if scale == "normalized":
        return round(value, 3)
    return round(value, 2)


def _text_fmt(scale: str) -> str:
    if scale == "percent":
        return "%{text:.1f}%"
    if scale == "normalized":
        return "%{text:.3f}"
    return "%{text:.2f}"


# ---------------------------------------------------------------------------
# EduScore (Yandex Education metric)
# ---------------------------------------------------------------------------

EDUSCORE_MAP = {3: 1.0, 2: 0.66, 1: 0.33, 0: 0.0}


def _eduscore_dialog(scores: dict[str, int]) -> float | None:
    """Compute EduScore for a single dialog.

    scores: {criterion: int score 0-3 or -1 for NA}
    Returns float 0-1 or None if no applicable criteria.
    """
    total = 0.0
    count = 0
    for c in CRITERIA:
        s = scores.get(c)
        if s is None or s < 0:
            continue
        total += EDUSCORE_MAP.get(s, 0.0)
        count += 1
    return round(total / count, 3) if count > 0 else None


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _find_runs() -> list[str]:
    runs = []
    for d in [RESULTS_DIR, LOGS_DIR]:
        if d.exists():
            for sub in sorted(d.iterdir(), reverse=True):
                if sub.is_dir() and sub.name not in runs:
                    runs.append(sub.name)
    return sorted(set(runs), reverse=True)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_summary(run_name: str) -> dict | None:
    p = LOGS_DIR / run_name / "summary.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _load_raw_scores(run_name: str) -> list[dict]:
    for parent in [RESULTS_DIR, LOGS_DIR]:
        p = parent / run_name / "all_raw_scores.jsonl"
        if p.exists():
            return _load_jsonl(p)
        p = parent / run_name / "raw_scores.jsonl"
        if p.exists():
            return _load_jsonl(p)
    return []


def _load_judge_decisions(run_name: str) -> list[dict]:
    for parent in [RESULTS_DIR, LOGS_DIR]:
        p = parent / run_name / "all_judge_decisions.jsonl"
        if p.exists():
            return _load_jsonl(p)
        p = parent / run_name / "judge_decisions.jsonl"
        if p.exists():
            return _load_jsonl(p)
    return []


def _load_errors(run_name: str) -> list[dict]:
    p = LOGS_DIR / run_name / "errors.jsonl"
    return _load_jsonl(p)


def _load_dialogs(run_name: str) -> dict[str, dict]:
    """Load dialog metadata keyed by dialog_id."""
    for parent in [RESULTS_DIR, LOGS_DIR]:
        p = parent / run_name / "dialogs.jsonl"
        if p.exists():
            records = _load_jsonl(p)
            return {r["dialog_id"]: r for r in records}
    return {}


def _find_math_results(run_name: str) -> list[dict]:
    """Find math_check_results.jsonl for a run, looking in math_* dirs and the run dir itself."""
    # First check the run dir itself
    for parent in [RESULTS_DIR, LOGS_DIR]:
        p = parent / run_name / "math_check_results.jsonl"
        if p.exists():
            return _load_jsonl(p)
    # Then check math_* dirs that may share dialog_ids
    for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if d.is_dir() and (d / "math_check_results.jsonl").exists():
            return _load_jsonl(d / "math_check_results.jsonl")
    return []


def _compute_error_rate(math_results: list[dict]) -> dict:
    """Compute error_rate and breakdown by error class per wiki formula."""
    if not math_results:
        return {}
    total = len(math_results)
    dialogs_with_error = 0
    error_classes = {"tutor_error": 0, "accepted_incorrect": 0, "rejected_correct": 0}

    for mr in math_results:
        has_error = False
        for v in mr.get("verifications", []):
            if v.get("is_correct") is False:
                has_error = True
                ec = v.get("error_class", "tutor_error")
                if ec in error_classes:
                    error_classes[ec] += 1
                else:
                    error_classes["tutor_error"] += 1
        if has_error:
            dialogs_with_error += 1

    return {
        "total_dialogs": total,
        "dialogs_with_error": dialogs_with_error,
        "error_rate": round(dialogs_with_error / total, 3) if total else 0,
        "by_class": error_classes,
    }


# ---------------------------------------------------------------------------
# Build DataFrames
# ---------------------------------------------------------------------------

def _raw_to_df(raw_scores: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in raw_scores:
        dialog_id = entry["dialog_id"]
        evaluator = entry.get("evaluator", "")
        temperature = entry.get("temperature", None)
        criteria = entry.get("criteria", {})
        for c in CRITERIA:
            cdata = criteria.get(c, {})
            score = cdata.get("score")
            if score is not None:
                rows.append({
                    "dialog_id": dialog_id,
                    "evaluator": evaluator,
                    "temperature": temperature,
                    "criterion": c,
                    "criterion_ru": CRITERIA_RU.get(c, c),
                    "score": int(score),
                    "reasoning": cdata.get("reasoning", ""),
                    "evidence": cdata.get("evidence", []),
                })
    return pd.DataFrame(rows)


def _judge_to_df(judge_decisions: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in judge_decisions:
        dialog_id = entry["dialog_id"]
        final = entry.get("final_scores", {})
        agreement = entry.get("agreement", {})
        for c in CRITERIA:
            cdata = final.get(c, {})
            score = cdata.get("score")
            if score is not None:
                rows.append({
                    "dialog_id": dialog_id,
                    "criterion": c,
                    "criterion_ru": CRITERIA_RU.get(c, c),
                    "final_score": int(score),
                    "reasoning": cdata.get("reasoning", ""),
                    "agreement": agreement.get(c, ""),
                })
    return pd.DataFrame(rows)


def _render_dialog_text(text: str):
    """Render dialog with colored role labels."""
    if not text:
        st.markdown("*текст не сохранён*")
        return
    import re
    # Split by role markers, keeping the delimiter
    parts = re.split(r"((?:Пользователь|Ассистент):)", text)
    html_parts = []
    for part in parts:
        part_stripped = part.strip()
        if not part_stripped:
            continue
        if part_stripped == "Пользователь:":
            html_parts.append(
                '<div style="margin-top:12px;">'
                '<span style="background:#2563eb;color:white;padding:2px 8px;border-radius:4px;font-weight:600;font-size:0.85em;">Пользователь</span>'
                '</div>'
            )
        elif part_stripped == "Ассистент:":
            html_parts.append(
                '<div style="margin-top:12px;">'
                '<span style="background:#16a34a;color:white;padding:2px 8px;border-radius:4px;font-weight:600;font-size:0.85em;">Ассистент</span>'
                '</div>'
            )
        else:
            # Escape HTML but preserve newlines
            escaped = part_stripped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            escaped = escaped.replace("\n", "<br>")
            html_parts.append(f'<div style="padding:4px 0 4px 12px;">{escaped}</div>')
    st.html("".join(html_parts))


def _model_from_dialog_id(dialog_id: str) -> str:
    """Extract student_model from dialog_id like 'medium_gemini3flash_task_123'."""
    parts = dialog_id.split("_")
    return parts[1] if len(parts) >= 2 else "unknown"


def _enrich_with_model(df: pd.DataFrame, dialogs: dict[str, dict]) -> pd.DataFrame:
    """Add student_model column from dialog metadata, with fallback to dialog_id parsing."""
    if df.empty:
        return df
    df = df.copy()
    df["student_model"] = df["dialog_id"].map(
        lambda did: dialogs.get(did, {}).get("student_model") or _model_from_dialog_id(did)
    )
    return df


# ---------------------------------------------------------------------------
# Pipeline reference
# ---------------------------------------------------------------------------

def _render_reference():
    st.subheader("Архитектура пайплайна")
    st.markdown("""
**Пайплайн оценки качества ИИ-тьютора**

```
XLSX-файлы с диалогами
        |
        v
   Загрузчик (loader)
   Парсит диалоги, извлекает метаданные
        |
        v
   3 оценщика (temperature=0):
   - Gemini 2.5 Flash (google/gemini-2.5-flash)
   - Gemini 3 Flash Preview (google/gemini-3-flash-preview)
   - Qwen 3.5 397B (qwen/qwen3.5-397b-a17b)
   Каждый оценивает по 9 критериям (0-3)
   + проверяет критические сбои (криты)
        |
        v
   Проверка единогласия (per criterion)
   Если все 3 совпали — принимаем автоматически
        |
        v
   Проверка критов (premature_end, prompt_leak, nonsense)
   Если хоть один оценщик нашёл крит → судья подтверждает
        |
        v
   Claude Sonnet 4 (судья)
   Вызывается для спорных критериев + подтверждения критов
   Подтверждённый крит → адекватность = 0
        |
        v
   Результаты: XLSX + JSONL + summary
```

**Почему 3 разных модели:**
- Разные архитектуры = разные «слепые зоны»
- Все на temperature=0 для детерминированности
- Gemini 2.5 Flash — стабильная baseline-модель
- Gemini 3 Flash — новая быстрая модель, хорошо работает с русским текстом
- Qwen 3.5 397B — большая модель, сильна в нюансах

**Критические сбои (криты):**
- `premature_end` — тьютор закрыл диалог до получения правильного ответа
- `prompt_leak` — утечка системного промпта в реплики
- `nonsense` — неуместные фразы, не соответствующие контексту
- Оценщики флагают → судья подтверждает/отклоняет → подтверждённый крит = адекватность 0

**Почему Claude как судья:**
- Четвёртая модель = полностью независимый взгляд
- Вызывается только когда оценщики расходятся или нашли крит
- Экономия токенов: unanimous критерии принимаются без судьи
""")

    st.subheader("9 критериев оценки")
    criteria_table = []
    for c in CRITERIA:
        criteria_table.append({
            "Критерий": CRITERIA_RU[c],
            "Ключ": c,
            "Шкала": "0-3",
        })
    st.dataframe(pd.DataFrame(criteria_table), use_container_width=True, hide_index=True)

    st.subheader("Критические сбои (криты)")
    crits_table = [
        {"Крит": "Преждевременное завершение", "Ключ": "premature_end",
         "Описание": "Тьютор закрыл диалог до получения правильного ответа"},
        {"Крит": "Утечка промпта", "Ключ": "prompt_leak",
         "Описание": "Фрагменты системных инструкций видны ученику"},
        {"Крит": "Неуместные фразы", "Ключ": "nonsense",
         "Описание": "Фразы не по контексту, галлюцинации, бессмыслица"},
    ]
    st.dataframe(pd.DataFrame(crits_table), use_container_width=True, hide_index=True)
    st.caption("Подтверждённый крит → адекватность автоматически = 0")

    st.subheader("EduScore")
    st.markdown("""
**EduScore** — агрегированная метрика образовательной эффективности ([wiki](https://wiki.yandex-team.ru/yandex-education/komandy-i-proekty/yandex-uchebnik/cmnds/rd/rabochie-materialy/zamery-modelejj/learnlm-zamery-kachestva/learnlm-rabota-nad-kachestvom/metriki/)).

Маппинг баллов:
| Балл (0-3) | Категория | EduScore |
|---|---|---|
| 3 | Да | 1.00 |
| 2 | Скорее да | 0.66 |
| 1 | Скорее нет | 0.33 |
| 0 | Нет | 0.00 |
| -1 | Не применимо | исключается |

$$\\text{EduScore}(d) = \\frac{\\sum_i s(c_i) \\cdot \\mathbf{1}[c_i \\neq \\text{NA}]}{\\sum_i \\mathbf{1}[c_i \\neq \\text{NA}]}$$

Значение 0-1, где 1 = максимальная образовательная ценность.
""")

    st.subheader("Промпт оценщика")
    eval_prompt_path = TUTOR_EVAL_DIR / "prompts" / "evaluator.md"
    if eval_prompt_path.exists():
        with st.expander("evaluator.md", expanded=False):
            st.markdown(eval_prompt_path.read_text(encoding="utf-8"))
    else:
        st.warning(f"Файл не найден: {eval_prompt_path}")

    st.subheader("Промпт судьи (критерии)")
    judge_prompt_path = TUTOR_EVAL_DIR / "prompts" / "judge.md"
    if judge_prompt_path.exists():
        with st.expander("judge.md", expanded=False):
            st.markdown(judge_prompt_path.read_text(encoding="utf-8"))
    else:
        st.warning(f"Файл не найден: {judge_prompt_path}")

    st.subheader("Промпт судьи (криты)")
    judge_crits_path = TUTOR_EVAL_DIR / "prompts" / "judge_crits.md"
    if judge_crits_path.exists():
        with st.expander("judge_crits.md", expanded=False):
            st.markdown(judge_crits_path.read_text(encoding="utf-8"))
    else:
        st.warning(f"Файл не найден: {judge_crits_path}")

    st.subheader("Рубрики")
    rubrics_path = TUTOR_EVAL_DIR / "rubrics.md"
    if rubrics_path.exists():
        with st.expander("rubrics.md", expanded=False):
            st.markdown(rubrics_path.read_text(encoding="utf-8"))
    else:
        st.warning(f"Файл не найден: {rubrics_path}")

    # --- Benchmark results ---
    st.subheader("Точность оценщиков (бенчмарк)")
    benchmark_dir = DATA_DIR / "results" / "benchmarks"
    benchmark_files = sorted(benchmark_dir.glob("benchmark_*.json"), reverse=True) if benchmark_dir.exists() else []

    # Pipeline benchmark
    pipeline_bench_files = sorted(benchmark_dir.glob("pipeline_benchmark_*.json"), reverse=True) if benchmark_dir.exists() else []
    if pipeline_bench_files:
        pr = json.loads(pipeline_bench_files[0].read_text(encoding="utf-8"))
        pipe = pr.get("pipeline", {})
        st.markdown(f"""
**Весь пайплайн (3 оценщика + судья) vs эксперт** — {pr.get('n_evaluated', '?')} диалогов, замер {pr.get('timestamp', '?')}

| Метрика | Значение |
|---|---|
| Exact match | **{pipe.get('exact_match', 0):.1%}** |
| Soft match | **{pipe.get('soft_match', 0):.1%}** |
| Unanimous (без судьи) | {pr.get('n_unanimous', 0)} из {pr.get('n_evaluated', 0)} |
| Судья вызван | {pr.get('n_judge_called', 0)} раз |
""")
        # Per-criterion with P/R/F1
        per_crit = pr.get("per_criterion", {})
        prf = pr.get("prf_per_criterion", {})
        if per_crit:
            with st.expander("Точность пайплайна по критериям", expanded=False):
                crit_rows = []
                for c in CRITERIA:
                    pc = per_crit.get(c, {})
                    p = prf.get(c, {})
                    if pc:
                        crit_rows.append({
                            "Критерий": CRITERIA_RU.get(c, c),
                            "Exact %": f"{pc.get('exact_pct', 0):.0f}%",
                            "Bias": f"{pc.get('mean_diff', 0):+.2f}",
                            "Precision": f"{p.get('precision', 0):.0%}",
                            "Recall": f"{p.get('recall', 0):.0%}",
                            "F1": f"{p.get('f1', 0):.0%}",
                        })
                st.dataframe(pd.DataFrame(crit_rows), use_container_width=True, hide_index=True)
                st.caption("Precision/Recall/F1 считаются для порога хор.(2-3) vs плох.(0-1)")

        st.markdown("---")

    st.markdown("**Точность отдельных оценщиков** (каждый vs эксперт)")

    if benchmark_files:
        report = json.loads(benchmark_files[0].read_text(encoding="utf-8"))
        st.caption(f"Золотой датасет: {report.get('n_examples', '?')} диалогов, размеченных экспертом. "
                   f"Замер: {report.get('timestamp', '?')}")

        models_data = report.get("models", [])
        if models_data:
            # Summary table
            summary_rows = []
            for m in models_data:
                summary_rows.append({
                    "Модель": m.get("model", "?"),
                    "Exact match": f"{m.get('exact_match', 0):.1%}",
                    "Soft match": f"{m.get('soft_match', 0):.1%}",
                    "Оценено": m.get("n_evaluated", 0),
                    "Ошибки": m.get("n_errors", 0),
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            st.caption("**Exact match** — точное совпадение с экспертом. "
                       "**Soft match** — exact=1.0, off-by-1=0.5, off-by-2+=0.0")

            # Per-criterion breakdown
            with st.expander("Детализация по критериям", expanded=False):
                for m in models_data:
                    st.markdown(f"#### {m.get('model', '?')}")
                    per_crit = m.get("per_criterion", {})
                    if per_crit:
                        crit_rows = []
                        for c in CRITERIA:
                            pc = per_crit.get(c, {})
                            if pc:
                                crit_rows.append({
                                    "Критерий": CRITERIA_RU.get(c, c),
                                    "Exact %": f"{pc.get('exact_pct', 0):.0f}%",
                                    "Средний сдвиг": f"{pc.get('mean_diff', 0):+.2f}",
                                    "N": pc.get("n", 0),
                                })
                        st.dataframe(pd.DataFrame(crit_rows), use_container_width=True, hide_index=True)
    else:
        st.info(
            "Бенчмарк не запускался. Запустите:\n\n"
            "```bash\n"
            "uv run python3 -m tutor_eval.benchmark "
            "--gold 'data/compare_products_final*.csv'\n"
            "```"
        )


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.title("Оценка тьютора")
st.caption("Пайплайн: Gemini 2.5 Flash + Gemini 3 Flash + Qwen 3.5 397B → Claude Sonnet 4 (судья)")

runs = _find_runs()

if not runs:
    st.info(
        "Нет данных. Запустите пайплайн:\n\n"
        "```bash\n"
        "./tutor_eval/run.sh --file path/to/dialogs.xlsx --sample 5\n"
        "```"
    )
    _render_reference()
    st.stop()

# --- Sidebar ---
selected_run = st.sidebar.selectbox("Прогон", runs, index=0)
scale_choice = st.sidebar.radio("Шкала", list(SCALE_OPTIONS.keys()), index=0, horizontal=True)
scale = SCALE_OPTIONS[scale_choice]

raw_scores = _load_raw_scores(selected_run)
judge_decisions = _load_judge_decisions(selected_run)
summary = _load_summary(selected_run)
errors = _load_errors(selected_run)
dialogs = _load_dialogs(selected_run)
math_results = _find_math_results(selected_run)
math_error_info = _compute_error_rate(math_results)

# ---------------------------------------------------------------------------
# Pre-compute overview data (used in overview bar + tabs)
# ---------------------------------------------------------------------------
crits_data = []
crits_by_dialog = {}  # dialog_id -> list of confirmed crit names
crits_categories_by_dialog = {}  # dialog_id -> list of "crit:category" strings
if judge_decisions:
    for jd in judge_decisions:
        cf = jd.get("critical_flags", {})
        confirmed = cf.get("confirmed", {})
        flagged = cf.get("flagged", {})
        if confirmed:
            crits_by_dialog[jd["dialog_id"]] = list(confirmed.keys())
            cats = []
            for crit_name, crit_info in confirmed.items():
                cat = crit_info.get("category") if isinstance(crit_info, dict) else None
                cats.append(f"{crit_name}:{cat}" if cat else crit_name)
            crits_categories_by_dialog[jd["dialog_id"]] = cats
        if confirmed or flagged:
            crits_data.append({
                "dialog_id": jd["dialog_id"],
                "confirmed": confirmed,
                "flagged": flagged,
            })

total_confirmed = sum(len(c["confirmed"]) for c in crits_data)
dialogs_with_crits = sum(1 for c in crits_data if c["confirmed"])

# Per-dialog final scores for quick lookups
dialog_scores = {}  # dialog_id -> {criterion: score, "_avg": float}
if judge_decisions:
    for jd in judge_decisions:
        final = jd.get("final_scores", {})
        scores = {}
        valid = []
        for c in CRITERIA:
            s = final.get(c, {}).get("score")
            if s is not None:
                scores[c] = int(s)
                if int(s) >= 0:
                    valid.append(int(s))
        scores["_avg"] = round(sum(valid) / len(valid), 2) if valid else 0
        scores["_eduscore"] = _eduscore_dialog(scores)
        dialog_scores[jd["dialog_id"]] = scores

# Global averages
all_avgs = [s["_avg"] for s in dialog_scores.values()]
global_avg = round(sum(all_avgs) / len(all_avgs), 2) if all_avgs else 0
all_eduscores = [s["_eduscore"] for s in dialog_scores.values() if s["_eduscore"] is not None]
global_eduscore = round(sum(all_eduscores) / len(all_eduscores), 3) if all_eduscores else 0
low_score_dialogs = sum(1 for s in dialog_scores.values() if s["_avg"] <= 1.0)

score_label = _score_label(scale)
score_lo, score_hi = _score_range(scale)
axis_range = _score_axis_range(scale)

# ---------------------------------------------------------------------------
# Overview bar (always visible above tabs)
# ---------------------------------------------------------------------------
st.markdown("---")
ov_cols = st.columns(8)
evaluated = len(judge_decisions) if judge_decisions else (summary.get("evaluated", 0) if summary else 0)
total_dialogs = summary.get("total_dialogs", evaluated) if summary else evaluated
ov_cols[0].metric("Диалогов", total_dialogs)
ov_cols[1].metric("Оценено", evaluated)
ov_cols[2].metric("EduScore", f"{global_eduscore:.2f}")
avg_display = f"{_to_display_score(global_avg, scale)}{('%' if scale == 'percent' else '')}"
ov_cols[3].metric("Средняя оценка", avg_display)
if math_error_info:
    er = math_error_info["error_rate"]
    ov_cols[4].metric("Error Rate", f"{er:.1%}", help="Доля диалогов с ≥1 мат. ошибкой")
else:
    ov_cols[4].metric("Error Rate", "—", help="Нет данных math_check")
ov_cols[5].metric("Низкие (<=1.0)", low_score_dialogs)
ov_cols[6].metric("Криты", f"{dialogs_with_crits} диал.")
ov_cols[7].metric("Ошибок", summary.get("errors", 0) if summary else 0)
st.markdown("---")

tab_charts, tab_models, tab_evaluators, tab_markup, tab_gold, tab_reference = st.tabs([
    "Результаты", "По моделям", "Анализ оценщиков", "Разметка", "Ручной замер", "Справка по пайплайну",
])

# ===================================================================
# TAB 1: Charts
# ===================================================================
with tab_charts:

    # --- Crits detail ---
    if crits_data:
        st.subheader("Критические сбои")

        # Crit type breakdown
        crit_type_counts = {"premature_end": 0, "prompt_leak": 0, "nonsense": 0}
        crit_type_flagged = {"premature_end": 0, "prompt_leak": 0, "nonsense": 0}
        for c in crits_data:
            for ctype in c["confirmed"]:
                if ctype in crit_type_counts:
                    crit_type_counts[ctype] += 1
            for ctype in c["flagged"]:
                if ctype in crit_type_flagged:
                    crit_type_flagged[ctype] += 1

        crit_ru = {
            "premature_end": "Преждевременное завершение",
            "prompt_leak": "Утечка промпта",
            "nonsense": "Неуместные фразы",
        }

        crit_df = pd.DataFrame([
            {
                "Тип": crit_ru.get(k, k),
                "Подтверждено": crit_type_counts[k],
                "Флагнуто (всего)": crit_type_flagged[k],
                "Отклонено": crit_type_flagged[k] - crit_type_counts[k],
            }
            for k in ["premature_end", "prompt_leak", "nonsense"]
        ])
        st.dataframe(crit_df, use_container_width=True, hide_index=True)

        # List of dialogs with confirmed crits
        if dialogs_with_crits > 0:
            with st.expander(f"Диалоги с подтверждёнными критами ({dialogs_with_crits})", expanded=False):
                for c in crits_data:
                    if c["confirmed"]:
                        cnames = ", ".join(crit_ru.get(k, k) for k in c["confirmed"])
                        st.markdown(f"- `{c['dialog_id']}` — {cnames}")

    if judge_decisions:
        df_judge = _judge_to_df(judge_decisions)

        if not df_judge.empty:
            df_jv = df_judge[df_judge["final_score"] >= 0].copy()
            df_jv["display_score"] = df_jv["final_score"].apply(
                lambda s: _to_display_score(s, scale)
            )

            # Bar chart: average final score per criterion
            st.subheader("Средний балл по критериям (финальный)")
            avg_by_crit = (
                df_jv.groupby("criterion_ru")["display_score"]
                .mean()
                .reset_index()
                .rename(columns={"display_score": "Средний балл"})
            )
            avg_by_crit["Средний балл"] = avg_by_crit["Средний балл"].apply(lambda v: _round_for_scale(v, scale))
            order = [CRITERIA_RU[c] for c in CRITERIA]
            avg_by_crit["criterion_ru"] = pd.Categorical(
                avg_by_crit["criterion_ru"], categories=order, ordered=True
            )
            avg_by_crit = avg_by_crit.sort_values("criterion_ru")

            fig_bar = px.bar(
                avg_by_crit,
                x="criterion_ru",
                y="Средний балл",
                color="Средний балл",
                color_continuous_scale="RdYlGn",
                range_color=[score_lo, score_hi],
                text="Средний балл",
                category_orders={"criterion_ru": order},
            )
            fmt = _text_fmt(scale)
            fig_bar.update_traces(texttemplate=fmt, textposition="outside")
            fig_bar.update_layout(
                xaxis_title="",
                yaxis_title=f"Балл ({score_label})",
                yaxis_range=axis_range,
                coloraxis_showscale=False,
                height=400,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Agreement distribution
            st.subheader("Уровень согласия оценщиков")
            agreement_counts = df_judge["agreement"].value_counts().reset_index()
            agreement_counts.columns = ["Уровень", "Кол-во"]
            fig_agree = px.pie(
                agreement_counts,
                values="Кол-во",
                names="Уровень",
                color="Уровень",
                color_discrete_map={
                    "unanimous": "#2ecc71",
                    "majority": "#f39c12",
                    "split": "#e74c3c",
                },
            )
            fig_agree.update_layout(height=350)
            st.plotly_chart(fig_agree, use_container_width=True)

    if raw_scores:
        df_raw = _raw_to_df(raw_scores)

        if not df_raw.empty:
            df_rv = df_raw[df_raw["score"] >= 0].copy()
            df_rv["display_score"] = df_rv["score"].apply(
                lambda s: _to_display_score(s, scale)
            )

            # Box plot per evaluator
            st.subheader("Распределение баллов по оценщикам")
            fig_box = px.box(
                df_rv,
                x="criterion_ru",
                y="display_score",
                color="evaluator",
                category_orders={"criterion_ru": [CRITERIA_RU[c] for c in CRITERIA]},
            )
            fig_box.update_layout(
                xaxis_title="",
                yaxis_title=f"Балл ({score_label})",
                yaxis_range=[-5 if scale == "percent" else (-0.05 if scale == "normalized" else -0.5), axis_range[1]],
                height=450,
                legend_title="Оценщик",
            )
            st.plotly_chart(fig_box, use_container_width=True)

            # Heatmap
            st.subheader("Тепловая карта: баллы по диалогам")
            first_eval = df_rv["evaluator"].unique()[0]
            df_heat = df_rv[df_rv["evaluator"] == first_eval].pivot(
                index="dialog_id", columns="criterion_ru", values="display_score"
            )
            cols_order = [CRITERIA_RU[c] for c in CRITERIA if CRITERIA_RU[c] in df_heat.columns]
            df_heat = df_heat[cols_order]

            fig_heat = px.imshow(
                df_heat.values,
                x=cols_order,
                y=list(df_heat.index),
                color_continuous_scale="RdYlGn",
                zmin=score_lo,
                zmax=score_hi,
                aspect="auto",
            )
            fig_heat.update_layout(
                xaxis_title="",
                yaxis_title="Диалог",
                height=max(300, len(df_heat) * 25),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

    if errors:
        st.subheader("Ошибки")
        st.dataframe(pd.DataFrame(errors), use_container_width=True)

    # --- Correctness (math errors) ---
    if math_error_info:
        st.markdown("---")
        st.subheader("Корректность (мат. ошибки)")
        st.markdown(
            "По [методологии wiki](https://wiki.yandex-team.ru/yandex-education/komandy-i-proekty/yandex-uchebnik/"
            "cmnds/rd/rabochie-materialy/zamery-modelejj/learnlm-zamery-kachestva/learnlm-rabota-nad-kachestvom/metriki/): "
            f"**error\\_rate** = доля диалогов с ≥1 ошибкой."
        )

        er_cols = st.columns(4)
        er_cols[0].metric("Error Rate", f"{math_error_info['error_rate']:.1%}")
        er_cols[1].metric(
            "Ошибка тьютора",
            math_error_info["by_class"].get("tutor_error", 0),
            help="Мат. ошибка в вычислениях/объяснениях тьютора",
        )
        er_cols[2].metric(
            "Принял неверное",
            math_error_info["by_class"].get("accepted_incorrect", 0),
            help="Тьютор принял неверный ответ ученика",
        )
        er_cols[3].metric(
            "Заругал верное",
            math_error_info["by_class"].get("rejected_correct", 0),
            help="Тьютор не принял верный ответ ученика",
        )


# ===================================================================
# TAB 2: By model
# ===================================================================
with tab_models:
    st.subheader("Оценки по моделям железного учителя")

    if not judge_decisions or not dialogs:
        st.warning("Нет данных (нужен прогон с сохранённой метаинформацией о диалогах)")
    else:
        df_judge = _judge_to_df(judge_decisions)
        df_judge = _enrich_with_model(df_judge, dialogs)
        df_jv = df_judge[df_judge["final_score"] >= 0].copy()

        if df_jv.empty:
            st.warning("Нет валидных оценок")
        else:
            df_jv["display_score"] = df_jv["final_score"].apply(
                lambda s: _to_display_score(s, scale)
            )

            models = sorted(df_jv["student_model"].unique())
            model_dialog_counts = (
                df_jv.groupby("student_model")["dialog_id"]
                .nunique()
                .sort_values(ascending=False)
            )
            count_parts = [f"{m}: {model_dialog_counts[m]}" for m in model_dialog_counts.index]
            st.caption(f"Модели ({len(df_jv['dialog_id'].unique())} диалогов): {', '.join(count_parts)}")

            # Grouped bar: model x criterion
            st.markdown("#### Средний балл: модель × критерий")
            avg_mc = (
                df_jv.groupby(["student_model", "criterion_ru"])["display_score"]
                .mean()
                .reset_index()
                .rename(columns={"display_score": "Средний балл"})
            )
            avg_mc["Средний балл"] = avg_mc["Средний балл"].apply(lambda v: _round_for_scale(v, scale))
            order = [CRITERIA_RU[c] for c in CRITERIA]
            avg_mc["criterion_ru"] = pd.Categorical(
                avg_mc["criterion_ru"], categories=order, ordered=True
            )

            fig_mc = px.bar(
                avg_mc,
                x="criterion_ru",
                y="Средний балл",
                color="student_model",
                barmode="group",
                text="Средний балл",
                category_orders={"criterion_ru": order},
            )
            fmt = _text_fmt(scale)
            fig_mc.update_traces(texttemplate=fmt, textposition="outside")
            fig_mc.update_layout(
                xaxis_title="",
                yaxis_title=f"Балл ({score_label})",
                yaxis_range=axis_range,
                height=450,
                legend_title="Модель",
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            # Overall average per model
            st.markdown("#### Общий средний балл по модели")
            avg_model = (
                df_jv.groupby("student_model")["display_score"]
                .mean()
                .reset_index()
                .rename(columns={"display_score": "Средний балл"})
                .sort_values("Средний балл", ascending=False)
            )
            avg_model["Средний балл"] = avg_model["Средний балл"].apply(lambda v: _round_for_scale(v, scale))

            fig_overall = px.bar(
                avg_model,
                x="student_model",
                y="Средний балл",
                color="Средний балл",
                color_continuous_scale="RdYlGn",
                range_color=[score_lo, score_hi],
                text="Средний балл",
            )
            fmt = _text_fmt(scale)
            fig_overall.update_traces(texttemplate=fmt, textposition="outside")
            fig_overall.update_layout(
                xaxis_title="Модель",
                yaxis_title=f"Балл ({score_label})",
                yaxis_range=axis_range,
                coloraxis_showscale=False,
                height=350,
            )
            st.plotly_chart(fig_overall, use_container_width=True)

            # EduScore per model
            st.markdown("#### EduScore по моделям")
            edu_rows = []
            for did, scores in dialog_scores.items():
                edu = scores.get("_eduscore")
                if edu is not None:
                    model = next(
                        (d.get("student_model") or _model_from_dialog_id(did)
                         for d in [dialogs.get(did, {})]
                        ),
                        _model_from_dialog_id(did),
                    )
                    edu_rows.append({"student_model": model, "EduScore": edu})
            if edu_rows:
                edu_df = pd.DataFrame(edu_rows)
                edu_avg = (
                    edu_df.groupby("student_model")["EduScore"]
                    .agg(["mean", "count"])
                    .reset_index()
                    .rename(columns={"mean": "EduScore", "count": "N"})
                    .sort_values("EduScore", ascending=False)
                )
                edu_avg["EduScore"] = edu_avg["EduScore"].round(3)
                fig_edu = px.bar(
                    edu_avg,
                    x="student_model",
                    y="EduScore",
                    color="EduScore",
                    color_continuous_scale="RdYlGn",
                    range_color=[0, 1],
                    text="EduScore",
                )
                fig_edu.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig_edu.update_layout(
                    xaxis_title="Модель",
                    yaxis_title="EduScore",
                    yaxis_range=[0, 1.15],
                    coloraxis_showscale=False,
                    height=350,
                )
                st.plotly_chart(fig_edu, use_container_width=True)

            # Radar chart per model
            st.markdown("#### Профиль моделей (радар)")
            fig_radar = go.Figure()
            for model in models:
                model_data = avg_mc[avg_mc["student_model"] == model].sort_values("criterion_ru")
                fig_radar.add_trace(go.Scatterpolar(
                    r=list(model_data["Средний балл"]) + [model_data["Средний балл"].iloc[0]],
                    theta=list(model_data["criterion_ru"]) + [model_data["criterion_ru"].iloc[0]],
                    name=model,
                    fill="toself",
                    opacity=0.5,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(range=[score_lo, score_hi])),
                height=500,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Table
            st.markdown("#### Сводная таблица")
            pivot = df_jv.pivot_table(
                index="student_model",
                columns="criterion_ru",
                values="display_score",
                aggfunc="mean",
            ).apply(lambda v: _round_for_scale(v, scale))
            pivot = pivot[[c for c in order if c in pivot.columns]]
            pivot["Среднее"] = pivot.mean(axis=1).apply(lambda v: _round_for_scale(v, scale))
            st.dataframe(pivot, use_container_width=True)


# ===================================================================
# TAB 3: Evaluator analysis
# ===================================================================
with tab_evaluators:
    st.subheader("Анализ оценщиков: кто завышает, кто занижает")

    if not raw_scores or not judge_decisions:
        st.warning("Нужны данные raw_scores и judge_decisions")
    else:
        df_raw = _raw_to_df(raw_scores)
        df_judge = _judge_to_df(judge_decisions)

        if df_raw.empty or df_judge.empty:
            st.warning("Нет валидных оценок")
        else:
            # Filter out -1
            df_rv = df_raw[df_raw["score"] >= 0].copy()
            df_jv = df_judge[df_judge["final_score"] >= 0].copy()

            # Merge raw with final to compute deviation
            df_merged = df_rv.merge(
                df_jv[["dialog_id", "criterion", "final_score"]],
                on=["dialog_id", "criterion"],
                how="inner",
            )
            df_merged["deviation"] = df_merged["score"] - df_merged["final_score"]

            evaluators = sorted(df_merged["evaluator"].unique())

            # --- Mean deviation per evaluator (bias) ---
            st.markdown("#### Средний сдвиг от финальной оценки")
            st.caption("Положительный = завышает, отрицательный = занижает")
            avg_dev = (
                df_merged.groupby("evaluator")["deviation"]
                .mean()
                .reset_index()
                .rename(columns={"deviation": "Средний сдвиг"})
                .sort_values("Средний сдвиг")
            )
            avg_dev["Средний сдвиг"] = avg_dev["Средний сдвиг"].round(3)

            colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in avg_dev["Средний сдвиг"]]
            fig_bias = px.bar(
                avg_dev,
                x="evaluator",
                y="Средний сдвиг",
                text="Средний сдвиг",
                color="evaluator",
            )
            fig_bias.update_traces(texttemplate="%{text:+.3f}", textposition="outside")
            fig_bias.update_layout(
                xaxis_title="Оценщик",
                yaxis_title="Сдвиг (оценка - финал)",
                yaxis_range=[-1.5, 1.5],
                showlegend=False,
                height=350,
            )
            st.plotly_chart(fig_bias, use_container_width=True)

            # --- Deviation per evaluator per criterion ---
            st.markdown("#### Сдвиг по критериям")
            order = [CRITERIA_RU[c] for c in CRITERIA]
            dev_by_crit = (
                df_merged.groupby(["evaluator", "criterion_ru"])["deviation"]
                .mean()
                .reset_index()
                .rename(columns={"deviation": "Сдвиг"})
            )
            dev_by_crit["Сдвиг"] = dev_by_crit["Сдвиг"].round(2)
            dev_by_crit["criterion_ru"] = pd.Categorical(
                dev_by_crit["criterion_ru"], categories=order, ordered=True
            )

            fig_dev_crit = px.bar(
                dev_by_crit,
                x="criterion_ru",
                y="Сдвиг",
                color="evaluator",
                barmode="group",
                text="Сдвиг",
                category_orders={"criterion_ru": order},
            )
            fig_dev_crit.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
            fig_dev_crit.update_layout(
                xaxis_title="",
                yaxis_title="Сдвиг (оценка - финал)",
                yaxis_range=[-2, 2],
                height=450,
                legend_title="Оценщик",
            )
            st.plotly_chart(fig_dev_crit, use_container_width=True)

            # --- Mean score per evaluator ---
            st.markdown("#### Средний балл по оценщикам")
            df_rv["display_score"] = df_rv["score"].apply(
                lambda s: _to_display_score(s, scale)
            )
            avg_eval = (
                df_rv.groupby("evaluator")["display_score"]
                .mean()
                .reset_index()
                .rename(columns={"display_score": "Средний балл"})
                .sort_values("Средний балл", ascending=False)
            )
            avg_eval["Средний балл"] = avg_eval["Средний балл"].apply(lambda v: _round_for_scale(v, scale))

            fig_avg_eval = px.bar(
                avg_eval,
                x="evaluator",
                y="Средний балл",
                color="Средний балл",
                color_continuous_scale="RdYlGn",
                range_color=[score_lo, score_hi],
                text="Средний балл",
            )
            fmt = _text_fmt(scale)
            fig_avg_eval.update_traces(texttemplate=fmt, textposition="outside")
            fig_avg_eval.update_layout(
                xaxis_title="Оценщик",
                yaxis_title=f"Балл ({score_label})",
                yaxis_range=axis_range,
                coloraxis_showscale=False,
                height=350,
            )
            st.plotly_chart(fig_avg_eval, use_container_width=True)

            # --- Agreement with final per evaluator ---
            st.markdown("#### Точное совпадение с финальным баллом")
            df_merged["exact_match"] = (df_merged["deviation"] == 0).astype(int)
            match_rate = (
                df_merged.groupby("evaluator")["exact_match"]
                .mean()
                .reset_index()
                .rename(columns={"exact_match": "Совпадение"})
                .sort_values("Совпадение", ascending=False)
            )
            match_rate["Совпадение %"] = (match_rate["Совпадение"] * 100).round(1)

            fig_match = px.bar(
                match_rate,
                x="evaluator",
                y="Совпадение %",
                text="Совпадение %",
                color="Совпадение %",
                color_continuous_scale="RdYlGn",
                range_color=[0, 100],
            )
            fig_match.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_match.update_layout(
                xaxis_title="Оценщик",
                yaxis_title="% совпадений",
                yaxis_range=[0, 110],
                coloraxis_showscale=False,
                height=350,
            )
            st.plotly_chart(fig_match, use_container_width=True)

            # --- Distribution of deviations ---
            st.markdown("#### Распределение отклонений")
            fig_hist = px.histogram(
                df_merged,
                x="deviation",
                color="evaluator",
                barmode="overlay",
                nbins=7,
                opacity=0.7,
            )
            fig_hist.update_layout(
                xaxis_title="Отклонение от финала",
                yaxis_title="Кол-во",
                height=350,
                legend_title="Оценщик",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # --- Pivot table ---
            st.markdown("#### Сводная таблица отклонений")
            pivot_dev = df_merged.pivot_table(
                index="evaluator",
                columns="criterion_ru",
                values="deviation",
                aggfunc="mean",
            ).round(2)
            pivot_dev = pivot_dev[[c for c in order if c in pivot_dev.columns]]
            pivot_dev["Среднее"] = pivot_dev.mean(axis=1).round(2)
            st.dataframe(
                pivot_dev.style.background_gradient(cmap="RdYlGn_r", vmin=-1, vmax=1),
                use_container_width=True,
            )


# ===================================================================
# TAB 4: Markup view
# ===================================================================
with tab_markup:
    st.subheader("Разметка: как принималось решение")

    if not judge_decisions:
        st.warning("Нет данных judge_decisions")
    else:
        # --- Filters ---
        all_dialog_ids = [d["dialog_id"] for d in judge_decisions]

        with st.expander("Фильтры", expanded=True):
            filter_cols = st.columns(4)

            with filter_cols[0]:
                problem_filter = st.multiselect("Проблемы", [
                    "Криты (подтверждённые)",
                    "Адекватность = 0",
                    "Средняя оценка <= 1.0",
                    "Есть split",
                    "Есть расхождения (не unanimous)",
                ], default=[])

            with filter_cols[1]:
                # Low score on specific criterion
                crit_filter = st.selectbox("Низкая оценка по критерию", ["Все"] + [CRITERIA_RU[c] for c in CRITERIA])
                crit_threshold = st.slider("Порог", 0, 3, 1, key="crit_threshold") if crit_filter != "Все" else None

            with filter_cols[2]:
                available_models = sorted(set(
                    dialogs.get(did, {}).get("student_model") or _model_from_dialog_id(did) for did in all_dialog_ids
                ))
                model_filter = st.multiselect("Модель ученика", available_models, default=[])

            with filter_cols[3]:
                available_types = sorted(set(
                    dialogs.get(did, {}).get("student_type", "unknown") for did in all_dialog_ids
                ))
                type_filter = st.multiselect("Тип ученика", available_types, default=[])

            # Second row of filters
            filter_cols2 = st.columns(4)
            with filter_cols2[0]:
                all_crit_cats = sorted(set(
                    cat for cats in crits_categories_by_dialog.values() for cat in cats
                ))
                crit_type_filter = st.multiselect("Тип крита", all_crit_cats, default=[])

        # Apply filters
        filtered_ids = all_dialog_ids.copy()

        if model_filter:
            filtered_ids = [did for did in filtered_ids
                           if (dialogs.get(did, {}).get("student_model") or _model_from_dialog_id(did)) in model_filter]
        if type_filter:
            filtered_ids = [did for did in filtered_ids
                           if dialogs.get(did, {}).get("student_type") in type_filter]

        for pf in problem_filter:
            if pf == "Криты (подтверждённые)":
                filtered_ids = [did for did in filtered_ids if did in crits_by_dialog]
            elif pf == "Адекватность = 0":
                filtered_ids = [did for did in filtered_ids
                               if dialog_scores.get(did, {}).get("adequacy") == 0]
            elif pf == "Средняя оценка <= 1.0":
                filtered_ids = [did for did in filtered_ids
                               if dialog_scores.get(did, {}).get("_avg", 99) <= 1.0]
            elif pf == "Есть split":
                filtered_ids = [did for did in filtered_ids
                               if any(jd.get("agreement", {}).get(c) == "split"
                                      for jd in judge_decisions if jd["dialog_id"] == did
                                      for c in CRITERIA)]
            elif pf == "Есть расхождения (не unanimous)":
                filtered_ids = [did for did in filtered_ids
                               if any(jd.get("agreement", {}).get(c) not in ("unanimous", "")
                                      for jd in judge_decisions if jd["dialog_id"] == did
                                      for c in CRITERIA)]

        if crit_type_filter:
            filtered_ids = [did for did in filtered_ids
                           if any(cat in crits_categories_by_dialog.get(did, []) for cat in crit_type_filter)]

        if crit_filter != "Все" and crit_threshold is not None:
            crit_key = next((k for k, v in CRITERIA_RU.items() if v == crit_filter), None)
            if crit_key:
                filtered_ids = [did for did in filtered_ids
                               if 0 <= dialog_scores.get(did, {}).get(crit_key, 99) <= crit_threshold]

        st.caption(f"Показано: {len(filtered_ids)} из {len(all_dialog_ids)} диалогов")

        if not filtered_ids:
            st.info("Нет диалогов, соответствующих фильтрам")
        else:
            # Format labels with score + crit badge
            def _label(did):
                ds = dialog_scores.get(did, {})
                edu = ds.get("_eduscore")
                edu_str = f"{edu:.2f}" if edu is not None else "?"
                parts = [f"{did} (EduScore={edu_str})"]
                if did in crits_by_dialog:
                    parts.append("КРИТ")
                return " ".join(parts)

            selected_dialog = st.selectbox(
                "Диалог",
                filtered_ids,
                format_func=_label,
            )

            # Show full dialog text
            dialog_meta = dialogs.get(selected_dialog, {})
            if dialog_meta:
                task_text = dialog_meta.get("task", "")
                grade = dialog_meta.get("grade_group", "")
                theme = dialog_meta.get("theme", "")
                subtheme = dialog_meta.get("subtheme", "")
                model = dialog_meta.get("student_model", "")
                stype = dialog_meta.get("student_type", "")

                st.caption(f"Модель: **{model}** | Тип ученика: **{stype}** | Класс: **{grade}** | Тема: {theme} → {subtheme}")

                with st.expander("Текст диалога", expanded=True):
                    if task_text:
                        st.markdown(f"**Задача:** {task_text}")
                        st.markdown("---")
                    _render_dialog_text(dialog_meta.get("text", ""))

            # Find judge decision for this dialog
            decision = next(
                (d for d in judge_decisions if d["dialog_id"] == selected_dialog),
                None,
            )
            dialog_raw = [r for r in raw_scores if r["dialog_id"] == selected_dialog]

            if decision:
                st.markdown("---")

                for c in CRITERIA:
                    c_ru = CRITERIA_RU[c]
                    final = decision.get("final_scores", {}).get(c, {})
                    agree = decision.get("agreement", {}).get(c, "")

                    final_score = final.get("score", "?")
                    final_reason = final.get("reasoning", "")

                    # Format score display
                    if isinstance(final_score, int) and final_score >= 0:
                        display = f"{_to_display_score(final_score, scale)}{('%' if scale == 'percent' else '')}"
                    elif isinstance(final_score, int) and final_score < 0:
                        display = "н/п"
                    else:
                        display = "?"

                    if agree == "unanimous":
                        agree_badge = ":green[unanimous]"
                    elif agree == "majority":
                        agree_badge = ":orange[majority]"
                    else:
                        agree_badge = ":red[split]"

                    st.markdown(f"### {c_ru} — **{display}** {agree_badge}")

                    eval_cols = st.columns(len(dialog_raw)) if dialog_raw else []
                    for idx, raw_entry in enumerate(dialog_raw):
                        with eval_cols[idx]:
                            evaluator_name = raw_entry.get("evaluator", f"eval_{idx}")
                            cdata = raw_entry.get("criteria", {}).get(c, {})
                            ev_score = cdata.get("score", "?")
                            ev_reason = cdata.get("reasoning", "")
                            ev_evidence = cdata.get("evidence", [])

                            if isinstance(ev_score, int) and ev_score >= 0:
                                ev_display = f"{_to_display_score(ev_score, scale)}{('%' if scale == 'percent' else '')}"
                            elif isinstance(ev_score, int) and ev_score < 0:
                                ev_display = "н/п"
                            else:
                                ev_display = f"{ev_score}"

                            st.markdown(f"**{evaluator_name}**: **{ev_display}**")
                            if ev_reason:
                                st.caption(ev_reason)
                            if ev_evidence:
                                for e in ev_evidence[:2]:
                                    st.markdown(f"> {e[:200]}")

                    if final_reason:
                        st.info(f"**Судья:** {final_reason}")

                    st.markdown("---")

                # Critical flags
                crits = decision.get("critical_flags", {})
                flagged = crits.get("flagged", {})
                confirmed = crits.get("confirmed", {})

                if flagged or confirmed:
                    st.subheader("Критические сбои")
                    for flag_name in ["premature_end", "prompt_leak", "nonsense"]:
                        flag_ru = {
                            "premature_end": "Преждевременное завершение",
                            "prompt_leak": "Утечка промпта",
                            "nonsense": "Неуместные фразы",
                        }.get(flag_name, flag_name)

                        if flag_name in confirmed:
                            entry = confirmed[flag_name]
                            cat = entry.get("category") if isinstance(entry, dict) else None
                            cat_label = f" [`{cat}`]" if cat else ""
                            st.error(f"**{flag_ru}**{cat_label} — ПОДТВЕРЖДЁН судьёй: {entry.get('reasoning', '')}")
                        elif flag_name in flagged:
                            entries = flagged[flag_name]
                            who = ", ".join(e.get("evaluator", "?") for e in entries)
                            st.warning(f"**{flag_ru}** — флагнули: {who} (судья отклонил)")

                overrides = decision.get("overrides", [])
                if overrides:
                    st.subheader("Переопределения судьи")
                    for o in overrides:
                        st.warning(o)

# ===================================================================
# TAB 5: Gold standard analysis
# ===================================================================
with tab_gold:
    st.subheader("Ручной замер (golden set)")

    import csv as csv_mod

    GOLD_SCORE_MAP = {"Да": 3, "да": 3, "Скорее да": 2, "скорее да": 2,
                      "Скорее нет": 1, "скорее нет": 1, "Нет": 0, "нет": 0,
                      "не применимо": -1}
    GOLD_COLS = {6: "expectations", 7: "transparency", 8: "learning_goal",
                 9: "adaptivity", 10: "simplicity", 11: "encourages_thinking",
                 12: "error_handling", 13: "friendly", 14: "adequacy"}

    gold_path = DATA_DIR.parent / "compare_products_final - железный учитель (gemini 3) (1).csv"

    if not gold_path.exists():
        st.warning(f"Golden set не найден: {gold_path}")
    else:
        @st.cache_data
        def _load_gold_data(path: str):
            examples = []
            with open(path, encoding="utf-8") as f:
                reader = csv_mod.reader(f)
                headers = next(reader)
                for row in reader:
                    if len(row) < 15:
                        continue
                    raw = [row[i].strip() for i in range(6, 15)]
                    if not any(s in GOLD_SCORE_MAP for s in raw):
                        continue
                    scores = {}
                    for col_idx, crit in GOLD_COLS.items():
                        scores[crit] = GOLD_SCORE_MAP.get(row[col_idx].strip(), -1)
                    edu = _eduscore_dialog(scores)
                    examples.append({
                        "task": row[1].strip(),
                        "role": row[2].strip(),
                        "dialog": row[5].strip() if len(row) > 5 and row[5].strip() else row[3].strip(),
                        "scores": scores,
                        "eduscore": edu,
                    })
            return examples

        gold_examples = _load_gold_data(str(gold_path))

        if not gold_examples:
            st.warning("Нет данных в golden set")
        else:
            # --- Overview metrics ---
            gold_eduscores = [e["eduscore"] for e in gold_examples if e["eduscore"] is not None]
            gold_avg_edu = sum(gold_eduscores) / len(gold_eduscores) if gold_eduscores else 0

            gcols = st.columns(4)
            gcols[0].metric("Диалогов", len(gold_examples))
            gcols[1].metric("EduScore (среднее)", f"{gold_avg_edu:.3f}")
            roles = {}
            for e in gold_examples:
                roles.setdefault(e["role"], 0)
                roles[e["role"]] += 1
            gcols[2].metric("Роли", ", ".join(f"{k}: {v}" for k, v in sorted(roles.items())))
            gcols[3].metric("Источник", "Эксперт (Gemini 3 Flash)")

            # --- EduScore distribution ---
            st.subheader("Распределение EduScore")
            edu_df = pd.DataFrame({"EduScore": gold_eduscores})
            fig_edu_hist = px.histogram(edu_df, x="EduScore", nbins=15, color_discrete_sequence=["#2ecc71"])
            fig_edu_hist.update_layout(
                xaxis_title="EduScore", yaxis_title="Кол-во диалогов",
                xaxis_range=[0, 1.05], height=300,
            )
            st.plotly_chart(fig_edu_hist, use_container_width=True)

            # --- EduScore by role ---
            st.subheader("EduScore по типу ученика")
            role_edu = pd.DataFrame([
                {"Роль": e["role"], "EduScore": e["eduscore"]}
                for e in gold_examples if e["eduscore"] is not None
            ])
            if not role_edu.empty:
                fig_role = px.box(role_edu, x="Роль", y="EduScore", color="Роль", points="all")
                fig_role.update_layout(yaxis_range=[0, 1.05], height=350, showlegend=False)
                st.plotly_chart(fig_role, use_container_width=True)

            # --- Средний балл по критериям ---
            st.subheader("Средний балл по критериям (экспертная оценка)")
            crit_scores = {c: [] for c in CRITERIA}
            for e in gold_examples:
                for c in CRITERIA:
                    s = e["scores"].get(c)
                    if s is not None and s >= 0:
                        crit_scores[c].append(s)

            crit_avg_rows = []
            for c in CRITERIA:
                vals = crit_scores[c]
                if vals:
                    avg = sum(vals) / len(vals)
                    display_val = _to_display_score(avg, scale)
                    crit_avg_rows.append({
                        "criterion_ru": CRITERIA_RU[c],
                        "Средний балл": display_val,
                        "N": len(vals),
                    })

            if crit_avg_rows:
                crit_avg_df = pd.DataFrame(crit_avg_rows)
                order = [CRITERIA_RU[c] for c in CRITERIA]
                crit_avg_df["criterion_ru"] = pd.Categorical(
                    crit_avg_df["criterion_ru"], categories=order, ordered=True
                )
                crit_avg_df = crit_avg_df.sort_values("criterion_ru")

                fig_gold_bar = px.bar(
                    crit_avg_df, x="criterion_ru", y="Средний балл",
                    color="Средний балл", color_continuous_scale="RdYlGn",
                    range_color=[score_lo, score_hi], text="Средний балл",
                    category_orders={"criterion_ru": order},
                )
                fmt = _text_fmt(scale)
                fig_gold_bar.update_traces(texttemplate=fmt, textposition="outside")
                fig_gold_bar.update_layout(
                    xaxis_title="", yaxis_title=f"Балл ({score_label})",
                    yaxis_range=axis_range, coloraxis_showscale=False, height=400,
                )
                st.plotly_chart(fig_gold_bar, use_container_width=True)

            # --- Score distribution per criterion ---
            st.subheader("Распределение оценок по критериям")
            dist_rows = []
            for e in gold_examples:
                for c in CRITERIA:
                    s = e["scores"].get(c)
                    if s is not None and s >= 0:
                        dist_rows.append({
                            "Критерий": CRITERIA_RU[c],
                            "Балл": s,
                        })
            if dist_rows:
                dist_df = pd.DataFrame(dist_rows)
                fig_dist = px.histogram(
                    dist_df, x="Балл", color="Критерий", barmode="group",
                    category_orders={"Критерий": [CRITERIA_RU[c] for c in CRITERIA]},
                    nbins=4,
                )
                fig_dist.update_layout(
                    xaxis_title="Балл (0-3)", yaxis_title="Кол-во",
                    height=400,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            # --- N/A distribution ---
            st.subheader("Не применимо (NA) по критериям")
            na_rows = []
            for c in CRITERIA:
                na_count = sum(1 for e in gold_examples if e["scores"].get(c) == -1)
                total = len(gold_examples)
                na_rows.append({
                    "Критерий": CRITERIA_RU[c],
                    "NA": na_count,
                    "NA %": f"{na_count / total * 100:.0f}%",
                    "Оценено": total - na_count,
                })
            st.dataframe(pd.DataFrame(na_rows), use_container_width=True, hide_index=True)

            # --- Benchmark comparison (if available) ---
            benchmark_dir = DATA_DIR / "results" / "benchmarks"
            pipeline_bench = sorted(benchmark_dir.glob("pipeline_benchmark_*.json"), reverse=True) if benchmark_dir.exists() else []

            if pipeline_bench:
                st.subheader("Сравнение: эксперт vs пайплайн")
                pr = json.loads(pipeline_bench[0].read_text(encoding="utf-8"))
                pipe = pr.get("pipeline", {})

                comp_cols = st.columns(3)
                comp_cols[0].metric("Pipeline Exact match", f"{pipe.get('exact_match', 0):.1%}")
                comp_cols[1].metric("Pipeline Soft match", f"{pipe.get('soft_match', 0):.1%}")

                # Compare avg scores per criterion
                st.markdown("#### Средний балл: эксперт vs пайплайн")
                comp_rows = []
                per_crit = pr.get("per_criterion", {})
                for c in CRITERIA:
                    vals = crit_scores.get(c, [])
                    gold_mean = sum(vals) / len(vals) if vals else 0
                    pc = per_crit.get(c, {})
                    bias = pc.get("mean_diff", 0)
                    pipeline_mean = gold_mean + bias
                    comp_rows.append({
                        "Критерий": CRITERIA_RU[c],
                        "Эксперт": round(gold_mean, 2),
                        "Пайплайн": round(pipeline_mean, 2),
                        "Сдвиг": f"{bias:+.2f}",
                        "Exact %": f"{pc.get('exact_pct', 0):.0f}%",
                    })
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

                # Grouped bar chart: pipeline vs expert (Gemini 3 dialogues)
                # Pipeline scores: from current run, filtered to gemini3flash dialogues
                gemini3_pipeline = {
                    did: scores for did, scores in dialog_scores.items()
                    if "gemini3flash" in did
                }
                if gemini3_pipeline:
                    chart_rows = []
                    for c in CRITERIA:
                        gold_vals = crit_scores.get(c, [])
                        pipe_vals = [
                            s[c] for s in gemini3_pipeline.values()
                            if s.get(c) is not None and s[c] >= 0
                        ]
                        if gold_vals and pipe_vals:
                            g_avg = sum(gold_vals) / len(gold_vals)
                            p_avg = sum(pipe_vals) / len(pipe_vals)
                            g_disp = _to_display_score(g_avg, scale)
                            p_disp = _to_display_score(p_avg, scale)
                            chart_rows.append({"criterion_ru": CRITERIA_RU[c], "Источник": "Ручная оценка", "Средний балл": g_disp})
                            chart_rows.append({"criterion_ru": CRITERIA_RU[c], "Источник": "Пайплайн", "Средний балл": p_disp})

                    if chart_rows:
                        st.markdown("#### Пайплайн VS Ручная оценка (Gemini 3)")
                        st.caption(f"Ручная оценка vs Пайплайн: {len(gemini3_pipeline)} диалогов Gemini")
                        chart_df = pd.DataFrame(chart_rows)
                        order = [CRITERIA_RU[c] for c in CRITERIA]
                        chart_df["criterion_ru"] = pd.Categorical(
                            chart_df["criterion_ru"], categories=order, ordered=True
                        )
                        fig_comp = px.bar(
                            chart_df,
                            x="criterion_ru",
                            y="Средний балл",
                            color="Источник",
                            barmode="group",
                            text="Средний балл",
                            category_orders={"criterion_ru": order},
                            color_discrete_map={
                                "Ручная оценка": "#2ecc71",
                                "Пайплайн": "#3498db",
                            },
                        )
                        fmt = _text_fmt(scale)
                        fig_comp.update_traces(texttemplate=fmt, textposition="outside")
                        fig_comp.update_layout(
                            xaxis_title="",
                            yaxis_title=f"Балл ({score_label})",
                            yaxis_range=axis_range,
                            height=450,
                            legend_title="Источник",
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)

            # --- Full data table ---
            with st.expander("Полные данные golden set", expanded=False):
                table_rows = []
                for i, e in enumerate(gold_examples):
                    row = {"#": i + 1, "Задача": e["task"][:60], "Роль": e["role"]}
                    for c in CRITERIA:
                        s = e["scores"].get(c)
                        row[CRITERIA_RU[c]] = s if s is not None else ""
                    row["EduScore"] = e["eduscore"]
                    table_rows.append(row)
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


# ===================================================================
# TAB 6: Pipeline reference
# ===================================================================
with tab_reference:
    _render_reference()
