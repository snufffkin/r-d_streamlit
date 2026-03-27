"""Dashboard: Math Correctness — view extracted claims and sympy verification results."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Мат. корректность", page_icon="🔢", layout="wide")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tutor_eval"
RESULTS_DIR = DATA_DIR / "results"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_math_results(path: str) -> list[dict]:
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


@st.cache_data
def load_dialogs(path: str) -> dict[str, dict]:
    """Load dialogs.jsonl into a dict keyed by dialog_id."""
    dialogs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                dialogs[d["dialog_id"]] = d
    return dialogs


def find_math_result_dirs() -> list[Path]:
    """Find all result dirs that contain math_check_results.jsonl."""
    candidates = []
    for p in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if p.is_dir() and (p / "math_check_results.jsonl").exists():
            candidates.append(p)
    return candidates


def find_dialog_file_for_run(run_dir: Path) -> Path | None:
    """Find the dialogs.jsonl that corresponds to this math run."""
    if (run_dir / "dialogs.jsonl").exists():
        return run_dir / "dialogs.jsonl"
    candidates = sorted(RESULTS_DIR.glob("*/dialogs.jsonl"), reverse=True)
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _claim_status(v: dict) -> str:
    """Return human-readable status for a verification entry."""
    if v.get("is_correct") is True:
        return "correct"
    elif v.get("is_correct") is False:
        return "incorrect"
    else:
        return "exec_error"


def _error_category(v: dict) -> str:
    """Classify exec error into a category."""
    err = v.get("error", "")
    if "NameError" in err:
        return "NameError"
    if "IndentationError" in err:
        return "IndentationError"
    if "SyntaxError" in err:
        return "SyntaxError"
    if "TypeError" in err:
        return "TypeError"
    if "NotImplementedError" in err:
        return "NotImplementedError"
    if "result" in err and "not set" in err:
        return "result not set"
    if err:
        return err.split(":")[0]
    return "unknown"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🔢 Математическая корректность тьютора")
st.caption("Извлечение математических утверждений из реплик тьютора и верификация через sympy")

# --- Select run ---
run_dirs = find_math_result_dirs()
if not run_dirs:
    st.warning("Нет результатов math check. Запустите `uv run python -m tutor_eval.run_math_check`")
    st.stop()

run_names = [p.name for p in run_dirs]
selected_run = st.sidebar.selectbox("Прогон", run_names)
run_dir = RESULTS_DIR / selected_run

results = load_math_results(str(run_dir / "math_check_results.jsonl"))

# Try to load dialogs for context
dialog_file = find_dialog_file_for_run(run_dir)
dialogs_map = load_dialogs(str(dialog_file)) if dialog_file else {}

# ---------------------------------------------------------------------------
# Precompute per-claim stats
# ---------------------------------------------------------------------------

all_verifications = []  # (dialog_result, verification, dialog_meta)
for r in results:
    d = dialogs_map.get(r["dialog_id"], {})
    for v in r.get("verifications", []):
        all_verifications.append((r, v, d))

claim_type_counts = Counter(v.get("type", "unknown") for _, v, _ in all_verifications)
error_cat_counts = Counter(
    _error_category(v) for _, v, _ in all_verifications if _claim_status(v) == "exec_error"
)

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

total_dialogs = len(results)
total_claims = sum(r.get("claims_count", 0) for r in results)
total_correct = sum(r.get("correct_count", 0) for r in results)
total_incorrect = sum(r.get("incorrect_count", 0) for r in results)
total_errors = sum(r.get("error_count", 0) for r in results)
dialogs_with_incorrect = sum(1 for r in results if r.get("incorrect_count", 0) > 0)
dialogs_with_exec_err = sum(1 for r in results if r.get("error_count", 0) > 0)
dialogs_no_claims = sum(1 for r in results if r.get("claims_count", 0) == 0)

st.markdown("### Сводка")
cols = st.columns(6)
cols[0].metric("Диалогов", total_dialogs)
cols[1].metric("Утверждений", total_claims)
cols[2].metric("Корректных", total_correct)
cols[3].metric("Некорректных", total_incorrect, delta=f"-{total_incorrect}" if total_incorrect else None, delta_color="inverse")
cols[4].metric("Ошибки sympy", total_errors)
cols[5].metric("Без утверждений", dialogs_no_claims)

if total_claims > 0:
    accuracy = total_correct / total_claims * 100
    st.progress(accuracy / 100, text=f"Accuracy: {accuracy:.1f}% ({total_correct}/{total_claims})")

# ---------------------------------------------------------------------------
# Error breakdown
# ---------------------------------------------------------------------------

if total_incorrect > 0 or total_errors > 0:
    st.markdown("### Разбивка ошибок")
    err_col1, err_col2, err_col3 = st.columns(3)

    with err_col1:
        st.markdown("**Incorrect по типу утверждения**")
        incorrect_by_type = Counter(
            v.get("type", "unknown") for _, v, _ in all_verifications if _claim_status(v) == "incorrect"
        )
        if incorrect_by_type:
            df_inc = pd.DataFrame(
                incorrect_by_type.most_common(),
                columns=["Тип", "Кол-во"],
            )
            st.dataframe(df_inc, hide_index=True, use_container_width=True)
        else:
            st.info("Нет incorrect")

    with err_col2:
        st.markdown("**Exec errors по категории**")
        if error_cat_counts:
            df_err = pd.DataFrame(
                error_cat_counts.most_common(),
                columns=["Категория", "Кол-во"],
            )
            st.dataframe(df_err, hide_index=True, use_container_width=True)
        else:
            st.info("Нет exec errors")

    with err_col3:
        st.markdown("**Диалоги с проблемами**")
        st.markdown(f"- С incorrect: **{dialogs_with_incorrect}** из {total_dialogs}")
        st.markdown(f"- С exec errors: **{dialogs_with_exec_err}** из {total_dialogs}")
        st.markdown(f"- Без утверждений: **{dialogs_no_claims}** из {total_dialogs}")

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

# Collect all unique metadata values
themes = set()
subthemes = set()
grade_groups = set()
student_types = set()
student_models = set()
for r in results:
    d = dialogs_map.get(r["dialog_id"], {})
    if d.get("theme"):
        themes.add(d["theme"])
    if d.get("subtheme"):
        subthemes.add(d["subtheme"])
    if d.get("grade_group"):
        grade_groups.add(d["grade_group"])
    if d.get("student_type"):
        student_types.add(d["student_type"])
    if d.get("student_model"):
        student_models.add(d["student_model"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Фильтры диалогов")

search_query = st.sidebar.text_input("Поиск (ID, задача, текст)", placeholder="введите текст...")

filter_mode = st.sidebar.radio(
    "Статус",
    ["Все", "Только с incorrect", "Только с exec errors", "С любыми проблемами", "Только корректные", "Без утверждений"],
    index=0,
)

selected_grade = st.sidebar.selectbox("Класс", ["Все"] + sorted(grade_groups))
selected_theme = st.sidebar.selectbox("Тема", ["Все"] + sorted(themes))

# Dynamic subtheme filter — show only subthemes matching selected theme
if selected_theme != "Все":
    matching_subthemes = set()
    for r in results:
        d = dialogs_map.get(r["dialog_id"], {})
        if d.get("theme") == selected_theme and d.get("subtheme"):
            matching_subthemes.add(d["subtheme"])
    selected_subtheme = st.sidebar.selectbox("Подтема", ["Все"] + sorted(matching_subthemes))
else:
    selected_subtheme = "Все"

selected_student_type = st.sidebar.selectbox("Тип ученика", ["Все"] + sorted(student_types))
selected_student_model = st.sidebar.selectbox("Модель ученика", ["Все"] + sorted(student_models))

st.sidebar.markdown("---")
st.sidebar.markdown("### Фильтры утверждений")

claim_types_available = sorted(claim_type_counts.keys())
selected_claim_type = st.sidebar.selectbox("Тип утверждения", ["Все"] + claim_types_available)

selected_claim_status = st.sidebar.radio(
    "Статус утверждения",
    ["Все", "Только incorrect", "Только exec errors", "Только корректные"],
    index=0,
)

selected_error_cat = "Все"
if error_cat_counts:
    error_cats_available = sorted(error_cat_counts.keys())
    selected_error_cat = st.sidebar.selectbox("Категория exec error", ["Все"] + error_cats_available)


def passes_dialog_filter(r: dict) -> bool:
    inc = r.get("incorrect_count", 0)
    err = r.get("error_count", 0)
    claims = r.get("claims_count", 0)
    if filter_mode == "Только с incorrect" and inc == 0:
        return False
    if filter_mode == "Только с exec errors" and err == 0:
        return False
    if filter_mode == "С любыми проблемами" and inc == 0 and err == 0:
        return False
    if filter_mode == "Только корректные" and (inc > 0 or err > 0):
        return False
    if filter_mode == "Без утверждений" and claims > 0:
        return False
    d = dialogs_map.get(r["dialog_id"], {})
    if selected_grade != "Все" and d.get("grade_group") != selected_grade:
        return False
    if selected_theme != "Все" and d.get("theme") != selected_theme:
        return False
    if selected_subtheme != "Все" and d.get("subtheme") != selected_subtheme:
        return False
    if selected_student_type != "Все" and d.get("student_type") != selected_student_type:
        return False
    if selected_student_model != "Все" and d.get("student_model") != selected_student_model:
        return False
    if search_query:
        q = search_query.lower()
        haystack = (
            r.get("dialog_id", "") + " "
            + d.get("task", "") + " "
            + d.get("text", "")
        ).lower()
        if q not in haystack:
            return False
    return True


def passes_claim_filter(v: dict) -> bool:
    status = _claim_status(v)
    if selected_claim_type != "Все" and v.get("type", "unknown") != selected_claim_type:
        return False
    if selected_claim_status == "Только incorrect" and status != "incorrect":
        return False
    if selected_claim_status == "Только exec errors" and status != "exec_error":
        return False
    if selected_claim_status == "Только корректные" and status != "correct":
        return False
    if selected_error_cat != "Все" and status == "exec_error" and _error_category(v) != selected_error_cat:
        return False
    return True


filtered = [r for r in results if passes_dialog_filter(r)]
st.markdown(f"**Показано {len(filtered)} из {total_dialogs} диалогов**")

# ---------------------------------------------------------------------------
# Table overview
# ---------------------------------------------------------------------------

st.markdown("### Обзор по диалогам")

table_rows = []
for r in filtered:
    d = dialogs_map.get(r["dialog_id"], {})
    table_rows.append({
        "dialog_id": r["dialog_id"],
        "Класс": d.get("grade_group", ""),
        "Тема": d.get("theme", ""),
        "Подтема": d.get("subtheme", ""),
        "Ученик": d.get("student_type", ""),
        "Модель": d.get("student_model", ""),
        "Утв.": r.get("claims_count", 0),
        "OK": r.get("correct_count", 0),
        "Fail": r.get("incorrect_count", 0),
        "Err": r.get("error_count", 0),
    })

if table_rows:
    df = pd.DataFrame(table_rows)

    def highlight_row(row):
        if row["Fail"] > 0:
            return ["background-color: #ffcccc"] * len(row)
        if row["Err"] > 0:
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df.style.apply(highlight_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# Detailed claim view
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### Детальный просмотр разметки")

if not filtered:
    st.info("Нет диалогов по текущим фильтрам")
    st.stop()

# Dialog selector
dialog_ids = [r["dialog_id"] for r in filtered]
selected_id = st.selectbox("Диалог", dialog_ids, index=0)

selected_result = next(r for r in filtered if r["dialog_id"] == selected_id)
selected_dialog = dialogs_map.get(selected_id, {})

# Show dialog metadata
if selected_dialog:
    meta_cols = st.columns(4)
    meta_cols[0].markdown(f"**Задача:** {selected_dialog.get('task', '')[:100]}")
    meta_cols[1].markdown(f"**Тема:** {selected_dialog.get('theme', '')} / {selected_dialog.get('subtheme', '')}")
    meta_cols[2].markdown(f"**Класс:** {selected_dialog.get('grade_group', '')}")
    meta_cols[3].markdown(f"**Модель ученика:** {selected_dialog.get('student_model', '')}")

# Show claims — apply claim-level filters
verifications = selected_result.get("verifications", [])
filtered_verifications = [v for v in verifications if passes_claim_filter(v)]

if not verifications:
    reason = selected_result.get("no_claims_reason", "")
    st.info(f"Нет проверяемых утверждений. {reason}")
elif not filtered_verifications:
    st.info(f"Все {len(verifications)} утверждений отфильтрованы. Измените фильтры утверждений.")
else:
    if len(filtered_verifications) < len(verifications):
        st.caption(f"Показано {len(filtered_verifications)} из {len(verifications)} утверждений (фильтр)")

    for i, v in enumerate(filtered_verifications):
        is_correct = v.get("is_correct")
        if is_correct is True:
            icon = "✅"
        elif is_correct is False:
            icon = "❌"
        else:
            icon = "⚠️"

        label = v.get("description", f"Утверждение {i+1}")
        claim_type = v.get("type", "")
        header = f"{icon} {label} [{claim_type}]"
        if is_correct is None:
            header += f" — {_error_category(v)}"

        with st.expander(header, expanded=(is_correct is not True)):
            st.markdown(f"**Цитата из диалога:**")
            st.markdown(f"> {v.get('quote', '')}")

            math_expr = v.get("math_expression", "")
            if math_expr:
                st.markdown(f"**Формализация:** `{math_expr}`")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**sympy-код проверки:**")
                st.code(v.get("sympy_check", ""), language="python")
                reasoning = v.get("codegen_reasoning", "")
                if reasoning:
                    st.caption(f"Стратегия: {reasoning}")
            with col2:
                st.markdown("**Результат:**")
                if is_correct is True:
                    st.success(f"Корректно (result = {v.get('actual_result', '')})")
                elif is_correct is False:
                    st.error(f"Некорректно (result = {v.get('actual_result', '')})")
                else:
                    st.warning(f"Ошибка выполнения: {v.get('error', '')}")

# Show full dialog text
if selected_dialog:
    with st.expander("Полный текст диалога", expanded=False):
        st.text(selected_dialog.get("text", ""))
