"""Dashboard: Manual Review of Evaluator Scores — verify judge decisions and critical flags."""

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Ревью оценок", page_icon="✏️", layout="wide")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tutor_eval"
RESULTS_DIR = DATA_DIR / "results"
REVIEWS_DIR = DATA_DIR / "reviews"
REVIEWS_DIR.mkdir(parents=True, exist_ok=True)

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
CRITICAL_FLAGS = ["premature_end", "prompt_leak", "nonsense"]
CRITICAL_FLAGS_RU = {
    "premature_end": "Преждевременное завершение",
    "prompt_leak": "Утечка промпта",
    "nonsense": "Неуместные фразы",
}

AGREEMENT_OPTIONS = ["—", "да", "скорее да", "скорее нет", "нет", "неприменимо"]
CRIT_OPTIONS = ["—", "крит", "не крит"]

REVIEWS_FILE = REVIEWS_DIR / "manual_reviews.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_dialog_text(text: str):
    """Render dialog with colored role labels and LaTeX formula support."""
    if not text:
        st.markdown("*текст не сохранён*")
        return
    parts = re.split(r"((?:Пользователь|Ассистент)(?:\s*\[[^\]]*\])*:)", text)
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        if stripped.startswith("Пользователь"):
            st.markdown("---")
            st.markdown(f":blue[**{stripped.rstrip(':')}**]")
        elif stripped.startswith("Ассистент"):
            st.markdown("---")
            st.markdown(f":green[**{stripped.rstrip(':')}**]")
        else:
            st.markdown(stripped)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_judge_decisions(path: str) -> dict[str, dict]:
    """Load judge decisions into dict keyed by dialog_id."""
    decisions = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                decisions[d["dialog_id"]] = d
    return decisions


@st.cache_data
def load_dialogs(path: str) -> dict[str, dict]:
    dialogs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                dialogs[d["dialog_id"]] = d
    return dialogs


def load_reviews() -> dict[str, dict]:
    """Load existing manual reviews keyed by dialog_id."""
    reviews = {}
    if REVIEWS_FILE.exists():
        with open(REVIEWS_FILE, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    reviews[r["dialog_id"]] = r
    return reviews


def save_review(review: dict):
    """Append a review to the JSONL file (overwrites previous for same dialog_id)."""
    existing = load_reviews()
    existing[review["dialog_id"]] = review
    with open(REVIEWS_FILE, "w", encoding="utf-8") as f:
        for r in existing.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def find_result_dirs() -> list[Path]:
    candidates = []
    for p in sorted(RESULTS_DIR.iterdir(), reverse=True):
        if p.is_dir() and (p / "all_judge_decisions.jsonl").exists():
            candidates.append(p)
    return candidates


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("✏️ Ручная проверка оценок")
st.caption("Проверка итоговых оценок judge и критических флагов по каждому диалогу")

# --- Select run ---
run_dirs = find_result_dirs()
if not run_dirs:
    st.warning("Нет результатов с judge decisions.")
    st.stop()

run_names = [p.name for p in run_dirs]
selected_run = st.sidebar.selectbox("Прогон", run_names)
run_dir = RESULTS_DIR / selected_run

decisions = load_judge_decisions(str(run_dir / "all_judge_decisions.jsonl"))
dialog_file = run_dir / "dialogs.jsonl"
dialogs_map = load_dialogs(str(dialog_file)) if dialog_file.exists() else {}
reviews = load_reviews()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### Фильтры")

review_status_filter = st.sidebar.radio(
    "Статус ревью",
    ["Все", "Не проверенные", "Проверенные"],
    index=1,
)

has_crit_filter = st.sidebar.radio(
    "Криты",
    ["Все", "С подтверждёнными критами", "С отклонёнными критами", "Без критов"],
    index=0,
)

selected_crit_types: list[str] = []
if has_crit_filter in ("С подтверждёнными критами", "С отклонёнными критами"):
    selected_crit_types = st.sidebar.multiselect(
        "Тип крита",
        options=CRITICAL_FLAGS,
        format_func=lambda f: CRITICAL_FLAGS_RU.get(f, f),
        default=[],
        help="Оставьте пустым для показа всех типов",
    )

# Metadata filters
themes = set()
grade_groups = set()
student_types = set()
file_names = set()
for did in decisions:
    d = dialogs_map.get(did, {})
    if d.get("theme"):
        themes.add(d["theme"])
    if d.get("grade_group"):
        grade_groups.add(d["grade_group"])
    if d.get("student_type"):
        student_types.add(d["student_type"])
    if d.get("file_name"):
        file_names.add(d["file_name"])

selected_file = st.sidebar.selectbox("Выгрузка", ["Все"] + sorted(file_names))
selected_grade = st.sidebar.selectbox("Класс", ["Все"] + sorted(grade_groups))
selected_theme = st.sidebar.selectbox("Тема", ["Все"] + sorted(themes))
selected_student_type = st.sidebar.selectbox("Тип ученика", ["Все"] + sorted(student_types))

search_query = st.sidebar.text_input("Поиск (ID, задача)", placeholder="введите текст...")


def passes_filter(dialog_id: str) -> bool:
    dec = decisions[dialog_id]
    d = dialogs_map.get(dialog_id, {})
    rev = reviews.get(dialog_id)

    if review_status_filter == "Не проверенные" and rev is not None:
        return False
    if review_status_filter == "Проверенные" and rev is None:
        return False

    confirmed = dec.get("critical_flags", {}).get("confirmed", {})
    has_confirmed_crit = bool(confirmed)
    # Check if review overrode crits
    rev_crits_overridden = False
    if rev:
        for flag in CRITICAL_FLAGS:
            rc = rev.get("crits", {}).get(flag, {})
            if rc.get("verdict") == "не крит":
                rev_crits_overridden = True

    if has_crit_filter == "С подтверждёнными критами":
        if not has_confirmed_crit:
            return False
        if selected_crit_types and not any(ct in confirmed for ct in selected_crit_types):
            return False
    if has_crit_filter == "С отклонёнными критами":
        if not rev_crits_overridden:
            return False
        if selected_crit_types and rev:
            has_match = any(
                rev.get("crits", {}).get(ct, {}).get("verdict") == "не крит"
                for ct in selected_crit_types
            )
            if not has_match:
                return False
    if has_crit_filter == "Без критов" and has_confirmed_crit:
        return False

    if selected_file != "Все" and d.get("file_name") != selected_file:
        return False
    if selected_grade != "Все" and d.get("grade_group") != selected_grade:
        return False
    if selected_theme != "Все" and d.get("theme") != selected_theme:
        return False
    if selected_student_type != "Все" and d.get("student_type") != selected_student_type:
        return False
    if search_query:
        q = search_query.lower()
        haystack = (dialog_id + " " + d.get("task", "")).lower()
        if q not in haystack:
            return False
    return True


filtered_ids = [did for did in decisions if passes_filter(did)]

# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

total_decisions = len(decisions)
total_reviewed = sum(1 for did in decisions if did in reviews)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Прогресс:** {total_reviewed} / {total_decisions} проверено")
st.sidebar.progress(total_reviewed / total_decisions if total_decisions else 0)

st.markdown(f"**Показано {len(filtered_ids)} диалогов** (всего {total_decisions}, проверено {total_reviewed})")

# ---------------------------------------------------------------------------
# Dialog selector
# ---------------------------------------------------------------------------

if not filtered_ids:
    st.info("Нет диалогов по фильтрам")
    st.stop()

# Show compact table for navigation
nav_rows = []
for did in filtered_ids:
    dec = decisions[did]
    d = dialogs_map.get(did, {})
    confirmed = dec.get("critical_flags", {}).get("confirmed", {})
    nav_rows.append({
        "dialog_id": did,
        "Класс": d.get("grade_group", ""),
        "Тема": d.get("theme", ""),
        "Крит": ", ".join(CRITICAL_FLAGS_RU.get(k, k) for k in confirmed) if confirmed else "—",
        "Проверен": "✅" if did in reviews else "",
    })

selected_id = st.selectbox(
    "Выберите диалог",
    filtered_ids,
    format_func=lambda did: f"{'✅ ' if did in reviews else ''}{did} | {dialogs_map.get(did, {}).get('theme', '')} | {'КРИТ' if decisions[did].get('critical_flags', {}).get('confirmed') else 'ok'}",
)

dec = decisions[selected_id]
dialog = dialogs_map.get(selected_id, {})
existing_review = reviews.get(selected_id, {})

# ---------------------------------------------------------------------------
# Dialog context
# ---------------------------------------------------------------------------

st.markdown("---")

meta_cols = st.columns(4)
meta_cols[0].markdown(f"**Задача:** {dialog.get('task', '')[:120]}")
meta_cols[1].markdown(f"**Тема:** {dialog.get('theme', '')} / {dialog.get('subtheme', '')}")
meta_cols[2].markdown(f"**Класс:** {dialog.get('grade_group', '')}")
meta_cols[3].markdown(f"**Ученик:** {dialog.get('student_type', '')} ({dialog.get('student_model', '')})")

with st.expander("Текст диалога", expanded=False):
    _render_dialog_text(dialog.get("text", ""))

# ---------------------------------------------------------------------------
# Section 1: Critical flags review
# ---------------------------------------------------------------------------

confirmed_crits = dec.get("critical_flags", {}).get("confirmed", {})
flagged_crits = dec.get("critical_flags", {}).get("flagged", {})

# Use dialog_id in widget keys so values reset on dialog switch
_dk = selected_id

with st.form(key=f"review_form_{_dk}"):
    # --- Critical flags ---
    if confirmed_crits or flagged_crits:
        st.markdown("### Критические флаги")

        all_crit_names = set(list(confirmed_crits.keys()) + list(flagged_crits.keys()))

        crit_reviews = {}
        for flag in CRITICAL_FLAGS:
            if flag not in all_crit_names:
                continue

            flag_ru = CRITICAL_FLAGS_RU.get(flag, flag)
            is_confirmed = flag in confirmed_crits
            conf_data = confirmed_crits.get(flag, {})
            flag_data = flagged_crits.get(flag, [])

            with st.container(border=True):
                st.markdown(f"#### {'🔴' if is_confirmed else '🟡'} {flag_ru} (`{flag}`)")

                if is_confirmed:
                    st.markdown(f"**Вердикт judge:** подтверждён")
                    st.markdown(f"**Обоснование:** {conf_data.get('reasoning', '')}")
                else:
                    st.markdown(f"**Вердикт judge:** не подтверждён (но оценщики флагнули)")

                if isinstance(flag_data, list):
                    for entry in flag_data:
                        st.caption(f"Оценщик {entry.get('evaluator', '?')}: «{entry.get('evidence', '')}» — {entry.get('reasoning', '')}")
                elif isinstance(flag_data, dict):
                    st.caption(f"«{flag_data.get('evidence', '')}» — {flag_data.get('reasoning', '')}")

                # Review form
                prev = existing_review.get("crits", {}).get(flag, {})
                col1, col2 = st.columns([1, 3])
                with col1:
                    verdict = st.selectbox(
                        "Решение",
                        CRIT_OPTIONS,
                        index=CRIT_OPTIONS.index(prev.get("verdict", "—")),
                        key=f"crit_verdict_{_dk}_{flag}",
                    )
                with col2:
                    comment = st.text_input(
                        "Комментарий",
                        value=prev.get("comment", ""),
                        key=f"crit_comment_{_dk}_{flag}",
                    )

                crit_reviews[flag] = {"verdict": verdict, "comment": comment}
    else:
        st.markdown("### Критические флаги")
        st.info("Нет критических флагов для этого диалога")
        crit_reviews = {}

    # --- Criteria scores ---
    st.markdown("### Оценки по критериям")

    final_scores = dec.get("final_scores", {})

    header_cols = st.columns([2, 1, 3, 1, 3])
    header_cols[0].markdown("**Критерий**")
    header_cols[1].markdown("**Балл judge**")
    header_cols[2].markdown("**Обоснование**")
    header_cols[3].markdown("**Ваша оценка**")
    header_cols[4].markdown("**Комментарий**")

    criteria_reviews = {}
    for criterion in CRITERIA:
        entry = final_scores.get(criterion, {})
        if isinstance(entry, dict):
            score = entry.get("score", "?")
            reasoning = entry.get("reasoning", "")
        else:
            score = entry
            reasoning = ""

        prev = existing_review.get("criteria", {}).get(criterion, {})

        cols = st.columns([2, 1, 3, 1, 3])
        cols[0].markdown(f"**{CRITERIA_RU.get(criterion, criterion)}**")

        if score == -1:
            cols[1].markdown("N/A")
        else:
            cols[1].markdown(f"**{score}**/3")

        cols[2].caption(reasoning[:200] if reasoning else "—")

        with cols[3]:
            agreement = st.selectbox(
                "Оценка",
                AGREEMENT_OPTIONS,
                index=AGREEMENT_OPTIONS.index(prev.get("agreement", "—")),
                key=f"agree_{_dk}_{criterion}",
                label_visibility="collapsed",
            )

        with cols[4]:
            comment = st.text_input(
                "Комментарий",
                value=prev.get("comment", ""),
                key=f"comment_{_dk}_{criterion}",
                label_visibility="collapsed",
            )

        criteria_reviews[criterion] = {"agreement": agreement, "comment": comment}

    # --- General comment ---
    st.markdown("### Общий комментарий")
    general_comment = st.text_area(
        "Общий комментарий по диалогу",
        value=existing_review.get("general_comment", ""),
        key=f"general_comment_{_dk}",
        height=80,
    )

    # --- Save (Enter in any field or button click) ---
    col_save, col_status = st.columns([1, 3])
    with col_save:
        submitted = st.form_submit_button(
            "💾 Сохранить ревью", type="primary", use_container_width=True,
        )

if submitted:
    review = {
        "dialog_id": selected_id,
        "run": selected_run,
        "criteria": criteria_reviews,
        "crits": crit_reviews,
        "general_comment": general_comment,
        "timestamp": datetime.now().isoformat(),
    }
    save_review(review)
    st.cache_data.clear()
    st.rerun()

if selected_id in reviews:
    ts = reviews[selected_id].get("timestamp", "")
    st.success(f"Проверен ({ts})")
else:
    st.info("Ещё не проверен")
