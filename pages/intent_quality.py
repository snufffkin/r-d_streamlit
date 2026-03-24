"""Анализ качества интентов: match rate, распределение, проблемные реплики."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Качество интентов", page_icon="🎯", layout="wide")

DATA_DIR = Path(__file__).parent.parent / "data" / "intent_eval"
RESULTS_PATH = DATA_DIR / "results.csv"

# ── Палитры и константы ──────────────────────────────────────────────────────

COLOR_MATCH = "#2ecc71"
COLOR_MISMATCH = "#e74c3c"
COLOR_WARN = "#f39c12"

DEFAULT_WEIGHTS = {
    "answer": 58,
    "set-problem": 14,
    "agree-with-tutor": 12,
    "chat": 5,
    "get-explanation": 4,
    "criticize-tutor": 3,
    "get-solution": 2,
    "find-mistake": 1,
    "end-dialog": 1,
    "thank-tutor": 0,
}

# 16 строк аудита: 7 подтипов answer (correct + 6 типов ошибок) + 9 остальных
AUDIT_INTENTS = [
    {"id": "answer (correct)", "name": "Верный ответ", "default_weight": "58% × P(верно)", "prompt_file": "intent_answer_correct.md"},
    {"id": "answer (careless)", "name": "Невнимательность", "default_weight": "58% × P(ош) × W", "prompt_file": "intent_answer_wrong.md"},
    {"id": "answer (procedure)", "name": "Незнание процедуры", "default_weight": "58% × P(ош) × W", "prompt_file": "intent_answer_wrong.md"},
    {"id": "answer (method)", "name": "Незнание способа", "default_weight": "58% × P(ош) × W", "prompt_file": "intent_answer_wrong.md"},
    {"id": "answer (unstable_proc)", "name": "Неустойч. процедура", "default_weight": "58% × P(ош) × W", "prompt_file": "intent_answer_wrong.md"},
    {"id": "answer (unstable_method)", "name": "Неустойч. способ", "default_weight": "58% × P(ош) × W", "prompt_file": "intent_answer_wrong.md"},
    {"id": "answer (misconception)", "name": "Неверная концепция", "default_weight": "58% × P(ош) × W", "prompt_file": "intent_answer_wrong.md"},
    {"id": "set-problem", "name": "Задать задачу", "default_weight": "14%", "prompt_file": "intent_set_problem.md"},
    {"id": "agree-with-tutor", "name": "Согласиться", "default_weight": "12%", "prompt_file": "intent_agree_with_tutor.md"},
    {"id": "chat", "name": "Болтовня", "default_weight": "5%", "prompt_file": "intent_chat.md"},
    {"id": "get-explanation", "name": "Просить объяснение", "default_weight": "4%", "prompt_file": "intent_get_explanation.md"},
    {"id": "criticize-tutor", "name": "Критиковать", "default_weight": "3%", "prompt_file": "intent_criticize_tutor.md"},
    {"id": "get-solution", "name": "Попросить ответ", "default_weight": "2%", "prompt_file": "intent_get_solution.md"},
    {"id": "find-mistake", "name": "Найти ошибку", "default_weight": "1%", "prompt_file": "intent_find_mistake.md"},
    {"id": "end-dialog", "name": "Закончить", "default_weight": "1%", "prompt_file": "intent_end_dialog.md"},
    {"id": "thank-tutor", "name": "Поблагодарить", "default_weight": "0%", "prompt_file": "intent_thank_tutor.md"},
]


# ── Загрузка данных ──────────────────────────────────────────────────────────


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data
def load_results(_mtime_key: float) -> pd.DataFrame | None:
    if not RESULTS_PATH.exists():
        return None
    df = pd.read_csv(RESULTS_PATH)
    # Нормализуем boolean
    if "match" in df.columns:
        df["match"] = df["match"].astype(str).str.lower().map(
            {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
        )
    if "confidence" in df.columns:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df


def _match_rate_color(rate: float) -> str:
    if rate >= 0.8:
        return COLOR_MATCH
    elif rate >= 0.6:
        return COLOR_WARN
    return COLOR_MISMATCH


# ── Заголовок ────────────────────────────────────────────────────────────────

st.title("🎯 Качество интентов")
st.caption("Анализ совпадения объявленных и фактических интентов в диалогах")

df = load_results(_mtime(RESULTS_PATH))

if df is None:
    st.warning(
        "Файл с результатами не найден.\n\n"
        f"**Ожидаемый путь:** `{RESULTS_PATH}`\n\n"
        "**Ожидаемые колонки:** `dialog_idx`, `turn_idx`, `intent_declared`, "
        "`student_text`, `teacher_text_before`, `match`, `confidence`, "
        "`reason`, `actual_intent`, `grade_group`, `task_id`\n\n"
        "Запустите пайплайн оценки интентов и положите `results.csv` в указанную папку."
    )
    st.stop()

# ── Табы ─────────────────────────────────────────────────────────────────────

tab_overview, tab_distrib, tab_problems, tab_answer, tab_audit = st.tabs(
    ["Обзор", "Распределение", "Проблемные реплики", "Answer детально", "Аудит"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1: Обзор
# ═══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    total_dialogs = df["dialog_idx"].nunique() if "dialog_idx" in df.columns else 0
    total_turns = len(df)
    overall_match = df["match"].mean() if "match" in df.columns else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Диалогов", f"{total_dialogs:,}")
    c2.metric("Реплик", f"{total_turns:,}")
    c3.metric("Общий match rate", f"{overall_match:.1%}")

    st.markdown("---")

    # ── Match rate по интентам (горизонтальные бары) ──
    st.subheader("Match rate по интентам")

    intent_stats = (
        df.groupby("intent_declared")
        .agg(
            n_turns=("match", "size"),
            match_rate=("match", "mean"),
        )
        .reset_index()
        .sort_values("match_rate", ascending=True)
    )

    intent_stats["color"] = intent_stats["match_rate"].apply(_match_rate_color)
    intent_stats["match_pct"] = (intent_stats["match_rate"] * 100).round(1)

    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            y=intent_stats["intent_declared"],
            x=intent_stats["match_pct"],
            orientation="h",
            marker_color=intent_stats["color"],
            text=intent_stats["match_pct"].apply(lambda v: f"{v:.1f}%"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Match rate: %{x:.1f}%<extra></extra>",
        )
    )
    fig_bar.update_layout(
        xaxis_title="Match rate, %",
        yaxis_title="",
        height=max(400, len(intent_stats) * 35),
        margin=dict(l=10, r=40, t=10, b=40),
        xaxis=dict(range=[0, 105]),
    )
    # Пороговые линии
    fig_bar.add_vline(x=80, line_dash="dash", line_color=COLOR_MATCH, annotation_text="80%", annotation_position="top")
    fig_bar.add_vline(x=60, line_dash="dash", line_color=COLOR_WARN, annotation_text="60%", annotation_position="top")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Таблица ──
    st.subheader("Статистика по интентам")

    # Топ причин несовпадения
    mismatch_df = df[df["match"] == False]
    if "reason" in mismatch_df.columns and not mismatch_df.empty:
        top_reasons = (
            mismatch_df.groupby("intent_declared")["reason"]
            .apply(lambda s: "; ".join(s.dropna().value_counts().head(3).index.tolist()))
            .reset_index()
            .rename(columns={"reason": "Топ причины несовпадения"})
        )
        table_df = intent_stats.merge(top_reasons, on="intent_declared", how="left")
    else:
        table_df = intent_stats.copy()
        table_df["Топ причины несовпадения"] = ""

    table_df = table_df.rename(
        columns={
            "intent_declared": "Интент",
            "n_turns": "Реплик",
            "match_pct": "Match rate, %",
        }
    ).sort_values("Match rate, %", ascending=False)

    st.dataframe(
        table_df[["Интент", "Реплик", "Match rate, %", "Топ причины несовпадения"]],
        use_container_width=True,
        hide_index=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2: Распределение
# ═══════════════════════════════════════════════════════════════════════════════

with tab_distrib:
    st.subheader("Фактическое распределение vs дефолтные веса")

    actual_counts = df["intent_declared"].value_counts()
    actual_pct = (actual_counts / actual_counts.sum() * 100).reset_index()
    actual_pct.columns = ["intent", "actual_pct"]

    weights_df = pd.DataFrame(list(DEFAULT_WEIGHTS.items()), columns=["intent", "default_weight"])
    weights_df["default_pct"] = weights_df["default_weight"] / weights_df["default_weight"].sum() * 100

    compare = weights_df.merge(actual_pct, on="intent", how="outer").fillna(0)
    compare["delta"] = (compare["actual_pct"] - compare["default_pct"]).round(2)
    compare = compare.sort_values("default_pct", ascending=False)

    # Grouped bar chart
    fig_dist = go.Figure()
    fig_dist.add_trace(
        go.Bar(
            name="Дефолтные веса, %",
            x=compare["intent"],
            y=compare["default_pct"],
            marker_color="#3498db",
        )
    )
    fig_dist.add_trace(
        go.Bar(
            name="Фактическое, %",
            x=compare["intent"],
            y=compare["actual_pct"],
            marker_color="#e67e22",
        )
    )
    fig_dist.update_layout(
        barmode="group",
        xaxis_title="Интент",
        yaxis_title="Доля, %",
        height=500,
        margin=dict(t=30, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Таблица с расхождениями
    st.subheader("Расхождения")
    compare_display = compare.copy()
    compare_display.columns = ["Интент", "Дефолтный вес", "Дефолт, %", "Факт, %", "Дельта, п.п."]
    compare_display = compare_display.sort_values("Дельта, п.п.", key=abs, ascending=False)

    st.dataframe(
        compare_display.style.map(
            lambda v: f"color: {COLOR_MISMATCH}" if isinstance(v, (int, float)) and abs(v) > 5 else "",
            subset=["Дельта, п.п."],
        ),
        use_container_width=True,
        hide_index=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3: Проблемные реплики
# ═══════════════════════════════════════════════════════════════════════════════

with tab_problems:
    st.subheader("Реплики с match = False")

    problem_df = df[df["match"] == False].copy()

    if problem_df.empty:
        st.success("Нет проблемных реплик — все интенты совпали!")
    else:
        # Фильтры
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            all_intents = sorted(problem_df["intent_declared"].dropna().unique().tolist())
            selected_intents = st.multiselect(
                "Фильтр по интенту",
                options=all_intents,
                default=all_intents,
                key="problem_intent_filter",
            )

        with col_f2:
            if "confidence" in problem_df.columns and problem_df["confidence"].notna().any():
                conf_min = float(problem_df["confidence"].min())
                conf_max = float(problem_df["confidence"].max())
                conf_thresh = st.slider(
                    "Порог уверенности (показать <=)",
                    min_value=conf_min,
                    max_value=conf_max,
                    value=conf_max,
                    step=0.05,
                    key="conf_slider",
                )
            else:
                conf_thresh = None

        filtered = problem_df[problem_df["intent_declared"].isin(selected_intents)]
        if conf_thresh is not None and "confidence" in filtered.columns:
            filtered = filtered[filtered["confidence"] <= conf_thresh]

        display_cols = [
            c
            for c in [
                "intent_declared",
                "student_text",
                "teacher_text_before",
                "reason",
                "actual_intent",
                "confidence",
            ]
            if c in filtered.columns
        ]

        col_rename = {
            "intent_declared": "Объявленный интент",
            "student_text": "Текст ученика",
            "teacher_text_before": "Текст учителя до",
            "reason": "Причина",
            "actual_intent": "Фактический интент",
            "confidence": "Уверенность",
        }

        st.caption(f"Показано {len(filtered)} из {len(problem_df)} проблемных реплик")
        st.dataframe(
            filtered[display_cols].rename(columns=col_rename),
            use_container_width=True,
            hide_index=True,
            height=600,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4: Answer детально
# ═══════════════════════════════════════════════════════════════════════════════

with tab_answer:
    st.subheader("Глубокий анализ интента answer")

    answer_df = df[df["intent_declared"] == "answer"].copy()

    if answer_df.empty:
        st.info("Нет данных по интенту answer.")
    else:
        n_answer = len(answer_df)
        n_match = answer_df["match"].sum()
        n_mismatch = n_answer - n_match
        rate = n_match / n_answer if n_answer > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Всего реплик answer", f"{n_answer:,}")
        c2.metric("Совпало", f"{n_match:,}")
        c3.metric("Не совпало", f"{n_mismatch:,}")
        c4.metric("Match rate", f"{rate:.1%}")

        st.markdown("---")

        # Что на самом деле скрывается за «неправильными» answer
        answer_mismatch = answer_df[answer_df["match"] == False]

        if "actual_intent" in answer_mismatch.columns and answer_mismatch["actual_intent"].notna().any():
            st.subheader("Чем на самом деле являются несовпавшие answer?")

            actual_dist = answer_mismatch["actual_intent"].value_counts().reset_index()
            actual_dist.columns = ["actual_intent", "count"]

            fig_pie = px.pie(
                actual_dist,
                names="actual_intent",
                values="count",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.3,
            )
            fig_pie.update_layout(
                height=450,
                margin=dict(t=20, b=20),
            )
            fig_pie.update_traces(
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Кол-во: %{value}<br>Доля: %{percent}<extra></extra>",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Таблица
            actual_dist["pct"] = (actual_dist["count"] / actual_dist["count"].sum() * 100).round(1)
            actual_dist.columns = ["Фактический интент", "Кол-во", "Доля, %"]
            st.dataframe(actual_dist, use_container_width=True, hide_index=True)
        else:
            st.info(
                "Колонка `actual_intent` пуста для несовпавших answer — "
                "нет данных о фактическом интенте."
            )

        # Распределение confidence для answer
        if "confidence" in answer_df.columns and answer_df["confidence"].notna().any():
            st.subheader("Распределение уверенности (answer)")
            fig_hist = px.histogram(
                answer_df,
                x="confidence",
                color=answer_df["match"].map({True: "Match", False: "Mismatch"}),
                color_discrete_map={"Match": COLOR_MATCH, "Mismatch": COLOR_MISMATCH},
                nbins=20,
                barmode="overlay",
                opacity=0.7,
                labels={"color": "Результат", "confidence": "Уверенность"},
            )
            fig_hist.update_layout(height=350, margin=dict(t=20, b=40))
            st.plotly_chart(fig_hist, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 5: Аудит
# ═══════════════════════════════════════════════════════════════════════════════

with tab_audit:
    st.subheader("Аудит-таблица по интентам")
    st.caption("Заполните данные аудита и сохраните в CSV.")

    audit_df = pd.DataFrame(AUDIT_INTENTS)
    audit_df = audit_df.rename(
        columns={
            "id": "Intent ID",
            "name": "Название",
            "default_weight": "Дефолтный вес",
            "prompt_file": "Промпт-файл",
        }
    )

    # Добавляем редактируемые колонки
    audit_df["Кто работал"] = ""
    audit_df["Что замеряли"] = ""
    audit_df["Размер пула"] = 0
    audit_df["Match rate, %"] = 0.0
    audit_df["Промпт переработан"] = False
    audit_df["Дельта, п.п."] = 0.0
    audit_df["Комментарии"] = ""

    edited = st.data_editor(
        audit_df,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Intent ID": st.column_config.TextColumn(disabled=True),
            "Название": st.column_config.TextColumn(disabled=True),
            "Дефолтный вес": st.column_config.NumberColumn(disabled=True),
            "Промпт-файл": st.column_config.TextColumn(disabled=True),
            "Кто работал": st.column_config.TextColumn(width="medium"),
            "Что замеряли": st.column_config.TextColumn(width="medium"),
            "Размер пула": st.column_config.NumberColumn(min_value=0, step=1),
            "Match rate, %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.1, format="%.1f"),
            "Промпт переработан": st.column_config.CheckboxColumn(),
            "Дельта, п.п.": st.column_config.NumberColumn(step=0.1, format="%.1f"),
            "Комментарии": st.column_config.TextColumn(width="large"),
        },
        key="audit_editor",
    )

    export_path = DATA_DIR / "audit_intents.csv"

    if st.button("💾 Сохранить аудит в CSV", type="primary"):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        edited.to_csv(export_path, index=False, encoding="utf-8-sig")
        st.success(f"Сохранено: `{export_path}`")
