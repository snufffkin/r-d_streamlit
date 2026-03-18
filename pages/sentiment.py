import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Эмоциональная окраска диалогов",
    page_icon="🎭",
    layout="wide",
)

DATA_DIR = Path(__file__).parent.parent / "data"

# ─── Цветовые палитры ───

SENTIMENT_COLORS = {
    "negative": "#EF4444",
    "slightly_negative": "#F97316",
    "neutral": "#9CA3AF",
    "slightly_positive": "#60A5FA",
    "positive": "#22C55E",
}

SENTIMENT_ORDER = ["negative", "slightly_negative", "neutral", "slightly_positive", "positive"]

SENTIMENT_RU = {
    "negative": "Негативный",
    "slightly_negative": "Слегка негативный",
    "neutral": "Нейтральный",
    "slightly_positive": "Слегка позитивный",
    "positive": "Позитивный",
}

ROLE_COLORS = {"tutor": "#8B5CF6", "student": "#F59E0B"}
ROLE_RU = {"tutor": "Тьютор", "student": "Ученик"}


# ─── Загрузка данных ───

@st.cache_data
def load_turns():
    df = pd.read_parquet(DATA_DIR / "turns.parquet")
    df["grade"] = df["grade"].fillna(0).astype(int)
    df["sentiment_label_ru"] = df["sentiment_label"].map(SENTIMENT_RU)
    df["role_ru"] = df["role"].map(ROLE_RU)
    return df


@st.cache_data
def load_dialogues():
    with open(DATA_DIR / "dialogues.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_summary():
    with open(DATA_DIR / "summary.json", encoding="utf-8") as f:
        return json.load(f)


turns = load_turns()
dialogues = load_dialogues()
summary = load_summary()

# DataFrame из dialogues для удобства
dial_rows = []
for d in dialogues:
    dial_rows.append({
        "dialogue_id": d["dialogue_id"],
        "topic": d["topic"],
        "grade": int(float(d["grade"])) if d["grade"] else 0,
        "total_turns": d["total_turns"],
        "avg_sentiment_tutor": d["tutor"]["avg_sentiment"],
        "avg_sentiment_student": d["student"]["avg_sentiment"],
        "tutor_turns": d["tutor"]["turns"],
        "student_turns": d["student"]["turns"],
        "tutor_trend": d["dynamics"]["tutor_trend"],
        "student_trend": d["dynamics"]["student_trend"],
        "correlation": d["dynamics"]["correlation"],
        "sentiment_gap": d["dynamics"]["sentiment_gap"],
        "negative_rate_tutor": d["dynamics"]["negative_rate_tutor"],
        "negative_rate_student": d["dynamics"]["negative_rate_student"],
    })
dial_df = pd.DataFrame(dial_rows)


# ─── Заголовок ───

st.title("🎭 Эмоциональная окраска диалогов ИИ-тьютора")
st.caption("Turn-level Sentiment Analysis · Январский прод «Изучи тему» · 698 диалогов, 11 849 реплик")

# ─── Табы ───

tab_overview, tab_dist, tab_dynamics, tab_grades, tab_topics, tab_problems, tab_explorer, tab_corrections = st.tabs([
    "📋 Обзор",
    "📊 Распределения",
    "📈 Динамика",
    "🎓 По классам",
    "📚 По темам",
    "🚨 Проблемные диалоги",
    "🔍 Просмотр диалога",
    "🔧 Коррекции",
])


# ═══════════════════════════════════════════
# TAB 1: ОБЗОР
# ═══════════════════════════════════════════

with tab_overview:
    st.header("Ключевые метрики")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Диалогов", f"{summary['total_dialogues']}")
    c2.metric("Реплик", f"{summary['total_turns']:,}")
    c3.metric("Ср. sentiment тьютора", f"{summary['corpus_avg_sentiment_tutor']:+.3f}")
    c4.metric("Ср. sentiment ученика", f"{summary['corpus_avg_sentiment_student']:+.3f}")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Распределение тональности по ролям")
        role_sent = turns.groupby(["role_ru", "sentiment_label"]).size().reset_index(name="count")
        role_sent["label_ru"] = role_sent["sentiment_label"].map(SENTIMENT_RU)
        fig = px.bar(
            role_sent,
            x="role_ru", y="count", color="label_ru",
            color_discrete_map={v: SENTIMENT_COLORS[k] for k, v in SENTIMENT_RU.items()},
            category_orders={"label_ru": [SENTIMENT_RU[s] for s in SENTIMENT_ORDER]},
            barmode="stack",
            labels={"role_ru": "Роль", "count": "Кол-во реплик", "label_ru": "Тональность"},
        )
        fig.update_layout(height=400, legend_title_text="Тональность")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Доля категорий (% от реплик роли)")
        for role, role_ru in ROLE_RU.items():
            role_data = turns[turns["role"] == role]
            total = len(role_data)
            pcts = role_data["sentiment_label"].value_counts(normalize=True).reindex(SENTIMENT_ORDER, fill_value=0) * 100
            fig = go.Figure(go.Bar(
                x=[pcts[s] for s in SENTIMENT_ORDER],
                y=[SENTIMENT_RU[s] for s in SENTIMENT_ORDER],
                orientation="h",
                marker_color=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER],
                text=[f"{pcts[s]:.1f}%" for s in SENTIMENT_ORDER],
                textposition="auto",
            ))
            fig.update_layout(
                title=f"{role_ru} ({total} реплик)",
                height=200, margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="% реплик",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Гистограмма sentiment_score")

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Тьютор", "Ученик"))
    for i, role in enumerate(["tutor", "student"], 1):
        role_data = turns[(turns["role"] == role) & (~turns["is_empty"])]
        fig.add_trace(
            go.Histogram(
                x=role_data["sentiment_score"],
                nbinsx=40,
                marker_color=ROLE_COLORS[role],
                opacity=0.8,
                name=ROLE_RU[role],
            ),
            row=1, col=i,
        )
    fig.update_layout(height=350, showlegend=True)
    fig.update_xaxes(title_text="Score", range=[-1.1, 1.1])
    fig.update_yaxes(title_text="Кол-во реплик")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 2: РАСПРЕДЕЛЕНИЯ
# ═══════════════════════════════════════════

with tab_dist:
    st.header("Детальные распределения")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Тьютор — категории тональности")
        tutor_counts = turns[turns["role"] == "tutor"]["sentiment_label"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
        fig = px.pie(
            names=[SENTIMENT_RU[s] for s in SENTIMENT_ORDER],
            values=[tutor_counts[s] for s in SENTIMENT_ORDER],
            color_discrete_sequence=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER],
            hole=0.4,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Ученик — категории тональности")
        student_counts = turns[turns["role"] == "student"]["sentiment_label"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
        fig = px.pie(
            names=[SENTIMENT_RU[s] for s in SENTIMENT_ORDER],
            values=[student_counts[s] for s in SENTIMENT_ORDER],
            color_discrete_sequence=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER],
            hole=0.4,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Box-plot score по ролям")

    fig = px.box(
        turns[~turns["is_empty"]],
        x="role_ru", y="sentiment_score",
        color="role_ru",
        color_discrete_map={ROLE_RU[k]: v for k, v in ROLE_COLORS.items()},
        labels={"role_ru": "Роль", "sentiment_score": "Sentiment Score"},
        points="outliers",
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Violin-plot: распределение score по ролям")

    fig = go.Figure()
    for role in ["tutor", "student"]:
        data = turns[(turns["role"] == role) & (~turns["is_empty"])]["sentiment_score"]
        fig.add_trace(go.Violin(
            y=data,
            name=ROLE_RU[role],
            marker_color=ROLE_COLORS[role],
            box_visible=True,
            meanline_visible=True,
        ))
    fig.update_layout(height=450, yaxis_title="Sentiment Score")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Эмоциональные маркеры (emotion model)")
    markers_flat = []
    for _, row in turns[turns["sentiment_markers"].apply(len) > 0].iterrows():
        for m in row["sentiment_markers"]:
            markers_flat.append({"marker": m, "role": row["role_ru"]})
    if markers_flat:
        mdf = pd.DataFrame(markers_flat)
        marker_counts = mdf.groupby(["marker", "role"]).size().reset_index(name="count")
        fig = px.bar(
            marker_counts, x="count", y="marker", color="role",
            orientation="h", barmode="group",
            color_discrete_map={ROLE_RU[k]: v for k, v in ROLE_COLORS.items()},
            labels={"count": "Кол-во", "marker": "Маркер", "role": "Роль"},
        )
        fig.update_layout(height=400, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 3: ДИНАМИКА
# ═══════════════════════════════════════════

with tab_dynamics:
    st.header("Динамика тональности внутри диалогов")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Тренд тональности тьютора по диалогам")
        fig = px.histogram(
            dial_df, x="tutor_trend", nbins=50,
            color_discrete_sequence=[ROLE_COLORS["tutor"]],
            labels={"tutor_trend": "Наклон тренда (slope)"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        pct_neg = (dial_df["tutor_trend"] < 0).mean() * 100
        st.caption(f"В {pct_neg:.0f}% диалогов тон тьютора ухудшается к концу")

    with col2:
        st.subheader("Тренд тональности ученика по диалогам")
        fig = px.histogram(
            dial_df, x="student_trend", nbins=50,
            color_discrete_sequence=[ROLE_COLORS["student"]],
            labels={"student_trend": "Наклон тренда (slope)"},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        pct_neg = (dial_df["student_trend"] < 0).mean() * 100
        st.caption(f"В {pct_neg:.0f}% диалогов тон ученика ухудшается к концу")

    st.divider()
    st.subheader("Scatter: средний sentiment тьютора vs ученика")

    fig = px.scatter(
        dial_df,
        x="avg_sentiment_tutor",
        y="avg_sentiment_student",
        color="negative_rate_student",
        size="total_turns",
        size_max=15,
        hover_data=["dialogue_id", "topic", "grade", "total_turns"],
        color_continuous_scale="RdYlGn_r",
        labels={
            "avg_sentiment_tutor": "Ср. sentiment тьютора",
            "avg_sentiment_student": "Ср. sentiment ученика",
            "negative_rate_student": "% негативных (ученик)",
            "total_turns": "Реплик",
        },
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Sentiment Gap: разрыв тональности (тьютор − ученик)")

    fig = px.histogram(
        dial_df, x="sentiment_gap", nbins=50,
        color_discrete_sequence=["#6366F1"],
        labels={"sentiment_gap": "Gap (тьютор − ученик)"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    mean_gap = dial_df["sentiment_gap"].mean()
    st.caption(f"Средний gap: {mean_gap:+.3f}. Положительный → тьютор позитивнее ученика.")

    st.divider()
    st.subheader("Корреляция тональности тьютор ↔ ученик")

    corr_valid = dial_df[dial_df["correlation"].notna()]
    fig = px.histogram(
        corr_valid, x="correlation", nbins=40,
        color_discrete_sequence=["#14B8A6"],
        labels={"correlation": "Корреляция Пирсона"},
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Средняя корреляция: {corr_valid['correlation'].mean():.3f}. "
               f"Положительная → тон тьютора и ученика синхронны.")


# ═══════════════════════════════════════════
# TAB 4: ПО КЛАССАМ
# ═══════════════════════════════════════════

with tab_grades:
    st.header("Анализ по классам")

    grades_valid = dial_df[dial_df["grade"].between(5, 11)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Средний sentiment ученика по классам")
        grade_student = grades_valid.groupby("grade")["avg_sentiment_student"].mean().reset_index()
        fig = px.bar(
            grade_student, x="grade", y="avg_sentiment_student",
            color="avg_sentiment_student",
            color_continuous_scale="RdYlGn",
            labels={"grade": "Класс", "avg_sentiment_student": "Ср. sentiment"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Средний sentiment тьютора по классам")
        grade_tutor = grades_valid.groupby("grade")["avg_sentiment_tutor"].mean().reset_index()
        fig = px.bar(
            grade_tutor, x="grade", y="avg_sentiment_tutor",
            color="avg_sentiment_tutor",
            color_continuous_scale="RdYlGn",
            labels={"grade": "Класс", "avg_sentiment_tutor": "Ср. sentiment"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Box-plot sentiment ученика по классам")

    turns_valid = turns[(turns["grade"].between(5, 11)) & (turns["role"] == "student") & (~turns["is_empty"])]
    fig = px.box(
        turns_valid, x="grade", y="sentiment_score",
        color_discrete_sequence=[ROLE_COLORS["student"]],
        labels={"grade": "Класс", "sentiment_score": "Sentiment Score"},
        category_orders={"grade": list(range(5, 12))},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Доля негативных реплик ученика по классам")

    neg_by_grade = grades_valid.groupby("grade")["negative_rate_student"].mean().reset_index()
    fig = px.bar(
        neg_by_grade, x="grade", y="negative_rate_student",
        color="negative_rate_student",
        color_continuous_scale="Reds",
        labels={"grade": "Класс", "negative_rate_student": "Средняя доля негативных"},
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Heatmap: sentiment по классу и роли")

    heat_data = turns[turns["grade"].between(5, 11)].groupby(
        ["grade", "role_ru"]
    )["sentiment_score"].mean().reset_index()
    heat_pivot = heat_data.pivot(index="role_ru", columns="grade", values="sentiment_score")
    fig = px.imshow(
        heat_pivot,
        color_continuous_scale="RdYlGn",
        labels=dict(x="Класс", y="Роль", color="Ср. score"),
        text_auto=".3f",
        aspect="auto",
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Кол-во диалогов по классам")

    grade_counts = grades_valid.groupby("grade").size().reset_index(name="count")
    fig = px.bar(
        grade_counts, x="grade", y="count",
        color_discrete_sequence=["#6366F1"],
        labels={"grade": "Класс", "count": "Кол-во диалогов"},
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 5: ПО ТЕМАМ
# ═══════════════════════════════════════════

with tab_topics:
    st.header("Анализ по темам")

    min_dialogues = st.slider("Мин. кол-во диалогов для отображения", 2, 20, 5)

    topic_stats = dial_df.groupby("topic").agg(
        dialogues=("dialogue_id", "count"),
        avg_student=("avg_sentiment_student", "mean"),
        avg_tutor=("avg_sentiment_tutor", "mean"),
        neg_rate=("negative_rate_student", "mean"),
        avg_turns=("total_turns", "mean"),
    ).reset_index()
    topic_stats = topic_stats[topic_stats["dialogues"] >= min_dialogues].sort_values("avg_student")

    st.subheader(f"Средний sentiment ученика по темам (≥ {min_dialogues} диалогов)")

    fig = px.bar(
        topic_stats,
        x="avg_student", y="topic",
        orientation="h",
        color="avg_student",
        color_continuous_scale="RdYlGn",
        hover_data=["dialogues", "neg_rate", "avg_tutor"],
        labels={
            "topic": "Тема", "avg_student": "Ср. sentiment ученика",
            "dialogues": "Диалогов", "neg_rate": "Доля негативных",
            "avg_tutor": "Ср. sentiment тьютора",
        },
    )
    fig.update_layout(height=max(400, len(topic_stats) * 28), yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Scatter: тема — sentiment ученика vs тьютора")

    fig = px.scatter(
        topic_stats,
        x="avg_tutor", y="avg_student",
        size="dialogues", size_max=25,
        color="neg_rate",
        color_continuous_scale="Reds",
        hover_data=["topic", "dialogues"],
        labels={
            "avg_tutor": "Ср. sentiment тьютора",
            "avg_student": "Ср. sentiment ученика",
            "neg_rate": "Доля негативных",
            "dialogues": "Диалогов",
        },
        text="topic",
    )
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(height=600)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Treemap: темы по кол-ву диалогов и sentiment")

    treemap_data = topic_stats[topic_stats["dialogues"] >= min_dialogues].copy()
    treemap_data["abs_student"] = treemap_data["avg_student"].abs()
    fig = px.treemap(
        treemap_data,
        path=["topic"],
        values="dialogues",
        color="avg_student",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        hover_data=["dialogues", "avg_student", "neg_rate"],
        labels={"avg_student": "Ср. sentiment", "dialogues": "Диалогов"},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 6: ПРОБЛЕМНЫЕ ДИАЛОГИ
# ═══════════════════════════════════════════

with tab_problems:
    st.header("Проблемные диалоги")
    st.caption("Диалоги с наибольшей долей негативных реплик ученика (score < -0.3)")

    top_n = st.slider("Показать топ-N", 5, 50, 20)
    top_problems = summary["top_20_problematic_dialogues"][:top_n]

    prob_df = pd.DataFrame(top_problems)
    fig = px.bar(
        prob_df, x="negative_rate_student", y="topic",
        orientation="h",
        color="avg_sentiment_student",
        color_continuous_scale="RdYlGn",
        hover_data=["dialogue_id", "total_turns"],
        labels={
            "negative_rate_student": "Доля негативных реплик",
            "topic": "Тема",
            "avg_sentiment_student": "Ср. sentiment ученика",
            "dialogue_id": "ID диалога",
            "total_turns": "Реплик",
        },
        text=prob_df["dialogue_id"].apply(lambda x: f"ID {x}"),
    )
    fig.update_layout(height=max(400, len(prob_df) * 30), yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Детали проблемных диалогов")

    st.dataframe(
        prob_df.rename(columns={
            "dialogue_id": "ID",
            "topic": "Тема",
            "negative_rate_student": "Доля негативных",
            "avg_sentiment_student": "Ср. sentiment",
            "total_turns": "Реплик",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Соотношение: длина диалога vs негативность ученика")

    fig = px.scatter(
        dial_df,
        x="total_turns", y="negative_rate_student",
        color="avg_sentiment_student",
        color_continuous_scale="RdYlGn",
        hover_data=["dialogue_id", "topic"],
        labels={
            "total_turns": "Кол-во реплик",
            "negative_rate_student": "Доля негативных (ученик)",
            "avg_sentiment_student": "Ср. sentiment ученика",
        },
        opacity=0.6,
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 7: ПРОСМОТР ДИАЛОГА
# ═══════════════════════════════════════════

with tab_explorer:
    st.header("Просмотр отдельного диалога")

    # Выбор диалога
    col_sel, col_info = st.columns([1, 2])

    with col_sel:
        dialogue_ids = sorted(turns["dialogue_id"].unique())
        selected_id = st.selectbox(
            "Выберите диалог (ID)",
            dialogue_ids,
            format_func=lambda x: f"ID {x} — {turns[turns['dialogue_id'] == x]['topic'].iloc[0]}"
        )

    dial_turns = turns[turns["dialogue_id"] == selected_id].sort_values("turn_index")
    dial_info = dial_df[dial_df["dialogue_id"] == selected_id].iloc[0]

    with col_info:
        st.markdown(f"""
        **Тема:** {dial_info['topic']}  ·  **Класс:** {dial_info['grade']}  ·
        **Реплик:** {dial_info['total_turns']}  ·
        **Ср. тьютор:** {dial_info['avg_sentiment_tutor']:+.3f}  ·
        **Ср. ученик:** {dial_info['avg_sentiment_student']:+.3f}
        """)

    st.divider()

    # Sentiment timeline
    st.subheader("Sentiment Timeline")

    fig = go.Figure()
    for role in ["tutor", "student"]:
        role_turns = dial_turns[dial_turns["role"] == role]
        fig.add_trace(go.Scatter(
            x=role_turns["turn_index"],
            y=role_turns["sentiment_score"],
            mode="lines+markers",
            name=ROLE_RU[role],
            line=dict(color=ROLE_COLORS[role], width=2),
            marker=dict(size=8),
            hovertext=role_turns["text_clean"].str[:100],
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        height=350,
        xaxis_title="Номер реплики",
        yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1.1, 1.1]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Реплики диалога
    st.subheader("Реплики")

    for _, turn in dial_turns.iterrows():
        score = turn["sentiment_score"]
        label = turn["sentiment_label"]
        role = turn["role"]
        role_ru = ROLE_RU[role]

        if score > 0.3:
            color = "#22C55E"
        elif score > 0.1:
            color = "#60A5FA"
        elif score > -0.1:
            color = "#9CA3AF"
        elif score > -0.3:
            color = "#F97316"
        else:
            color = "#EF4444"

        badge = f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:12px;">{score:+.2f} {SENTIMENT_RU[label]}</span>'

        align = "left" if role == "student" else "right"
        bg = "#F3E8FF" if role == "tutor" else "#FEF3C7"

        markers_str = ""
        markers = turn["sentiment_markers"]
        if isinstance(markers, (list, np.ndarray)) and len(markers) > 0:
            markers_str = f' · <span style="font-size:11px;color:#666;">{", ".join(str(m) for m in markers)}</span>'

        correction_str = ""
        if "correction_rule" in turn and turn.get("correction_rule"):
            correction_str = f' · <span style="background:#DBEAFE;color:#1E40AF;padding:1px 6px;border-radius:8px;font-size:10px;">🔧 {turn["correction_rule"]}</span>'

        text_preview = turn["text_clean"][:300]
        if len(turn["text_clean"]) > 300:
            text_preview += "..."

        st.markdown(
            f'<div style="text-align:{align};margin-bottom:8px;">'
            f'<div style="display:inline-block;background:{bg};padding:10px 14px;border-radius:12px;max-width:80%;text-align:left;">'
            f'<b>{role_ru}</b> {badge}{markers_str}{correction_str}<br>'
            f'<span style="font-size:14px;">{text_preview}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════
# TAB 8: КОРРЕКЦИИ
# ═══════════════════════════════════════════

with tab_corrections:
    st.header("Rule-based коррекции поверх BERT")
    st.caption("Детерминированные правила, исправляющие систематические ошибки sentiment-модели в образовательном контексте")

    has_corrections = "correction_rule" in turns.columns
    if not has_corrections:
        st.warning("Колонка `correction_rule` отсутствует в данных. Перезапустите пайплайн с модулем corrections.py.")
    else:
        corrected = turns[turns["correction_rule"].astype(str).ne("")]
        total_corrected = len(corrected)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Всего скорректировано", f"{total_corrected:,}")
        c2.metric("% от всех реплик", f"{total_corrected / len(turns) * 100:.1f}%")
        corrected_tutor = len(corrected[corrected["role"] == "tutor"])
        corrected_student = len(corrected[corrected["role"] == "student"])
        c3.metric("Тьютор", f"{corrected_tutor:,}")
        c4.metric("Ученик", f"{corrected_student:,}")

        st.divider()
        st.subheader("Количество коррекций по правилам")

        RULE_DESCRIPTIONS = {
            "math_expression": "Мат. выражение → neutral",
            "topic_request": "Запрос темы → neutral",
            "short_neutral_student": "Короткий ответ без эмоций → neutral",
            "gratitude_detected": "Благодарность/интерес → slightly_positive",
            "frustration_detected": "Фрустрация обнаружена → slightly_negative",
            "frustration_softened": "Фрустрация смягчена (было слишком negative)",
            "frustration_with_emoji": "Фрустрация + грустный эмодзи",
            "tutor_praise_strong": "Похвала тьютора (2+ слова) → positive",
            "tutor_praise": "Похвала тьютора (1 слово) → slightly_positive",
            "tutor_instruction": "Инструкция тьютора → neutral",
            "tutor_definition": "Определение/теория → neutral",
            "tutor_soft_correction": "Мягкая коррекция ('Почти!') → slightly_positive",
            "tutor_formula_neutral": "Тьютор + формулы → neutral",
            "profanity": "Ругательства → negative",
            "homework_copypaste": "Копипаста задания → neutral",
            "student_question": "Учебный вопрос → neutral",
            "student_math_answer_long": "Длинный мат. ответ → neutral",
            "empty_turn": "Пустая реплика → neutral",
        }

        rule_counts = corrected["correction_rule"].value_counts().reset_index()
        rule_counts.columns = ["rule", "count"]
        rule_counts["description"] = rule_counts["rule"].map(
            lambda r: RULE_DESCRIPTIONS.get(r, r)
        )

        fig = px.bar(
            rule_counts,
            x="count", y="description",
            orientation="h",
            color="count",
            color_continuous_scale="Blues",
            labels={"count": "Кол-во", "description": "Правило"},
            text="count",
        )
        fig.update_layout(
            height=max(400, len(rule_counts) * 35),
            yaxis=dict(categoryorder="total ascending"),
            showlegend=False,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Коррекции по ролям")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Тьютор**")
            tutor_rules = corrected[corrected["role"] == "tutor"]["correction_rule"].value_counts()
            if not tutor_rules.empty:
                tutor_rule_df = tutor_rules.reset_index()
                tutor_rule_df.columns = ["rule", "count"]
                tutor_rule_df["description"] = tutor_rule_df["rule"].map(
                    lambda r: RULE_DESCRIPTIONS.get(r, r)
                )
                fig = px.pie(
                    tutor_rule_df, names="description", values="count",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Set3,
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Ученик**")
            student_rules = corrected[corrected["role"] == "student"]["correction_rule"].value_counts()
            if not student_rules.empty:
                student_rule_df = student_rules.reset_index()
                student_rule_df.columns = ["rule", "count"]
                student_rule_df["description"] = student_rule_df["rule"].map(
                    lambda r: RULE_DESCRIPTIONS.get(r, r)
                )
                fig = px.pie(
                    student_rule_df, names="description", values="count",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Примеры скорректированных реплик")

        selected_rule = st.selectbox(
            "Фильтр по правилу",
            ["Все"] + list(rule_counts["rule"]),
            format_func=lambda r: f"{RULE_DESCRIPTIONS.get(r, r)}" if r != "Все" else "Все правила",
        )

        examples = corrected if selected_rule == "Все" else corrected[corrected["correction_rule"] == selected_rule]
        examples = examples.head(30)

        for _, row in examples.iterrows():
            score = row["sentiment_score"]
            rule = row["correction_rule"]
            role_ru = ROLE_RU.get(row["role"], row["role"])

            if score > 0.3:
                color = "#22C55E"
            elif score > 0.1:
                color = "#60A5FA"
            elif score > -0.1:
                color = "#9CA3AF"
            elif score > -0.3:
                color = "#F97316"
            else:
                color = "#EF4444"

            badge = f'<span style="background:{color};color:white;padding:2px 8px;border-radius:10px;font-size:12px;">{score:+.2f}</span>'
            rule_badge = f'<span style="background:#DBEAFE;color:#1E40AF;padding:2px 6px;border-radius:8px;font-size:11px;">🔧 {RULE_DESCRIPTIONS.get(rule, rule)}</span>'

            text_preview = str(row["text_clean"])[:200]
            if len(str(row["text_clean"])) > 200:
                text_preview += "..."

            st.markdown(
                f'<div style="margin-bottom:6px;padding:8px 12px;background:#F9FAFB;border-radius:8px;border-left:3px solid {color};">'
                f'<b>{role_ru}</b> {badge} {rule_badge}<br>'
                f'<span style="font-size:13px;color:#374151;">{text_preview}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
