"""Анализ эмоций в диалогах: BERT sentiment + Gemini emotions."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Эмоции в диалогах", page_icon="🧠", layout="wide")

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Палитры и константы ────────────────────────────────────────────────────

SENTIMENT_COLOR = {
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
EMOTION_COLORS = {
    "anxiety": "#EF4444",
    "boredom": "#9CA3AF",
    "curiosity": "#3498db",
    "confusion": "#E67E22",
    "frustration": "#C0392B",
    "joy": "#F1C40F",
    "neutral": "#BDC3C7",
    "flow": "#22C55E",
    "delight": "#F39C12",
}
EMOTION_RU = {
    "anxiety": "Тревога",
    "boredom": "Скука",
    "curiosity": "Любопытство",
    "confusion": "Замешательство",
    "frustration": "Фрустрация",
    "joy": "Радость",
    "neutral": "Нейтрально",
    "flow": "Поток",
    "delight": "Восторг",
}
LP_COLORS = {"productive": "#22C55E", "disengaged": "#EF4444", "stuck": "#F97316", "n/a": "#9CA3AF"}
LP_RU = {"productive": "Продуктивно", "disengaged": "Отвлечён", "stuck": "Застрял", "n/a": "Н/д"}


# ── Загрузка данных ────────────────────────────────────────────────────────

def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data
def load_summary(_mtime_key: float) -> dict | None:
    p = DATA_DIR / "summary.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_dialogues(_mtime_key: float) -> list[dict] | None:
    p = DATA_DIR / "dialogues.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_turns(_mtime_key: float) -> pd.DataFrame | None:
    p = DATA_DIR / "turns.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


@st.cache_data
def load_annotations(_mtime_key: float) -> pd.DataFrame | None:
    p = DATA_DIR / "annotated_turns.jsonl"
    if not p.exists():
        return None
    records = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "error" in obj:
                continue
            if "emotion" in obj:
                obj["emotion"] = str(obj["emotion"]).lower()
            records.append(obj)
    return pd.DataFrame(records) if records else None


def no_data(name: str) -> None:
    st.warning(f"Файл **{name}** не найден. Убедитесь, что пайплайн был запущен.")


def sentiment_badge(label: str, score: float) -> str:
    colors = {
        "positive": ("#22C55E", "#fff"),
        "slightly_positive": ("#60A5FA", "#fff"),
        "neutral": ("#9CA3AF", "#fff"),
        "slightly_negative": ("#F97316", "#fff"),
        "negative": ("#EF4444", "#fff"),
    }
    bg, fg = colors.get(label, ("#9CA3AF", "#fff"))
    ru = SENTIMENT_RU.get(label, label)
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:10px;font-size:0.8em">{ru} ({score:+.2f})</span>'
    )


def emotion_badge(emotion: str, valence: float) -> str:
    bg = EMOTION_COLORS.get(emotion, "#9CA3AF")
    ru = EMOTION_RU.get(emotion, emotion)
    return (
        f'<span style="background:{bg};color:#fff;padding:2px 8px;'
        f'border-radius:10px;font-size:0.8em">{ru} (v={valence:+.2f})</span>'
    )


# ── Данные ─────────────────────────────────────────────────────────────────

summary = load_summary(_mtime(DATA_DIR / "summary.json"))
dialogues = load_dialogues(_mtime(DATA_DIR / "dialogues.json"))
turns_df = load_turns(_mtime(DATA_DIR / "turns.parquet"))
annotations_df = load_annotations(_mtime(DATA_DIR / "annotated_turns.jsonl"))

# Построим список диалогов из аннотаций (полный набор)
ann_dialogue_ids: list[int] = []
ann_topic_map: dict[int, str] = {}
if annotations_df is not None and "activity_id" in annotations_df.columns:
    ann_topic_map = (
        annotations_df.groupby("activity_id")["topic"]
        .first()
        .to_dict()
    )
    ann_dialogue_ids = sorted(ann_topic_map.keys())

# ── Заголовок ──────────────────────────────────────────────────────────────

st.title("🧠 Эмоциональный анализ диалогов")
st.caption("BERT sentiment + Gemini emotion annotation · Пилот «Изучи тему»")

# ── Табы ───────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Обзор корпуса",
    "🎭 Эмоции (Gemini)",
    "📈 Динамика диалогов",
    "⚖️ Valence по диалогам",
    "🔍 Проводник диалогов",
    "📖 Справка",
])

# ═══════════════════════════════════════════════════════════════════════════
# Таб 1 — Обзор корпуса
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    if annotations_df is None:
        no_data("annotated_turns.jsonl")
    else:
        n_dialogues = annotations_df["activity_id"].nunique()
        n_turns = len(annotations_df)
        avg_valence = annotations_df["valence"].mean()
        avg_helpfulness = annotations_df["helpfulness"].mean() if "helpfulness" in annotations_df.columns else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Диалогов", n_dialogues)
        c2.metric("Реплик (ученик)", n_turns)
        c3.metric("Ср. valence", f"{avg_valence:+.3f}")
        c4.metric("Ср. helpfulness", f"{avg_helpfulness:.2f}")
        st.divider()

        st.subheader("Средняя valence ученика по диалогам")
        dlg_valence = (
            annotations_df.groupby("activity_id")
            .agg(avg_valence=("valence", "mean"), topic=("topic", "first"))
            .reset_index()
            .sort_values("activity_id")
        )
        dlg_valence["Знак"] = dlg_valence["avg_valence"].apply(
            lambda v: "Позитивный" if v >= 0 else "Негативный"
        )
        fig = px.bar(
            dlg_valence, x="activity_id", y="avg_valence", color="Знак",
            color_discrete_map={"Позитивный": "#22C55E", "Негативный": "#EF4444"},
            hover_data=["topic"],
            labels={"activity_id": "ID диалога", "avg_valence": "Ср. valence"},
        )
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

        # Проблемные диалоги (lowest valence)
        st.subheader("Проблемные диалоги (самая низкая valence)")
        problem = (
            annotations_df.groupby("activity_id")
            .agg(
                topic=("topic", "first"),
                avg_valence=("valence", "mean"),
                n_turns=("turn_idx", "count"),
                negative_share=("valence", lambda x: (x < -0.2).mean()),
            )
            .reset_index()
            .sort_values("avg_valence")
            .head(20)
            .rename(columns={
                "activity_id": "ID",
                "topic": "Тема",
                "avg_valence": "Ср. valence",
                "n_turns": "Реплик",
                "negative_share": "% негатива",
            })
        )
        problem["% негатива"] = (problem["% негатива"] * 100).round(1)
        st.dataframe(problem, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════
# Таб 2 — Эмоции (Gemini)
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    if annotations_df is None:
        no_data("annotated_turns.jsonl")
    else:
        st.subheader("Разметка эмоций (Gemini)")

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Распределение эмоций**")
            emo_counts = annotations_df["emotion"].value_counts().reset_index()
            emo_counts.columns = ["emotion", "count"]
            emo_counts["label"] = emo_counts["emotion"].map(
                lambda e: EMOTION_RU.get(e, e)
            )
            fig = px.pie(
                emo_counts, names="label", values="count",
                color="emotion", color_discrete_map=EMOTION_COLORS,
            )
            fig.update_traces(textinfo="label+percent")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            if "learning_potential" in annotations_df.columns:
                st.markdown("**Учебный потенциал (learning_potential)**")
                lp = annotations_df["learning_potential"].value_counts().reset_index()
                lp.columns = ["lp", "count"]
                lp["label"] = lp["lp"].map(lambda v: LP_RU.get(v, v))
                fig = px.pie(
                    lp, names="label", values="count",
                    color="lp", color_discrete_map=LP_COLORS,
                )
                fig.update_traces(textinfo="label+percent")
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Scatter: valence × arousal
        if {"valence", "arousal"}.issubset(annotations_df.columns):
            st.markdown("**Valence × Arousal** (размер = helpfulness)")
            sdf = annotations_df.dropna(subset=["valence", "arousal"]).copy()
            sdf["emotion_ru"] = sdf["emotion"].map(lambda e: EMOTION_RU.get(e, e))
            fig = px.scatter(
                sdf, x="valence", y="arousal",
                color="emotion_ru",
                size="helpfulness" if "helpfulness" in sdf.columns else None,
                color_discrete_map={EMOTION_RU.get(k, k): v for k, v in EMOTION_COLORS.items()},
                hover_data=["topic", "user_reply"] if "topic" in sdf.columns else None,
                labels={"valence": "Валентность", "arousal": "Возбуждение", "emotion_ru": "Эмоция"},
            )
            fig.update_layout(xaxis=dict(range=[-1.1, 1.1]), yaxis=dict(range=[-1.1, 1.1]))
            fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
            fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            if "helpfulness" in annotations_df.columns:
                st.markdown("**Средняя helpfulness по эмоциям**")
                hbe = (
                    annotations_df.groupby("emotion")["helpfulness"]
                    .mean().reset_index()
                    .sort_values("helpfulness", ascending=False)
                )
                hbe["emotion_ru"] = hbe["emotion"].map(lambda e: EMOTION_RU.get(e, e))
                fig = px.bar(
                    hbe, x="emotion_ru", y="helpfulness",
                    color="emotion", color_discrete_map=EMOTION_COLORS,
                    labels={"emotion_ru": "Эмоция", "helpfulness": "Ср. helpfulness"},
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if {"reply_type", "helpfulness"}.issubset(annotations_df.columns):
                st.markdown("**Средняя helpfulness по reply_type**")
                hbr = (
                    annotations_df.groupby("reply_type")["helpfulness"]
                    .mean().reset_index()
                    .sort_values("helpfulness", ascending=False)
                )
                fig = px.bar(
                    hbr, x="reply_type", y="helpfulness",
                    labels={"reply_type": "Тип ответа", "helpfulness": "Ср. helpfulness"},
                )
                st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Таб 3 — Динамика диалогов (по Gemini valence)
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    if annotations_df is None or not ann_dialogue_ids:
        no_data("annotated_turns.jsonl")
    else:
        st.subheader("Динамика valence/эмоций в диалогах")

        dlg_opts = {did: f'{did}: {ann_topic_map.get(did, "?")}' for did in ann_dialogue_ids}
        sel = st.selectbox(
            "Выберите диалог", ann_dialogue_ids,
            format_func=lambda x: dlg_opts[x], key="dyn_dlg",
        )

        dt = annotations_df[annotations_df["activity_id"] == sel].sort_values("turn_idx")
        dt = dt.copy()
        dt["emotion_ru"] = dt["emotion"].map(lambda e: EMOTION_RU.get(e, e))

        # Valence line
        fig = px.line(
            dt, x="turn_idx", y="valence", markers=True,
            labels={"turn_idx": "Номер реплики (ученик)", "valence": "Valence"},
            hover_data=["emotion_ru", "user_reply"],
        )
        fig.update_traces(line_color="#F59E0B", marker_color="#F59E0B")
        fig.update_layout(xaxis=dict(dtick=1), yaxis=dict(range=[-1.1, 1.1]))
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)

        # Arousal line
        fig2 = px.line(
            dt, x="turn_idx", y="arousal", markers=True,
            labels={"turn_idx": "Номер реплики (ученик)", "arousal": "Arousal"},
            hover_data=["emotion_ru", "user_reply"],
        )
        fig2.update_traces(line_color="#8B5CF6", marker_color="#8B5CF6")
        fig2.update_layout(xaxis=dict(dtick=1), yaxis=dict(range=[-1.1, 1.1]))
        fig2.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        st.plotly_chart(fig2, use_container_width=True)

        # Emotion sequence
        st.markdown("**Последовательность эмоций**")
        emo_seq = dt[["turn_idx", "emotion", "emotion_ru", "valence", "arousal", "user_reply"]].copy()
        emo_seq.columns = ["Турн", "Эмоция (код)", "Эмоция", "Valence", "Arousal", "Реплика"]
        st.dataframe(emo_seq, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Тепловая карта valence ученика")

        piv = annotations_df.pivot_table(
            index="activity_id", columns="turn_idx",
            values="valence", aggfunc="first",
        )
        y_labels = [f"{i}: {ann_topic_map.get(i, '?')}" for i in piv.index]

        fig = go.Figure(go.Heatmap(
            z=piv.values,
            x=[str(c) for c in piv.columns],
            y=y_labels,
            colorscale="RdYlGn", zmid=0,
            colorbar=dict(title="Valence"),
        ))
        fig.update_layout(
            xaxis_title="Номер реплики ученика",
            yaxis_title="Диалог",
            height=max(400, len(piv) * 22),
        )
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Таб 4 — Valence по диалогам (замена Тьютор vs Ученик)
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    if annotations_df is None:
        no_data("annotated_turns.jsonl")
    else:
        st.subheader("Средняя valence и helpfulness по диалогам")

        dlg_stats = (
            annotations_df.groupby("activity_id")
            .agg(
                topic=("topic", "first"),
                avg_valence=("valence", "mean"),
                avg_arousal=("arousal", "mean"),
                avg_helpfulness=("helpfulness", "mean"),
                n_turns=("turn_idx", "count"),
            )
            .reset_index()
            .sort_values("activity_id")
        )

        fig = px.bar(
            dlg_stats, x="activity_id", y="avg_valence",
            color="avg_valence", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
            hover_data=["topic", "avg_helpfulness", "n_turns"],
            labels={"activity_id": "ID диалога", "avg_valence": "Ср. valence"},
        )
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Распределение эмоций по диалогам")

        emo_per_dlg = (
            annotations_df.groupby(["activity_id", "emotion"])
            .size().reset_index(name="count")
        )
        totals = emo_per_dlg.groupby("activity_id")["count"].transform("sum")
        emo_per_dlg["share"] = emo_per_dlg["count"] / totals
        emo_per_dlg["emotion_ru"] = emo_per_dlg["emotion"].map(lambda e: EMOTION_RU.get(e, e))

        fig = px.bar(
            emo_per_dlg, x="activity_id", y="share", color="emotion_ru",
            color_discrete_map={EMOTION_RU.get(k, k): v for k, v in EMOTION_COLORS.items()},
            labels={"activity_id": "ID диалога", "share": "Доля", "emotion_ru": "Эмоция"},
        )
        fig.update_layout(xaxis=dict(dtick=1), barmode="stack")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Valence vs длина диалога")

        fig = px.scatter(
            dlg_stats, x="n_turns", y="avg_valence",
            text="activity_id", hover_data=["topic"],
            size="avg_helpfulness",
            labels={"n_turns": "Реплик", "avg_valence": "Ср. valence"},
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Таб 5 — Проводник диалогов
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    if annotations_df is None or not ann_dialogue_ids:
        no_data("annotated_turns.jsonl")
    else:
        st.subheader("Проводник диалогов")

        dlg_opts5 = {did: f'{did}: {ann_topic_map.get(did, "?")}' for did in ann_dialogue_ids}
        sel5 = st.selectbox(
            "Выберите диалог", ann_dialogue_ids,
            format_func=lambda x: dlg_opts5[x], key="expl_dlg",
        )

        ann_turns = annotations_df[annotations_df["activity_id"] == sel5].sort_values("turn_idx")

        # Если есть BERT-данные для этого диалога — используем их для полного отображения
        has_bert = turns_df is not None and sel5 in turns_df["dialogue_id"].values

        if has_bert:
            dt5 = turns_df[turns_df["dialogue_id"] == sel5].sort_values("turn_index")
            ann_map: dict[int, dict] = {}
            for _, r in ann_turns.iterrows():
                ann_map[int(r["turn_idx"])] = r.to_dict()

            for _, turn in dt5.iterrows():
                role = turn["role"]
                avatar = "🤖" if role == "tutor" else "🧑‍🎓"
                with st.chat_message(role, avatar=avatar):
                    badge = sentiment_badge(turn["sentiment_label"], turn["sentiment_score"])
                    st.markdown(f'{turn["text"]}  {badge}', unsafe_allow_html=True)

                    ann = ann_map.get(int(turn["turn_index"]))
                    if ann:
                        with st.expander("🔍 Gemini-разметка"):
                            gc1, gc2, gc3, gc4 = st.columns(4)
                            emo_raw = ann.get("emotion", "—")
                            gc1.metric("Эмоция", EMOTION_RU.get(str(emo_raw), str(emo_raw)))
                            gc2.metric("Валентность", f"{ann.get('valence', 0):+.1f}")
                            gc3.metric("Возбуждение", f"{ann.get('arousal', 0):+.1f}")
                            gc4.metric("Helpfulness", str(ann.get("helpfulness", "—")))

                            sig = ann.get("signal_description", "")
                            if sig:
                                st.caption(f"**Сигнал:** {sig}")
                            bp = ann.get("bot_prompt", "")
                            if bp:
                                st.caption(f"**Промпт бота:** {bp}")
        else:
            # Отображаем из аннотаций (только реплики ученика + бот-промпт)
            for _, ann in ann_turns.iterrows():
                bot_prompt = ann.get("bot_prompt", "")
                if bot_prompt:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(bot_prompt)

                with st.chat_message("user", avatar="🧑‍🎓"):
                    badge = emotion_badge(ann["emotion"], ann.get("valence", 0))
                    st.markdown(f'{ann["user_reply"]}  {badge}', unsafe_allow_html=True)

                    with st.expander("🔍 Gemini-разметка"):
                        gc1, gc2, gc3, gc4 = st.columns(4)
                        emo_raw = ann.get("emotion", "—")
                        gc1.metric("Эмоция", EMOTION_RU.get(str(emo_raw), str(emo_raw)))
                        gc2.metric("Валентность", f"{ann.get('valence', 0):+.1f}")
                        gc3.metric("Возбуждение", f"{ann.get('arousal', 0):+.1f}")
                        gc4.metric("Helpfulness", str(ann.get("helpfulness", "—")))

                        sig = ann.get("signal_description", "")
                        if sig:
                            st.caption(f"**Сигнал:** {sig}")

                        lp = ann.get("learning_potential", "")
                        if lp:
                            st.caption(f"**Учебный потенциал:** {LP_RU.get(lp, lp)}")

# ═══════════════════════════════════════════════════════════════════════════
# Таб 6 — Справка
# ═══════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Как читать показатели")
    st.markdown(
        "Данные получены в два этапа: BERT-модель размечает sentiment каждой реплики, "
        "затем Gemini 2.5 Flash классифицирует эмоциональное состояние ученика "
        "с помощью 3-pass ensemble (три прогона с разной температурой, результат — "
        "голосование большинства)."
    )

    st.divider()

    # ── Числовые шкалы ──────────────────────────────────────────────────
    st.subheader("Числовые шкалы")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Valence (валентность)")
        st.markdown(
            "Эмоциональная окраска реплики на шкале **от −1.0 до +1.0** "
            "(Pekrun Control-Value Theory).\n\n"
            "| Значение | Интерпретация |\n"
            "|----------|---------------|\n"
            "| −1.0 | Крайне негативно (отчаяние, сильная фрустрация) |\n"
            "| −0.5 | Негативно (растерянность, тревога) |\n"
            "| 0.0 | Нейтрально |\n"
            "| +0.5 | Позитивно (уверенность, интерес) |\n"
            "| +1.0 | Крайне позитивно (инсайт, энтузиазм) |"
        )

    with col2:
        st.markdown("#### Arousal (возбуждение)")
        st.markdown(
            "Уровень активации и вовлечённости на шкале **от −1.0 до +1.0**.\n\n"
            "| Значение | Интерпретация |\n"
            "|----------|---------------|\n"
            "| −1.0 | Полная апатия, отстранённость |\n"
            "| 0.0 | Нейтральное вовлечение |\n"
            "| +1.0 | Максимальное возбуждение / концентрация |"
        )

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Helpfulness (полезность)")
        st.markdown(
            "Насколько текущий ход продуктивен для обучения. "
            "Шкала Ликерта **от 1 до 9**.\n\n"
            "| Значение | Интерпретация |\n"
            "|----------|---------------|\n"
            "| 1 | Полностью тормозит обучение |\n"
            "| 5 | Нейтрально |\n"
            "| 9 | Максимально продуктивный ход |"
        )

    with col4:
        st.markdown("#### Confidence (conf)")
        st.markdown(
            "Уверенность модели в классификации, скорректированная на согласованность "
            "между проходами. **от 0.0 до 1.0**.\n\n"
            "| Значение | Интерпретация |\n"
            "|----------|---------------|\n"
            "| 0.8–1.0 | Явные текстовые + поведенческие сигналы совпадают |\n"
            "| 0.5–0.7 | Один сильный сигнал или несколько слабых |\n"
            "| 0.3–0.5 | Только поведенческие сигналы, текст нейтрален |\n"
            "| < 0.3 | Почти нет сигналов (типично для math-реплик) |"
        )

    st.divider()

    # ── Таксономия эмоций ───────────────────────────────────────────────
    st.subheader("Таксономия эмоций (7 классов)")
    st.markdown(
        "Основана на D'Mello & Graesser (2012) — динамика аффективных состояний в ITS, "
        "и Pekrun Control-Value Theory (2006) — два измерения: валентность и активация."
    )

    emo_data = [
        ("Flow / Поток", "Позитивная", "Высокая", "#22C55E",
         "Ученик вовлечён, сосредоточен, работает уверенно. "
         "Сигналы: быстрый верный ответ, уточняющие вопросы, развёрнутые рассуждения."),
        ("Confusion / Замешательство", "Негативная", "Высокая", "#E67E22",
         "Когнитивный дисбаланс. Потенциально продуктивно, если ученик пытается разобраться. "
         "Сигналы: неверный ответ с попыткой рассуждать, «не знаю», частичный ответ."),
        ("Frustration / Фрустрация", "Негативная", "Высокая–средняя", "#C0392B",
         "Ученик застрял, нарастает раздражение. Требует интервенции. "
         "Сигналы: повтор ошибки, «???», резкие минимальные ответы."),
        ("Boredom / Скука", "Негативная", "Низкая", "#9CA3AF",
         "Отстранённость, потеря интереса. "
         "Сигналы: минимальный ввод без смысла, навигация вместо ответа, ранний выход."),
        ("Delight / Восторг", "Позитивная", "Высокая", "#F39C12",
         "Инсайт, eureka-момент после усилия. "
         "Сигналы: развёрнутый ответ после серии ошибок, «о, понял!», самостоятельное обобщение."),
        ("Anxiety / Тревога", "Негативная", "Высокая", "#EF4444",
         "Тревога от высоких ставок. Преимущественно в начале сессии. "
         "Сигналы: упоминание контрольной, экзамена, срочности, страха ошибки."),
        ("Neutral / Нейтрально", "Нейтральная", "Средняя", "#BDC3C7",
         "Базовое рабочее состояние, нет выраженных эмоций. "
         "Сигналы: деловые навигационные реплики, выбор темы."),
    ]
    for name, val, act, color, desc in emo_data:
        st.markdown(
            f'<div style="border-left:4px solid {color};padding:8px 12px;margin-bottom:8px">'
            f'<strong>{name}</strong> &nbsp; '
            f'<span style="font-size:0.85em;color:#888">валентность: {val} · активация: {act}</span>'
            f'<br><span style="font-size:0.9em">{desc}</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Учебный потенциал ───────────────────────────────────────────────
    st.subheader("Учебный потенциал (learning_potential)")
    st.markdown(
        "Оценка того, насколько текущий момент может привести к обучению."
    )
    lp_data = [
        ("Productive / Продуктивно", "#22C55E",
         "Ученик активно работает — ошибки ведут к осмыслению и прогрессу."),
        ("Stuck / Застрял", "#F97316",
         "Повторяет ту же ошибку без прогресса — нужна помощь или смена подхода."),
        ("Disengaged / Отвлечён", "#EF4444",
         "Ученик не включён в процесс — минимальный ввод, переключение, ранний выход."),
        ("N/A / Н/д", "#9CA3AF",
         "Невозможно определить (навигационная реплика, выбор темы и т.п.)."),
    ]
    for name, color, desc in lp_data:
        st.markdown(
            f'<div style="border-left:4px solid {color};padding:8px 12px;margin-bottom:8px">'
            f'<strong>{name}</strong><br>'
            f'<span style="font-size:0.9em">{desc}</span></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Типы реплик ─────────────────────────────────────────────────────
    st.subheader("Тип реплики (reply_type)")
    st.markdown(
        "| Тип | Описание |\n"
        "|-----|----------|\n"
        "| **math** | Математическое выражение, число, формула |\n"
        "| **nav** | Навигационная реплика: «Да», «Нет», «Изучить тему», выбор из меню |\n"
        "| **free_text** | Свободный текст — рассуждение, вопрос, развёрнутый ответ |"
    )

    st.divider()

    # ── Как читать графики ──────────────────────────────────────────────
    st.subheader("Как читать графики")

    st.markdown("#### Тепловая карта valence")
    st.markdown(
        "Каждая строка — диалог, каждый столбец — номер реплики ученика. "
        "Цвет ячейки отражает valence: **зелёный** — позитивный, **красный** — негативный, "
        "**жёлтый** — нейтральный. Позволяет быстро найти диалоги, где ученик «скатывался» "
        "в негатив, и на каком турне это произошло."
    )

    st.markdown("#### Valence × Arousal (scatter)")
    st.markdown(
        "Каждая точка — реплика ученика. Горизонтальная ось — валентность (плохо ← → хорошо), "
        "вертикальная — возбуждение (апатия ↓ ↑ вовлечённость). Размер точки — helpfulness. "
        "Цвет — эмоция.\n\n"
        "- **Правый верхний** квадрант (V+, A+): flow, delight — идеальная зона обучения\n"
        "- **Левый верхний** (V−, A+): confusion, frustration, anxiety — высокая активация, но негативная\n"
        "- **Левый нижний** (V−, A−): boredom — ученик отключился\n"
        "- **Центр**: neutral — рабочее состояние без выраженных эмоций"
    )

    st.markdown("#### Динамика valence/arousal")
    st.markdown(
        "Линейный график по турнам диалога. Показывает, как менялось эмоциональное состояние "
        "ученика по ходу беседы. Падение valence может указывать на нарастающую фрустрацию, "
        "рост — на вовлечение и прогресс."
    )

    st.markdown("#### Распределение эмоций по диалогам (stacked bar)")
    st.markdown(
        "Доля каждой эмоции внутри диалога. Позволяет сравнить эмоциональный профиль "
        "разных сессий — например, где преобладал flow, а где — confusion и boredom."
    )

    st.divider()

    # ── Методология ─────────────────────────────────────────────────────
    st.subheader("Методология")
    st.markdown(
        "**Этап 1 — BERT Sentiment.** Многоязычная BERT-модель (cointegrated/rubert-tiny-sentiment) "
        "присваивает каждой реплике sentiment-скор от −1 до +1 и категорию "
        "(negative / slightly_negative / neutral / slightly_positive / positive).\n\n"
        "**Этап 2 — Gemini Emotion Annotation.** Каждая реплика ученика проходит через "
        "Gemini 2.5 Flash с 3-pass ensemble:\n"
        "- Три прогона с температурами 0.1, 0.2, 0.3\n"
        "- Эмоция — голосование большинства (majority vote)\n"
        "- Числовые шкалы (valence, arousal, helpfulness) — медиана\n"
        "- Confidence = средняя уверенность × доля согласия между проходами\n\n"
        "**Теоретическая база:**\n"
        "- D'Mello, S. K., & Graesser, A. C. (2012). Dynamics of affective states during complex learning. *Learning and Instruction*, 22(2), 145–157.\n"
        "- Pekrun, R. (2006). The control-value theory of achievement emotions. *Educational Psychology Review*, 18(4), 315–341."
    )

# ── Футер ──────────────────────────────────────────────────────────────────
st.divider()
st.caption("Sentiment: BERT (multilingual) · Эмоции: Gemini 2.5 Flash · Streamlit + Plotly")
