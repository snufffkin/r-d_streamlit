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
}
EMOTION_RU = {
    "anxiety": "Тревога",
    "boredom": "Скука",
    "curiosity": "Любопытство",
    "confusion": "Замешательство",
    "frustration": "Фрустрация",
    "joy": "Радость",
    "neutral": "Нейтрально",
}
LP_COLORS = {"productive": "#22C55E", "disengaged": "#EF4444", "n/a": "#9CA3AF"}
LP_RU = {"productive": "Продуктивно", "disengaged": "Отвлечён", "n/a": "Н/д"}


# ── Загрузка данных ────────────────────────────────────────────────────────

def _mtime(p: Path) -> float:
    """Время модификации файла — используется как ключ кеша."""
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


# ── Данные ─────────────────────────────────────────────────────────────────

summary = load_summary(_mtime(DATA_DIR / "summary.json"))
dialogues = load_dialogues(_mtime(DATA_DIR / "dialogues.json"))
turns_df = load_turns(_mtime(DATA_DIR / "turns.parquet"))
annotations_df = load_annotations(_mtime(DATA_DIR / "annotated_turns.jsonl"))

# ── Заголовок ──────────────────────────────────────────────────────────────

st.title("🧠 Эмоциональный анализ диалогов")
st.caption("BERT sentiment + Gemini emotion annotation · Пилот «Изучи тему»")

# ── Табы ───────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Обзор корпуса",
    "🎭 Эмоции (Gemini)",
    "📈 Динамика диалогов",
    "⚖️ Тьютор vs Ученик",
    "🔍 Проводник диалогов",
])

# ═══════════════════════════════════════════════════════════════════════════
# Таб 1 — Обзор корпуса
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    if summary is None:
        no_data("summary.json")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Диалогов", summary["total_dialogues"])
        c2.metric("Реплик", summary["total_turns"])
        c3.metric("Ср. sentiment тьютора", f"{summary['corpus_avg_sentiment_tutor']:+.3f}")
        c4.metric("Ср. sentiment ученика", f"{summary['corpus_avg_sentiment_student']:+.3f}")
        st.divider()

    if dialogues is None:
        no_data("dialogues.json")
    else:
        st.subheader("Средний sentiment ученика по диалогам")
        bar_data = pd.DataFrame([
            {
                "ID": d["dialogue_id"],
                "Тема": d["topic"],
                "Ср. sentiment": d["student"]["avg_sentiment"],
                "Знак": "Позитивный" if d["student"]["avg_sentiment"] >= 0 else "Негативный",
            }
            for d in dialogues
        ])
        fig = px.bar(
            bar_data, x="ID", y="Ср. sentiment", color="Знак",
            color_discrete_map={"Позитивный": "#22C55E", "Негативный": "#EF4444"},
            hover_data=["Тема"],
        )
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

        if summary and "top_20_problematic_dialogues" in summary:
            st.subheader("Проблемные диалоги")
            prob = pd.DataFrame(summary["top_20_problematic_dialogues"]).rename(columns={
                "dialogue_id": "ID", "topic": "Тема",
                "negative_rate_student": "% негатива",
                "avg_sentiment_student": "Ср. sentiment",
                "total_turns": "Реплик",
            })
            st.dataframe(prob, use_container_width=True, hide_index=True)

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
# Таб 3 — Динамика диалогов
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    if turns_df is None or dialogues is None:
        no_data("turns.parquet / dialogues.json")
    else:
        st.subheader("Динамика sentiment в диалогах")

        dlg_opts = {d["dialogue_id"]: f'{d["dialogue_id"]}: {d["topic"]}' for d in dialogues}
        sel = st.selectbox(
            "Выберите диалог", list(dlg_opts.keys()),
            format_func=lambda x: dlg_opts[x], key="dyn_dlg",
        )

        dt = turns_df[turns_df["dialogue_id"] == sel].sort_values("turn_index")
        fig = px.line(
            dt, x="turn_index", y="sentiment_score", color="role",
            markers=True,
            color_discrete_map={"tutor": "#8B5CF6", "student": "#F59E0B"},
            labels={"turn_index": "Номер реплики", "sentiment_score": "Sentiment", "role": "Роль"},
        )
        fig.update_layout(xaxis=dict(dtick=1), yaxis=dict(range=[-1.1, 1.1]))
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)

        dm = next((d for d in dialogues if d["dialogue_id"] == sel), None)
        if dm:
            dyn = dm["dynamics"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Тренд тьютора", f"{dyn['tutor_trend']:+.4f}")
            mc2.metric("Тренд ученика", f"{dyn['student_trend']:+.4f}")
            corr = dyn["correlation"]
            mc3.metric("Корреляция", f"{corr:.2f}" if corr is not None else "н/д")
            mc4.metric("Sentiment gap", f"{dyn['sentiment_gap']:+.3f}")

        st.divider()
        st.subheader("Тепловая карта sentiment ученика")

        stu = turns_df[turns_df["role"] == "student"]
        piv = stu.pivot_table(index="dialogue_id", columns="turn_index",
                              values="sentiment_score", aggfunc="first")
        topic_map = {d["dialogue_id"]: d["topic"] for d in dialogues}
        y_labels = [topic_map.get(i, str(i)) for i in piv.index]

        fig = go.Figure(go.Heatmap(
            z=piv.values,
            x=[str(c) for c in piv.columns],
            y=y_labels,
            colorscale="RdYlGn", zmid=0,
            colorbar=dict(title="Sentiment"),
        ))
        fig.update_layout(xaxis_title="Номер реплики ученика", yaxis_title="Диалог", height=400)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Таб 4 — Тьютор vs Ученик
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    if dialogues is None:
        no_data("dialogues.json")
    else:
        st.subheader("Сравнение sentiment: тьютор vs ученик")

        rows = []
        for d in dialogues:
            rows.append({
                "ID": d["dialogue_id"], "Тема": d["topic"],
                "Тьютор": d["tutor"]["avg_sentiment"],
                "Ученик": d["student"]["avg_sentiment"],
            })
        cdf = pd.DataFrame(rows).melt(
            id_vars=["ID", "Тема"], value_vars=["Тьютор", "Ученик"],
            var_name="Роль", value_name="Ср. sentiment",
        )
        fig = px.bar(
            cdf, x="ID", y="Ср. sentiment", color="Роль", barmode="group",
            color_discrete_map={"Тьютор": "#8B5CF6", "Ученик": "#F59E0B"},
            hover_data=["Тема"],
        )
        fig.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Распределение категорий sentiment")

        stack_rows = []
        for d in dialogues:
            for rk, rr in [("tutor", "Тьютор"), ("student", "Ученик")]:
                dist = d[rk]["sentiment_distribution"]
                total = d[rk]["turns"]
                for cat in SENTIMENT_ORDER:
                    stack_rows.append({
                        "ID": d["dialogue_id"], "Роль": rr,
                        "Категория": SENTIMENT_RU[cat],
                        "Доля": dist.get(cat, 0) / total if total else 0,
                    })
        sdf = pd.DataFrame(stack_rows)
        fig = px.bar(
            sdf, x="ID", y="Доля", color="Категория", facet_row="Роль",
            color_discrete_map={SENTIMENT_RU[k]: v for k, v in SENTIMENT_COLOR.items()},
            category_orders={"Категория": [SENTIMENT_RU[c] for c in SENTIMENT_ORDER]},
        )
        fig.update_layout(xaxis=dict(dtick=1), height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Sentiment gap vs длина диалога")

        gap_rows = [{
            "ID": d["dialogue_id"], "Тема": d["topic"],
            "Sentiment gap": d["dynamics"]["sentiment_gap"],
            "Реплик": d["total_turns"],
        } for d in dialogues]
        gdf = pd.DataFrame(gap_rows)
        fig = px.scatter(
            gdf, x="Реплик", y="Sentiment gap", text="ID", hover_data=["Тема"],
        )
        fig.update_traces(textposition="top center", marker=dict(size=12))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# Таб 5 — Проводник диалогов
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    if turns_df is None or dialogues is None:
        no_data("turns.parquet / dialogues.json")
    else:
        st.subheader("Проводник диалогов")

        dlg_opts5 = {d["dialogue_id"]: f'{d["dialogue_id"]}: {d["topic"]}' for d in dialogues}
        sel5 = st.selectbox(
            "Выберите диалог", list(dlg_opts5.keys()),
            format_func=lambda x: dlg_opts5[x], key="expl_dlg",
        )

        dt5 = turns_df[turns_df["dialogue_id"] == sel5].sort_values("turn_index")

        # Индекс Gemini-аннотаций
        ann_map: dict[int, dict] = {}
        if annotations_df is not None and "activity_id" in annotations_df.columns:
            for _, r in annotations_df[annotations_df["activity_id"] == sel5].iterrows():
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

# ── Футер ──────────────────────────────────────────────────────────────────
st.divider()
st.caption("Sentiment: BERT (multilingual) · Эмоции: Gemini 2.5 Flash · Streamlit + Plotly")
