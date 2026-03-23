import re
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="Сложность текста диалогов",
    page_icon="📐",
    layout="wide",
)

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "complexity.csv"

# ─── Палитры ──────────────────────────────────────────────────────────────────

COMPLEXITY_COLORS = {
    "Низкая (1–3)": "#22C55E",
    "Средняя (4–7)": "#F59E0B",
    "Высокая (8–10)": "#EF4444",
}
ROLE_COLORS = {"tutor": "#8B5CF6", "student": "#F59E0B"}
ROLE_RU = {"tutor": "Тьютор", "student": "Ученик"}

GRADE_PALETTE = px.colors.qualitative.Pastel

# ─── Бейджи и парсинг диалога ─────────────────────────────────────────────────

_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[.\d]*Z\s*")
_MSG_RE = re.compile(r"^(user|bot|User|Bot):\s*", re.IGNORECASE)


def asl_badge(asl: float | None) -> str:
    if asl is None:
        return ""
    if asl < 8:
        bg, label = "#22C55E", "Кор."
    elif asl < 15:
        bg, label = "#F59E0B", "Ср."
    else:
        bg, label = "#EF4444", "Дл."
    return (
        f'<span style="background:{bg};color:#fff;padding:1px 7px;'
        f'border-radius:8px;font-size:0.78em;margin-left:4px">'
        f'ASL {asl:.1f} ({label})</span>'
    )


def ttr_badge(ttr: float | None) -> str:
    if ttr is None:
        return ""
    if ttr >= 0.7:
        bg = "#22C55E"
    elif ttr >= 0.4:
        bg = "#F59E0B"
    else:
        bg = "#EF4444"
    return (
        f'<span style="background:{bg};color:#fff;padding:1px 7px;'
        f'border-radius:8px;font-size:0.78em;margin-left:4px">'
        f'TTR {ttr:.2f}</span>'
    )


def parse_dialog_messages(
    dialog_text: str,
    tutor_asl_raw: str,
    tutor_ttr_raw: str,
    student_asl_raw: str,
    student_ttr_raw: str,
) -> list[dict]:
    """Разбивает текст диалога на реплики, прикрепляет метрики ASL/TTR."""
    tutor_asl = dict(parse_reply_metrics(tutor_asl_raw))
    tutor_ttr = dict(parse_reply_metrics(tutor_ttr_raw))
    student_asl = dict(parse_reply_metrics(student_asl_raw))
    student_ttr = dict(parse_reply_metrics(student_ttr_raw))

    # Убираем временны́е метки, затем разбиваем по переносам строк
    clean = _TIMESTAMP_RE.sub("", dialog_text).strip()
    lines = [l.strip() for l in re.split(r"[\r\n]+", clean) if l.strip()]

    messages: list[dict] = []
    tutor_idx = 0
    student_idx = 0

    for line in lines:
        m = _MSG_RE.match(line)
        if not m:
            # продолжение предыдущей реплики
            if messages:
                messages[-1]["text"] += "\n" + line
            continue
        role_raw = m.group(1).lower()
        text = line[m.end():].strip()
        if not text:
            continue

        if role_raw == "bot":
            tutor_idx += 1
            messages.append({
                "role": "tutor",
                "reply_num": tutor_idx,
                "text": text,
                "asl": tutor_asl.get(tutor_idx),
                "ttr": tutor_ttr.get(tutor_idx),
            })
        else:
            student_idx += 1
            messages.append({
                "role": "student",
                "reply_num": student_idx,
                "text": text,
                "asl": student_asl.get(student_idx),
                "ttr": student_ttr.get(student_idx),
            })

    return messages


# ─── Парсинг метрик ───────────────────────────────────────────────────────────


def parse_reply_metrics(text: str) -> list[tuple[int, float]]:
    if not text or pd.isna(text):
        return []
    results = []
    for line in re.split(r"[\r\n]+", str(text).strip()):
        line = line.strip().strip("[]")
        if not line or line == "|":
            continue
        m = re.match(r"^(\d+)-(\d+(?:[.,]\d+)?)$", line)
        if m:
            try:
                results.append((int(m.group(1)), float(m.group(2).replace(",", "."))))
            except ValueError:
                continue
    return results


def parse_complexity_score(text: str) -> tuple[float | None, str]:
    if not text or pd.isna(text):
        return None, ""
    text = str(text).strip()
    m = re.match(r"^(\d+(?:[.,]\d+)?)\s*(?:\((.+)\))?", text, re.DOTALL)
    if m:
        score = float(m.group(1).replace(",", "."))
        justification = re.sub(r"^Краткое обоснование:\s*", "", m.group(2) or "").strip()
        return score, justification
    return None, ""


def extract_grade(dialog_text: str) -> int | None:
    if not dialog_text:
        return None
    m = re.search(r"user:\s*.+?\((\d+)\s*класс\)", dialog_text)
    if m:
        return int(m.group(1))
    m = re.search(r"\((\d+)(?:-\d+)?\s*класс\)", dialog_text)
    return int(m.group(1)) if m else None


def extract_topic(dialog_text: str) -> str:
    if not dialog_text:
        return "Неизвестно"
    m = re.search(r'Начинаем изучение темы\s*.([^»"]+)[»"]', dialog_text)
    if m:
        return m.group(1).strip()
    m = re.search(r"user:\s*(.+?\(\d+\s*класс\))", dialog_text)
    if m:
        return m.group(1).strip()
    return "Неизвестно"


def complexity_band(score: float | None) -> str:
    if score is None:
        return "Неизвестно"
    if score <= 3:
        return "Низкая (1–3)"
    if score <= 7:
        return "Средняя (4–7)"
    return "Высокая (8–10)"


# ─── Загрузка ─────────────────────────────────────────────────────────────────


def _detect_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Авто-определение формата CSV по количеству и содержимому колонок.

    Новый формат (6 колонок): dialog | tutor_asl | tutor_ttr | complexity | student_asl | student_ttr
    Старый формат (7+ колонок): dialog | old_result | tutor_asl | tutor_ttr | complexity | student_asl | student_ttr
    """
    ncols = len(df.columns)
    # Ищем колонку с оценкой сложности: первая колонка (кроме dialog),
    # в которой значения начинаются с цифры, за которой идёт пробел или скобка
    def looks_like_complexity(series: pd.Series) -> bool:
        sample = series.dropna().astype(str).head(10)
        return sample.str.match(r"^\d+\s*[\(\[]").any()

    if ncols >= 7:
        # Старый формат: пропускаем col1 (old_result)
        df = df.iloc[:, :7].copy()
        df.columns = ["dialog", "old_result", "tutor_asl_raw", "tutor_ttr_raw",
                      "complexity_raw", "student_asl_raw", "student_ttr_raw"]
    else:
        # Новый формат: 6 колонок без old_result
        df = df.iloc[:, :6].copy()
        df.columns = ["dialog", "tutor_asl_raw", "tutor_ttr_raw",
                      "complexity_raw", "student_asl_raw", "student_ttr_raw"]
    return df


@st.cache_data(show_spinner=False)
def load_and_parse(csv_bytes: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(io.BytesIO(csv_bytes), header=0, on_bad_lines="skip")
    raw = raw.iloc[:, :7]  # берём первые 7 колонок максимум
    raw = _detect_and_rename(raw)
    raw = raw[raw["dialog"].notna() & (raw["dialog"].str.strip() != "")].reset_index(drop=True)

    dialog_rows, reply_rows = [], []

    for idx, row in raw.iterrows():
        dialog_text = str(row["dialog"])
        grade = extract_grade(dialog_text)
        topic = extract_topic(dialog_text)
        score, justification = parse_complexity_score(row["complexity_raw"])

        tutor_asl = parse_reply_metrics(row["tutor_asl_raw"])
        tutor_ttr = parse_reply_metrics(row["tutor_ttr_raw"])
        student_asl = parse_reply_metrics(row["student_asl_raw"])
        student_ttr = parse_reply_metrics(row["student_ttr_raw"])

        t_asl_v = [v for _, v in tutor_asl]
        t_ttr_v = [v for _, v in tutor_ttr]
        s_asl_v = [v for _, v in student_asl]
        s_ttr_v = [v for _, v in student_ttr]

        dialog_rows.append({
            "dialog_id": idx,
            "topic": topic,
            "grade": grade,
            "grade_label": f"{grade} кл." if grade else "—",
            "complexity_score": score,
            "complexity_band": complexity_band(score),
            "complexity_justification": justification,
            "dialog_text": dialog_text,
            "tutor_asl_raw": str(row["tutor_asl_raw"]) if pd.notna(row["tutor_asl_raw"]) else "",
            "tutor_ttr_raw": str(row["tutor_ttr_raw"]) if pd.notna(row["tutor_ttr_raw"]) else "",
            "student_asl_raw": str(row["student_asl_raw"]) if pd.notna(row["student_asl_raw"]) else "",
            "student_ttr_raw": str(row["student_ttr_raw"]) if pd.notna(row["student_ttr_raw"]) else "",
            "tutor_asl_mean": np.mean(t_asl_v) if t_asl_v else None,
            "tutor_asl_max": np.max(t_asl_v) if t_asl_v else None,
            "tutor_ttr_mean": np.mean(t_ttr_v) if t_ttr_v else None,
            "student_asl_mean": np.mean(s_asl_v) if s_asl_v else None,
            "student_ttr_mean": np.mean(s_ttr_v) if s_ttr_v else None,
            "num_tutor_replies": len(t_asl_v),
            "num_student_replies": len(s_asl_v),
        })

        asl_d, ttr_d = dict(tutor_asl), dict(tutor_ttr)
        for rn in sorted(set(asl_d) | set(ttr_d)):
            reply_rows.append({"dialog_id": idx, "topic": topic, "grade": grade,
                                "grade_label": f"{grade} кл." if grade else "—",
                                "role": "tutor", "reply_num": rn,
                                "asl": asl_d.get(rn), "ttr": ttr_d.get(rn)})

        asl_s, ttr_s = dict(student_asl), dict(student_ttr)
        for rn in sorted(set(asl_s) | set(ttr_s)):
            reply_rows.append({"dialog_id": idx, "topic": topic, "grade": grade,
                                "grade_label": f"{grade} кл." if grade else "—",
                                "role": "student", "reply_num": rn,
                                "asl": asl_s.get(rn), "ttr": ttr_s.get(rn)})

    return pd.DataFrame(dialog_rows), pd.DataFrame(reply_rows)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Данные")
    uploaded = st.file_uploader(
        "Загрузить новый CSV",
        type=["csv"],
        help="Формат: тот же, что и 'Классификация - ВТОРЫЕ РЕЗУЛЬТАТЫ.csv'",
    )
    if uploaded is not None:
        csv_bytes = uploaded.read()
        st.success(f"Загружен: {uploaded.name}")
    elif DEFAULT_CSV.exists():
        with open(DEFAULT_CSV, "rb") as f:
            csv_bytes = f.read()
        st.info(f"Встроенный датасет ({DEFAULT_CSV.name})")
    else:
        st.error("Файл данных не найден. Загрузите CSV выше.")
        st.stop()

with st.spinner("Парсинг данных…"):
    df_dialog_full, df_replies_full = load_and_parse(csv_bytes)

with st.sidebar:
    st.divider()
    st.subheader("Фильтры")

    all_grades = sorted(df_dialog_full["grade"].dropna().unique().astype(int).tolist())
    grade_options = [f"{g}" for g in all_grades]
    selected_grade_strs = st.multiselect(
        "Класс",
        options=grade_options,
        default=grade_options,
        help="Оставьте пустым — выбраны все",
    )
    selected_grades = [int(s) for s in selected_grade_strs] if selected_grade_strs else all_grades

    st.divider()
    st.caption(
        "**ASL** — средняя длина предложения (слов)\n\n"
        "**TTR** — лексическое разнообразие (0–1)\n\n"
        "**Сложность** — оценка по матрице 1–10"
    )

# ─── Применяем фильтры ────────────────────────────────────────────────────────

df_dialog = df_dialog_full[df_dialog_full["grade"].isin(selected_grades)].copy()
df_replies = df_replies_full[df_replies_full["grade"].isin(selected_grades)].copy()

# ─── Заголовок + KPI ──────────────────────────────────────────────────────────

st.title("📐 Сложность текста диалогов")
st.caption("Анализ метрик ASL и TTR по репликам тьютора и ученика")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Диалогов", len(df_dialog))
c2.metric(
    "Средняя сложность",
    f"{df_dialog['complexity_score'].mean():.1f}" if df_dialog["complexity_score"].notna().any() else "—",
    help="Оценка по матрице 1–10",
)
c3.metric(
    "ASL тьютора (ср.)",
    f"{df_dialog['tutor_asl_mean'].mean():.1f}" if df_dialog["tutor_asl_mean"].notna().any() else "—",
)
c4.metric(
    "TTR тьютора (ср.)",
    f"{df_dialog['tutor_ttr_mean'].mean():.2f}" if df_dialog["tutor_ttr_mean"].notna().any() else "—",
)
c5.metric(
    "ASL ученика (ср.)",
    f"{df_dialog['student_asl_mean'].mean():.1f}" if df_dialog["student_asl_mean"].notna().any() else "—",
)
c6.metric(
    "TTR ученика (ср.)",
    f"{df_dialog['student_ttr_mean'].mean():.2f}" if df_dialog["student_ttr_mean"].notna().any() else "—",
)

st.divider()

# ─── Вкладки ──────────────────────────────────────────────────────────────────

tab_agg, tab_overview, tab_tutor, tab_student, tab_compare, tab_detail, tab_help = st.tabs([
    "📈 По классам", "📊 Обзор", "🎓 Тьютор", "📚 Ученик", "⚖️ Сравнение", "🔍 Диалог", "📖 Справка",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 0 — ПО КЛАССАМ (aggregate)
# ════════════════════════════════════════════════════════════════════════════
with tab_agg:
    if df_dialog.empty:
        st.warning("Нет данных для выбранных классов.")
    else:
        # ── Агрегат по классам (уровень диалога) ────────────────────────────
        agg = (
            df_dialog.groupby("grade")
            .agg(
                n=("dialog_id", "count"),
                complexity_mean=("complexity_score", "mean"),
                complexity_std=("complexity_score", "std"),
                tutor_asl_mean=("tutor_asl_mean", "mean"),
                tutor_ttr_mean=("tutor_ttr_mean", "mean"),
                student_asl_mean=("student_asl_mean", "mean"),
                student_ttr_mean=("student_ttr_mean", "mean"),
            )
            .reset_index()
        )
        agg["grade_label"] = agg["grade"].apply(lambda g: f"{g} кл.")

        # ── Строка 1: Сложность + Диалоги ───────────────────────────────────
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Средняя сложность тьютора по классам")
            fig = px.bar(
                agg,
                x="grade_label",
                y="complexity_mean",
                error_y="complexity_std",
                color="grade_label",
                color_discrete_sequence=GRADE_PALETTE,
                text=agg["complexity_mean"].round(1),
                labels={"grade_label": "Класс", "complexity_mean": "Ср. сложность"},
                category_orders={"grade_label": agg.sort_values("grade")["grade_label"].tolist()},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                yaxis_range=[0, 10.5],
                showlegend=False,
                height=360,
            )
            # Референсные зоны
            fig.add_hrect(y0=0, y1=3, fillcolor="#22C55E", opacity=0.07, line_width=0,
                          annotation_text="Низкая", annotation_position="left")
            fig.add_hrect(y0=3, y1=7, fillcolor="#F59E0B", opacity=0.07, line_width=0,
                          annotation_text="Средняя", annotation_position="left")
            fig.add_hrect(y0=7, y1=10, fillcolor="#EF4444", opacity=0.07, line_width=0,
                          annotation_text="Высокая", annotation_position="left")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("Количество диалогов по классам")
            fig = px.bar(
                agg,
                x="grade_label",
                y="n",
                color="grade_label",
                color_discrete_sequence=GRADE_PALETTE,
                text="n",
                labels={"grade_label": "Класс", "n": "Диалогов"},
                category_orders={"grade_label": agg.sort_values("grade")["grade_label"].tolist()},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, height=360)
            st.plotly_chart(fig, use_container_width=True)

        # ── Строка 2: ASL тьютора и ученика по классам ──────────────────────
        st.subheader("Средняя ASL по классам: тьютор vs ученик")
        agg_melt_asl = agg.melt(
            id_vars="grade_label",
            value_vars=["tutor_asl_mean", "student_asl_mean"],
            var_name="role",
            value_name="asl",
        )
        agg_melt_asl["role"] = agg_melt_asl["role"].map(
            {"tutor_asl_mean": "Тьютор", "student_asl_mean": "Ученик"}
        )
        fig = px.bar(
            agg_melt_asl,
            x="grade_label",
            y="asl",
            color="role",
            barmode="group",
            color_discrete_map={"Тьютор": "#8B5CF6", "Ученик": "#F59E0B"},
            text=agg_melt_asl["asl"].round(1),
            labels={"grade_label": "Класс", "asl": "Средняя ASL (слов/пред.)", "role": "Роль"},
            category_orders={
                "grade_label": agg.sort_values("grade")["grade_label"].tolist(),
                "role": ["Тьютор", "Ученик"],
            },
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

        # ── Строка 3: TTR по классам ─────────────────────────────────────────
        st.subheader("Средняя TTR по классам: тьютор vs ученик")
        agg_melt_ttr = agg.melt(
            id_vars="grade_label",
            value_vars=["tutor_ttr_mean", "student_ttr_mean"],
            var_name="role",
            value_name="ttr",
        )
        agg_melt_ttr["role"] = agg_melt_ttr["role"].map(
            {"tutor_ttr_mean": "Тьютор", "student_ttr_mean": "Ученик"}
        )
        fig = px.bar(
            agg_melt_ttr,
            x="grade_label",
            y="ttr",
            color="role",
            barmode="group",
            color_discrete_map={"Тьютор": "#8B5CF6", "Ученик": "#F59E0B"},
            text=agg_melt_ttr["ttr"].round(2),
            labels={"grade_label": "Класс", "ttr": "Средняя TTR (0–1)", "role": "Роль"},
            category_orders={
                "grade_label": agg.sort_values("grade")["grade_label"].tolist(),
                "role": ["Тьютор", "Ученик"],
            },
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_range=[0, 1.15], height=360)
        st.plotly_chart(fig, use_container_width=True)

        # ── Строка 4: Box ASL/TTR тьютора по классам (из реплик) ────────────
        tutor_rep = df_replies[df_replies["role"] == "tutor"].copy()
        if not tutor_rep.empty:
            col_l2, col_r2 = st.columns(2)
            with col_l2:
                st.subheader("Разброс ASL тьютора по классам")
                asl_box = tutor_rep.dropna(subset=["asl"])
                if not asl_box.empty:
                    fig = px.box(
                        asl_box.sort_values("grade"),
                        x="grade_label",
                        y="asl",
                        color="grade_label",
                        color_discrete_sequence=GRADE_PALETTE,
                        points="all",
                        hover_data=["topic"],
                        labels={"grade_label": "Класс", "asl": "ASL"},
                    )
                    fig.update_layout(showlegend=False, height=360)
                    st.plotly_chart(fig, use_container_width=True)

            with col_r2:
                st.subheader("Разброс TTR тьютора по классам")
                ttr_box = tutor_rep.dropna(subset=["ttr"])
                if not ttr_box.empty:
                    fig = px.box(
                        ttr_box.sort_values("grade"),
                        x="grade_label",
                        y="ttr",
                        color="grade_label",
                        color_discrete_sequence=GRADE_PALETTE,
                        points="all",
                        hover_data=["topic"],
                        labels={"grade_label": "Класс", "ttr": "TTR"},
                    )
                    fig.update_layout(showlegend=False, yaxis_range=[0, 1.05], height=360)
                    st.plotly_chart(fig, use_container_width=True)

        # ── Строка 5: Scatter сложность vs ASL тьютора по классам ───────────
        st.subheader("Сложность vs ASL тьютора (каждая точка — диалог)")
        sc = df_dialog.dropna(subset=["tutor_asl_mean", "complexity_score"]).copy()
        sc["num_tutor_replies"] = sc["num_tutor_replies"].fillna(1).clip(lower=1)
        if not sc.empty:
            fig = px.scatter(
                sc.sort_values("grade"),
                x="tutor_asl_mean",
                y="complexity_score",
                color="grade_label",
                color_discrete_sequence=GRADE_PALETTE,
                size="num_tutor_replies",
                size_max=20,
                hover_data={"topic": True, "tutor_ttr_mean": ":.2f", "num_tutor_replies": True},
                labels={
                    "tutor_asl_mean": "Средняя ASL тьютора",
                    "complexity_score": "Оценка сложности",
                    "grade_label": "Класс",
                    "num_tutor_replies": "Реплик тьютора",
                },
            )
            fig.update_layout(height=400, yaxis_range=[0, 10.5])
            fig.add_hrect(y0=0, y1=3, fillcolor="#22C55E", opacity=0.06, line_width=0)
            fig.add_hrect(y0=3, y1=7, fillcolor="#F59E0B", opacity=0.06, line_width=0)
            fig.add_hrect(y0=7, y1=10, fillcolor="#EF4444", opacity=0.06, line_width=0)
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — ОБЗОР
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:
    if df_dialog.empty:
        st.warning("Нет данных для выбранных классов.")
    else:
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Распределение оценок сложности")
            score_counts = (
                df_dialog["complexity_score"].dropna()
                .value_counts().sort_index().reset_index()
            )
            score_counts.columns = ["score", "count"]
            score_counts["band"] = score_counts["score"].apply(complexity_band)
            fig = px.bar(
                score_counts,
                x="score", y="count",
                color="band",
                color_discrete_map=COMPLEXITY_COLORS,
                text="count",
                labels={"score": "Оценка", "count": "Диалогов", "band": "Уровень"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                xaxis=dict(tickmode="linear", dtick=1),
                legend_title_text="Уровень",
                height=340,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("ASL vs TTR тьютора (по диалогам)")
            sc = df_dialog.dropna(subset=["tutor_asl_mean", "tutor_ttr_mean"]).copy()
            sc["_size"] = sc["complexity_score"].fillna(sc["complexity_score"].median()).clip(lower=1)
            if not sc.empty:
                fig = px.scatter(
                    sc.sort_values("grade"),
                    x="tutor_asl_mean",
                    y="tutor_ttr_mean",
                    color="grade_label",
                    color_discrete_sequence=GRADE_PALETTE,
                    size="_size",
                    size_max=18,
                    hover_data={"topic": True, "grade": True, "complexity_score": True, "_size": False},
                    labels={
                        "tutor_asl_mean": "ASL тьютора",
                        "tutor_ttr_mean": "TTR тьютора",
                        "grade_label": "Класс",
                        "complexity_score": "Сложность",
                    },
                )
                fig.update_layout(height=340, yaxis_range=[0, 1.05])
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Сводная таблица диалогов")
        disp = df_dialog[[
            "topic", "grade", "complexity_score", "complexity_band",
            "tutor_asl_mean", "tutor_ttr_mean",
            "student_asl_mean", "student_ttr_mean",
            "num_tutor_replies", "num_student_replies",
        ]].rename(columns={
            "topic": "Тема", "grade": "Класс",
            "complexity_score": "Сложность", "complexity_band": "Уровень",
            "tutor_asl_mean": "ASL тьютора", "tutor_ttr_mean": "TTR тьютора",
            "student_asl_mean": "ASL ученика", "student_ttr_mean": "TTR ученика",
            "num_tutor_replies": "Реплик тьютора", "num_student_replies": "Реплик ученика",
        })
        for col in ["ASL тьютора", "TTR тьютора", "ASL ученика", "TTR ученика"]:
            disp[col] = disp[col].round(2)
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — ТЬЮТОР
# ════════════════════════════════════════════════════════════════════════════
with tab_tutor:
    tutor_df = df_replies[df_replies["role"] == "tutor"].copy()

    if tutor_df.empty:
        st.warning("Нет данных о репликах тьютора.")
    else:
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("ASL по номеру реплики")
            asl_df = tutor_df.dropna(subset=["asl"])
            if not asl_df.empty:
                fig = px.box(
                    asl_df, x="reply_num", y="asl", points="all",
                    color="grade_label", color_discrete_sequence=GRADE_PALETTE,
                    hover_data=["topic"],
                    labels={"reply_num": "№ реплики тьютора", "asl": "ASL", "grade_label": "Класс"},
                )
                fig.update_layout(height=360)
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("TTR по номеру реплики")
            ttr_df = tutor_df.dropna(subset=["ttr"])
            if not ttr_df.empty:
                fig = px.box(
                    ttr_df, x="reply_num", y="ttr", points="all",
                    color="grade_label", color_discrete_sequence=GRADE_PALETTE,
                    hover_data=["topic"],
                    labels={"reply_num": "№ реплики тьютора", "ttr": "TTR", "grade_label": "Класс"},
                )
                fig.update_layout(height=360, yaxis_range=[0, 1.05])
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Прогрессия ASL тьютора по номеру реплики (среднее ± std по классу)")
        asl_prog = tutor_df.dropna(subset=["asl", "grade_label"])
        if not asl_prog.empty:
            asl_agg = (
                asl_prog.groupby(["grade_label", "reply_num"])["asl"]
                .agg(mean="mean", std="std", n="count").reset_index()
            )
            asl_agg["std"] = asl_agg["std"].fillna(0)
            fig = go.Figure()
            grades_sorted = sorted(asl_agg["grade_label"].unique(),
                                   key=lambda x: int(x.split()[0]) if x.split()[0].isdigit() else 99)
            palette = GRADE_PALETTE
            for i, gl in enumerate(grades_sorted):
                d = asl_agg[asl_agg["grade_label"] == gl].sort_values("reply_num")
                color = palette[i % len(palette)]
                fig.add_trace(go.Scatter(
                    x=d["reply_num"], y=d["mean"] + d["std"],
                    mode="lines", line=dict(width=0), showlegend=False,
                    hoverinfo="skip", fillcolor=color.replace("rgb", "rgba").replace(")", ",0.15)") if color.startswith("rgb") else color,
                ))
                fig.add_trace(go.Scatter(
                    x=d["reply_num"], y=d["mean"] - d["std"],
                    mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=color.replace("rgb", "rgba").replace(")", ",0.15)") if color.startswith("rgb") else color,
                    showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=d["reply_num"], y=d["mean"], mode="lines+markers",
                    name=gl, line=dict(color=color, width=2),
                    customdata=d[["n"]].values,
                    hovertemplate="Реплика %{x}<br>ASL: %{y:.1f}<br>Диалогов: %{customdata[0]}<extra>" + gl + "</extra>",
                ))
            fig.update_layout(
                height=380,
                xaxis_title="№ реплики тьютора",
                yaxis_title="Средняя ASL",
                legend_title="Класс",
                xaxis=dict(dtick=1),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Прогрессия TTR тьютора по номеру реплики (среднее ± std по классу)")
        ttr_prog = tutor_df.dropna(subset=["ttr", "grade_label"])
        if not ttr_prog.empty:
            ttr_agg = (
                ttr_prog.groupby(["grade_label", "reply_num"])["ttr"]
                .agg(mean="mean", std="std", n="count").reset_index()
            )
            ttr_agg["std"] = ttr_agg["std"].fillna(0)
            fig = go.Figure()
            grades_sorted = sorted(ttr_agg["grade_label"].unique(),
                                   key=lambda x: int(x.split()[0]) if x.split()[0].isdigit() else 99)
            for i, gl in enumerate(grades_sorted):
                d = ttr_agg[ttr_agg["grade_label"] == gl].sort_values("reply_num")
                color = palette[i % len(palette)]
                fig.add_trace(go.Scatter(
                    x=d["reply_num"], y=(d["mean"] + d["std"]).clip(upper=1),
                    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=d["reply_num"], y=(d["mean"] - d["std"]).clip(lower=0),
                    mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=color.replace("rgb", "rgba").replace(")", ",0.15)") if color.startswith("rgb") else color,
                    showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=d["reply_num"], y=d["mean"], mode="lines+markers",
                    name=gl, line=dict(color=color, width=2),
                    customdata=d[["n"]].values,
                    hovertemplate="Реплика %{x}<br>TTR: %{y:.3f}<br>Диалогов: %{customdata[0]}<extra>" + gl + "</extra>",
                ))
            fig.update_layout(
                height=380,
                xaxis_title="№ реплики тьютора",
                yaxis_title="Средняя TTR",
                yaxis_range=[0, 1.05],
                legend_title="Класс",
                xaxis=dict(dtick=1),
            )
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — УЧЕНИК
# ════════════════════════════════════════════════════════════════════════════
with tab_student:
    student_df = df_replies[df_replies["role"] == "student"].copy()

    if student_df.empty:
        st.warning("Нет данных о репликах ученика.")
    else:
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("ASL ученика: распределение")
            asl_hist = student_df.dropna(subset=["asl"])
            if not asl_hist.empty:
                fig = px.histogram(
                    asl_hist, x="asl",
                    color="grade_label", barmode="overlay",
                    color_discrete_sequence=GRADE_PALETTE,
                    nbins=15,
                    labels={"asl": "ASL", "grade_label": "Класс"},
                )
                fig.update_layout(height=320)
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("TTR ученика: распределение")
            ttr_hist = student_df.dropna(subset=["ttr"])
            if not ttr_hist.empty:
                fig = px.histogram(
                    ttr_hist, x="ttr",
                    color="grade_label", barmode="overlay",
                    color_discrete_sequence=GRADE_PALETTE,
                    nbins=15,
                    labels={"ttr": "TTR", "grade_label": "Класс"},
                )
                fig.update_layout(height=320, xaxis_range=[0, 1.05])
                st.plotly_chart(fig, use_container_width=True)

        col_l2, col_r2 = st.columns(2)

        with col_l2:
            st.subheader("ASL по номеру реплики")
            asl_df = student_df.dropna(subset=["asl"])
            if not asl_df.empty:
                fig = px.box(
                    asl_df, x="reply_num", y="asl", points="all",
                    color="grade_label", color_discrete_sequence=GRADE_PALETTE,
                    hover_data=["topic"],
                    labels={"reply_num": "№ реплики ученика", "asl": "ASL", "grade_label": "Класс"},
                )
                fig.update_layout(height=340)
                st.plotly_chart(fig, use_container_width=True)

        with col_r2:
            st.subheader("TTR по номеру реплики")
            ttr_df = student_df.dropna(subset=["ttr"])
            if not ttr_df.empty:
                fig = px.box(
                    ttr_df, x="reply_num", y="ttr", points="all",
                    color="grade_label", color_discrete_sequence=GRADE_PALETTE,
                    hover_data=["topic"],
                    labels={"reply_num": "№ реплики ученика", "ttr": "TTR", "grade_label": "Класс"},
                )
                fig.update_layout(height=340, yaxis_range=[0, 1.05])
                st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — СРАВНЕНИЕ
# ════════════════════════════════════════════════════════════════════════════
with tab_compare:
    if df_dialog.empty:
        st.warning("Нет данных для выбранных классов.")
    else:
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Средняя ASL по классам: тьютор vs ученик")
            comp_asl = df_dialog[["grade", "grade_label", "tutor_asl_mean", "student_asl_mean"]].dropna()
            if not comp_asl.empty:
                asl_by_grade = (
                    comp_asl.groupby("grade_label")
                    .agg(tutor=("tutor_asl_mean", "mean"), student=("student_asl_mean", "mean"))
                    .reset_index()
                    .merge(comp_asl[["grade", "grade_label"]].drop_duplicates(), on="grade_label")
                    .sort_values("grade")
                )
                grade_order = asl_by_grade["grade_label"].tolist()
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Тьютор", x=asl_by_grade["grade_label"], y=asl_by_grade["tutor"],
                    marker_color="#8B5CF6",
                    text=asl_by_grade["tutor"].round(1), textposition="outside",
                ))
                fig.add_trace(go.Bar(
                    name="Ученик", x=asl_by_grade["grade_label"], y=asl_by_grade["student"],
                    marker_color="#F59E0B",
                    text=asl_by_grade["student"].round(1), textposition="outside",
                ))
                fig.update_layout(
                    barmode="group",
                    xaxis=dict(title="Класс", categoryorder="array", categoryarray=grade_order),
                    yaxis_title="Средняя ASL",
                    legend_title="Роль", height=380,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("Средняя TTR по классам: тьютор vs ученик")
            comp_ttr = df_dialog[["grade", "grade_label", "tutor_ttr_mean", "student_ttr_mean"]].dropna()
            if not comp_ttr.empty:
                ttr_by_grade = (
                    comp_ttr.groupby("grade_label")
                    .agg(tutor=("tutor_ttr_mean", "mean"), student=("student_ttr_mean", "mean"))
                    .reset_index()
                    .merge(comp_ttr[["grade", "grade_label"]].drop_duplicates(), on="grade_label")
                    .sort_values("grade")
                )
                grade_order = ttr_by_grade["grade_label"].tolist()
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Тьютор", x=ttr_by_grade["grade_label"], y=ttr_by_grade["tutor"],
                    marker_color="#8B5CF6",
                    text=ttr_by_grade["tutor"].round(2), textposition="outside",
                ))
                fig.add_trace(go.Bar(
                    name="Ученик", x=ttr_by_grade["grade_label"], y=ttr_by_grade["student"],
                    marker_color="#F59E0B",
                    text=ttr_by_grade["student"].round(2), textposition="outside",
                ))
                fig.update_layout(
                    barmode="group",
                    xaxis=dict(title="Класс", categoryorder="array", categoryarray=grade_order),
                    yaxis_title="Средняя TTR",
                    yaxis_range=[0, 1.15], legend_title="Роль", height=380,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Соотношение ASL тьютора и ученика")
        sc = df_dialog.dropna(subset=["tutor_asl_mean", "student_asl_mean"]).copy()
        sc["_size"] = sc["complexity_score"].fillna(sc["complexity_score"].median()).clip(lower=1)
        if not sc.empty:
            max_val = max(sc["tutor_asl_mean"].max(), sc["student_asl_mean"].max()) * 1.15
            fig = px.scatter(
                sc.sort_values("grade"),
                x="student_asl_mean", y="tutor_asl_mean",
                color="grade_label", color_discrete_sequence=GRADE_PALETTE,
                size="_size", size_max=18,
                hover_data={"topic": True, "complexity_score": True, "_size": False},
                labels={
                    "student_asl_mean": "ASL ученика",
                    "tutor_asl_mean": "ASL тьютора",
                    "grade_label": "Класс",
                    "complexity_score": "Сложность",
                },
            )
            fig.add_shape(
                type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                line=dict(dash="dot", color="gray", width=1),
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Диагональ — паритет. Точки выше → тьютор пишет длиннее ученика.")

        st.subheader("Дельта ASL (тьютор − ученик) по классам")
        delta_df = df_dialog.copy()
        delta_df["asl_delta"] = delta_df["tutor_asl_mean"] - delta_df["student_asl_mean"]
        delta_df["ttr_delta"] = delta_df["tutor_ttr_mean"] - delta_df["student_ttr_mean"]
        delta_clean = delta_df.dropna(subset=["asl_delta", "grade_label"]).sort_values("grade")
        if not delta_clean.empty:
            grade_order = delta_clean.drop_duplicates("grade_label").sort_values("grade")["grade_label"].tolist()
            fig = px.box(
                delta_clean,
                x="grade_label", y="asl_delta",
                color="grade_label", color_discrete_sequence=GRADE_PALETTE,
                points="outliers",
                hover_data=["topic"],
                labels={"grade_label": "Класс", "asl_delta": "ASL тьютора − ASL ученика"},
                category_orders={"grade_label": grade_order},
            )
            fig.add_hline(y=0, line_dash="dot", line_color="gray",
                          annotation_text="паритет", annotation_position="right")
            fig.update_layout(showlegend=False, height=360)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Выше нуля → тьютор пишет длиннее ученика; ниже нуля → наоборот.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — ДЕТАЛИ ДИАЛОГА
# ════════════════════════════════════════════════════════════════════════════
with tab_detail:
    if df_dialog.empty:
        st.warning("Нет данных для выбранных классов.")
    else:
        topics = df_dialog["topic"].tolist()
        selected_topic = st.selectbox("Выберите диалог", topics)
        sel = df_dialog[df_dialog["topic"] == selected_topic].iloc[0]
        dialog_id = sel["dialog_id"]

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Класс", f"{sel['grade']} кл." if sel["grade"] else "—")
        col_m2.metric("Сложность", f"{sel['complexity_score']:.0f}/10" if pd.notna(sel["complexity_score"]) else "—")
        col_m3.metric("Уровень", sel["complexity_band"])
        col_m4.metric("ASL тьютора", f"{sel['tutor_asl_mean']:.1f}" if pd.notna(sel["tutor_asl_mean"]) else "—")
        col_m5.metric("TTR тьютора", f"{sel['tutor_ttr_mean']:.2f}" if pd.notna(sel["tutor_ttr_mean"]) else "—")

        if sel["complexity_justification"]:
            st.info(f"**Обоснование оценки:** {sel['complexity_justification']}")

        dial_replies = df_replies_full[df_replies_full["dialog_id"] == dialog_id].copy()

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("ASL по репликам")
            asl_d = dial_replies.dropna(subset=["asl"])
            if not asl_d.empty:
                fig = px.line(
                    asl_d, x="reply_num", y="asl",
                    color="role", markers=True,
                    color_discrete_map=ROLE_COLORS,
                    labels={"reply_num": "№ реплики", "asl": "ASL", "role": "Роль"},
                    category_orders={"role": ["tutor", "student"]},
                )
                fig.for_each_trace(lambda t: t.update(name=ROLE_RU.get(t.name, t.name)))
                fig.update_layout(height=320, legend_title="Роль")
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("TTR по репликам")
            ttr_d = dial_replies.dropna(subset=["ttr"])
            if not ttr_d.empty:
                fig = px.line(
                    ttr_d, x="reply_num", y="ttr",
                    color="role", markers=True,
                    color_discrete_map=ROLE_COLORS,
                    labels={"reply_num": "№ реплики", "ttr": "TTR", "role": "Роль"},
                    category_orders={"role": ["tutor", "student"]},
                )
                fig.for_each_trace(lambda t: t.update(name=ROLE_RU.get(t.name, t.name)))
                fig.update_layout(height=320, yaxis_range=[0, 1.05], legend_title="Роль")
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Данные по репликам")
        reply_table = dial_replies[["role", "reply_num", "asl", "ttr"]].copy()
        reply_table["role"] = reply_table["role"].map(ROLE_RU)
        reply_table = reply_table.rename(
            columns={"role": "Роль", "reply_num": "№ реплики", "asl": "ASL", "ttr": "TTR"}
        )
        st.dataframe(reply_table.round(3), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("💬 Диалог с метриками")

        msgs = parse_dialog_messages(
            sel["dialog_text"],
            sel["tutor_asl_raw"],
            sel["tutor_ttr_raw"],
            sel["student_asl_raw"],
            sel["student_ttr_raw"],
        )

        if not msgs:
            st.info("Не удалось разобрать текст диалога.")
        else:
            for msg in msgs:
                role = msg["role"]
                avatar = "🤖" if role == "tutor" else "🧑‍🎓"
                with st.chat_message(role if role == "assistant" else ("assistant" if role == "tutor" else "user"),
                                     avatar=avatar):
                    badges = asl_badge(msg["asl"]) + ttr_badge(msg["ttr"])
                    rn_label = f'<span style="font-size:0.75em;color:#888">#{msg["reply_num"]}</span> '
                    st.markdown(
                        rn_label + msg["text"] + ("&nbsp;" + badges if badges else ""),
                        unsafe_allow_html=True,
                    )
                    if msg["asl"] is not None or msg["ttr"] is not None:
                        with st.expander("📊 Метрики реплики", expanded=False):
                            mc1, mc2 = st.columns(2)
                            mc1.metric(
                                "ASL",
                                f"{msg['asl']:.1f}" if msg["asl"] is not None else "—",
                                help="Средняя длина предложения (слов)",
                            )
                            mc2.metric(
                                "TTR",
                                f"{msg['ttr']:.3f}" if msg["ttr"] is not None else "—",
                                help="Лексическое разнообразие (0–1)",
                            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — СПРАВКА
# ════════════════════════════════════════════════════════════════════════════
with tab_help:
    st.header("📖 Справка по методологии")
    st.caption(
        "Анализ выполнялся языковой моделью **Gemini Flash** по строгому промпту. "
        "Ниже описаны правила расчёта метрик и матрица оценки сложности."
    )

    # ── 1. Как проводился замер ─────────────────────────────────────────────
    st.subheader("Как проводился замер")
    st.markdown("""
Каждый диалог (лог переписки ученика и бота-репетитора) передавался в LLM вместе с детальным промптом.
Модель выполняла три задачи:

1. **Очистка текста** — из каждой реплики удалялись временны́е метки
   (`2026-01-22T13:28:33Z`), имена спикеров (`user:`, `bot:`), Markdown-разметка
   (`**`, `-`, `*`, `$`) и пунктуация.
2. **Расчёт метрик** — для каждой реплики тьютора и ученика по отдельности
   вычислялись ASL и TTR (см. ниже).
3. **Оценка сложности** — на основе совокупности реплик тьютора и с учётом класса
   ученика модель выставляла оценку от 1 до 10 по матрице (см. ниже).
""")

    # ── 2. Термины и формулы ────────────────────────────────────────────────
    st.subheader("Метрики и термины")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
**ASL — Average Sentence Length (Средняя длина предложения, СДП)**

$$\\text{ASL} = \\frac{\\text{Общее количество слов}}{\\text{Общее количество предложений}}$$

Считается отдельно для каждой реплики.

**Что считается словом:** любая непрерывная последовательность букв или цифр.
Предлоги и союзы — отдельные слова.

**Что считается предложением:** последовательность слов, завершающаяся
`.`, `?` или `!` (или конец реплики, если знака нет).
Элементы списка считаются отдельными предложениями.
""")

    with col_r:
        st.markdown("""
**TTR — Type-Token Ratio (Лексическое разнообразие)**

$$\\text{TTR} = \\frac{\\text{Уникальные слова (леммы)}}{\\text{Общее количество слов}}$$

Значение от **0** до **1**:
- **~1.0** — почти каждое слово встречается впервые (высокое разнообразие)
- **~0.3–0.5** — умеренное повторение слов
- **< 0.3** — текст сильно повторяется (шаблонные фразы)

Перед подсчётом слова приводятся к **начальной форме (лемматизация)**.
""")

    st.info(
        "**Оценка сложности** выставляется **только для тьютора** и отражает, "
        "насколько язык репетитора соответствует уровню ученика. "
        "Метрики ASL и TTR для ученика фиксируются как дополнительный контекст."
    )

    # ── 3. Матрица сложности ────────────────────────────────────────────────
    st.subheader("Матрица оценки сложности тьютора")
    st.markdown(
        "Оценка выставляется **в совокупности по всем репликам тьютора** в диалоге. "
        "Класс ученика определяется из текста диалога и задаёт нужную строку матрицы. "
        "Действует принцип **«слабого звена»**: если ASL высокая — оценка идёт вверх "
        "даже при скромном наборе терминов."
    )

    matrix_data = {
        "Класс": [
            "5–6 кл.", "5–6 кл.", "5–6 кл.",
            "7–8 кл.", "7–8 кл.", "7–8 кл.",
            "9 кл.",   "9 кл.",   "9 кл.",
            "10–11 кл.", "10–11 кл.", "10–11 кл.",
        ],
        "Уровень": [
            "Низкая (1–3)", "Средняя (4–7)", "Высокая (8–10)",
            "Низкая (1–3)", "Средняя (4–7)", "Высокая (8–10)",
            "Низкая (1–3)", "Средняя (4–7)", "Высокая (8–10)",
            "Низкая (1–3)", "Средняя (4–7)", "Высокая (8–10)",
        ],
        "ASL (слов/пред.)": [
            "5–7", "8–15", "15+",
            "6–10", "12–18", "20+",
            "до 10", "15–20", "22+",
            "8–12", "18–25", "25+",
        ],
        "TTR": [
            "низкий", "средний", "высокий",
            "низкий", "средний", "высокий",
            "низкий", "средний", "высокий",
            "низкий", "средний", "высокий",
        ],
        "Кол-во терминов": [
            "0–1", "1–2", "3+",
            "1", "2–3", "4+",
            "1–2", "3", "4+",
            "1–2", "3–4", "5+",
        ],
        "Характер текста": [
            "Короткие прямые команды, бытовой контекст, нет абстракций",
            "Сюжетные задачи в 2–3 действия, конструкции «Если…, то…», правила с примерами",
            "Логические задачи, длинные условия без чисел, сухая теория без примеров",
            "Отработка формул, простая констатация («Дано…»)",
            "Стандартные теоремы, описания построений, связи «так как…, следовательно…»",
            "Доказательства от противного, параметры, вложенные конструкции",
            "Типовые задания 1 части ОГЭ, чёткие алгоритмы",
            "Сюжетные задачи, свойства функций, алгебра + геометрия",
            "Задачи 2 части ОГЭ, теория вероятностей, параметры",
            "Базовые вычисления, справочные данные",
            "Физ./геом. смысл, оптимизация, 3D-объекты",
            "Строгие доказательства, сложные проценты, нестандартные тела вращения",
        ],
    }

    df_matrix = pd.DataFrame(matrix_data)

    def color_level(val):
        colors = {
            "Низкая (1–3)": "background-color: #d1fae5; color: #065f46",
            "Средняя (4–7)": "background-color: #fef3c7; color: #92400e",
            "Высокая (8–10)": "background-color: #fee2e2; color: #991b1b",
        }
        return colors.get(val, "")

    styled = df_matrix.style.map(color_level, subset=["Уровень"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=460)

    # ── 4. Интерпретация значений ────────────────────────────────────────────
    st.subheader("Интерпретация значений")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**ASL — ориентиры**")
        st.markdown("""
| ASL | Что означает |
|-----|-------------|
| < 5 | Очень короткие фразы |
| 5–10 | Простые предложения |
| 10–15 | Средняя сложность |
| 15–20 | Развёрнутые объяснения |
| > 20 | Сложные многосоставные конструкции |
""")

    with col2:
        st.markdown("**TTR — ориентиры**")
        st.markdown("""
| TTR | Что означает |
|-----|-------------|
| < 0.4 | Низкое разнообразие (шаблон) |
| 0.4–0.6 | Умеренное разнообразие |
| 0.6–0.8 | Хорошее разнообразие |
| > 0.8 | Высокое разнообразие |
| 1.0 | Каждое слово уникально |
""")

    with col3:
        st.markdown("**Оценка сложности (1–10)**")
        st.markdown("""
| Балл | Уровень | Что означает |
|------|---------|-------------|
| 1–3 | 🟢 Низкая | Тьютор адаптируется к уровню; язык простой |
| 4–7 | 🟡 Средняя | Нормативный уровень объяснений |
| 8–10 | 🔴 Высокая | Язык сложнее нормы для данного класса |
""")

    # ── 5. Важные оговорки ──────────────────────────────────────────────────
    st.subheader("Важные оговорки")
    st.warning("""
- Метрики рассчитывались **языковой моделью**, а не детерминированным алгоритмом — возможны погрешности в лемматизации и разбиении на предложения.
- **Математические формулы** (LaTeX-выражения) очищались из текста перед подсчётом, что снижает ASL для технических реплик.
- Оценка сложности — **экспертная интерпретация** матрицы моделью, а не формальный алгоритм. При пограничных значениях модель применяла принцип «слабого звена».
- Текущий датасет содержит малое количество диалогов; агрегирующие графики станут значимыми при наличии 20+ диалогов.
""")
