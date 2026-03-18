import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Генераторы — оценка трудоёмкости", page_icon="🔬", layout="wide")

# --- Data Loading ---

DATA_PATH = Path(__file__).parent.parent / "data" / "prod_errors_may_aug.csv"

GENERATOR_COMPLEXITY = {
    "Уравнения (и их системы)": 1.5,
    "Числовые выражения (примеры)": 1,
    "Буквенные выражения": 1,
    "Неравенства (и их системы)": 2,
    "Десятичные дроби": 1,
    "Натуральные числа": 1,
    "Обыкновенные дроби": 1,
}

NON_ALGORITHMIC = {
    "Решение задач", "Кружок (нестандартные задачи, изюм)",
    "Функции и графики", "Четырёхугольники", "Треугольники",
    "Прямые", "Логика", "Окружность", "Построение",
    "Многоугольники", "Призма",
}

THEME_COMPLEXITY = {
    "Линейные уравнения": 1,
    "Порядок действий и скобки": 1,
    "Числовые выражения с корнями": 1,
    "Умножение и деление": 1,
    "Сложение и вычитание": 1,
    "Квадратный трёхчлен": 1,
    "Степень с натуральным показателем": 1,
    "Проценты": 1,
    "Вычислять значение выражения": 1,
    "Алгебраические дроби": 1,
    "Задачи на линейное уравнение": 2,
    "Дробно-рациональные уравнения": 2,
    "Задачи на систему уравнений": 2,
    "Обыкновенные дроби. Понятие": 1,
    "Умножение обыкновенных дробей": 1,
    "Сравнение ": 1,
    "Округление (десятичных и натуральных)": 1,
    "Модуль": 2,
    "Деление обыкновенных дробей": 1,
    "Динамические задачи на дробно-рациональное уравнение": 2,
}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    # Pandas reads True/False as booleans; convert to bool explicitly
    df["answer_correct_flg"] = df["answer_correct_flg"].map({True: True, False: False}).fillna(True).astype(bool)
    df["schema_correct_flg"] = df["schema_correct_flg"].map({True: True, False: False}).fillna(True).astype(bool)
    df["task_problem"] = df["task_problem"].map({True: True, False: False}).fillna(False).astype(bool)
    return df


@st.cache_data
def filter_data(df):
    mask = (
        (df["cls_grade_group"] != "1-4 кл.")
        & (df["cls_grade_group"].notna())
        & (df["cls_grade_group"] != "")
        & (~df["task_problem"])
    )
    return df[mask].copy()


df_raw = load_data()
df = filter_data(df_raw)

# --- Header ---

st.title("🔬 Оценка трудоёмкости генераторов")
st.markdown(
    "Анализ ошибок схемогена на проде (май–август). "
    "**Фильтр:** без 1-4 кл., без задач с некорректными условиями."
)

# --- KPI Row ---

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Исходный датасет", f"{len(df_raw):,}")
with col2:
    st.metric("После фильтрации", f"{len(df):,}")
with col3:
    ans_err = (df["answer_correct_flg"] == False).sum()
    st.metric("Ошибки ответа", f"{ans_err}", f"{100 * ans_err / len(df):.1f}%")
with col4:
    alg_sections = set(GENERATOR_COMPLEXITY.keys())
    alg_count = df[df["Раздел"].isin(alg_sections)].shape[0]
    st.metric("Покрываемые генераторами", f"{alg_count}", f"{100 * alg_count / len(df):.0f}%")
with col5:
    non_alg = len(df) - alg_count
    st.metric("Нужен синт. пайплайн", f"{non_alg}", f"{100 * non_alg / len(df):.0f}%")

st.divider()

# === TAB LAYOUT ===

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📝 Вывод",
    "📊 По разделам",
    "📋 По темам",
    "🎓 По классам",
    "🎯 ROI генераторов",
    "🚫 Почему не генератор",
    "🔍 Детализация",
])

# =====================
# TAB 0: Вывод
# =====================
with tab0:
    st.subheader("Решение: алгоритмические генераторы математических задач")

    st.markdown(
        "**Проблема:** схемоген обучается на синтетических данных, в которых просачивается брак — "
        "задачи без ответов, с неверными ответами, из другого раздела. Это портит качество модели."
    )
    st.markdown(
        "**Решение:** для алгоритмизируемых тем (алгебра, арифметика) написать питоновские скрипты-генераторы, "
        "которые создают задачи-аналоги с **гарантированно правильными ответами** и передают ML-команде "
        "связку: оригинальные схемы + 20-30 сгенерированных аналогов + схемы к ним."
    )

    st.divider()

    # --- Key numbers ---
    alg_sections = set(GENERATOR_COMPLEXITY.keys())
    alg_count = df[df["Раздел"].isin(alg_sections)].shape[0]
    non_alg_count = len(df) - alg_count

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Датасет после фильтрации", f"{len(df)} задач",
                  help="Без 1-4 кл. и некорректных условий. Исходный: 1985")
    with col2:
        st.metric("Покрываются генераторами", f"{alg_count} задач ({100 * alg_count / len(df):.0f}%)")
    with col3:
        st.metric("Нужен синт. пайплайн", f"{non_alg_count} задач ({100 * non_alg_count / len(df):.0f}%)")

    st.divider()

    # --- Phase plan ---
    st.subheader("План по фазам")

    phase_data = [
        {
            "Фаза": "Фаза 1 — Quick Wins",
            "Генераторов": 7,
            "Ч/ч": "5–7",
            "Задач": 164,
            "% датасета": f"{100 * 164 / len(df):.0f}%",
            "Темы": "Линейные уравнения, порядок действий, выражения с корнями, умножение/деление, сложение/вычитание, квадратный трёхчлен, степени",
        },
        {
            "Фаза": "Фаза 2 — Расширение",
            "Генераторов": 8,
            "Ч/ч": "8–15",
            "Задач": 97,
            "% датасета": f"{100 * 97 / len(df):.0f}%",
            "Темы": "Проценты, значение выражения, алгебраические дроби, задачи на лин. уравнение, дробно-рац. уравнения, системы, обыкн. дроби",
        },
        {
            "Фаза": "Итого",
            "Генераторов": 15,
            "Ч/ч": "13–22",
            "Задач": 261,
            "% датасета": f"{100 * 261 / len(df):.0f}%",
            "Темы": "",
        },
    ]
    st.dataframe(pd.DataFrame(phase_data), use_container_width=True, hide_index=True)

    st.divider()

    # --- Detailed breakdown: section → theme → generators ---
    st.subheader("Детальная раскладка: раздел → тема → генераторы")

    st.caption(
        "Внутри каждого раздела — несколько тем. На каждую тему может потребоваться "
        "отдельный генератор (или несколько, если тема содержит разные подтипы задач). "
        "Ниже — полная карта с оценкой."
    )

    # Per-theme generator assessment
    THEME_GENERATOR_MAP = [
        # Уравнения (и их системы) — 86 задач
        ("Уравнения", "Линейные уравнения", 32, True, "1–2", "0.5–1",
         "ax+b=c, со скобками, дробные коэфф. Может потребоваться 2 генератора: простые + со скобками/дробями"),
        ("Уравнения", "Квадратный трёхчлен", 10, True, "1–2", "0.5–1",
         "ax²+bx+c=0, формулы Виета, разложение. Один генератор с параметром сложности"),
        ("Уравнения", "Сложение и вычитание", 7, True, "1", "0.5",
         "Простые уравнения вида (x−5)·3=9. Покрывается генератором линейных уравнений"),
        ("Уравнения", "Умножение и деление", 6, True, "1", "0.5",
         "x÷5=x−1000. Покрывается генератором линейных уравнений"),
        ("Уравнения", "Системы уравнений", 5, True, "2", "1–2",
         "Системы 2×2 и 3×3 линейных. Отдельный генератор с матричным решением"),
        ("Уравнения", "Модуль", 5, True, "2", "1–1.5",
         "|x+a|=b. Раскрытие по определению, 2 случая. Отдельный генератор"),
        ("Уравнения", "Степень (показательные ур.)", 5, True, "2", "1",
         "aˣ=b, сводятся к одному основанию. Отдельный генератор"),
        ("Уравнения", "Выражения с корнями (иррац. ур.)", 5, True, "2", "1",
         "√(ax+b)=c. Возведение в квадрат + проверка. Отдельный генератор"),
        ("Уравнения", "Дроби (обыкн. в уравнениях)", 4, True, "1", "0.5",
         "Покрывается генератором линейных/квадратных уравнений с дробными коэфф."),
        ("Уравнения", "Логарифмические уравнения", 3, True, "2–3", "1.5–2",
         "log_a(f(x))=log_a(g(x)). Отдельный генератор, нужна проверка ОДЗ"),
        ("Уравнения", "Дробно-рациональные", 2, True, "2", "1–2",
         "P(x)/Q(x)=0. Отдельный генератор с проверкой ОДЗ"),
        ("Уравнения", "Системы нелинейных", 2, False, "3", "—",
         "Слишком разнородные, 2 задачи — нет ROI"),

        # Числовые выражения (примеры) — 61 задач
        ("Числовые выражения", "Порядок действий и скобки", 25, True, "1", "0.5–1",
         "Генерация AST-дерева выражений. 1 генератор с уровнями вложенности"),
        ("Числовые выражения", "Выражения с корнями", 17, True, "1–2", "0.5–1",
         "√a±√b, вынесение из-под корня. 1 генератор, строит из упрощённой формы"),
        ("Числовые выражения", "Степень с нат. показателем", 9, True, "1", "0.5",
         "Свойства степеней: aⁿ·aᵐ, (aⁿ)ᵐ. 1 генератор"),
        ("Числовые выражения", "Читать и записывать выражения", 3, False, "—", "—",
         "Текстовые формулировки: «что произойдёт с объёмом конуса». Не алгоритмизируется"),
        ("Числовые выражения", "Сравнение", 2, True, "1", "0.5",
         "Покрывается генератором порядка действий или дробей"),
        ("Числовые выражения", "Степень с рац./целым показ.", 4, True, "1", "0.5",
         "Покрывается генератором степеней с расширением на отрицательные/дробные"),
        ("Числовые выражения", "Вычислить удобным способом", 1, False, "—", "—",
         "1 задача, нет ROI"),

        # Буквенные выражения — 45 задач
        ("Буквенные выражения", "Вычислить значение выражения", 13, True, "1", "0.5",
         "Подставить числа в формулу. 1 генератор: создаёт выражение + значения переменных"),
        ("Буквенные выражения", "Алгебраические дроби", 12, True, "1–2", "1",
         "Сложение/вычитание/упрощение дробей с переменными. 1 генератор, строит из ответа"),
        ("Буквенные выражения", "Упрощение выражений", 4, True, "1", "0.5",
         "Раскрытие скобок, приведение подобных. Покрывается генератором многочленов"),
        ("Буквенные выражения", "Многочлены", 3, True, "1", "0.5",
         "Раскрытие скобок. 1 генератор: создаёт произведение → раскрывает"),
        ("Буквенные выражения", "Одночлены", 3, True, "1", "0.5",
         "Степени одночленов. Покрывается генератором степеней"),
        ("Буквенные выражения", "Составлять выражения", 3, False, "—", "—",
         "Текстовая формулировка: «записать неполный квадрат суммы». Не алгоритмизируется"),
        ("Буквенные выражения", "Свойства степеней", 3, True, "1", "0.5",
         "Покрывается генератором степеней"),
        ("Буквенные выражения", "Другие (≤2 задачи)", 4, True, "1", "0.5",
         "Упрощение по свойствам, разложение на множители. Частично покрываются другими генераторами"),

        # Неравенства — 20 задач
        ("Неравенства", "Квадратные неравенства", 6, True, "2", "1",
         "Метод интервалов: (ax−b)(cx−d)≥0. 1 генератор"),
        ("Неравенства", "Линейные неравенства", 6, True, "1", "0.5",
         "ax+b≥c. Покрывается генератором линейных уравнений с изменением знака"),
        ("Неравенства", "С модулем", 2, True, "2", "1",
         "|x−a|+|b−x|>c. Отдельный генератор или расширение генератора модулей"),
        ("Неравенства", "Показательные неравенства", 2, True, "2", "1",
         "aˣ>b. Расширение генератора показательных уравнений"),
        ("Неравенства", "Другие (≤2 задачи)", 4, False, "—", "—",
         "Действия с неравенствами, иррациональные. Малый объём, разнородные"),

        # Десятичные дроби — 18 задач
        ("Десятичные дроби", "Деление", 5, True, "1", "0.5",
         "Деление десятичных дробей. Покрывается общим арифметическим генератором"),
        ("Десятичные дроби", "Округление", 4, True, "1", "0.5",
         "Округление до разряда. 1 простой генератор"),
        ("Десятичные дроби", "Умножение", 4, True, "1", "0.5",
         "Умножение десятичных. Покрывается арифметическим генератором"),
        ("Десятичные дроби", "Сложение и вычитание", 2, True, "1", "0.5",
         "Покрывается арифметическим генератором"),
        ("Десятичные дроби", "Другие (≤1 задачи)", 3, True, "1", "0.5",
         "Перевод, сравнение, периодические. Простые генераторы"),

        # Натуральные числа — 18 задач
        ("Натуральные числа", "Умножение и деление", 11, True, "1", "0.5",
         "Умножение больших чисел. Покрывается арифметическим генератором"),
        ("Натуральные числа", "Сложение и вычитание", 4, True, "1", "0.5",
         "Покрывается арифметическим генератором"),
        ("Натуральные числа", "Другие (≤1 задачи)", 3, False, "—", "—",
         "Римские числа, разряды, сравнение. По 1 задаче — нет ROI"),

        # Обыкновенные дроби — 13 задач
        ("Обыкновенные дроби", "Сокращение дробей", 4, True, "1", "0.5",
         "Основное свойство дроби. Покрывается генератором дробей"),
        ("Обыкновенные дроби", "Сложение и вычитание", 4, True, "1", "0.5",
         "Приведение к общему знаменателю. Покрывается генератором дробей"),
        ("Обыкновенные дроби", "Умножение", 4, True, "1", "0.5",
         "a/b × c/d. Покрывается генератором дробей"),
        ("Обыкновенные дроби", "Сравнение", 1, True, "1", "0.5",
         "Покрывается генератором дробей"),
    ]

    gen_df = pd.DataFrame(THEME_GENERATOR_MAP, columns=[
        "Раздел", "Тема", "Задач", "Генератор возможен", "Сложность", "Ч/ч", "Комментарий",
    ])

    # Summary metrics
    gen_possible = gen_df[gen_df["Генератор возможен"]]
    gen_not = gen_df[~gen_df["Генератор возможен"]]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего тем в алг. разделах", len(gen_df))
    with col2:
        st.metric("Тем с генератором", len(gen_possible))
    with col3:
        st.metric("Задач покрыто", int(gen_possible["Задач"].sum()))
    with col4:
        st.metric("Тем без генератора", len(gen_not), f"{int(gen_not['Задач'].sum())} задач")

    st.markdown("")

    # Sunburst chart
    sun_df = gen_df.copy()
    sun_df["Статус"] = sun_df["Генератор возможен"].map({True: "Генератор", False: "Нет генератора"})
    fig_sun = px.sunburst(
        sun_df,
        path=["Раздел", "Тема"],
        values="Задач",
        color="Статус",
        color_discrete_map={"Генератор": "#2ecc71", "Нет генератора": "#e74c3c"},
        title="Карта: раздел → тема (размер = задачи, цвет = возможность генератора)",
    )
    fig_sun.update_layout(height=550)
    st.plotly_chart(fig_sun, use_container_width=True)

    # Detailed table per section
    for section in gen_df["Раздел"].unique():
        sec_rows = gen_df[gen_df["Раздел"] == section]
        total_tasks = int(sec_rows["Задач"].sum())
        gen_tasks = int(sec_rows[sec_rows["Генератор возможен"]]["Задач"].sum())
        unique_gens = len(sec_rows[sec_rows["Генератор возможен"]])

        with st.expander(
            f"**{section}** — {total_tasks} задач, {unique_gens} генераторов, покрыто {gen_tasks}",
            expanded=False,
        ):
            display_df = sec_rows[["Тема", "Задач", "Генератор возможен", "Сложность", "Ч/ч", "Комментарий"]].copy()
            display_df["Генератор возможен"] = display_df["Генератор возможен"].map({True: "✅", False: "❌"})
            display_df.columns = ["Тема", "Задач", "Генератор", "Сложность", "Ч/ч", "Комментарий"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Total hours estimate from detailed data
    st.markdown("---")
    st.markdown(
        "**Ключевое наблюдение:** внутри одного раздела несколько тем, и не каждая тема = 1 генератор. "
        "Некоторые темы покрываются одним и тем же генератором (например, «сложение и вычитание» "
        "и «умножение и деление» в уравнениях — оба покрываются генератором линейных уравнений). "
        "Реальное количество уникальных скриптов — **15–20**, а не 40+."
    )

    st.divider()

    # --- What's NOT covered ---
    st.subheader("Что остаётся за пределами генераторов")

    not_covered = [
        {"Раздел": "Решение задач", "Задач": 103, "Причина": "Текстовые условия с уникальным контекстом — нужен LLM"},
        {"Раздел": "Кружок (олимпиадные)", "Задач": 30, "Причина": "Уникальная логика, нет шаблона. 70% ошибок ответа — худший показатель"},
        {"Раздел": "Функции и графики", "Задач": 23, "Причина": "Визуальный компонент, исследование функций"},
        {"Раздел": "Геометрия (всё)", "Задач": 25, "Причина": "Чертежи, пространственное мышление, проверка корректности фигур"},
        {"Раздел": "Малые группы", "Задач": 70, "Причина": "Тригонометрия, производные, делимость, комбинаторика и др. — по 3-8 задач, нет ROI"},
    ]
    st.dataframe(pd.DataFrame(not_covered), use_container_width=True, hide_index=True)

    st.markdown(f"**Итого не покрывается:** ~{non_alg_count} задач ({100 * non_alg_count / len(df):.0f}%) → нужен синтетический пайплайн (ревизия пайплайна Айрата)")

    st.divider()

    # --- Recommendation ---
    st.subheader("Рекомендация")

    st.info(
        "**13–22 часа работы** на 15 генераторов закроют **51% задач** (261 из 512) "
        "с гарантированным качеством. Все генераторы фазы 1 укладываются в порог Павла "
        "(< 1 часа на генератор). Начинать с линейных уравнений — максимальный ROI (40 задач)."
    )

    st.warning(
        "**Оставшиеся 49%** (текстовые задачи, кружок, геометрия) требуют другого подхода: "
        "ревизия пайплайна синтетической генерации Айрата. Отдельно — олимпиадные задачи "
        "(30 шт., 70% ошибок ответа): написать образцовые схемы по классическим сборникам."
    )

    # --- Visual summary ---
    summary_fig = go.Figure()
    summary_fig.add_trace(go.Bar(
        name="Фаза 1 (генераторы)", x=["План"], y=[164],
        marker_color="#2ecc71", text=["164 задач<br>5–7 ч"], textposition="inside",
    ))
    summary_fig.add_trace(go.Bar(
        name="Фаза 2 (генераторы)", x=["План"], y=[97],
        marker_color="#3498db", text=["97 задач<br>8–15 ч"], textposition="inside",
    ))
    summary_fig.add_trace(go.Bar(
        name="Не покрывается", x=["План"], y=[non_alg_count],
        marker_color="#e74c3c", text=[f"{non_alg_count} задач<br>синт. пайплайн"], textposition="inside",
    ))
    summary_fig.update_layout(
        barmode="stack",
        title="Покрытие датасета генераторами",
        yaxis_title="Задач",
        height=400,
        showlegend=True,
    )
    st.plotly_chart(summary_fig, use_container_width=True)

# =====================
# TAB 1: По разделам
# =====================
with tab1:
    st.subheader("Распределение задач по разделам математики")

    sec_stats = (
        df.groupby("Раздел")
        .agg(
            total=("id", "count"),
            ans_errors=("answer_correct_flg", lambda x: (x == False).sum()),
        )
        .reset_index()
        .sort_values("total", ascending=False)
    )
    sec_stats["ans_pct"] = (100 * sec_stats["ans_errors"] / sec_stats["total"]).round(1)
    sec_stats["type"] = sec_stats["Раздел"].apply(
        lambda x: "Алгоритмизируемый" if x in GENERATOR_COMPLEXITY else (
            "Не алгоритмизируемый" if x in NON_ALGORITHMIC else "Пограничный"
        )
    )

    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.bar(
            sec_stats,
            x="total",
            y="Раздел",
            color="type",
            orientation="h",
            color_discrete_map={
                "Алгоритмизируемый": "#2ecc71",
                "Не алгоритмизируемый": "#e74c3c",
                "Пограничный": "#f39c12",
            },
            title="Количество задач по разделам",
            labels={"total": "Задач", "Раздел": "", "type": "Тип"},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.bar(
            sec_stats.sort_values("ans_pct", ascending=False),
            x="ans_pct",
            y="Раздел",
            color="type",
            orientation="h",
            color_discrete_map={
                "Алгоритмизируемый": "#2ecc71",
                "Не алгоритмизируемый": "#e74c3c",
                "Пограничный": "#f39c12",
            },
            title="% ошибок ответа по разделам",
            labels={"ans_pct": "% ошибок ответа", "Раздел": "", "type": "Тип"},
        )
        fig2.update_layout(yaxis=dict(autorange="reversed"), height=600)
        st.plotly_chart(fig2, use_container_width=True)

    # Pie: algorithmizable vs not
    pie_data = sec_stats.groupby("type")["total"].sum().reset_index()
    fig3 = px.pie(
        pie_data,
        values="total",
        names="type",
        title="Алгоритмизируемые vs не алгоритмизируемые",
        color="type",
        color_discrete_map={
            "Алгоритмизируемый": "#2ecc71",
            "Не алгоритмизируемый": "#e74c3c",
            "Пограничный": "#f39c12",
        },
    )
    st.plotly_chart(fig3, use_container_width=True)

# =====================
# TAB 2: По темам
# =====================
with tab2:
    st.subheader("Распределение по темам (внутри разделов)")

    min_tasks = st.slider("Минимум задач в теме", 1, 20, 5, key="min_tasks")

    theme_stats = (
        df.groupby(["Темы", "Раздел"])
        .agg(
            total=("id", "count"),
            ans_errors=("answer_correct_flg", lambda x: (x == False).sum()),
        )
        .reset_index()
    )
    theme_stats["ans_pct"] = (100 * theme_stats["ans_errors"] / theme_stats["total"]).round(1)
    theme_stats["algorithmizable"] = theme_stats["Темы"].apply(
        lambda x: "Да" if x.strip() in THEME_COMPLEXITY else "Нет"
    )
    theme_stats = theme_stats[theme_stats["total"] >= min_tasks].sort_values("total", ascending=False)

    col_a, col_b = st.columns(2)

    with col_a:
        fig = px.bar(
            theme_stats.head(20),
            x="total",
            y="Темы",
            color="algorithmizable",
            orientation="h",
            color_discrete_map={"Да": "#2ecc71", "Нет": "#e74c3c"},
            title="Топ-20 тем по количеству задач",
            labels={"total": "Задач", "Темы": "", "algorithmizable": "Генератор возможен"},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=700)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = px.bar(
            theme_stats.sort_values("ans_pct", ascending=False).head(20),
            x="ans_pct",
            y="Темы",
            color="algorithmizable",
            orientation="h",
            color_discrete_map={"Да": "#2ecc71", "Нет": "#e74c3c"},
            title="Топ-20 тем по % ошибок ответа",
            labels={"ans_pct": "% ошибок ответа", "Темы": "", "algorithmizable": "Генератор возможен"},
        )
        fig2.update_layout(yaxis=dict(autorange="reversed"), height=700)
        st.plotly_chart(fig2, use_container_width=True)

    # Treemap
    st.subheader("Карта тем по разделам")
    tree_data = theme_stats[theme_stats["total"] >= min_tasks].copy()
    fig3 = px.treemap(
        tree_data,
        path=["Раздел", "Темы"],
        values="total",
        color="ans_pct",
        color_continuous_scale="RdYlGn_r",
        title="Размер = количество задач, цвет = % ошибок ответа",
        labels={"ans_pct": "% ошибок ответа"},
    )
    fig3.update_layout(height=600)
    st.plotly_chart(fig3, use_container_width=True)

# =====================
# TAB 3: По классам
# =====================
with tab3:
    st.subheader("Распределение по классам")

    grade_stats = (
        df.groupby("cls_grade_group")
        .agg(
            total=("id", "count"),
            ans_errors=("answer_correct_flg", lambda x: (x == False).sum()),
        )
        .reset_index()
    )
    grade_stats["ans_pct"] = (100 * grade_stats["ans_errors"] / grade_stats["total"]).round(1)

    col_a, col_b, col_c = st.columns(3)
    for i, (_, row) in enumerate(grade_stats.iterrows()):
        col = [col_a, col_b, col_c][i]
        with col:
            st.metric(
                row["cls_grade_group"],
                f"{row['total']} задач",
                f"Ошибок ответа: {row['ans_errors']} ({row['ans_pct']}%)",
            )

    # By grade × section
    grade_sec = (
        df.groupby(["cls_grade_group", "Раздел"])
        .agg(total=("id", "count"))
        .reset_index()
    )
    fig = px.bar(
        grade_sec,
        x="cls_grade_group",
        y="total",
        color="Раздел",
        title="Разделы по классам",
        labels={"total": "Задач", "cls_grade_group": "Класс"},
        barmode="stack",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap: grade × section → answer error rate
    st.subheader("Тепловая карта: % ошибок ответа (класс × раздел)")
    heatmap_data = (
        df.groupby(["cls_grade_group", "Раздел"])
        .agg(
            total=("id", "count"),
            ans_errors=("answer_correct_flg", lambda x: (x == False).sum()),
        )
        .reset_index()
    )
    heatmap_data["ans_pct"] = (100 * heatmap_data["ans_errors"] / heatmap_data["total"]).round(0)
    heatmap_data = heatmap_data[heatmap_data["total"] >= 3]

    pivot = heatmap_data.pivot_table(index="Раздел", columns="cls_grade_group", values="ans_pct")
    pivot = pivot.reindex(columns=["5-6 кл.", "7-9 кл.", "10-11 кл."])

    fig2 = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="RdYlGn_r",
        title="% ошибок ответа (только разделы с ≥3 задач)",
        labels={"color": "% ошибок"},
        aspect="auto",
    )
    fig2.update_layout(height=600)
    st.plotly_chart(fig2, use_container_width=True)

# =====================
# TAB 4: ROI генераторов
# =====================
with tab4:
    st.subheader("ROI: что писать в первую очередь")

    roi_data = []
    for theme, complexity in THEME_COMPLEXITY.items():
        theme_rows = df[df["Темы"].str.strip() == theme.strip()]
        if len(theme_rows) == 0:
            continue
        total = len(theme_rows)
        ans_err = (theme_rows["answer_correct_flg"] == False).sum()
        hours_est = 0.5 if complexity == 1 else 1.5
        roi = total / hours_est
        roi_data.append({
            "Тема": theme.strip(),
            "Задач": total,
            "Ошибок ответа": ans_err,
            "% ошибок": round(100 * ans_err / total, 0),
            "Сложность": complexity,
            "Ч/ч (оценка)": hours_est,
            "ROI (задач/час)": round(roi, 1),
            "Фаза": "Фаза 1" if theme.strip() in {
                "Линейные уравнения", "Порядок действий и скобки",
                "Числовые выражения с корнями", "Умножение и деление",
                "Сложение и вычитание", "Квадратный трёхчлен",
                "Степень с натуральным показателем",
            } else "Фаза 2",
        })

    roi_df = pd.DataFrame(roi_data).sort_values("ROI (задач/час)", ascending=False)

    # Summary metrics
    phase1 = roi_df[roi_df["Фаза"] == "Фаза 1"]
    phase2 = roi_df[roi_df["Фаза"] == "Фаза 2"]

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Фаза 1: генераторов", f"{len(phase1)}")
    with col_b:
        st.metric("Фаза 1: задач", f"{phase1['Задач'].sum()}")
    with col_c:
        st.metric("Фаза 1: часов", f"{phase1['Ч/ч (оценка)'].sum():.0f}")
    with col_d:
        st.metric("Всего задач (Ф1+Ф2)", f"{roi_df['Задач'].sum()}")

    # ROI chart
    fig = px.bar(
        roi_df,
        x="ROI (задач/час)",
        y="Тема",
        color="Фаза",
        orientation="h",
        color_discrete_map={"Фаза 1": "#2ecc71", "Фаза 2": "#3498db"},
        title="ROI генераторов (задач, покрываемых за час работы)",
        labels={"Тема": ""},
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Bubble chart: tasks × error rate × hours
    fig2 = px.scatter(
        roi_df,
        x="Задач",
        y="% ошибок",
        size="ROI (задач/час)",
        color="Фаза",
        text="Тема",
        color_discrete_map={"Фаза 1": "#2ecc71", "Фаза 2": "#3498db"},
        title="Пузырьковая диаграмма: объём × % ошибок ответа × ROI",
        labels={"Задач": "Количество задач", "% ошибок": "% ошибок ответа"},
    )
    fig2.update_traces(textposition="top center", textfont_size=9)
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, use_container_width=True)

    # Table
    st.subheader("Таблица ROI")
    st.dataframe(roi_df, use_container_width=True, hide_index=True)

    # Cumulative coverage chart
    st.subheader("Кумулятивное покрытие")
    cum_df = roi_df.sort_values("ROI (задач/час)", ascending=False).copy()
    cum_df["Часов (кум.)"] = cum_df["Ч/ч (оценка)"].cumsum()
    cum_df["Задач (кум.)"] = cum_df["Задач"].cumsum()
    cum_df["% покрытия"] = (100 * cum_df["Задач (кум.)"] / len(df)).round(1)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=cum_df["Часов (кум.)"],
        y=cum_df["% покрытия"],
        mode="lines+markers+text",
        text=cum_df["Тема"],
        textposition="top left",
        textfont_size=8,
        name="% покрытия датасета",
        line=dict(color="#2ecc71", width=3),
    ))
    fig3.update_layout(
        title="Кумулятивное покрытие: сколько задач закроем за N часов",
        xaxis_title="Часов работы (кумулятивно)",
        yaxis_title="% задач датасета покрыто генераторами",
        height=450,
    )
    st.plotly_chart(fig3, use_container_width=True)

# =====================
# TAB 5: Почему не генератор
# =====================
with tab5:
    st.subheader("Почему эти разделы не покрываются генераторами")

    BLOCKER_REASONS = {
        "Решение задач": {
            "tasks": 103,
            "reason": "Текстовые условия с уникальным контекстом",
            "detail": (
                "Каждая задача — уникальная текстовая формулировка с сюжетом, персонажами, "
                "предметной областью. Генератор не может создавать осмысленные тексты: "
                "«Барон Мюнгаузен похвастался...», «Малыш, Карлсон и Винни-Пух съели торт...», "
                "«Имеются два сосуда с раствором кислоты...». "
                "Шаблонизация возможна лишь для простейших подтипов (задачи на линейное уравнение)."
            ),
            "blocker": "Нужен NLP/LLM для генерации осмысленного текста условия",
            "examples": [
                ("Барон Мюнгаузен похвастался, что если Мери Поппинс будет бежать со скоростью 7 км/ч, а он начнет догонять её со скоростью 6 км/ч через час, то догонит...", "Не поверила, потребуется 7 часов"),
                ("Малыш, Карлсон и Винни-Пух съели торт. Каждый ел с постоянной скоростью. Малышу досталась только ⅓...", "52 минуты"),
                ("Два сосуда: 10 кг и 16 кг раствора кислоты различной концентрации. При смешивании — 55%...", "87%"),
            ],
        },
        "Кружок (нестандартные задачи, изюм)": {
            "tasks": 30,
            "reason": "Уникальная логика, нет шаблона решения",
            "detail": (
                "Олимпиадные и нестандартные задачи. Каждая требует креативного подхода, "
                "нет повторяющихся алгоритмов. Типы: задачи на перебор, взвешивания, "
                "задачи в целых числах, оценка + пример. 70% ошибок ответа — худший показатель."
            ),
            "blocker": "Нет алгоритма решения → нет алгоритма генерации",
            "examples": [
                ("На крыше сидят воробьи. Напротив каждого — 3 воробья. Сколько воробьев?", "4"),
                ("В «Детском мире» продавали 2- и 3-колёсные велосипеды. 12 рулей и 27 колёс. Сколько 3-колёсных?", "3"),
                ("Можно максимально поставить 8 ферм, каждая стоит 250 и принесёт 50...", "−6000 единиц"),
            ],
        },
        "Функции и графики": {
            "tasks": 23,
            "reason": "Визуальный компонент + исследование",
            "detail": (
                "Половина задач (12 из 23) — исследование функции: ООД, монотонность, экстремумы. "
                "Остальные — построение графиков, работа с графиком, составление уравнения функции. "
                "Генератор не может создать задачу «определи по графику» без самого графика."
            ),
            "blocker": "Графический компонент; исследование требует символьных CAS-вычислений",
            "examples": [
                ("y = log₇(4x − x²). Область определения?", "(0; 4)"),
                ("Найти функцию, обратную к y = 3x + 4", "y = (x − 4) / 3"),
                ("Парабола y = x² − 4x + 3. В каких четвертях?", "I, II, IV"),
            ],
        },
        "Четырёхугольники": {
            "tasks": 12,
            "reason": "Геометрия: чертежи и пространственное мышление",
            "detail": (
                "Задачи на параллелограммы (5), прямоугольники (3), трапеции. "
                "Требуют понимания взаимного расположения фигур, биссектрис, диагоналей. "
                "Генератор может создать числовые параметры, но не может гарантировать "
                "корректность геометрической конфигурации (существование фигуры, непротиворечивость)."
            ),
            "blocker": "Нужна проверка геометрической корректности; часто нужен чертёж",
            "examples": [
                ("Прямоугольник периметра 180 разрезали на 8 одинаковых прямоугольников. Сумма разрезов = 110. Площадь?", "103.75 см²"),
                ("В параллелограмме ABCD биссектриса угла A пересекает BC в K. AB = 5, периметр...", "36/11"),
            ],
        },
        "Треугольники": {
            "tasks": 7,
            "reason": "Геометрия: координаты, углы, типы треугольников",
            "detail": (
                "Задачи на определение типа треугольника по координатам, поиск углов, "
                "периметр, площадь. Требуют как вычислений, так и геометрической интерпретации."
            ),
            "blocker": "Геометрическая корректность + малый объём (7 задач — нет ROI)",
            "examples": [
                ("Треугольник с вершинами A(0;1;2), B(−2;−1;0), C(1;0;1) — остро-, прямо- или тупоугольный?", "Тупоугольный"),
                ("Один из острых углов прямоугольного треугольника равен 23°. Найдите другой.", "67°"),
            ],
        },
        "Производные": {
            "tasks": 8,
            "reason": "Пределы, определения, практический смысл",
            "detail": (
                "Половина задач — пределы (lim), не собственно производные. "
                "Задачи на определение, непрерывность, дифференцируемость требуют "
                "теоретических знаний, а не вычислений. Практический смысл — интерпретация."
            ),
            "blocker": "Разнородные типы задач; малый объём (8 задач)",
            "examples": [
                ("lim(x→3)(x² + 2x − 12)", "88"),
                ("При каких параметрах f(x) непрерывна/дифференцируема на R?", "α = −2, β = −2"),
                ("lim(x→π) (1 − sin(x/2)) / (π − x)", "1/2"),
            ],
        },
        "Тригонометрия": {
            "tasks": 8,
            "reason": "Вычисления + уравнения + углы в окружности",
            "detail": (
                "Смесь вычислительных (cos²75° + cos30° − sin²75°) и уравнительных "
                "(2cos²x + 2sin2x = 3) задач, плюс углы в окружности (геометрия). "
                "Вычислительные частично алгоритмизируемы, но объём мал."
            ),
            "blocker": "Малый объём (8 задач); смесь алгебры и геометрии",
            "examples": [
                ("cos²75° + cos30° − sin²75°", "0"),
                ("√3·cos(π/6) + 2sin(π/3) − (√3/2)·ctg(π/6)", "√3"),
            ],
        },
        "Делимость": {
            "tasks": 8,
            "reason": "НОД/НОК алгоритмизируемы, но объём мал",
            "detail": (
                "НОД и НОК (4 задачи) — алгоритмизируемы. Признаки делимости (2) — тоже. "
                "Но всего 8 задач — ROI написания генератора низкий."
            ),
            "blocker": "Малый объём (8 задач). Частично алгоритмизируемый — кандидат на фазу 3",
            "examples": [
                ("НОК(5; 10; 16) =", "80"),
                ("НОД(18, 24) × НОК(18, 24) и 18 × 24", "432"),
            ],
        },
        "Комбинаторика": {
            "tasks": 6,
            "reason": "Перебор вариантов + текстовые формулировки",
            "detail": (
                "Вычислительная комбинаторика (C⁹⁸₁₀₀) алгоритмизируема, но текстовые "
                "задачи на перебор («Можно ли из домино выложить ряд?») — нет. "
                "Малый объём."
            ),
            "blocker": "Малый объём (6 задач); текстовые задачи на перебор не шаблонизируются",
            "examples": [
                ("Вычислить C⁹⁸₁₀₀", "4950"),
                ("Из домино выбросили пустышки. Можно ли оставшиеся выложить в ряд?", "Нет"),
                ("Сколько 8-значных чисел из одной 1, двух 2 и пяти 5?", "168"),
            ],
        },
        "Логика": {
            "tasks": 5,
            "reason": "Теоретические знания + нестандартные формулировки",
            "detail": (
                "4 из 5 задач — «теоретические знания»: задачи, замаскированные под другие разделы "
                "(уравнения, неравенства), но требующие логического рассуждения. "
                "Одна задача — истинные/ложные высказывания."
            ),
            "blocker": "Нестандартные формулировки; задачи не из типичного кодификатора",
            "examples": [
                ("Из посёлка выехал мотоциклист 60 км/ч, навстречу велосипедист 15 км/ч. Кто ближе к городу при встрече?", "Велосипедист (90 км)"),
            ],
        },
    }

    # Summary bar chart
    blocker_df = pd.DataFrame([
        {"Раздел": k, "Задач": v["tasks"], "Причина": v["reason"]}
        for k, v in BLOCKER_REASONS.items()
    ]).sort_values("Задач", ascending=False)

    fig = px.bar(
        blocker_df,
        x="Задач",
        y="Раздел",
        orientation="h",
        color="Причина",
        title="Не алгоритмизируемые разделы: объём и причина",
        labels={"Раздел": ""},
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.4),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Итого не покрывается:** {sum(v['tasks'] for v in BLOCKER_REASONS.values())} задач")
    st.divider()

    # Detailed cards for each section
    for section, info in BLOCKER_REASONS.items():
        with st.expander(f"**{section}** — {info['tasks']} задач | {info['reason']}", expanded=False):
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.markdown(info["detail"])
                st.markdown(f"**Блокер:** {info['blocker']}")
            with col_r:
                # Show actual data from dataset
                sec_df_local = df[df["Раздел"] == section]
                ans_err_local = (sec_df_local["answer_correct_flg"] == False).sum()
                st.metric("Ошибок ответа", f"{ans_err_local} ({100 * ans_err_local / len(sec_df_local):.0f}%)")
                themes_local = sec_df_local["Темы"].value_counts().head(3)
                st.markdown("**Топ темы:**")
                for t, c in themes_local.items():
                    st.markdown(f"- {t} ({c})")

            st.markdown("**Примеры задач из датасета:**")
            for task_text, answer in info["examples"]:
                st.markdown(f"> {task_text}")
                st.markdown(f"> **Ответ:** {answer}")
                st.markdown("")

# =====================
# TAB 6: Детализация
# =====================
with tab6:
    st.subheader("Детализация по разделу")

    sections = sorted(df["Раздел"].unique())
    selected_section = st.selectbox("Выберите раздел", sections)

    sec_df = df[df["Раздел"] == selected_section]
    sec_ans_err = (sec_df["answer_correct_flg"] == False).sum()

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Задач", len(sec_df))
    with col_b:
        st.metric("Ошибок ответа", f"{sec_ans_err} ({100 * sec_ans_err / len(sec_df):.0f}%)")
    with col_c:
        is_alg = selected_section in GENERATOR_COMPLEXITY
        st.metric("Генератор", "Алгоритмизируемый" if is_alg else "Синтетический пайплайн")

    # Themes within section
    sec_themes = (
        sec_df.groupby("Темы")
        .agg(
            total=("id", "count"),
            ans_errors=("answer_correct_flg", lambda x: (x == False).sum()),
        )
        .reset_index()
        .sort_values("total", ascending=False)
    )
    sec_themes["ans_pct"] = (100 * sec_themes["ans_errors"] / sec_themes["total"]).round(1)

    fig = px.bar(
        sec_themes,
        x="total",
        y="Темы",
        orientation="h",
        color="ans_pct",
        color_continuous_scale="RdYlGn_r",
        title=f"Темы в разделе «{selected_section}»",
        labels={"total": "Задач", "Темы": "", "ans_pct": "% ошибок ответа"},
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=max(300, len(sec_themes) * 30 + 100))
    st.plotly_chart(fig, use_container_width=True)

    # Grade distribution within section
    sec_grades = sec_df["cls_grade_group"].value_counts().reset_index()
    sec_grades.columns = ["Класс", "Задач"]
    fig2 = px.pie(sec_grades, values="Задач", names="Класс", title="Распределение по классам")
    st.plotly_chart(fig2, use_container_width=True)

    # Sample tasks
    st.subheader("Примеры задач")
    sample = sec_df[["id", "task", "answer", "Темы", "cls_grade_group"]].head(10)
    sample.columns = ["ID", "Условие", "Ответ", "Тема", "Класс"]
    # Truncate long task texts
    sample["Условие"] = sample["Условие"].str[:200]
    sample["Ответ"] = sample["Ответ"].str[:100]
    st.dataframe(sample, use_container_width=True, hide_index=True)
