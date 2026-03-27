import streamlit as st

st.cache_data.clear()

st.set_page_config(
    page_title="R&D Аналитика",
    page_icon="📊",
    layout="wide",
)

st.title("📊 R&D Аналитика")
st.markdown("Результаты аналитической работы.")

st.page_link("pages/generators.py", label="Оценка трудоёмкости генераторов", icon="🔬")
st.page_link("pages/sentiment.py", label="Эмоциональная окраска диалогов", icon="🎭")
st.page_link("pages/emotions.py", label="Эмоциональный анализ (BERT + Gemini)", icon="🧠")
st.page_link("pages/complexity.py", label="Сложность текста диалогов (ASL / TTR)", icon="📐")
st.page_link("pages/intent_quality.py", label="Качество интентов", icon="🎯")
st.page_link("pages/tutor_eval.py", label="Оценка тьютора (3x Gemini + Claude Judge)", icon="🎓")
st.page_link("pages/math_correctness.py", label="Математическая корректность тьютора", icon="🔢")
st.page_link("pages/review_eval.py", label="Ручная проверка оценок", icon="✏️")
