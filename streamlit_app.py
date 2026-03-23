import streamlit as st

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
