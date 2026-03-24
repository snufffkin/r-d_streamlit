#!/usr/bin/env bash
# Запуск оценки интентов одной командой.
#
# Использование:
#   ./eval_intents/eval.sh                  # полный прогон (1564 реплики)
#   ./eval_intents/eval.sh --sample 50      # тест на 50 репликах
#   ./eval_intents/eval.sh --sample 10      # быстрый тест
#
# Требования:
#   - uv (https://docs.astral.sh/uv/)
#   - .env файл с GOOGLE_API_KEY в корне проекта

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Проверка uv
if ! command -v uv &>/dev/null; then
    echo "❌ uv не найден. Установи: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Проверка .env
if [ ! -f .env ]; then
    echo "❌ Файл .env не найден. Создай его с GOOGLE_API_KEY=..."
    exit 1
fi

if ! grep -q "GOOGLE_API_KEY" .env; then
    echo "❌ GOOGLE_API_KEY не найден в .env"
    exit 1
fi

# Проверка данных
INPUT="data/intent_eval/source_dialogs.xlsx"
if [ ! -f "$INPUT" ]; then
    echo "❌ Файл данных не найден: $INPUT"
    echo "   Скопируй XLSX с диалогами в эту папку."
    exit 1
fi

OUTPUT="data/intent_eval/results.csv"

echo "🔧 Устанавливаю зависимости..."
uv sync --quiet

echo "🚀 Запускаю оценку интентов..."
uv run python3 eval_intents/run.py --input "$INPUT" --output "$OUTPUT" "$@"

echo ""
echo "✅ Результаты: $OUTPUT"
echo "📊 Открой дашборд: uv run streamlit run streamlit_app.py"
