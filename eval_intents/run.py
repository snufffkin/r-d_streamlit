#!/usr/bin/env python3
"""
Pipeline for evaluating intent compliance of synthetic student turns.

Reads XLSX with synthetic dialogs, parses user turns with intent tags,
calls Gemini to evaluate whether each reply matches its declared intent,
and saves results to CSV.

Usage:
    python run.py --input dialogs.xlsx --output results.csv
    python run.py --input dialogs.xlsx --output results.csv --sample 50
    python run.py --input dialogs.xlsx --output results.csv --concurrency 10
"""

import argparse
import asyncio
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (r-d_streamlit/)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import openpyxl
from google import genai
from google.genai import types
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UserTurn:
    """A single parsed user turn from a dialog."""
    dialog_idx: int
    turn_idx: int
    intent_declared: str
    student_text: str
    teacher_text_before: str  # last assistant utterance before this turn
    grade_group: str
    task_id: str


@dataclass
class EvalResult:
    """Evaluation result for a single turn."""
    dialog_idx: int
    turn_idx: int
    intent_declared: str
    student_text: str
    teacher_text_before: str
    match: bool
    confidence: float
    reason: str
    actual_intent: str
    grade_group: str
    task_id: str


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Matches lines like: Пользователь [answer]: текст
USER_TURN_RE = re.compile(
    r"^Пользователь\s*\[([^\]]+)\]\s*:\s*(.+)$", re.MULTILINE
)
# Matches lines like: Пользователь: текст  (first turn, no intent)
USER_TURN_NO_INTENT_RE = re.compile(
    r"^Пользователь\s*:\s*(.+)$", re.MULTILINE
)
# Matches assistant lines
ASSISTANT_RE = re.compile(
    r"^Ассистент\s*:\s*(.+?)(?=\n(?:Пользователь|Ассистент)|\Z)",
    re.MULTILINE | re.DOTALL,
)


def parse_dialog(dialog_text: str) -> list[dict]:
    """Parse dialog text into a list of turns with role, intent, text."""
    turns: list[dict] = []
    # Split into lines and rebuild turns
    # We need ordered turns, so we scan line by line
    lines = dialog_text.strip().split("\n")
    current_role = None
    current_intent = None
    current_text_lines: list[str] = []

    def flush():
        if current_role and current_text_lines:
            turns.append({
                "role": current_role,
                "intent": current_intent,
                "text": "\n".join(current_text_lines).strip(),
            })

    for line in lines:
        # Check for user turn with intent
        m = re.match(r"^Пользователь\s*\[([^\]]+)\]\s*:\s*(.*)$", line)
        if m:
            flush()
            current_role = "user"
            current_intent = m.group(1).strip()
            current_text_lines = [m.group(2).strip()] if m.group(2).strip() else []
            continue

        # Check for user turn without intent
        m = re.match(r"^Пользователь\s*:\s*(.*)$", line)
        if m:
            flush()
            current_role = "user"
            current_intent = None
            current_text_lines = [m.group(1).strip()] if m.group(1).strip() else []
            continue

        # Check for assistant turn
        m = re.match(r"^Ассистент\s*:\s*(.*)$", line)
        if m:
            flush()
            current_role = "assistant"
            current_intent = None
            current_text_lines = [m.group(1).strip()] if m.group(1).strip() else []
            continue

        # Continuation line
        if current_role:
            current_text_lines.append(line)

    flush()
    return turns


def extract_user_turns(
    dialog_text: str,
    dialog_idx: int,
    grade_group: str,
    task_id: str,
) -> list[UserTurn]:
    """Extract evaluable user turns (those with intent tags) from a dialog."""
    turns = parse_dialog(dialog_text)
    result: list[UserTurn] = []
    turn_counter = 0

    for i, turn in enumerate(turns):
        if turn["role"] != "user":
            continue
        turn_counter += 1

        # Skip first user turn (no intent tag — it's the task statement)
        if turn["intent"] is None:
            continue

        # Find preceding assistant text
        teacher_before = ""
        for j in range(i - 1, -1, -1):
            if turns[j]["role"] == "assistant":
                teacher_before = turns[j]["text"]
                break

        result.append(UserTurn(
            dialog_idx=dialog_idx,
            turn_idx=turn_counter,
            intent_declared=turn["intent"],
            student_text=turn["text"],
            teacher_text_before=teacher_before,
            grade_group=grade_group,
            task_id=task_id,
        ))

    return result


def build_context_window(dialog_text: str, target_turn_idx: int) -> str:
    """Build context: last 3 teacher+student exchanges before the target turn."""
    turns = parse_dialog(dialog_text)
    # Find the target user turn (by counting user turns with intents)
    user_count = 0
    target_pos = -1
    for i, t in enumerate(turns):
        if t["role"] == "user":
            user_count += 1
            if t["intent"] is not None and user_count == target_turn_idx:
                target_pos = i
                break

    if target_pos < 0:
        return ""

    # Collect up to 6 preceding turns (3 exchanges = 3 assistant + 3 user)
    context_turns = turns[max(0, target_pos - 6):target_pos]
    lines = []
    for t in context_turns:
        role = "Ученик" if t["role"] == "user" else "Учитель"
        intent_tag = f" [{t['intent']}]" if t.get("intent") else ""
        lines.append(f"{role}{intent_tag}: {t['text']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# XLSX reading
# ---------------------------------------------------------------------------

def read_xlsx(path: str) -> list[dict]:
    """Read XLSX and return list of dialog records.

    The file has: row 1 = headers, row 2 = type descriptors, rows 3+ = data.
    We skip rows 1-2 and start from row 3.
    """
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active
    records = []
    for row_idx in range(3, ws.max_row + 1):
        row = [ws.cell(row=row_idx, column=c).value for c in range(1, ws.max_column + 1)]
        if len(row) < 19:
            continue
        dialog_text = row[18]  # col 19 (0-indexed: 18)
        if not dialog_text or not str(dialog_text).strip():
            continue
        # Skip the type-descriptor row (contains "string", "int64", etc.)
        if str(dialog_text).strip() in ("string", "any", "int64", "float64"):
            continue
        records.append({
            "dialog_idx": row_idx - 2,  # 1-based, skipping header+types
            "output": row[0],           # col 1
            "grade_group": str(row[2] or ""),   # col 3
            "task": str(row[4] or ""),          # col 5
            "task_id": str(row[8] or ""),       # col 9
            "dialog": str(dialog_text),         # col 19
        })
    wb.close()
    return records


# ---------------------------------------------------------------------------
# Gemini evaluation
# ---------------------------------------------------------------------------

EVAL_PROMPT_TEMPLATE = """\
Ты — эксперт по оценке качества синтетических диалогов «ученик-репетитор» по математике.

Тебе дан фрагмент диалога (контекст) и конкретная реплика ученика с объявленным интентом.

## Контекст диалога (последние реплики перед оцениваемой):
{context}

## Оцениваемая реплика:
- Реплика репетитора перед ней: {teacher_before}
- Реплика ученика: {student_text}
- Объявленный интент: [{intent}]

## Задача:
Определи, соответствует ли реплика ученика объявленному интенту [{intent}].

## Справка по интентам:

- **answer** — ученик отвечает на последний вопрос или задание репетитора. Это конкретный ответ (числовой результат, следующий шаг решения, ответ на вопрос по теме). Ответ может быть правильным или неправильным — это не важно. Важно что ученик ПЫТАЕТСЯ ответить, а не задаёт вопрос, не соглашается, не просит что-то.

- **get-explanation** — ученик НЕ решает и НЕ даёт ответ. Он не понимает что-то из последней реплики репетитора и задаёт уточняющий вопрос или просит объяснить иначе. Ключевое: опирается на конкретное место в реплике репетитора. НЕ пытается считать или решать.

- **get-solution** — ученик хочет, чтобы репетитор дал готовый ответ или решил за него. Просит показать решение, дать ответ. Может выражать нежелание думать, усталость, нетерпение. НЕ решает сам.

- **agree-with-tutor** — ученик коротко (одно-два слова) даёт знать, что воспринял сказанное репетитором. Это НЕ ответ на вопрос, НЕ объяснение, НЕ вопрос — просто подтверждение: «понял», «ок», «ага», «да».

- **chat** — ученику скучно, он уводит разговор от учёбы. Реплика не по теме математики — зацепился за что-то в словах репетитора и ушёл в сторону. НЕ решает задачу, НЕ отвечает по существу.

- **thank-tutor** — ученик благодарит репетитора за конкретное действие: объяснение, подсказку, исправление, пример. Не просто «спасибо» в пустоту, а за что-то конкретное.

- **set-problem** — ученик ИГНОРИРУЕТ текущий вопрос репетитора и предлагает свою задачу или тему. Придумывает конкретную задачу по математике и просит разобрать.

- **criticize-tutor** — ученик выражает недовольство репликой репетитора: непонятное объяснение, слишком сложно, скучная подача, раздражает тон. НЕ решает задачу, НЕ отвечает на вопросы.

## Правила оценки:
- Для **answer**: ученик должен именно ПЫТАТЬСЯ дать ответ (правильный или нет). Если вместо этого он задаёт вопрос, соглашается или просит что-то — это НЕ answer.
- Для **get-explanation** vs **get-solution**: get-explanation — просит ОБЪЯСНИТЬ непонятное; get-solution — просит ДАТЬ ГОТОВЫЙ ОТВЕТ. Разная мотивация.
- Для **agree-with-tutor**: максимально короткая реплика-подтверждение. Если ученик при этом пытается решать — это answer, не agree.

Ответь строго в формате JSON (без markdown-блоков):
{{"match": true/false, "confidence": 0.0-1.0, "reason": "краткое объяснение", "actual_intent": "интент если отличается, иначе пустая строка"}}
"""


async def evaluate_turn(
    client: genai.Client,
    turn: UserTurn,
    dialog_text: str,
    semaphore: asyncio.Semaphore,
    model: str = "gemini-3-flash-preview",
    max_retries: int = 3,
) -> EvalResult:
    """Evaluate a single user turn via Gemini."""
    context = build_context_window(dialog_text, turn.turn_idx)

    prompt = EVAL_PROMPT_TEMPLATE.format(
        context=context or "(начало диалога)",
        teacher_before=turn.teacher_text_before or "(нет)",
        student_text=turn.student_text,
        intent=turn.intent_declared,
    )

    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=2048,
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )

                text = response.text.strip()
                # Clean potential markdown wrapping
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)

                data = json.loads(text)

                return EvalResult(
                    dialog_idx=turn.dialog_idx,
                    turn_idx=turn.turn_idx,
                    intent_declared=turn.intent_declared,
                    student_text=turn.student_text,
                    teacher_text_before=turn.teacher_text_before,
                    match=bool(data.get("match", False)),
                    confidence=float(data.get("confidence", 0.0)),
                    reason=str(data.get("reason", "")),
                    actual_intent=str(data.get("actual_intent", "")),
                    grade_group=turn.grade_group,
                    task_id=turn.task_id,
                )

            except json.JSONDecodeError as e:
                # Bad JSON from model — retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return EvalResult(
                    dialog_idx=turn.dialog_idx,
                    turn_idx=turn.turn_idx,
                    intent_declared=turn.intent_declared,
                    student_text=turn.student_text,
                    teacher_text_before=turn.teacher_text_before,
                    match=False,
                    confidence=0.0,
                    reason=f"JSON parse error: {e}; raw: {text[:200]}",
                    actual_intent="",
                    grade_group=turn.grade_group,
                    task_id=turn.task_id,
                )

            except Exception as e:
                err_name = type(e).__name__
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"\n  Retry {attempt+1}/{max_retries} for dialog {turn.dialog_idx} "
                          f"turn {turn.turn_idx} ({err_name}), waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                return EvalResult(
                    dialog_idx=turn.dialog_idx,
                    turn_idx=turn.turn_idx,
                    intent_declared=turn.intent_declared,
                    student_text=turn.student_text,
                    teacher_text_before=turn.teacher_text_before,
                    match=False,
                    confidence=0.0,
                    reason=f"API error after {max_retries} retries: {err_name}: {e}",
                    actual_intent="",
                    grade_group=turn.grade_group,
                    task_id=turn.task_id,
                )

    # Should not reach here
    raise RuntimeError("Unreachable")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(
    input_path: str,
    output_path: str,
    sample_n: int | None = None,
    concurrency: int = 5,
):
    # Validate API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable is not set.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Step 1: Read XLSX
    print(f"Reading {input_path}...")
    records = read_xlsx(input_path)
    print(f"  Found {len(records)} dialogs.")

    # Step 2: Parse all user turns
    all_turns: list[tuple[UserTurn, str]] = []  # (turn, dialog_text)
    for rec in records:
        turns = extract_user_turns(
            rec["dialog"], rec["dialog_idx"], rec["grade_group"], rec["task_id"]
        )
        for t in turns:
            all_turns.append((t, rec["dialog"]))

    print(f"  Extracted {len(all_turns)} user turns with intent tags.")

    # Step 3: Sample if requested
    if sample_n is not None and sample_n < len(all_turns):
        random.seed(42)
        all_turns = random.sample(all_turns, sample_n)
        print(f"  Sampled {sample_n} turns for evaluation.")

    # Step 4: Evaluate with Gemini
    semaphore = asyncio.Semaphore(concurrency)
    print(f"Evaluating {len(all_turns)} turns (concurrency={concurrency})...")

    pbar = tqdm(total=len(all_turns), desc="Evaluating", unit="turn")
    results: list[EvalResult] = []

    async def eval_with_progress(turn: UserTurn, dialog: str) -> EvalResult:
        result = await evaluate_turn(client, turn, dialog, semaphore)
        pbar.update(1)
        return result

    tasks = [
        eval_with_progress(turn, dialog)
        for turn, dialog in all_turns
    ]
    results = await asyncio.gather(*tasks)
    pbar.close()

    # Step 5: Write CSV
    print(f"Writing results to {output_path}...")
    fieldnames = [
        "dialog_idx", "turn_idx", "intent_declared", "student_text",
        "teacher_text_before", "match", "confidence", "reason",
        "actual_intent", "grade_group", "task_id",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "dialog_idx": r.dialog_idx,
                "turn_idx": r.turn_idx,
                "intent_declared": r.intent_declared,
                "student_text": r.student_text,
                "teacher_text_before": r.teacher_text_before,
                "match": r.match,
                "confidence": r.confidence,
                "reason": r.reason,
                "actual_intent": r.actual_intent,
                "grade_group": r.grade_group,
                "task_id": r.task_id,
            })

    # Summary stats
    total = len(results)
    matched = sum(1 for r in results if r.match)
    errors = sum(1 for r in results if "error" in r.reason.lower())
    avg_conf = sum(r.confidence for r in results) / total if total else 0

    print(f"\nDone! Results: {output_path}")
    print(f"  Total turns evaluated: {total}")
    print(f"  Match: {matched}/{total} ({100*matched/total:.1f}%)")
    print(f"  Avg confidence: {avg_conf:.3f}")
    if errors:
        print(f"  Errors: {errors}")

    # Per-intent breakdown
    intent_stats: dict[str, dict] = {}
    for r in results:
        key = r.intent_declared
        if key not in intent_stats:
            intent_stats[key] = {"total": 0, "match": 0}
        intent_stats[key]["total"] += 1
        if r.match:
            intent_stats[key]["match"] += 1

    print("\n  Per-intent breakdown:")
    for intent, stats in sorted(intent_stats.items()):
        pct = 100 * stats["match"] / stats["total"] if stats["total"] else 0
        print(f"    [{intent}]: {stats['match']}/{stats['total']} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate intent compliance of synthetic student turns via Gemini."
    )
    parser.add_argument(
        "--input", required=True, help="Path to XLSX file with synthetic dialogs."
    )
    parser.add_argument(
        "--output", required=True, help="Path for output CSV file."
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Evaluate only N random turns (for testing)."
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max concurrent Gemini API calls (default: 5)."
    )
    args = parser.parse_args()

    asyncio.run(run_pipeline(
        input_path=args.input,
        output_path=args.output,
        sample_n=args.sample,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()
