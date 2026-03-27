"""Run math correctness check on dialog files — extract claims via LLM, verify with sympy."""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from tutor_eval.config import DATA_DIR, RESULTS_DIR
from tutor_eval.loader import Dialog
from tutor_eval.math_check import check_dialog_math, result_to_dict
from tutor_eval.providers.gemini_provider import OpenRouterEvalProvider


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EXTRACTOR_MODEL = "google/gemini-3-flash-preview"
EXTRACTOR_NAME = "gemini-3-flash"
CODEGEN_MODEL = "google/gemini-3-flash-preview"
CODEGEN_NAME = "gemini-3-flash"
DIALOG_CONCURRENCY = 3  # parallel dialogs (each spawns N parallel codegen calls)


# ---------------------------------------------------------------------------
# Load dialogs from the latest results run (dialogs.jsonl)
# ---------------------------------------------------------------------------

def load_dialogs_jsonl(path: Path) -> list[Dialog]:
    """Load dialogs from a JSONL file (as saved by the eval pipeline)."""
    dialogs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            dialogs.append(Dialog(
                dialog_id=d["dialog_id"],
                text=d["text"],
                task=d.get("task", ""),
                task_id=d.get("task_id", d["dialog_id"]),
                file_name=d.get("file_name", ""),
                student_type=d.get("student_type", ""),
                student_model=d.get("student_model", ""),
                grade_group=d.get("grade_group", ""),
                theme=d.get("theme", ""),
                subtheme=d.get("subtheme", ""),
                skill=d.get("skill", ""),
            ))
    return dialogs


def find_latest_dialogs_jsonl() -> Path | None:
    """Find the most recent dialogs.jsonl in results directory."""
    candidates = sorted(RESULTS_DIR.glob("*/dialogs.jsonl"), reverse=True)
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(dialogs: list[Dialog], output_dir: Path, limit: int | None = None):
    """Run math check on all dialogs and save results."""
    if limit:
        dialogs = dialogs[:limit]

    extractor = OpenRouterEvalProvider(
        model=EXTRACTOR_MODEL,
        name=EXTRACTOR_NAME,
        temperature=0.0,
    )
    codegen = OpenRouterEvalProvider(
        model=CODEGEN_MODEL,
        name=CODEGEN_NAME,
        temperature=0.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "math_check_results.jsonl"

    sem = asyncio.Semaphore(DIALOG_CONCURRENCY)
    total = len(dialogs)
    done = 0
    errors = 0
    t0 = time.time()

    async def process_one(dialog: Dialog) -> dict | None:
        nonlocal done, errors
        async with sem:
            try:
                result = await check_dialog_math(dialog, extractor, codegen)
                done += 1
                status = f"[{done}/{total}]"
                if result.incorrect_count > 0:
                    print(f"  {status} {dialog.dialog_id}: {result.incorrect_count} INCORRECT claims!")
                elif result.claims_count == 0:
                    print(f"  {status} {dialog.dialog_id}: no claims")
                else:
                    print(f"  {status} {dialog.dialog_id}: {result.correct_count}/{result.claims_count} correct")
                return result_to_dict(result)
            except Exception as e:
                done += 1
                errors += 1
                print(f"  [{done}/{total}] ERROR {dialog.dialog_id}: {e}")
                return {
                    "dialog_id": dialog.dialog_id,
                    "error": str(e),
                    "claims_count": 0,
                    "correct_count": 0,
                    "incorrect_count": 0,
                    "error_count": 0,
                }

    print(f"Running math check on {total} dialogs")
    print(f"  Extractor: {EXTRACTOR_MODEL}, Codegen: {CODEGEN_MODEL}")
    tasks = [process_one(d) for d in dialogs]
    results = await asyncio.gather(*tasks)

    # Write results
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            if r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Results: {out_path}")
    print(f"  Total: {total}, Errors: {errors}")

    # Quick summary
    total_claims = sum(r.get("claims_count", 0) for r in results if r)
    total_correct = sum(r.get("correct_count", 0) for r in results if r)
    total_incorrect = sum(r.get("incorrect_count", 0) for r in results if r)
    total_exec_errors = sum(r.get("error_count", 0) for r in results if r)
    print(f"  Claims: {total_claims}, Correct: {total_correct}, Incorrect: {total_incorrect}, Exec errors: {total_exec_errors}")
    if total_claims > 0:
        print(f"  Accuracy: {total_correct / total_claims * 100:.1f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Math correctness check for tutor dialogs")
    parser.add_argument("--input", type=str, help="Path to dialogs.jsonl (default: latest in results/)")
    parser.add_argument("--limit", type=int, default=None, help="Max dialogs to process")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: data/tutor_eval/results/math_<timestamp>)")
    args = parser.parse_args()

    # Find input
    if args.input:
        input_path = Path(args.input)
    else:
        input_path = find_latest_dialogs_jsonl()
        if not input_path:
            print("No dialogs.jsonl found in results/. Use --input to specify.")
            sys.exit(1)
    print(f"Input: {input_path}")

    dialogs = load_dialogs_jsonl(input_path)
    print(f"Loaded {len(dialogs)} dialogs")

    # Output dir
    if args.output:
        output_dir = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output_dir = RESULTS_DIR / f"math_{ts}"

    asyncio.run(run(dialogs, output_dir, args.limit))


if __name__ == "__main__":
    main()
