"""Judge logic: build judge prompt from evaluator results and parse decision."""

from tutor_eval.config import (
    CRITERIA,
    CRITICAL_FLAGS,
    RUBRICS_PATH,
    JUDGE_PROMPT_PATH,
    JUDGE_CRITS_PROMPT_PATH,
)
from tutor_eval.loader import Dialog
from tutor_eval.providers.base import EvalResult


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_evaluator_scores(result: EvalResult, only_criteria: list[str] | None = None) -> str:
    """Format a single evaluator's scores for the judge prompt."""
    criteria_to_format = only_criteria if only_criteria else CRITERIA
    lines = []
    for criterion in criteria_to_format:
        entry = result.scores.get(criterion, {})
        if not isinstance(entry, dict):
            entry = {}
        score = entry.get("score", "?")
        reasoning = entry.get("reasoning", "")
        evidence = entry.get("evidence", [])
        lines.append(f"**{criterion}**: {score}/3")
        lines.append(f"  Обоснование: {reasoning}")
        if evidence:
            for e in evidence[:2]:
                lines.append(f"  Цитата: \u00ab{e[:200]}\u00bb")
        lines.append("")
    notes = result.overall_notes
    if notes:
        lines.append(f"Общие замечания: {notes}")
    return "\n".join(lines)


def _format_evaluator_crits(result: EvalResult) -> str:
    """Format critical flags from a single evaluator."""
    crits = result.critical_flags
    if not crits:
        return "(нет флагов)"
    lines = []
    for flag_name in CRITICAL_FLAGS:
        entry = crits.get(flag_name, {})
        if not isinstance(entry, dict):
            continue
        found = entry.get("found", False)
        if found:
            evidence = entry.get("evidence", "")
            reasoning = entry.get("reasoning", "")
            category = entry.get("category", "")
            cat_label = f" (категория: {category})" if category else ""
            lines.append(f"**{flag_name}**: НАЙДЕН{cat_label}")
            if evidence:
                lines.append(f"  Цитата: \u00ab{evidence[:300]}\u00bb")
            if reasoning:
                lines.append(f"  Обоснование: {reasoning}")
            lines.append("")
    return "\n".join(lines) if lines else "(нет флагов)"


# ---------------------------------------------------------------------------
# Criteria classification
# ---------------------------------------------------------------------------

def _classify_criteria(results: list[EvalResult]) -> tuple[dict, dict, list[str]]:
    """Classify each criterion as unanimous or disputed."""
    final_scores = {}
    agreement = {}
    disputed = []

    for criterion in CRITERIA:
        scores = []
        for r in results:
            entry = r.scores.get(criterion, {})
            if not isinstance(entry, dict):
                continue
            s = entry.get("score")
            if s is not None:
                scores.append(int(s))

        if len(scores) < 2:
            disputed.append(criterion)
            continue

        if len(set(scores)) == 1:
            final_scores[criterion] = {
                "score": scores[0],
                "reasoning": f"Все {len(scores)} оценщика единогласны: {scores[0]}/3",
            }
            agreement[criterion] = "unanimous"
        else:
            disputed.append(criterion)

    return final_scores, agreement, disputed


# ---------------------------------------------------------------------------
# Critical flags collection
# ---------------------------------------------------------------------------

def _collect_critical_flags(results: list[EvalResult]) -> dict[str, list[dict]]:
    """Collect critical flags from all evaluators.

    Returns: {flag_name: [list of evaluator entries that flagged it]}
    """
    flagged = {}
    for flag_name in CRITICAL_FLAGS:
        entries = []
        for r in results:
            crits = r.critical_flags
            if not crits or not isinstance(crits, dict):
                continue
            entry = crits.get(flag_name, {})
            if isinstance(entry, dict) and entry.get("found", False):
                entries.append({
                    "evaluator": r.evaluator,
                    "evidence": entry.get("evidence", ""),
                    "reasoning": entry.get("reasoning", ""),
                    "category": entry.get("category"),
                })
        if entries:
            flagged[flag_name] = entries
    return flagged


def _build_crits_judge_prompt(dialog: Dialog, results: list[EvalResult], flagged: dict) -> str:
    """Build prompt for judge to confirm/reject critical flags."""
    template = JUDGE_CRITS_PROMPT_PATH.read_text(encoding="utf-8")

    # Format evaluator flags
    labels = ["A", "B", "C"]
    flag_sections = []
    for i, result in enumerate(results):
        label = labels[i] if i < len(labels) else f"#{i+1}"
        flag_sections.append(f"### Оценщик {label} ({result.evaluator})")
        flag_sections.append(_format_evaluator_crits(result))

    prompt = (
        template
        .replace("{task}", dialog.task)
        .replace("{grade_group}", dialog.grade_group)
        .replace("{theme}", f"{dialog.theme} \u2192 {dialog.subtheme}")
        .replace("{dialog}", dialog.text)
        .replace("{evaluator_flags}", "\n\n".join(flag_sections))
    )
    return prompt


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_judge_prompt(dialog: Dialog, results: list[EvalResult], only_criteria: list[str] | None = None) -> str:
    """Build the full prompt for Claude judge."""
    rubrics = RUBRICS_PATH.read_text(encoding="utf-8")
    template = JUDGE_PROMPT_PATH.read_text(encoding="utf-8")

    labels = ["A", "B", "C"]
    evaluator_sections = {}
    for i, result in enumerate(results):
        label = labels[i] if i < len(labels) else f"#{i+1}"
        key = f"evaluator_{label.lower()}"
        evaluator_sections[key] = (
            f"({result.evaluator}, model={result.model})\n"
            + _format_evaluator_scores(result, only_criteria)
        )

    prompt = (
        template
        .replace("{rubrics}", rubrics)
        .replace("{task}", dialog.task)
        .replace("{grade_group}", dialog.grade_group)
        .replace("{theme}", f"{dialog.theme} \u2192 {dialog.subtheme}")
        .replace("{dialog}", dialog.text)
    )
    for key, value in evaluator_sections.items():
        prompt = prompt.replace(f"{{{key}}}", value)

    if only_criteria:
        criteria_list = ", ".join(only_criteria)
        prompt += (
            f"\n\nВАЖНО: Оцени ТОЛЬКО следующие критерии (по остальным уже есть единогласие): {criteria_list}. "
            f"В JSON включи только эти критерии."
        )

    return prompt


# ---------------------------------------------------------------------------
# Main judge logic
# ---------------------------------------------------------------------------

async def run_judge(
    dialog: Dialog,
    eval_results: list[EvalResult],
    judge_provider,
) -> dict:
    """Run the judge on evaluator results.

    1. Collect critical flags — if any flagged, send to judge for confirmation
    2. Classify criteria as unanimous/disputed
    3. Send disputed criteria to judge
    4. If any crit confirmed — override adequacy to 0
    """
    # Step 1: Critical flags
    flagged = _collect_critical_flags(eval_results)
    confirmed_crits = {}

    if flagged:
        crits_prompt = _build_crits_judge_prompt(dialog, eval_results, flagged)
        crits_decision = await judge_provider.judge(crits_prompt)
        judge_crits = crits_decision.get("critical_flags", {})

        for flag_name in CRITICAL_FLAGS:
            entry = judge_crits.get(flag_name, {})
            if isinstance(entry, dict) and entry.get("verdict") == "confirmed":
                confirmed_crits[flag_name] = {
                    "verdict": "confirmed",
                    "category": entry.get("category"),
                    "reasoning": entry.get("reasoning", ""),
                }

    # Step 2: Classify criteria
    unanimous_scores, unanimous_agreement, disputed = _classify_criteria(eval_results)

    # Step 3: Judge disputed criteria (if any)
    if not disputed:
        decision = {
            "final_scores": unanimous_scores,
            "agreement": unanimous_agreement,
            "overrides": [],
            "_meta": {
                "model": judge_provider.model_id,
                "tokens_used": 0,
                "latency_ms": 0,
                "shortcut": "unanimous" if not flagged else "unanimous_with_crits",
            },
        }
    else:
        prompt = build_judge_prompt(dialog, eval_results, only_criteria=disputed)
        decision = await judge_provider.judge(prompt)

        final_scores = {**unanimous_scores}
        agreement = {**unanimous_agreement}
        overrides = decision.get("overrides", [])

        judge_finals = decision.get("final_scores", {})
        judge_agreement = decision.get("agreement", {})

        for c in disputed:
            if c in judge_finals:
                final_scores[c] = judge_finals[c]
                agreement[c] = judge_agreement.get(c, "split")

        decision["final_scores"] = final_scores
        decision["agreement"] = agreement
        decision["overrides"] = overrides

    # Step 4: If crits confirmed — override adequacy to 0
    if confirmed_crits:
        crit_names = ", ".join(confirmed_crits.keys())
        decision["final_scores"]["adequacy"] = {
            "score": 0,
            "reasoning": f"Автоматический 0: подтверждены критические сбои ({crit_names})",
        }
        decision.setdefault("overrides", []).append(
            f"Адекватность принудительно обнулена из-за критов: {crit_names}"
        )

    # Attach crits to decision
    decision["critical_flags"] = {
        "flagged": {k: v for k, v in flagged.items()},
        "confirmed": confirmed_crits,
    }

    return decision
