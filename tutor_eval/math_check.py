"""Math correctness checker: two-stage pipeline.

Stage 1 (extract): LLM reads dialog → outputs math claims (quote + formalized expression)
Stage 2 (verify):  LLM generates sympy code for each claim → sandbox executes it
"""

import asyncio
import json
import re as _re
import time
from dataclasses import dataclass
from pathlib import Path

from tutor_eval.config import (
    DATA_DIR,
    LOGS_DIR,
    RESULTS_DIR,
)
from tutor_eval.loader import Dialog
from tutor_eval.providers.json_utils import parse_json_response

MATH_EXTRACT_PROMPT_PATH = Path(__file__).parent / "prompts" / "math_extract.md"
MATH_VERIFY_PROMPT_PATH = Path(__file__).parent / "prompts" / "math_verify.md"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class MathClaim:
    """A single verifiable math claim extracted from a dialog."""
    quote: str
    math_expression: str
    claim_type: str  # arithmetic, algebraic_identity, equation_solution, ...
    description: str
    error_class: str = "none"  # tutor_error, accepted_incorrect, rejected_correct, none
    sympy_check: str = ""  # filled by stage 2


@dataclass
class MathVerification:
    """Result of verifying a single claim with sympy."""
    claim: MathClaim
    is_correct: bool | None  # None = execution error
    error: str | None  # sympy execution error if any
    actual_result: str  # what sympy returned
    codegen_reasoning: str = ""  # LLM's reasoning about how it verifies


@dataclass
class DialogMathResult:
    """All math verification results for one dialog."""
    dialog_id: str
    claims_count: int
    correct_count: int
    incorrect_count: int
    error_count: int
    no_claims_reason: str  # if no claims found
    verifications: list[MathVerification]
    extractor_model: str
    codegen_model: str
    latency_ms: int


# ---------------------------------------------------------------------------
# Sympy execution sandbox
# ---------------------------------------------------------------------------

def _run_sympy_check(code: str) -> tuple[bool | None, str | None, str]:
    """Execute sympy check code and return (result, error, actual_output).

    Runs in a restricted namespace with only sympy and math available.
    """
    # Strip import statements — we pre-inject everything into the namespace.
    cleaned_lines = []
    for line in code.splitlines():
        parts = [p.strip() for p in line.split(";")]
        kept = [p for p in parts if p and not _re.match(r"(from\s+\S+\s+import|import\s+)", p)]
        if kept:
            cleaned_lines.append("; ".join(kept))
    code = "\n".join(cleaned_lines)

    namespace = {"__builtins__": {}}

    # Pre-import sympy into the namespace
    import sympy
    import math as math_module
    from fractions import Fraction
    namespace["sympy"] = sympy
    namespace["math"] = math_module
    namespace["Fraction"] = Fraction

    # Inject commonly used sympy names
    for name in [
        "symbols", "Symbol", "Rational", "Integer", "Float",
        "sqrt", "cbrt", "Abs", "sign",
        "sin", "cos", "tan", "cot", "asin", "acos", "atan", "atan2",
        "log", "ln", "exp",
        "pi", "E", "I", "oo",
        "simplify", "expand", "factor", "collect", "cancel",
        "diff", "integrate", "limit", "series",
        "solve", "solveset", "Eq", "Ne", "Lt", "Gt", "Le", "Ge",
        "Matrix", "det",
        "S", "Poly",
        "trigsimp", "radsimp", "powsimp",
        "nsimplify", "N",
        "Sum", "Product",
        "binomial", "factorial",
        "gcd", "lcm",
        "FiniteSet",
        "true", "false",
    ]:
        if hasattr(sympy, name):
            namespace[name] = getattr(sympy, name)

    # Allow basic builtins
    safe_builtins = {
        "True": True, "False": False, "None": None,
        "int": int, "float": float, "str": str, "bool": bool,
        "list": list, "tuple": tuple, "set": set, "dict": dict,
        "len": len, "abs": abs, "min": min, "max": max,
        "range": range, "enumerate": enumerate, "zip": zip,
        "sorted": sorted, "sum": sum, "all": all, "any": any,
        "round": round, "pow": pow,
        "isinstance": isinstance, "type": type,
        "print": print,
        "ValueError": ValueError, "TypeError": TypeError,
    }
    namespace["__builtins__"] = safe_builtins

    try:
        exec(code, namespace)
        result = namespace.get("result")
        if result is None:
            return None, "Variable 'result' not set by check code", ""
        return bool(result), None, str(result)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", ""


def verify_claims(claims: list[MathClaim]) -> list[MathVerification]:
    """Run pre-filled sympy_check code for a list of claims (no LLM needed)."""
    results = []
    for claim in claims:
        if not claim.sympy_check.strip():
            results.append(MathVerification(
                claim=claim, is_correct=None,
                error="Empty sympy_check code", actual_result="",
            ))
            continue
        is_correct, error, actual = _run_sympy_check(claim.sympy_check)
        results.append(MathVerification(
            claim=claim, is_correct=is_correct, error=error, actual_result=actual,
        ))
    return results


# ---------------------------------------------------------------------------
# Stage 1: Extract claims
# ---------------------------------------------------------------------------

def build_extract_prompt(dialog: Dialog) -> str:
    """Build the prompt for extracting math claims from a dialog."""
    template = MATH_EXTRACT_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template
        .replace("{task}", dialog.task)
        .replace("{grade_group}", dialog.grade_group or "не указан")
        .replace("{theme}", f"{dialog.theme} → {dialog.subtheme}")
        .replace("{dialog}", dialog.text)
    )


def _parse_claims(raw_json: dict) -> list[MathClaim]:
    """Parse extracted claims from stage 1 JSON response."""
    claims = []
    for item in raw_json.get("claims", []):
        claims.append(MathClaim(
            quote=item.get("quote", ""),
            math_expression=item.get("math_expression", ""),
            claim_type=item.get("type", "other"),
            description=item.get("description", ""),
            error_class=item.get("error_class", "none"),
        ))
    return claims


async def extract_claims(dialog: Dialog, provider) -> tuple[list[MathClaim], str]:
    """Stage 1: extract math claims from dialog.

    Returns (claims, no_claims_reason).
    """
    prompt = build_extract_prompt(dialog)
    raw_response = await provider.evaluate(prompt)
    raw_text = raw_response.raw_response if hasattr(raw_response, "raw_response") else str(raw_response)

    try:
        parsed = parse_json_response(raw_text)
    except Exception:
        return [], "Failed to parse JSON from extractor response"

    claims = _parse_claims(parsed)
    no_claims_reason = parsed.get("no_claims_reason", "") if not claims else ""
    return claims, no_claims_reason


# ---------------------------------------------------------------------------
# Stage 2: Generate sympy code for each claim
# ---------------------------------------------------------------------------

def build_verify_prompt(claim: MathClaim, task: str, grade_group: str) -> str:
    """Build the prompt for generating sympy verification code."""
    template = MATH_VERIFY_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        template
        .replace("{task}", task)
        .replace("{grade_group}", grade_group or "не указан")
        .replace("{quote}", claim.quote)
        .replace("{math_expression}", claim.math_expression)
        .replace("{type}", claim.claim_type)
        .replace("{description}", claim.description)
    )


async def generate_and_verify(
    claim: MathClaim,
    task: str,
    grade_group: str,
    provider,
) -> MathVerification:
    """Stage 2: generate sympy code for a claim and execute it."""
    prompt = build_verify_prompt(claim, task, grade_group)
    raw_response = await provider.evaluate(prompt)
    raw_text = raw_response.raw_response if hasattr(raw_response, "raw_response") else str(raw_response)

    try:
        parsed = parse_json_response(raw_text)
    except Exception:
        return MathVerification(
            claim=claim,
            is_correct=None,
            error=f"Failed to parse codegen JSON",
            actual_result="",
        )

    sympy_code = parsed.get("sympy_check", "")
    reasoning = parsed.get("reasoning", "")
    claim.sympy_check = sympy_code

    if not sympy_code.strip():
        return MathVerification(
            claim=claim,
            is_correct=None,
            error="Empty sympy_check from codegen",
            actual_result="",
            codegen_reasoning=reasoning,
        )

    is_correct, error, actual = _run_sympy_check(sympy_code)
    return MathVerification(
        claim=claim,
        is_correct=is_correct,
        error=error,
        actual_result=actual,
        codegen_reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

async def check_dialog_math(
    dialog: Dialog,
    extractor_provider,
    codegen_provider=None,
) -> DialogMathResult:
    """Two-stage math correctness check for a dialog.

    Stage 1: extractor_provider extracts claims from dialog text.
    Stage 2: codegen_provider generates sympy code per claim & verifies.

    If codegen_provider is None, uses extractor_provider for both stages.
    """
    if codegen_provider is None:
        codegen_provider = extractor_provider

    t0 = time.time()

    # Stage 1: extract claims
    claims, no_claims_reason = await extract_claims(dialog, extractor_provider)

    if not claims:
        return DialogMathResult(
            dialog_id=dialog.dialog_id,
            claims_count=0,
            correct_count=0,
            incorrect_count=0,
            error_count=0,
            no_claims_reason=no_claims_reason,
            verifications=[],
            extractor_model=getattr(extractor_provider, "model_id", "unknown"),
            codegen_model=getattr(codegen_provider, "model_id", "unknown"),
            latency_ms=int((time.time() - t0) * 1000),
        )

    # Stage 2: generate code + verify (parallel per claim)
    verify_tasks = [
        generate_and_verify(claim, dialog.task, dialog.grade_group, codegen_provider)
        for claim in claims
    ]
    verifications = await asyncio.gather(*verify_tasks)

    correct = sum(1 for v in verifications if v.is_correct is True)
    incorrect = sum(1 for v in verifications if v.is_correct is False)
    errors = sum(1 for v in verifications if v.is_correct is None)

    return DialogMathResult(
        dialog_id=dialog.dialog_id,
        claims_count=len(claims),
        correct_count=correct,
        incorrect_count=incorrect,
        error_count=errors,
        no_claims_reason="",
        verifications=list(verifications),
        extractor_model=getattr(extractor_provider, "model_id", "unknown"),
        codegen_model=getattr(codegen_provider, "model_id", "unknown"),
        latency_ms=int((time.time() - t0) * 1000),
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def result_to_dict(r: DialogMathResult) -> dict:
    """Serialize DialogMathResult to a JSON-compatible dict."""
    return {
        "dialog_id": r.dialog_id,
        "claims_count": r.claims_count,
        "correct_count": r.correct_count,
        "incorrect_count": r.incorrect_count,
        "error_count": r.error_count,
        "no_claims_reason": r.no_claims_reason,
        "extractor_model": r.extractor_model,
        "codegen_model": r.codegen_model,
        "latency_ms": r.latency_ms,
        "verifications": [
            {
                "quote": v.claim.quote,
                "math_expression": v.claim.math_expression,
                "sympy_check": v.claim.sympy_check,
                "type": v.claim.claim_type,
                "error_class": v.claim.error_class,
                "description": v.claim.description,
                "is_correct": v.is_correct,
                "error": v.error,
                "actual_result": v.actual_result,
                "codegen_reasoning": v.codegen_reasoning,
            }
            for v in r.verifications
        ],
    }
