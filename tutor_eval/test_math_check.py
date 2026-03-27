"""Quick smoke test for math_check module — verifies sympy sandbox on hand-crafted claims."""

from tutor_eval.math_check import MathClaim, verify_claims, _run_sympy_check


def test_sympy_sandbox():
    """Test that the sympy sandbox can execute basic checks."""
    # Simple arithmetic
    ok, err, out = _run_sympy_check("result = (720 / 4 == 180)")
    assert ok is True, f"720/4=180 failed: {err}"

    # Sympy Rational
    ok, err, out = _run_sympy_check("result = (Rational(3,7) > Rational(2,5))")
    assert ok is True, f"3/7 > 2/5 failed: {err}"

    # Derivative via simplify
    ok, err, out = _run_sympy_check("""
x = symbols('x')
result = simplify(diff(sin(x), x) - cos(x)) == 0
""")
    assert ok is True, f"d/dx sin(x) = cos(x) failed: {err}"

    # Symbolic identity via simplify
    ok, err, out = _run_sympy_check("""
x = symbols('x')
result = simplify((4 + x)*x - (4*x + x**2)) == 0
""")
    assert ok is True, f"(4+x)*x = 4x+x^2 failed: {err}"

    # Power identity with positive base and real exponents
    ok, err, out = _run_sympy_check("""
a = symbols('a', positive=True)
m, n = symbols('m n', real=True)
result = simplify((a**m)**n - a**(m*n)) == 0
""")
    assert ok is True, f"(a^m)^n = a^(mn) with positive/real failed: {err}"

    # Incorrect claim should return False
    ok, err, out = _run_sympy_check("result = (2 + 2 == 5)")
    assert ok is False, f"2+2=5 should be False, got {ok}"

    # Bad code should return None + error
    ok, err, out = _run_sympy_check("result = undefined_var")
    assert ok is None, f"Bad code should give None, got {ok}"
    assert err is not None

    # Import stripping
    ok, err, out = _run_sympy_check("from sympy import Rational; result = Rational(720, 4) == 180")
    assert ok is True, f"Import stripping failed: {err}"

    print("All sandbox tests passed!")


def test_verify_claims():
    """Test verify_claims on a batch of claims."""
    claims = [
        MathClaim(
            quote="720 ÷ 4 = 180",
            math_expression="720 / 4 = 180",
            claim_type="arithmetic",
            description="Деление расстояния на время",
            sympy_check="result = (720 / 4 == 180)",
        ),
        MathClaim(
            quote="180 × 6 = 1080",
            math_expression="180 * 6 = 1080",
            claim_type="arithmetic",
            description="Скорость на время",
            sympy_check="result = (180 * 6 == 1080)",
        ),
        MathClaim(
            quote="25% × 3.6° = 90°",
            math_expression="25 * 3.6 = 90",
            claim_type="arithmetic",
            description="Проценты в градусы",
            sympy_check="result = (25 * 3.6 == 90.0)",
        ),
        MathClaim(
            quote="Производная (3x-7)^{-1/3} = -(3x-7)^{-4/3}",
            math_expression="d/dx (3x-7)^(-1/3) = -(3x-7)^(-4/3)",
            claim_type="derivative",
            description="Производная сложной функции",
            sympy_check="""
x = symbols('x')
f = (3*x - 7)**Rational(-1, 3)
deriv = diff(f, x)
expected = -1 * (3*x - 7)**Rational(-4, 3)
result = simplify(deriv - expected) == 0
""",
        ),
    ]

    results = verify_claims(claims)
    for v in results:
        status = "ok" if v.is_correct else ("FAIL" if v.is_correct is False else "ERR")
        print(f"  [{status}] {v.claim.description}: is_correct={v.is_correct}, error={v.error}")

    correct = sum(1 for v in results if v.is_correct is True)
    print(f"\n{correct}/{len(results)} claims verified as correct")


if __name__ == "__main__":
    test_sympy_sandbox()
    print()
    test_verify_claims()
