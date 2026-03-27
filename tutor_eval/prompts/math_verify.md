# Шаг 2: Генерация sympy-кода для верификации математического утверждения

Ты — эксперт по sympy. Тебе дано математическое утверждение из диалога тьютора. Напиши Python-код, который проверяет его корректность.

## Контекст

**Задача из диалога:** {task}
**Класс:** {grade_group}

## Утверждение для проверки

**Цитата тьютора:** {quote}
**Формализация:** {math_expression}
**Тип:** {type}
**Описание:** {description}

## Правила написания кода

Код должен присвоить переменной `result` значение `True` (утверждение верно) или `False` (неверно).

### Что доступно в namespace (НЕ пиши import!)

Все имена из sympy уже доступны: `symbols`, `Symbol`, `Rational`, `Integer`, `sqrt`, `sin`, `cos`, `tan`, `log`, `exp`, `pi`, `E`, `oo`, `simplify`, `expand`, `factor`, `diff`, `integrate`, `solve`, `solveset`, `Eq`, `Matrix`, `N`, `FiniteSet`, `Abs`, `binomial`, `factorial`, `gcd`, `lcm`, и т.д.

Также доступны: `math` (Python), `Fraction` (из fractions), базовые built-in (`int`, `float`, `set`, `list`, `abs`, `round`, `len`, `all`, `any`, `sorted`, `sum`, `min`, `max`).

### КРИТИЧНО: правила сравнения

**1. Чистая арифметика (только числа, без символов)**
```python
result = (720 / 4 == 180)
result = (25 * 3.6 == 90.0)
```

**2. Дроби — используй Rational для точности**
```python
result = (Rational(360, 100) == Rational(18, 5))  # НЕ сравнивай Rational с float!
# Или через чистый Python:
result = (360 / 100 == 3.6)
```

**3. Символьные равенства — ВСЕГДА через simplify**
```python
x = symbols('x')
# ПРАВИЛЬНО:
result = simplify((4 + x)*x - (4*x + x**2)) == 0
# НЕПРАВИЛЬНО (вернёт False!):
# result = (4 + x)*x == 4*x + x**2
```

**4. Степенные тождества — указывай assumptions**
```python
# (a^m)^n = a^(m*n) верно только для positive a
a = symbols('a', positive=True)
m, n = symbols('m n')
result = simplify((a**m)**n - a**(m*n)) == 0
```

**5. Логарифмы**
```python
# log(a, b) — логарифм a по основанию b
result = simplify(log(8, 2) - 3) == 0
# Тождества — через simplify
result = simplify(log(2, 12) - 1/log(12, 2)) == 0
```

**6. Производные**
```python
x = symbols('x')
f = (3*x - 7)**Rational(-1, 3)
result = simplify(diff(f, x) - (-(3*x - 7)**Rational(-4, 3))) == 0
```

**7. Решение уравнений**
```python
x = symbols('x')
result = set(solve(x**2 - 5*x + 6, x)) == {2, 3}
```

**8. Числовые сравнения**
```python
result = (Rational(3, 7) > Rational(2, 5))
```

**9. Тригонометрия**
```python
result = simplify(sin(pi/6) - Rational(1, 2)) == 0
```

### Частые ошибки (НЕ делай так!)

```python
# ОШИБКА: Rational vs float
result = (Rational(360, 100) == 3.6)  # False!
# ИСПРАВЛЕНИЕ:
result = (360 / 100 == 3.6)  # или Rational(360,100) == Rational(36,10)

# ОШИБКА: символьное == без simplify
x = symbols('x')
result = ((x+1)**2 == x**2 + 2*x + 1)  # False!
# ИСПРАВЛЕНИЕ:
result = simplify((x+1)**2 - (x**2 + 2*x + 1)) == 0

# ОШИБКА: свободный символ вместо подстановки
S, a, b = symbols('S a b')
result = simplify(S - a*b) == 0  # False! S — независимый символ
# ИСПРАВЛЕНИЕ: это определение, не вычислимый факт — не нужно проверять
```

### LaTeX → sympy

- `\frac{a}{b}` → `Rational(a, b)` (для числовых) или `a / b` (для символьных)
- `\sqrt{x}` → `sqrt(x)`
- `\sqrt[3]{x}` → `x**Rational(1, 3)` или `cbrt(x)`
- `\log_a b` → `log(b, a)`
- `\sin x` → `sin(x)`

## Формат ответа

Ответь строго в формате JSON:

```json
{
  "sympy_check": "<Python-код, многострочный если нужно>",
  "reasoning": "<1 предложение: как именно проверяешь>"
}
```

ВАЖНО:
- НЕ пиши import — всё уже доступно
- Код ОБЯЗАН присвоить `result` (bool)
- Ответь ТОЛЬКО JSON
