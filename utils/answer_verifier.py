"""Boxed-answer extraction and verification for FrugalOPL."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from fractions import Fraction

try:
    from math_verify import parse as parse_math
    from math_verify import verify as verify_math
except ImportError:
    parse_math = None
    verify_math = None


_BOXED_RE = re.compile(r"\\boxed\s*\{")
_CHOICE_LINE_RE = re.compile(r"(?m)^\s*([A-J])\s*[.)：:]\s+\S")
_STYLE_COMMAND_RE = re.compile(
    r"\\(?:text|textrm|textnormal|mathrm|mathbf|mathit|mathsf|mathtt|"
    r"textbf|operatorname|mbox)\s*\{"
)
_DECIMAL_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def _repair(text: object) -> str:
    """Normalize common formatting damage in model and dataset text."""

    value = "" if text is None else str(text)
    return (
        value.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
        .replace("\ufe68", "\\")
        .replace("\uff3c", "\\")
        .replace("\x08oxed", r"\boxed")
        .replace("\x0crac", r"\frac")
        .replace("d\x0crac", r"\dfrac")
        .replace("t\x0crac", r"\tfrac")
        .replace("−", "-")
        .replace("–", "-")
        .replace("＋", "+")
    )


def _is_escaped(text: str, position: int) -> bool:
    count = 0
    position -= 1
    while position >= 0 and text[position] == "\\":
        count += 1
        position -= 1
    return bool(count % 2)


def _matching_brace(text: str, opening: int) -> int | None:
    if opening >= len(text) or text[opening] != "{":
        return None
    depth = 1
    for position in range(opening + 1, len(text)):
        if _is_escaped(text, position):
            continue
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return position
    return None


def extract_final_answer(response: str) -> tuple[str, bool]:
    """Extract the final complete, non-empty boxed payload."""

    response = _repair(response)
    matches = list(_BOXED_RE.finditer(response))
    if not matches:
        return "", False
    brace_start = matches[-1].end() - 1
    brace_end = _matching_brace(response, brace_start)
    if brace_end is None:
        return "", False
    answer = response[brace_start + 1 : brace_end].strip()
    return answer, bool(answer)


def extract_boxed_answers(response: str) -> list[str]:
    """Return all complete boxed payloads in textual-marker order."""

    response = _repair(response)
    answers: list[str] = []
    for match in _BOXED_RE.finditer(response):
        brace_start = match.end() - 1
        brace_end = _matching_brace(response, brace_start)
        if brace_end is not None:
            answers.append(response[brace_start + 1 : brace_end].strip())
    return answers


def _strip_surrounding_markup(text: object) -> str:
    value = _repair(text).strip()
    changed = True
    while value and changed:
        changed = False
        for left, right in (
            ("$$", "$$"),
            (r"\[", r"\]"),
            (r"\(", r"\)"),
            ("$", "$"),
            ("**", "**"),
            ("__", "__"),
            ("\x60", "\x60"),
        ):
            if value.startswith(left) and value.endswith(right):
                value = value[len(left) : len(value) - len(right)].strip()
                changed = True
                break
        if value.startswith("{") and _matching_brace(value, 0) == len(value) - 1:
            value = value[1:-1].strip()
            changed = True
    return value


def _replace_style_commands(text: str) -> str:
    value = text
    while True:
        match = _STYLE_COMMAND_RE.search(value)
        if match is None:
            return value
        opening = match.end() - 1
        closing = _matching_brace(value, opening)
        if closing is None:
            return value
        value = value[: match.start()] + value[opening + 1 : closing] + value[closing + 1 :]


def extract_choice_label(answer: object) -> str:
    """Extract an explicit A--J label, without guessing from option content."""

    value = _strip_surrounding_markup(answer)
    value = _replace_style_commands(value)
    value = _strip_surrounding_markup(value)
    value = value.replace("**", "").replace("__", "").replace("\x60", "")
    value = re.sub(r"^[\s#>*✅☑🎯]+", "", value).strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(
        r"^(?:(?:the\s+)?(?:final|correct)\s+)?(?:answer|choice|option)"
        r"\s*(?:is\s*|[:=\-]\s*)?",
        "",
        value,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    match = re.match(
        r"^[\[(]?\s*([A-J])\s*[\])]?\s*(?=$|[.:：\-]|\s)",
        value,
        flags=re.IGNORECASE,
    )
    if match is None:
        return ""
    if match.group(1).islower() and value[match.end() :].strip():
        return ""
    return match.group(1).upper()


def choice_labels(question: str) -> list[str]:
    """Return printed option labels, or an empty list for an open task."""

    labels: list[str] = []
    for match in _CHOICE_LINE_RE.finditer(_repair(question)):
        label = match.group(1).upper()
        if label not in labels:
            labels.append(label)
    return labels if len(labels) >= 2 else []


def inspect_answer_format(question: str, response: str) -> dict[str, object]:
    """Report boxed-answer and question-level multiple-choice conformance."""

    labels = choice_labels(question)
    boxed_answers = extract_boxed_answers(response)
    final_answer, valid = extract_final_answer(response)
    choice_label = extract_choice_label(final_answer)
    return {
        "valid": valid,
        "multiple_choice": bool(labels),
        "choice_labels": labels,
        "box_count": len(boxed_answers),
        "boxed_answer": final_answer,
        "normalized_choice_label": choice_label,
        "choice_label_valid": not labels or choice_label in labels,
    }


def _normalize(text: object) -> str:
    value = _strip_surrounding_markup(text)
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = value.replace(r"\displaystyle", "")
    value = _replace_style_commands(value)
    value = _strip_surrounding_markup(value)
    for spacing in (r"\,", r"\!", r"\;", r"\:", r"\quad", r"\qquad"):
        value = value.replace(spacing, "")
    value = re.sub(r"[，。.,;!?]+$", "", value.strip())
    return re.sub(r"\s+", "", value).casefold()


def _decimal_value(text: str) -> Decimal | None:
    if _DECIMAL_RE.fullmatch(text) is None:
        return None
    try:
        value = Decimal(text)
        return value if value.is_finite() else None
    except InvalidOperation:
        return None


def _math_payload(text: object) -> str:
    return _strip_surrounding_markup(text).replace(r"\dfrac", r"\frac").replace(
        r"\tfrac", r"\frac"
    )


def _replace_constant_latex(text: str) -> str | None:
    value = text
    command_re = re.compile(r"\\(?:frac|sqrt)\b")
    while True:
        matches = list(command_re.finditer(value))
        if not matches:
            return value
        match = matches[-1]
        command = match.group(0)[1:]
        cursor = match.end()
        while cursor < len(value) and value[cursor].isspace():
            cursor += 1
        if command == "frac":
            if cursor >= len(value) or value[cursor] != "{":
                return None
            numerator_end = _matching_brace(value, cursor)
            if numerator_end is None:
                return None
            denominator_start = numerator_end + 1
            while denominator_start < len(value) and value[denominator_start].isspace():
                denominator_start += 1
            if denominator_start >= len(value) or value[denominator_start] != "{":
                return None
            denominator_end = _matching_brace(value, denominator_start)
            if denominator_end is None:
                return None
            numerator = value[cursor + 1 : numerator_end]
            denominator = value[denominator_start + 1 : denominator_end]
            value = (
                value[: match.start()]
                + f"(({numerator})/({denominator}))"
                + value[denominator_end + 1 :]
            )
            continue
        if cursor >= len(value) or value[cursor] != "{":
            return None
        radicand_end = _matching_brace(value, cursor)
        if radicand_end is None:
            return None
        radicand = value[cursor + 1 : radicand_end]
        value = value[: match.start()] + f"sqrt({radicand})" + value[radicand_end + 1 :]


def _constant_math_expression(text: object) -> str | None:
    value = _math_payload(text)
    if "=" in value:
        value = value.rsplit("=", 1)[-1].strip()
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\displaystyle", "")
    value = value.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    value = value.replace(r"\pi", "pi")
    value = value.replace(r"^\circ", "").replace(r"\degree", "").replace("°", "")
    value = value.replace(r"\approx", "").replace(r"\sim", "")
    for spacing in (r"\,", r"\!", r"\;", r"\:", r"\quad", r"\qquad"):
        value = value.replace(spacing, "")
    value = _replace_constant_latex(value)
    if value is None:
        return None
    value = value.replace("{", "(").replace("}", ")").replace("^", "**")
    value = value.replace("$", "").replace("&", "")
    value = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value)
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"(\d|\))(?=(?:sqrt|pi|\())", r"\1*", value)
    value = re.sub(r"(pi|\))(?=(?:\d|\())", r"\1*", value)
    if not value or len(value) > 512 or "\\" in value:
        return None
    return value


def _evaluate_constant_node(node: ast.AST) -> Fraction | float:
    if isinstance(node, ast.Expression):
        return _evaluate_constant_node(node.body)
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
        if isinstance(node.value, int):
            return Fraction(node.value)
        if isinstance(node.value, float) and math.isfinite(node.value):
            return Fraction(str(node.value))
        raise ValueError("unsupported constant")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_constant_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate_constant_node(node.left)
        right = _evaluate_constant_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ValueError("division by zero")
            return left / right
        if isinstance(node.op, ast.Pow):
            if not isinstance(right, Fraction) or right.denominator != 1:
                raise ValueError("non-integer power")
            if abs(right.numerator) > 1000:
                raise ValueError("exponent too large")
            return left**right.numerator
        raise ValueError("unsupported operator")
    if isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sqrt"
        and len(node.args) == 1
        and not node.keywords
    ):
        argument = _evaluate_constant_node(node.args[0])
        if argument < 0:
            raise ValueError("negative square root")
        if isinstance(argument, Fraction):
            numerator = math.isqrt(argument.numerator)
            denominator = math.isqrt(argument.denominator)
            if numerator * numerator == argument.numerator and denominator * denominator == argument.denominator:
                return Fraction(numerator, denominator)
        return math.sqrt(float(argument))
    raise ValueError("unsupported expression")


def _constant_math_value(text: object) -> Fraction | float | None:
    expression = _constant_math_expression(text)
    if expression is None:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 128:
            return None
        value = _evaluate_constant_node(tree)
        return value if math.isfinite(float(value)) else None
    except (ArithmeticError, OverflowError, SyntaxError, TypeError, ValueError):
        return None


def _constant_math_equal(prediction: object, target: object) -> bool:
    left = _constant_math_value(prediction)
    right = _constant_math_value(target)
    if left is None or right is None:
        return False
    if isinstance(left, Fraction) and isinstance(right, Fraction):
        return left == right
    return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)


def _symbolic_equal(prediction: str, target: str) -> bool:
    if parse_math is None or verify_math is None:
        return False
    try:
        parsed_prediction = parse_math("$" + _math_payload(prediction) + "$")
        parsed_target = parse_math("$" + _math_payload(target) + "$")
        return bool(
            parsed_prediction
            and parsed_target
            and verify_math(parsed_target, parsed_prediction)
        )
    except Exception:
        return False


def answers_equivalent(prediction: str, target: str) -> bool:
    """Compare one extracted answer payload with a gold answer."""

    target_normalized = _normalize(target)
    target_label = extract_choice_label(target)
    if target_label and target_normalized == target_label.casefold():
        prediction_label = extract_choice_label(prediction)
        return bool(prediction_label) and prediction_label == target_label

    prediction_normalized = _normalize(prediction)
    if not prediction_normalized or not target_normalized:
        return False
    if prediction_normalized == target_normalized:
        return True
    prediction_number = _decimal_value(prediction_normalized)
    target_number = _decimal_value(target_normalized)
    if prediction_number is not None and target_number is not None:
        return prediction_number == target_number
    if _constant_math_equal(prediction, target):
        return True
    return _symbolic_equal(prediction, target)


def verify_response_answer(response: str, answer: str) -> float:
    """Return 1.0 only when the final boxed answer is correct."""

    prediction, valid_format = extract_final_answer(response)
    if not valid_format:
        return 0.0
    return float(answers_equivalent(prediction, str(answer)))


def verify_response_answers(
    responses: Sequence[str],
    answers: Sequence[str],
) -> list[float]:
    """Verify aligned response and answer lists in input order."""

    if isinstance(responses, (str, bytes)) or not isinstance(responses, Sequence):
        raise TypeError("responses must be a sequence of response strings.")
    if isinstance(answers, (str, bytes)) or not isinstance(answers, Sequence):
        raise TypeError("answers must be a sequence of answer strings.")
    if len(responses) != len(answers):
        raise ValueError(
            "responses and answers must have the same length; "
            f"got {len(responses)} and {len(answers)}."
        )
    return [
        verify_response_answer(response, answer)
        for response, answer in zip(responses, answers, strict=True)
    ]


__all__ = [
    "answers_equivalent",
    "choice_labels",
    "extract_boxed_answers",
    "extract_choice_label",
    "extract_final_answer",
    "inspect_answer_format",
    "verify_response_answer",
    "verify_response_answers",
]
