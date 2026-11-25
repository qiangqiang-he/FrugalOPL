"""Model-agnostic task prompt templates.

This module owns model-visible task text only.  A definition is either one
format string for a ``user`` turn or an immutable sequence of role/content
format templates.  Model services render those messages and use the target
tokenizer's ``apply_chat_template`` method to add model-specific control
tokens.
"""

from __future__ import annotations

from collections.abc import Mapping


MATH_PROMPT = r"""Solve the problem step by step. Put the final answer in \boxed{{...}}.

For a multiple-choice problem, put the option label in the box.

### Problem
{question}"""


EXPLICIT_STEP_PROMPT = r"""Solve the problem step by step. Organize the reasoning with headings `### Step 1`, `### Step 2`, and so on. Put the final answer in \boxed{{...}}.

For a multiple-choice problem, put the option label in the box.

### Problem
{question}"""



OPSD_TEACHER_PROMPT = r"""Solve the problem step by step. Put the final answer in \boxed{{...}}.

For a multiple-choice problem, put the option label in the box.

### Problem
{question}

### Reference Solution
{solution}

The reference solution above is correct. Understand the reasoning behind it, but do not simply copy or paraphrase it. Solve the problem independently using the reference solution as privileged information, and derive the same final answer."""



# Official OPSD reason_first=False prompts (siyan-zhao/OPSD, ae7d2519).
# Keep separate from the existing PG-OPSD templates so that selecting the new
# objective does not change prior experiment prompts.
KL_OPSD_STUDENT_PROMPT = (
    "Problem: {question}\n\n"
    r"Please reason step by step, and put your final answer within \boxed{{}}."
)

KL_OPSD_TEACHER_PROMPT = (
    "Problem: {question}\n\n"
    "Here is a reference solution to this problem:\n"
    "=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
    "\n\nAfter reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step — do not copy or paraphrase it. Now, using your "
    "own words and independent reasoning, derive the same final answer to the problem above. "
    "Think step by step, explore different approaches, and don't be afraid to backtrack "
    "or reconsider if something doesn't work out:\n\n"
    r"Please reason step by step, and put your final answer within \boxed{{}}."
)

TRA_TRUE_ANSWER_PROMPT = (
    '{student_prompt}\n\n'
    'The correct answer to this problem is: {answer}.\n'
    'Use this known correct answer to guide your reasoning on the problem above. '
    'Explain the steps that lead to this answer.'
)


# The previous question and solution are represented as actual conversation
# history.  In particular, no model-visible text labels either turn as an
# "example" or asks the model to refer to it.
MATH_ONESHOT_MESSAGES = (
    ("user", MATH_PROMPT.replace("{question}", "{example_question}")),
    ("assistant", "{example_solution}"),
    ("user", MATH_PROMPT),
)




def render_prompt_from_record(template: str, record: Mapping[str, object], /) -> str:
    """Fill named placeholders from a complete item without filtering its data.

    Python's ordinary format-map semantics use only the fields referenced by
    the template. Extra fields, including nested metadata, remain untouched.
    """

    if not isinstance(template, str):
        raise TypeError("Prompt templates must be strings.")
    if not isinstance(record, Mapping):
        raise TypeError("Prompt records must be mappings.")
    try:
        return template.format_map(record)
    except KeyError as error:
        raise ValueError(f"Missing prompt variable {error.args[0]!r}.") from error


def render_prompt(template: str, /, **variables: str) -> str:
    """Render a prompt template from non-empty named text variables.

    The function intentionally knows nothing about a template's semantics.
    Callers choose the template and supply its named placeholders, while model
    services later convert the returned text into a user message and apply the
    tokenizer's chat template.
    """

    if not isinstance(template, str):
        raise TypeError("Prompt templates must be strings.")

    normalized_variables: dict[str, str] = {}
    for name, value in variables.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Prompt variable {name!r} must be a non-empty string.")
        normalized_variables[name] = value.strip()

    try:
        return template.format(**normalized_variables)
    except KeyError as error:
        raise ValueError(f"Missing prompt variable {error.args[0]!r}.") from error


__all__ = [
    "EXPLICIT_STEP_PROMPT",
    "KL_OPSD_STUDENT_PROMPT",
    "KL_OPSD_TEACHER_PROMPT",
    "MATH_PROMPT",
    "MATH_ONESHOT_MESSAGES",
    "OPSD_TEACHER_PROMPT",
    "TRA_TRUE_ANSWER_PROMPT",
    "render_prompt",
    "render_prompt_from_record",
]
