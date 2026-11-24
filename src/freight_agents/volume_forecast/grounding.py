"""Check that an answer states only numbers its sources gave.

A number counts as grounded when it appears in a tool result the agent received
or in the question itself. Thousands separators, trailing zeros and leading
zeros are normalized first, so "1,855", "11.40" and "05" match "1855", "11.4"
and "5". Signs are ignored: a source's -4.2 grounds an answer's "down 4.2%".
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Linear patterns with no nested quantifiers.
NUMBER = re.compile(r"\d+(?:\.\d+)?")
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def _normal(token: str) -> str:
    whole, _, fraction = token.partition(".")
    fraction = fraction.rstrip("0")
    return f"{int(whole)}.{fraction}" if fraction else str(int(whole))


def numbers_in(text: str) -> set[str]:
    return {_normal(t) for t in NUMBER.findall(THOUSANDS.sub("", text))}


def ungrounded(answer: str, sources: Iterable[str]) -> set[str]:
    allowed: set[str] = set()
    for source in sources:
        allowed |= numbers_in(source)
    return numbers_in(answer) - allowed
