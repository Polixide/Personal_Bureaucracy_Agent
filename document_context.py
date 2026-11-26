"""Bounded source selection retaining identity, first-page fields and label context."""

import re

from document_evidence import fold, normalized_snippet


def select_context(text: str, max_chars: int, *, quotes=(), patterns=()) -> tuple[str, dict]:
    if len(text) <= max_chars:
        return text, {"pages": list(range(1, text.count("\f") + 2)), "characters": len(text), "full_text": True}
    lines = text.split("\n")
    pages, page = [], 1
    for line in lines:
        page += line.count("\f")
        pages.append(page)
    selected = set()
    used = 0
    def add(indices):
        nonlocal used
        for index in indices:
            if 0 <= index < len(lines) and index not in selected and used + len(lines[index]) + 1 <= max_chars:
                selected.add(index)
                used += len(lines[index]) + 1
    # Identity quotes first; each quote is still original text, not model prose.
    for quote in quotes:
        first = normalized_snippet(quote.splitlines()[0])
        for index, line in enumerate(lines):
            if first and first in normalized_snippet(line):
                add(range(max(0, index - 1), index + len(quote.splitlines()) + 2))
                break
    # Preserve the complete first page where budget allows; forms put fields here.
    add(index for index, page in enumerate(pages) if page == 1)
    for index, line in enumerate(lines):
        if any(re.search(pattern, fold(line)) for pattern in patterns):
            add(range(max(0, index - 2), min(len(lines), index + 6)))
    for page in sorted(set(pages)):
        add([index for index, number in enumerate(pages) if number == page][:20])
    context = "\n".join(lines[index] for index in sorted(selected))
    return context, {"pages": sorted({pages[index] for index in selected}), "characters": len(context), "full_text": False}
