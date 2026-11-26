"""Distinguish a document's own identity from instructions and attachment lists."""

import re

from document_evidence import fold
from document_profiles import IDENTITY_FIELDS, IDENTITY_PURPOSES, indicator_types

REFERENCE_SECTION = re.compile(
    r"\b(?:checklist|attachment(?:s| list)?|supporting documents|supporting evidence|required (?:supporting )?documents|"
    r"documents (?:required|requested|to submit)|required attachments|supporting documentation|legal requirements|gerekli belgeler|ekler|istenen belgeler)\b")
REFERENCE_LINE = re.compile(
    r"\b(?:must|should|required|submit|attach|provide|enclose|including|pursuant|legislation|legislative|"
    r"gereklidir|sunulmalidir|ibraz|eklenmelidir)\b")
BODY_SECTION = re.compile(r"^(?:applicant|student|policy|coverage|personal) (?:details|information)\s*:?")
LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])(?:\s+|$)")
OTHER_DOCUMENT_HEADING = re.compile(r"\b(?:declaration|beyan|consent|muvafakat|invoice|receipt|checklist|kontrol)\b")


def identity_evidence(text: str) -> dict:
    """Collect role-based evidence, never choose a type by keyword frequency."""
    lines = text.split("\n")
    primary, references = [], []
    structures = {kind: {} for kind in IDENTITY_FIELDS}
    reference_section = False
    other_identity = []
    nonempty_seen = 0
    page_line = 0
    for index, line in enumerate(lines):
        if "\f" in line:
            reference_section = False
            page_line = 0
        normalized = " ".join(fold(line).split())
        if not normalized:
            continue
        nonempty_seen += 1
        page_line += 1
        if (nonempty_seen <= 8 and len(normalized.split()) <= 12 and
                OTHER_DOCUMENT_HEADING.search(normalized) and not indicator_types(normalized) and
                not re.search(r"[.!?]\s|[.!?]$", normalized)):
            other_identity.append(dict(quote=line, line=index + 1))
        if BODY_SECTION.search(normalized):
            reference_section = False
        # A mention inside a sentence is not a persistent section heading.
        section_header = (bool(REFERENCE_SECTION.search(normalized)) and len(normalized) <= 100 and
                          len(normalized.split()) <= 12 and not re.search(r"[.!?]\s|[.!?]$", normalized))
        if section_header:
            reference_section = True
        purpose_kinds = [kind for kind, pattern in IDENTITY_PURPOSES.items() if re.search(pattern, normalized)]
        # A self-description can share a PDF line with a later instructional sentence.
        if purpose_kinds and not reference_section:
            for kind in purpose_kinds:
                primary.append(dict(type=kind, role="purpose", quote=line, line=index + 1))
        contextual = reference_section or bool(REFERENCE_LINE.search(normalized)) or bool(LIST_ITEM.search(line))
        kinds = indicator_types(normalized)
        if contextual:
            references.append(dict(quote=line, line=index + 1, types=kinds))
            continue
        for kind, pattern in IDENTITY_PURPOSES.items():
            if re.search(pattern, normalized):
                primary.append(dict(type=kind, role="purpose", quote=line, line=index + 1))
        # A title must look like a heading, rather than a sentence mentioning a form.
        if len(normalized) <= 160 and len(normalized.split()) <= 18 and not re.search(r"[.!?]\s|[.!?]$", normalized):
            for kind in kinds:
                primary.append(dict(type=kind, role="title", quote=line, line=index + 1))
        # PDF extraction often wraps a short title across adjacent lines.
        for width in (2, 3):
            if kinds:
                break
            window = lines[index:index + width]
            if len(window) != width or any("\f" in item for item in window):
                break
            if any(LIST_ITEM.search(item) or REFERENCE_SECTION.search(fold(item)) or
                   REFERENCE_LINE.search(fold(item)) for item in window):
                continue
            if any(indicator_types(" ".join(fold(item).split())) for item in window[1:]):
                continue
            heading = " ".join(fold(" ".join(window)).split())
            if (len(heading) <= 160 and len(heading.split()) <= 18 and
                    not REFERENCE_SECTION.search(heading) and not REFERENCE_LINE.search(heading) and
                    not re.search(r"[.!?]\s|[.!?]$", heading)):
                for kind in indicator_types(heading):
                    if kind not in kinds:
                        primary.append(dict(type=kind, role="title" if page_line <= 12 else "heading",
                                            quote="\n".join(window), line=index + 1))
        # Two distinct populated field roles can establish an otherwise untitled document.
        adjacent = []
        for following in lines[index + 1:index + 4]:
            if "\f" in following:
                break
            adjacent.append(following)
        block = normalized + " " + " ".join(fold(" ".join(adjacent)).split())
        for kind, fields in IDENTITY_FIELDS.items():
            for role, pattern in fields.items():
                match = re.match(r"(?:" + pattern + r")\s*[:：]\s*(\S.+|\S)$", normalized)
                quote = line
                if not match:
                    match = re.match(r"(?:" + pattern + r")\s*[:：]\s*(\S.+|\S)$", block)
                    quote += "\n" + "\n".join(adjacent)
                value = match.group(1).strip() if match else ""
                populated = value not in {"", "-", "n/a", "none", "null", "unknown", "not provided", "not supplied"}
                if role in {"student_id", "application_id", "policy_id", "passport"}:
                    populated = populated and bool(re.fullmatch(r"[\w./-]+", value)) and any(c.isdigit() for c in value)
                if match and populated and not REFERENCE_LINE.search(value):
                    structures[kind].setdefault(role, dict(type=kind, role="structure", quote=quote, line=index + 1))
    for kind, fields in structures.items():
        # A repeated ID alone is corroboration, not two independent structural roles.
        if len(fields) >= 2:
            primary.extend(fields.values())
    authoritative = {item["type"] for item in primary if item["role"] in {"title", "purpose", "structure"}}
    if authoritative:
        # A wrapped mid-page field heading is not a rival document identity when
        # another type has its own title/purpose/structural evidence.
        for item in primary:
            if item["type"] not in authoritative:
                references.append(dict(quote=item["quote"], line=item["line"], types=[item["type"]]))
        primary = [item for item in primary if item["type"] in authoritative]
    return {"primary": primary, "references": references, "other_identity": other_identity,
            "types": sorted({item["type"] for item in primary})}
