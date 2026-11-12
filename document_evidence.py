"""Conservative source candidates and grounding; no model calls or JSON eval."""

from datetime import date
import re
import unicodedata


DATE_PATTERN = re.compile(r"(?<!\d)(?:\d{4}\s*-\s*\d{2}\s*-\s*\d{2}|\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{4})(?!\d)")
LEGAL_REFERENCE = re.compile(r"\b(?:kanun|yonetmelik|mevzuat|law|regulation|sayili|resmi gazete|referans|reference)\b")
UNRELATED_FIELD = re.compile(r"\b(?:dogum|birth|vergi|tax|kimlik|identity|telefon|phone|sayfa|page)\b")


def normalized_snippet(text: str) -> str:
    """Ignore layout whitespace and Unicode presentation, not words or digits."""
    text = " ".join(unicodedata.normalize("NFKC", text).split())
    return re.sub(r"\s*([:/.-])\s*", r"\1", text)


def snippet_in_source(snippet: str, text: str) -> bool:
    return bool(normalized_snippet(snippet)) and normalized_snippet(snippet) in normalized_snippet(text)


def label_text(text: str) -> str:
    return re.sub(r"\s+", " ", fold(text)).strip()


def folded_label_end(text: str, end: int) -> int:
    """Map a match in accent-folded text back to original Unicode positions."""
    positions = [index + 1 for index, character in enumerate(text) for _ in fold(character)]
    return positions[end - 1]


def fold(text: str) -> str:
    """Normalize Turkish case/accents for labels, never for source quotes."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold().replace("ı", "i"))
                   if not unicodedata.combining(c))


def normalize_date(raw: str, order: str | None = None) -> str | None:
    """Normalize calendar-valid numeric dates; ambiguous day/month stays null."""
    raw = re.sub(r"\s*([/.-])\s*", r"\1", unicodedata.normalize("NFKC", raw).strip())
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            return date.fromisoformat(raw).isoformat()
        if not re.fullmatch(r"\d{1,2}([/.-])\d{1,2}\1\d{4}", raw):
            return None
        first, second, year = map(int, re.split(r"[/.-]", raw))
        if order == "dmy" or (order is None and first > 12):
            return date(year, second, first).isoformat()
        if order == "mdy":
            return date(year, first, second).isoformat()
        if first == second:
            return date(year, second, first).isoformat()
    except ValueError:
        pass
    return None


def date_order(text: str) -> str | None:
    """Use explicit notation or consistent unambiguous numeric source dates."""
    nonlegal_text = "\n".join(line for line in text.splitlines()
                              if not LEGAL_REFERENCE.search(fold(line)) and not UNRELATED_FIELD.search(fold(line)))
    normalized = fold(nonlegal_text)
    dmy = bool(re.search(r"dd/mm/yyyy|gg/aa/yyyy", normalized))
    mdy = "mm/dd/yyyy" in normalized
    for raw in DATE_PATTERN.findall(nonlegal_text):
        if re.match(r"\d{4}\s*-", raw):
            continue
        first, second, _ = map(int, re.split(r"[/.-]", raw))
        dmy |= 12 < first <= 31 and 1 <= second <= 12
        mdy |= 1 <= first <= 12 and 12 < second <= 31
    return "dmy" if dmy and not mdy else "mdy" if mdy and not dmy else None


def date_label(line: str) -> str | None:
    label = label_text(line)
    if LEGAL_REFERENCE.search(label) or UNRELATED_FIELD.search(label):
        return None
    if (("baslangic" in label and "bitis" in label and "tarih" in label) or
            re.search(r"\b(?:coverage|travel) period\b", label)):
        return "coverage_period"
    if re.search(r"police (?:onay|tanzim|duzenleme) tarihi|policy (?:issue|approval) date|\bissue date\b", label):
        return "issue_date"
    if re.search(r"duzenleme tarihi|tanzim tarihi|document issuance date", label):
        return "document_issue_date"
    if re.search(r"(?:sigorta|teminat|police) baslangic tarihi|coverage start date", label):
        return "coverage_start_date"
    if re.search(r"(?:sigorta|teminat|police) bitis tarihi|coverage end date", label):
        return "coverage_end_date"
    if re.search(r"\bexpiry date\b|\bexpiration date\b|son gecerlilik tarihi", label):
        return "expiry_date"
    return None


def collect_evidence(text: str, *, paginated: bool = False) -> dict:
    """Keep original strings/line references and associate only recognized labels."""
    lines = text.split("\n")
    page_numbers, page = [], 1
    for line in lines:
        page += line.count("\f")
        page_numbers.append(page if paginated else None)
    order = date_order(text)
    evidence = []
    assigned = set()
    for index, line in enumerate(lines):
        label = date_label(line)
        block = [(index, line)]
        # A split Turkish period heading is common in digitally extracted PDFs.
        if (label is None and "baslangic" in fold(line) and not DATE_PATTERN.search(line)
                and index + 1 < len(lines) and "\f" not in lines[index + 1]):
            combined = line + " " + lines[index + 1]
            if date_label(combined) == "coverage_period":
                label = "coverage_period"
                block.append((index + 1, lines[index + 1]))
        if label is None:
            continue
        for offset in range(len(block), len(block) + 4):
            if index + offset >= len(lines):
                break
            following = lines[index + offset]
            if ("\f" in following or date_label(following) or
                    LEGAL_REFERENCE.search(fold(following)) or UNRELATED_FIELD.search(fold(following))):
                break
            # Only dates, times, separators, or blank lines can continue a value.
            remainder = DATE_PATTERN.sub("", unicodedata.normalize("NFKC", following))
            remainder = re.sub(r"\b\d{2}:\d{2}(?::\d{2})?\b", "", remainder)
            if remainder.strip(" \t\r:;.,-/–—()[]"):
                break
            block.append((index + offset, following))
        matches = [(number, match) for number, value in block for match in DATE_PATTERN.finditer(value)]
        snippet = "\n".join(value for _, value in block)
        for position, (number, match) in enumerate(matches):
            field = label
            if label == "coverage_period":
                # More/fewer than two dates cannot establish a unique period.
                field = ("coverage_start_date", "coverage_end_date")[position] if len(matches) == 2 else "coverage_period"
            evidence.append(dict(field=field, original_value=match.group(),
                                 normalized_value=normalize_date(match.group(), order) if field != "coverage_period" else None,
                                 snippet=snippet, line_number=number + 1, page_number=page_numbers[number]))
            assigned.add((number, match.start()))
        if not matches:
            evidence.append(dict(field=label, original_value="", normalized_value=None,
                                 snippet=snippet, line_number=index + 1, page_number=page_numbers[index]))
    # Unassigned dates are audit candidates only, never automatic field values.
    for number, line in enumerate(lines):
        for match in DATE_PATTERN.finditer(line):
            if (number, match.start()) not in assigned:
                evidence.append(dict(field=None, original_value=match.group(),
                                     normalized_value=normalize_date(match.group(), order),
                                     snippet=line, line_number=number + 1, page_number=page_numbers[number]))
    policy_label = re.compile(r"\b(?:police\s+(?:no(?:su)?|numarasi)|policy\s+(?:number|no))\b")
    for number, line in enumerate(lines):
        match = policy_label.search(fold(line))
        if not match:
            continue
        raw_value = line[folded_label_end(line, match.end()):].strip()
        snippet = line
        if not raw_value.strip(" :：.#\t"):
            for offset in (1, 2):
                if number + offset >= len(lines):
                    break
                following = lines[number + offset]
                if ("\f" in following or LEGAL_REFERENCE.search(fold(following)) or
                        UNRELATED_FIELD.search(fold(following)) or date_label(following)):
                    break
                snippet += "\n" + following
                raw_value = following.strip()
                if raw_value.strip(" :：.#\t"):
                    break
        value = unicodedata.normalize("NFKC", raw_value).lstrip(" :.#\t")
        # Only a single labeled token with a digit; no cross-field number search.
        valid = (re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", value) and
                 any(character.isdigit() for character in value) and not DATE_PATTERN.fullmatch(value))
        evidence.append(dict(field="policy_number", original_value=raw_value,
                             normalized_value=value if valid else None, snippet=snippet,
                             line_number=number + 1, page_number=page_numbers[number]))
    normalized = re.sub(r"\s+", " ", fold(text))
    insurance = bool(re.search(r"saglik sigorta(?:si)? policesi|health insurance (?:certificate|policy)", normalized))
    enrollment = "student enrollment certificate" in normalized or "ogrenci belgesi" in normalized
    return {"date_order": order, "insurance_heading": insurance,
            "conflicting_headings": insurance and enrollment, "candidates": evidence}


def field_candidates(field: str, payload: dict, evidence: dict) -> list[dict]:
    """Use the same semantic evidence roles for reconciliation and final auditing."""
    selected = [item for item in evidence["candidates"] if item["field"] == field]
    if field == "issue_date" and not selected and payload.get("document_type") == "health_insurance_certificate":
        selected = [item for item in evidence["candidates"] if item["field"] == "document_issue_date"]
    if field == "expiry_date" and not selected and payload.get("document_type") == "health_insurance_certificate":
        selected = [item for item in evidence["candidates"] if item["field"] == "coverage_end_date"]
    if field.startswith("coverage_") and not selected:
        selected = [item for item in evidence["candidates"] if item["field"] == "coverage_period"]
    return selected


def independently_supported(field: str, payload: dict, evidence: dict, text: str) -> bool:
    """Final field validity is independent of an earlier model citation failure."""
    value = payload.get(field)
    if value is None or field in payload.get("ambiguous_fields", []):
        return False
    selected = field_candidates(field, payload, evidence)
    related_fields = {field} | {item["field"] for item in selected}
    if field == "expiry_date" and "coverage_end_date" in related_fields:
        related_fields.add("coverage_start_date")
    if any(any(name in issue for name in related_fields) for issue in payload.get("validation_issues", [])):
        return False
    values = {item["normalized_value"] for item in selected if item["normalized_value"] is not None}
    unresolved = any(item["normalized_value"] is None and item["original_value"] for item in selected)
    quote = payload.get("source_snippets", {}).get(field, "")
    return (values == {value} and not unresolved and snippet_in_source(quote, text) and
            any(item["normalized_value"] == value and
                normalized_snippet(item["snippet"]) == normalized_snippet(quote) for item in selected))


def ground_payload(payload: dict, evidence: dict) -> dict:
    """Use unique labeled evidence; retain disagreements as review issues."""
    payload = dict(payload)
    issues = []
    corrections = []
    ambiguous = list(payload.get("ambiguous_fields") or [])
    snippets = dict(payload.get("source_snippets") or {})
    candidates = evidence["candidates"]
    if evidence["conflicting_headings"]:
        payload["document_type"] = "unknown"
        issues.append("Conflicting document headings; manual review needed")
    elif evidence["insurance_heading"]:
        payload["document_type"] = "health_insurance_certificate"
    for field in ("issue_date", "coverage_start_date", "coverage_end_date", "expiry_date", "policy_number"):
        selected = field_candidates(field, payload, evidence)
        model_value = payload.get(field)
        # Preserve strict schema rejection of numeric/boolean dates.
        if field != "policy_number" and isinstance(model_value, (int, float, bool)):
            continue
        insurance = payload.get("document_type") == "health_insurance_certificate"
        # Preserve enrollment behavior: a missing model field still needs review.
        if model_value is None and payload.get("document_type") != "health_insurance_certificate":
            continue
        values = {item["normalized_value"] for item in selected if item["normalized_value"] is not None}
        unresolved = any(item["normalized_value"] is None and item["original_value"] for item in selected)
        unique_source = len(values) == 1 and not unresolved
        if field in ambiguous and not (insurance and unique_source):
            payload[field] = None
            continue
        invalid_model_date = (field != "policy_number" and isinstance(model_value, str) and
                              normalize_date(model_value, evidence["date_order"]) is None)
        if invalid_model_date and not (insurance and unique_source):
            payload[field] = None
            ambiguous.append(field)
            issues.append(f"{field}: invalid date returned by model; manual review needed")
            continue
        if unique_source:
            value = next(iter(values))
            payload[field] = value
            if field in ambiguous:
                ambiguous.remove(field)
                corrections.append(f"{field}: model ambiguity resolved by unique labeled source evidence")
            model_equivalent = (normalize_date(model_value, evidence["date_order"])
                                if field != "policy_number" and isinstance(model_value, str) else model_value)
            if invalid_model_date or model_equivalent not in (None, "", value):
                note = f"{field}: model value rejected; using unique labeled source evidence"
                corrections.append(note)
                if not insurance:
                    issues.append(note + "; manual review needed")
            chosen = next(item for item in selected if item["normalized_value"] == value)
            # Verify the model quote, then use the precise deterministic field quote.
            existing_quote = snippets.get(field)
            if existing_quote:
                supported_quote = (value in existing_quote if field == "policy_number" else
                                   any(normalize_date(raw, evidence["date_order"]) == value
                                       for raw in DATE_PATTERN.findall(existing_quote)))
                if not supported_quote:
                    note = f"{field}: source snippet does not support selected value; replaced with labeled source quote"
                    (corrections if insurance else issues).append(note)
            snippets[field] = chosen["snippet"]
        elif selected or model_value is not None:
            payload[field] = None
            snippets.pop(field, None)
            if selected and all(not item["original_value"] for item in selected) and model_value is None:
                continue
            ambiguous.append(field)
            reason = "ambiguous, invalid, or conflicting labeled evidence" if selected else "no supporting labeled source evidence"
            issues.append(f"{field}: {reason}; manual review needed")
    payload["ambiguous_fields"] = list(dict.fromkeys(ambiguous))
    for field in ("issue_date", "coverage_start_date", "coverage_end_date", "expiry_date", "policy_number"):
        if payload.get(field) is None:
            snippets.pop(field, None)
    payload["missing_fields"] = []
    payload["validation_issues"] = issues
    payload["correction_notes"] = list(dict.fromkeys(corrections))
    payload["source_snippets"] = snippets
    payload["source_evidence"] = candidates
    return payload
