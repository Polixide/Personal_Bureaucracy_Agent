"""Deterministic source evidence and reconciliation; no model calls or legal decisions.

Label, snippet, layout and correction concepts adapted selectively from
origin/test/integration-abc:document_evidence.py. Canonical bounded TextPages,
document-specific schemas and the existing European-date helper are preserved.
"""
from copy import deepcopy
from datetime import date
import re
import unicodedata

def normalized(text):
    text = text.casefold().replace("\u0131", "i")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


DATE_PATTERN = re.compile(r"(?<![\w])(?:\d{2}\.\d{2}\.\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})(?![\w])")
IDENTIFIER_LABELS = {
    "policy_number": r"policy(?:\s*(?:number|no\.?))?|police\s*(?:no\.?|numarasi)",
    "passport_or_foreign_id": r"passport(?:\s*(?:number|no\.?))?|foreign(?:er)?\s*(?:id|identity)(?:\s*(?:number|no\.?))?|pasaport(?:\s*(?:no\.?|numarasi))?|yabanci\s*kimlik(?:\s*(?:no\.?|numarasi))?",
    "application_number": r"application\s*(?:number|no\.?|id)|basvuru\s*(?:no\.?|numarasi)",
    "student_id": r"student\s*(?:id|identifier|number)|ogrenci\s*(?:no\.?|numarasi)",
}
NAME_LABEL = re.compile(r"^(?:insured(?:\s+person)?|policy\s*holder|holder\s*name|applicant(?:\s*name)?|full\s*name|name\s*(?:/|and)\s*surname|ad(?:i)?\s*/\s*soyad(?:i)?|ad(?:i)?\s+soyad(?:i)?|name)(?:\s*:\s*|\s*$)(.*)$")


def normalized_date(value):
    """Parse only the three supported source formats, with broad quality bounds."""
    if not isinstance(value, str):
        return None
    value = unicodedata.normalize("NFKC", value)
    if not DATE_PATTERN.fullmatch(value):
        return None
    try:
        if "." in value or "/" in value:
            day, month, year = re.split(r"[./]", value)
            parsed = date(int(year), int(month), int(day))
        else:
            parsed = date.fromisoformat(value)
        return parsed.isoformat() if 1900 <= parsed.year <= 2100 else None
    except ValueError:
        return None


def _basic_candidates(chunks):
    """Conservative candidates from selected text only; no semantic date roles."""
    candidates, issues = [], []
    for chunk_index, chunk in enumerate(chunks):
        for match in DATE_PATTERN.finditer(chunk.text):
            value = normalized_date(match.group())
            if value is None:
                issues.append(f"date: invalid source date {match.group()!r} on page {chunk.number}")
                continue
            nearby = " ".join(chunk.text[max(0, match.start()-35):match.end()+20].split())
            candidates.append(dict(kind="date", value=value, page=chunk.number, nearby=nearby,
                                   original_value=match.group(), chunk_index=chunk_index, line_number=chunk.text[:match.start()].count("\n")+1))
        lines = chunk.text.splitlines()
        name_parts = {"first": [], "last": []}
        for index, line in enumerate(lines):
            folded = normalized(line)
            for kind, label in IDENTIFIER_LABELS.items():
                labelled = line
                if re.search(rf"\b(?:{label})\s*[:#]?\s*$", folded) and index + 1 < len(lines):
                    labelled += " " + lines[index+1]
                pattern = rf"\b(?:{label})(?:\s*[:#]\s*|\s+)([a-z0-9][a-z0-9./-]{{2,39}})\b"
                for match in re.finditer(pattern, normalized(labelled)):
                    value = labelled[folded_label_end(labelled, match.start(1)):folded_label_end(labelled, match.end(1))]
                    if any(c.isdigit() for c in value) and not DATE_PATTERN.fullmatch(value):
                        candidates.append(dict(kind=kind, value=value, page=chunk.number, nearby=line.strip()[:90]))
            for part, labels in (("first", "name|ad|adi"), ("last", "surname|soyad|soyadi")):
                if re.match(rf"^(?:{labels})(?:\s*:|\s*$)", folded.strip()):
                    value = line.split(":", 1)[1].strip() if ":" in line else ""
                    if not value and index + 1 < len(lines):
                        value = lines[index+1].strip()
                    if value and all(w.isalpha() for w in value.split()) and len(value.split()) <= 3:
                        name_parts[part].append(value)
            match = NAME_LABEL.match(folded.strip())
            if match:
                # Labels and values may occupy adjacent lines in PDF text.
                value = line.strip()[match.start(1):].strip()
                if not value and index + 1 < len(lines):
                    value = lines[index+1].strip()
                value = re.split(r"[;|\t]", value)[0].strip()
                words = value.replace("-", " ").replace("'", " ").split()
                if 2 <= len(words) <= 6 and all(word.isalpha() for word in words):
                    candidates.append(dict(kind="name", value=value, page=chunk.number, nearby=line.strip()[:90]))
        if len(name_parts["first"]) == len(name_parts["last"]) == 1:
            value = name_parts["first"][0] + " " + name_parts["last"][0]
            candidates.append(dict(kind="name", value=value, page=chunk.number, nearby="Name / surname labels"))
    return candidates, issues



EVIDENCE_DATE_PATTERN = re.compile(r"(?<!\d)(?:\d{4}\s*-\s*\d{2}\s*-\s*\d{2}|\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{4})(?!\d)")
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


def source_date(raw: str, order: str | None = None) -> str | None:
    """Normalize calendar-valid numeric dates; ambiguous day/month stays null."""
    raw = re.sub(r"\s*([/.-])\s*", r"\1", unicodedata.normalize("NFKC", raw).strip())
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
            return date.fromisoformat(raw).isoformat() if 1900 <= int(raw[:4]) <= 2100 else None
        if not re.fullmatch(r"\d{1,2}([/.-])\d{1,2}\1\d{4}", raw):
            return None
        first, second, year = map(int, re.split(r"[/.-]", raw))
        if not 1900 <= year <= 2100:
            return None
        if order == "dmy" or "." in raw or (order is None and first > 12):
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
    for raw in EVIDENCE_DATE_PATTERN.findall(nonlegal_text):
        if re.match(r"\d{4}\s*-", raw):
            continue
        first, second, _ = map(int, re.split(r"[/.-]", raw))
        dmy |= ("." in raw and 1 <= first <= 31 and 1 <= second <= 12) or (12 < first <= 31 and 1 <= second <= 12)
        mdy |= 1 <= first <= 12 and 12 < second <= 31
    return "dmy" if dmy and not mdy else "mdy" if mdy and not dmy else None


def date_label(line: str) -> str | None:
    label = label_text(line)
    if LEGAL_REFERENCE.search(label) or UNRELATED_FIELD.search(label):
        return None
    if re.search(r"requested (?:start|begin)(?: date)?", label):
        return "requested_start_date"
    if re.search(r"requested end(?: date)?", label):
        return "requested_end_date"
    if "requested period" in label:
        return "requested_period"
    if (("baslangic" in label and "bitis" in label and "tarih" in label) or
            re.search(r"\b(?:coverage|travel) period\b|^coverage\s*:", label)):
        return "coverage_period"
    if re.search(r"police (?:onay|tanzim|duzenleme) tarihi|policy (?:issue|approval) date|\bissue date\b", label):
        return "issue_date"
    if re.search(r"duzenleme tarihi|tanzim tarihi|document issuance date", label):
        return "document_issue_date"
    if re.search(r"(?:sigorta|teminat|police) baslangic tarihi|coverage start(?: date)?", label):
        return "coverage_start_date"
    if re.search(r"(?:sigorta|teminat|police) bitis tarihi|coverage end(?: date)?", label):
        return "coverage_end_date"
    if re.search(r"\bexpiry date\b|\bexpiration date\b|son gecerlilik tarihi", label):
        return "expiry_date"
    return None


def collect_evidence(text: str, *, paginated: bool = False, order_hint=None) -> list[dict]:
    """Keep original strings/line references and associate only recognized labels."""
    lines = text.split("\n")
    page_numbers, page = [], 1
    for line in lines:
        page += line.count("\f")
        page_numbers.append(page if paginated else None)
    order = order_hint or date_order(text)
    evidence = []
    assigned = set()
    for index, line in enumerate(lines):
        label = date_label(line)
        block = [(index, line)]
        # A split Turkish period heading is common in digitally extracted PDFs.
        if (label is None and "baslangic" in fold(line) and not EVIDENCE_DATE_PATTERN.search(line)
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
            remainder = EVIDENCE_DATE_PATTERN.sub("", unicodedata.normalize("NFKC", following))
            remainder = re.sub(r"\b\d{2}:\d{2}(?::\d{2})?\b", "", remainder)
            if remainder.strip(" \t\r:;.,-/–—()[]"):
                break
            block.append((index + offset, following))
        matches = [(number, match) for number, value in block for match in EVIDENCE_DATE_PATTERN.finditer(value)]
        snippet = "\n".join(value for _, value in block)
        for position, (number, match) in enumerate(matches):
            field = label
            if label in ("coverage_period", "requested_period"):
                # More/fewer than two dates cannot establish a unique period.
                prefix = "coverage" if label == "coverage_period" else "requested"
                field = (prefix + "_start_date", prefix + "_end_date")[position] if len(matches) == 2 else label
            evidence.append(dict(field=field, original_value=match.group(),
                                 normalized_value=source_date(match.group(), order) if field not in ("coverage_period", "requested_period") else None,
                                 snippet=snippet, line_number=number + 1, page_number=page_numbers[number]))
            assigned.add((number, match.start()))
        if not matches:
            evidence.append(dict(field=label, original_value="", normalized_value=None,
                                 snippet=snippet, line_number=index + 1, page_number=page_numbers[index]))
    # Unassigned dates are audit candidates only, never automatic field values.
    for number, line in enumerate(lines):
        for match in EVIDENCE_DATE_PATTERN.finditer(line):
            if (number, match.start()) not in assigned:
                evidence.append(dict(field=None, original_value=match.group(),
                                     normalized_value=source_date(match.group(), order),
                                     snippet=line, line_number=number + 1, page_number=page_numbers[number]))
    policy_label = re.compile(r"\b(?:police\s+(?:no(?:su)?|numarasi)|policy(?:\s+(?:number|no)|(?=\s*:)))\b")
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
                 any(character.isdigit() for character in value) and not EVIDENCE_DATE_PATTERN.fullmatch(value))
        evidence.append(dict(field="policy_number", original_value=raw_value,
                             normalized_value=value if valid else None, snippet=snippet,
                             line_number=number + 1, page_number=page_numbers[number]))
    return evidence


def _record(kind, field, original, value, snippet, page, line):
    return dict(kind=kind, field=field, value=value, page=page, nearby=snippet,
                original_value=original, normalized_value=value, snippet=snippet,
                page_number=page, line_number=line)


def source_candidates(chunks):
    """Bounded TextPages -> original/normalized evidence with semantic labels."""
    basic, issues = _basic_candidates(chunks)
    candidates = []
    order = date_order("\n".join(chunk.text for chunk in chunks))
    for c in basic:
        if c["kind"] == "policy_number":
            continue  # The stronger, whole-token policy-label parser below owns this role.
        if c["kind"] == "date":
            chunk = chunks[c["chunk_index"]]
            quote = chunk.text.splitlines()[c["line_number"]-1]
            record = _record("date", None, c["original_value"], source_date(c["original_value"], order), quote, c["page"], c["line_number"])
            record["value"] = c["value"]  # Preserve the canonical European-date candidate API.
            candidates.append(record)
            continue
        chunk = next(chunk for chunk in chunks if chunk.number == c["page"] and
                     (c["value"] in chunk.text or c["kind"] == "name"))
        matching_lines = [(i, line) for i, line in enumerate(chunk.text.splitlines(), 1)
                          if normalized(c["value"]) in normalized(line)]
        line, quote = matching_lines[0] if matching_lines else (1, chunk.text)
        candidates.append(_record(c["kind"], c["kind"], c["value"], c["value"], quote, c["page"], line))
    for chunk in chunks:
        for e in collect_evidence(chunk.text, order_hint=order):
            candidates.append(_record("policy_number" if e["field"] == "policy_number" else "date" if e["normalized_value"] is not None else "date_evidence",
                                      e["field"], e["original_value"], e["normalized_value"],
                                      e["snippet"], chunk.number, e["line_number"]))
        # Companies are accepted from explicit labels; unlabelled companies may
        # be interpreted by the model only when its exact value is a source line.
        lines = chunk.text.splitlines()
        for index, line in enumerate(lines, 1):
            match = re.match(r"^(insurer|insurance provider|insurance company|sigorta sirketi|sigortaci|institution|university)\s*[:\uff1a]\s*", fold(line))
            if match:
                end = folded_label_end(line, match.end())
                raw = line[end:].strip()
                quote = line
                if not raw and index < len(lines):
                    following = lines[index]
                    if ":" not in following and not date_label(following) and not LEGAL_REFERENCE.search(fold(following)):
                        raw = following.strip()
                        quote += "\n" + following
                field = "institution" if match.group(1) in ("institution", "university") else "insurer"
                candidates.append(_record(field, field, raw, raw if any(c.isalpha() for c in raw) else None, quote, chunk.number, index))
    # Evidence is deterministic; identical repeated observations corroborate.
    unique = []
    for c in candidates:
        if c not in unique:
            unique.append(c)
    return unique, issues


def candidate_hint(candidates, limit):
    text = "[SOURCE CANDIDATES]"
    groups = [[c for c in candidates if c["kind"] == kind] for kind in
              ("date", "policy_number", "insurer", "passport_or_foreign_id", "application_number", "student_id", "name")]
    for index in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if index < len(group):
                c = group[index]
                label = c.get("field") or c["kind"]
                row = f"\n{c['kind']}={c['value']} ({label}) | p{c['page']} | {c['nearby'][:50]}"
                if len(text) + len(row) <= limit:
                    text += row
    return text if "\n" in text else ""


def field_candidates(field, schema, candidates):
    if field in schema.DATES:
        selected = [c for c in candidates if c.get("field") == field]
        if field == "issue_date" and not selected:
            selected = [c for c in candidates if c.get("field") == "document_issue_date"]
        if not selected and field.startswith(("coverage_", "requested_")):
            prefix = field.split("_", 1)[0]
            selected = [c for c in candidates if c.get("field") == prefix + "_period"]
        # Keep canonical enrollment/application interpretation of unlabelled
        # dates, but never let insurance use legal/birth dates as policy dates.
        if not selected and schema.__name__ != "HealthInsurance":
            selected = [c for c in candidates if c["kind"] == "date" and c.get("field") is None
                        and not LEGAL_REFERENCE.search(fold(c["snippet"]))
                        and not UNRELATED_FIELD.search(fold(c["snippet"]))]
        return selected
    kind = "name" if field in ("holder_name", "applicant_name") else field
    return [c for c in candidates if c["kind"] == kind]


def _equivalent(value, date_field=False):
    if date_field:
        return normalized_date(value)
    return " ".join(normalized(value).split()) if isinstance(value, str) else None


def ground_fields(payload, schema, candidates, issues, chunks=()):
    """Reconcile source evidence, keeping corrections separate from review issues.

    Only insurance receives automatic fills/corrections. Enrollment and IKAMET
    retain null-on-model-uncertainty behavior; conflicting source stays null for
    every document type. Input metadata is never trusted as source evidence.
    """
    insurance = schema.__name__ == "HealthInsurance"
    ambiguous = list(payload.get("ambiguous_fields") or [])
    corrections = []
    quotes = payload.get("source_snippets") or {}
    if not isinstance(quotes, dict):
        quotes = {}
        issues.append("Invalid source_snippets discarded")
    sources = payload.get("source_pages") or {}
    if not isinstance(sources, dict):
        sources = {}
        issues.append("Invalid source_pages discarded")
    sources = dict(sources)
    snippets = {}
    for field in schema.observation_fields():
        value = payload.get(field)
        selected = field_candidates(field, schema, candidates)
        known_kind = field in schema.DATES or field in {"holder_name", "applicant_name", "insurer", "institution", *IDENTIFIER_LABELS}
        if not known_kind:
            continue
        # Literal whole-line support for company/institution names interpreted
        # by the model, never for arbitrary numbers or legal-reference dates.
        if not selected and field in ("insurer", "institution") and isinstance(value, str):
            for chunk in chunks:
                for index, line in enumerate(chunk.text.splitlines(), 1):
                    if _equivalent(line) == _equivalent(value):
                        selected.append(_record(field, field, line, line.strip(), line, chunk.number, index))
        nonempty = [c for c in selected if c["original_value"]]
        values = {c["normalized_value"] for c in nonempty if c["normalized_value"] is not None}
        unresolved = any(c["normalized_value"] is None for c in nonempty)
        unique = len(values) == 1 and not unresolved
        # For non-insurance, match a single source value without filling nulls.
        if value is None and not insurance and not (unresolved or len(values) > 1):
            continue
        if field in ambiguous and not (insurance and unique):
            payload[field] = None
            continue
        if unresolved or len(values) > 1:
            payload[field] = None
            ambiguous.append(field)
            issues.append(f"{field}: ambiguous, invalid, or conflicting labelled source evidence; manual review needed")
            continue
        if not unique:
            if value is not None:
                payload[field] = None
                ambiguous.append(field)
                issues.append(f"{field}: value absent from reliable source candidates; discarded")
            continue
        chosen = next(c for c in nonempty if c["normalized_value"] in values)
        source_value = chosen["normalized_value"]
        same = _equivalent(value, field in schema.DATES) == _equivalent(source_value, field in schema.DATES)
        if not insurance and not same:
            payload[field] = None
            issues.append(f"{field}: value absent from reliable source candidates; discarded")
            continue
        payload[field] = source_value
        snippets[field] = chosen["snippet"]
        if field not in sources:
            sources[field] = chosen["page_number"]
        if not same:
            corrections.append(f"{field}: model value {value!r} replaced by unique labelled source evidence {source_value!r}")
        if field in ambiguous:
            ambiguous.remove(field)
            corrections.append(f"{field}: model ambiguity resolved by unique labelled source evidence")
    # Preserve unsupported quote claims for the final audit (never in final snippets).
    payload["source_snippets"] = snippets
    payload["correction_notes"] = list(dict.fromkeys(corrections))
    payload["ambiguous_fields"] = list(dict.fromkeys(ambiguous))
    payload["source_pages"] = sources
    payload["source_evidence"] = [
        {key: c[key] for key in ("field", "original_value", "normalized_value", "snippet", "page_number", "line_number")}
        for c in candidates]
    return deepcopy(quotes)


def audit_snippets(info, model_quotes, chunks):
    """Keep only real, field-supporting quotes; never let a fabricated citation pass."""
    source = "\n".join(chunk.text for chunk in chunks)
    for field, quote in model_quotes.items():
        supported = (isinstance(quote, str) and snippet_in_source(quote, source)
                     and field in info.observation_fields() and getattr(info, field) is not None
                     and field in info.source_snippets
                     and normalized_snippet(quote) == normalized_snippet(info.source_snippets[field]))
        if supported:
            continue
        # A bad model quote is informational only if final independent evidence
        # survived schema/date checks and has no field-related review issue.
        independent = (field in info.observation_fields() and getattr(info, field) is not None
                       and field in info.source_snippets
                       and snippet_in_source(info.source_snippets[field], source)
                       and not any(field in issue for issue in info.validation_issues))
        message = f"{field}: model source snippet not found or does not support the field"
        if independent:
            info.correction_notes.append(message + "; replaced with independently verified source evidence")
        else:
            info.validation_issues.append(message + "; manual review needed")
    for field, quote in list(info.source_snippets.items()):
        if field not in info.observation_fields() or getattr(info, field) is None or not snippet_in_source(quote, source):
            del info.source_snippets[field]
    info.validation_issues = list(dict.fromkeys(info.validation_issues))
    info.correction_notes = list(dict.fromkeys(info.correction_notes))
    return info
