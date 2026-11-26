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
    "policy_number": r"policy(?:\s*(?:number|no\.?|id))?|police\s*(?:no\.?|numarasi)",
    "passport_or_foreign_id": r"passport(?:\s*(?:number|no\.?))?|foreign(?:er)?\s*(?:id|identity)(?:\s*(?:number|no\.?))?|pasaport(?:\s*(?:no\.?|numarasi))?|yabanci\s*kimlik(?:\s*(?:no\.?|numarasi))?",
    "application_number": r"application\s*(?:number|no\.?|id)|basvuru\s*(?:no\.?|numarasi)|kayit\s*numarasi|registration\s*(?:number|no)",
    "student_id": r"student\s*(?:id|identifier|number)|ogrenci\s*(?:no\.?|numarasi)",
}
NAME_LABEL = re.compile(r"^(?:insured(?:\s+person)?|sigortali(?:nin)?(?:\s+(?:adi?\s*(?:/|ve)?\s*soyadi?|adi?\s+soyadi?))?|sigorta\s*ettiren(?:\s+adi?\s*soyadi?)?|policy\s*holder|holder\s*name|applicant(?:\s*name)?|full\s*name|name\s*(?:/|and)\s*surname|ad(?:i)?\s*/\s*soyad(?:i)?|ad(?:i)?\s+soyad(?:i)?|name)(?:\s*:\s*|\s*$)(.*)$")


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
            for kind,label in IDENTIFIER_LABELS.items():
                found=re.search(rf"\b(?:{label})\b",folded)
                if found and not any(c.isdigit() for c in folded[found.end():]) and index+1<len(lines):
                    value=lines[index+1].strip().lstrip(": ")
                    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9./-]{2,39}",value) and any(c.isdigit() for c in value) and not EVIDENCE_DATE_PATTERN.fullmatch(value):
                        candidates.append(dict(kind=kind,value=value,page=chunk.number,nearby=line.strip()[:90]))
            for part, labels in (("first", "name|given name|ad|adi"), ("last", "surname|sumame|family name|soyad|soyadi")):
                columns=[part.strip() for part in re.split(r"\s{2,}",line.strip()) if part.strip()]
                first_column=normalized(columns[0]) if columns else ""
                if re.match(rf"^(?:{labels})(?:\s*:|\s*$)", folded.strip()) or re.fullmatch(rf"(?:{labels})",first_column):
                    value = line.split(":", 1)[1].strip() if ":" in line else columns[1] if len(columns)>1 else ""
                    def person_part(text):
                        words=text.split()
                        return (1<=len(words)<=3 and all(w.isalpha() and len(w)>=2 for w in words)
                            and not re.search(r"\b(?:nationality|citizenship|born|country|name|surname|sumame|other|previous|place|date|adi|soyadi|uyrugu|dogum|bilgileri)\b", normalized(text))
                            and not _TEXT_EXCLUSIONS.search(normalized(text)))
                    if not person_part(value):
                        options=[lines[index+1].strip()] if index+1<len(lines) else []
                        options += [lines[j].strip() for j in range(index-1,max(-1,index-4),-1)]
                        value=next((v for v in options if person_part(v)), "")
                    if value:
                        name_parts[part].append((value,index))
            match = NAME_LABEL.match(folded.strip())
            if match:
                # Labels and values may occupy adjacent lines in PDF text.
                value = line.strip()[match.start(1):].strip()
                if not value and index + 1 < len(lines):
                    value = lines[index+1].strip()
                value = re.split(r"[;|\t]", value)[0].strip()
                words = value.replace("-", " ").replace("'", " ").split()
                if 2 <= len(words) <= 6 and all(word.isalpha() for word in words) and _person_row(value) and not {"nationality", "citizenship", "surname", "country"}.intersection(normalized(value).split()):
                    candidates.append(dict(kind="name", value=value, page=chunk.number, nearby=line.strip()[:90]))
        first={normalized(value):(value,index) for value,index in name_parts["first"]}
        last={normalized(value):(value,index) for value,index in name_parts["last"]}
        if len(first)==len(last)==1:
            given,first_index=next(iter(first.values()));surname,last_index=next(iter(last.values()))
            value=given+" "+surname
            quote="\n".join(lines[max(0,min(first_index,last_index)-3):max(first_index,last_index)+2])
            candidates.append(dict(kind="name",value=value,page=chunk.number,nearby=quote,source_snippet=quote,original_value=quote))
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
    dmy = bool(re.search(r"dd/mm/yyyy|gg/aa/yyyy|\b(?:baslangic|bitis|tanzim|duzenleme) tarihi\b", normalized))
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
    if re.search(r"(?:program(?:me)?|exchange) start date|program baslangic tarihi",label):
        return "program_start_date"
    if re.search(r"(?:program(?:me)?|exchange) end date|program bitis tarihi",label):
        return "program_end_date"
    if ("requested start date" in label and "requested end date" in label) or ("talep edilen" in label and "baslangic" in label and "bitis" in label):
        return "requested_period"
    if re.search(r"requested (?:start|begin)(?: date)?|talep edilen.*baslangic.*tarihi", label):
        return "requested_start_date"
    if re.search(r"requested end(?: date)?|talep edilen.*bitis.*tarihi", label):
        return "requested_end_date"
    if "requested period" in label:
        return "requested_period"
    if (("baslangic" in label and "bitis" in label and "tarih" in label) or
            re.search(r"\b(?:coverage|travel) period\b|^coverage\s*:", label)):
        return "coverage_period"
    if re.search(r"police (?:onay|tanzim|duzenleme) tarihi|policy (?:issue|approval) date|\bissue date\b", label):
        return "issue_date"
    if re.search(r"duzenleme tarihi|tanzim tarihi|document issuance date|evrak tarih(?: ve sayisi)?|belge tarihi", label):
        return "document_issue_date"
    if re.search(r"(?:(?:sigorta|teminat|police|seyahat) )?baslangic tarihi|coverage start(?: date)?|inception date|effective (?:start )?date", label):
        return "coverage_start_date"
    if re.search(r"(?:(?:sigorta|teminat|police|seyahat) )?bitis tarihi|coverage end(?: date)?|coverage expires|policy expiration date", label):
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
    paired_labels = set()
    for index, line in enumerate(lines):
        if index in paired_labels:
            continue
        label = date_label(line)
        block = [(index, line)]
        # Portable table layout: adjacent start/end headers followed by a row
        # of two dates. Never associate all values with the last header.
        if label in {"coverage_start_date", "requested_start_date"} and not EVIDENCE_DATE_PATTERN.search(line) and index + 2 < len(lines):
            end_label = label.replace("start", "end")
            if date_label(lines[index + 1]) == end_label and not EVIDENCE_DATE_PATTERN.search(lines[index + 1]):
                numeric = []
                for row in range(index + 2, min(index + 5, len(lines))):
                    if date_label(lines[row]) or LEGAL_REFERENCE.search(fold(lines[row])) or UNRELATED_FIELD.search(fold(lines[row])):
                        break
                    rest = EVIDENCE_DATE_PATTERN.sub("", lines[row]).strip(" \t|:;/.-")
                    if rest:
                        break
                    numeric.extend((row, match) for match in EVIDENCE_DATE_PATTERN.finditer(lines[row]))
                if len(numeric) == 2:
                    snippet = "\n".join(lines[index:numeric[-1][0] + 1])
                    for role, (row, match) in zip((label, end_label), numeric):
                        evidence.append(dict(field=role, original_value=match.group(), normalized_value=source_date(match.group(), order),
                            snippet=snippet, line_number=row + 1, page_number=page_numbers[row]))
                        assigned.add((row, match.start()))
                    paired_labels.add(index + 1)
                    continue
        # A split Turkish period heading is common in digitally extracted PDFs.
        if (label is None and "baslangic" in fold(line) and not EVIDENCE_DATE_PATTERN.search(line)
                and index + 1 < len(lines) and "\f" not in lines[index + 1]):
            combined = line + " " + lines[index + 1]
            if date_label(combined) == "coverage_period":
                label = "coverage_period"
                block.append((index + 1, lines[index + 1]))
                paired_labels.add(index + 1)
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
        column=re.match(r"^([A-Za-z0-9][A-Za-z0-9._/-]*)(?:\s{2,}|\t)(?=[^\d\r\n:]{2,}[:#])",value)
        if column:value=column.group(1);raw_value=value
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


# Restricted textual evidence for insurance layouts, not a general name detector.
_TEXT_EXCLUSIONS = re.compile(
    r"\b(?:policy|police|coverage|certificate|application|form|period|validity|"
    r"terms|conditions|information|important|please|read|carefully|this|is|the|"
    r"document|country|nationality|citizenship|kanun\w*|law|regulation|legal|"
    r"yonetmelik\w*|geregince|kapsaminda|tarafindan|pursuant|according)\b")
_NONPERSON = re.compile(r"\b(?:insurance|sigorta|company|university|limited|ltd|inc|corp)\b")
_PERSON_CONTEXT = re.compile(r"^(?:insured|person|policy holder|sigortali)(?: (?:person|information|details|bilgileri))?\s*:?$")
_COUNTRY_ROWS = {"united states", "united kingdom", "united arab emirates", "south africa",
                 "new zealand", "republic of turkey", "republic of turkiye", "south korea", "north korea"}


def _person_row(value):
    if re.search(r"\b(?:sigortali|sigorta|adi|soyadi|bilgileri|insured|surname|holder|information)\b", fold(value)):
        return False
    words = value.replace("-", " ").replace("'", " ").split()
    folded = fold(value)
    return (2 <= len(words) <= 5 and all(len(word) >= 2 and word.isalpha() for word in words)
            and (value.isupper() or all(word[0].isupper() for word in words))
            and not _TEXT_EXCLUSIONS.search(folded) and not _NONPERSON.search(folded)
            and folded not in _COUNTRY_ROWS)


def _insurance_company_row(value):
    folded = fold(value)
    tokens=re.findall(r"[^\W\d_]+",value)
    if not tokens or not all(word[0].isupper() or word.casefold() in {"ve","and","of"} for word in tokens):return False
    if re.search(r"\b(?:birligi|association|union|federation)\b",folded):return False
    if (len(value) > 100 or not re.search(r"\b(?:sigorta|insurance)\b", folded)
            or _TEXT_EXCLUSIONS.search(folded) or LEGAL_REFERENCE.search(folded)
            or any(char in value for char in ":;=/") or EVIDENCE_DATE_PATTERN.search(value)):
        return False
    words = re.findall(r"[a-z]+", folded)
    generic = {"sigorta", "insurance", "health", "saglik", "company", "a", "s", "as", "inc", "ltd", "limited", "co", "corp", "corporation",
               "travel", "medical", "private", "personal", "general", "genel", "international"}
    boilerplate = {"for", "foreigners", "sartlari", "contract", "agreement", "under", "issued"}
    return (2 <= len(words) <= 9 and words[0] not in generic
            and not boilerplate.intersection(words)
            and any(len(word) > 1 and word not in generic for word in words))


def _company_span(value):
    """Literal company span; product headings and legal prose are not providers."""
    segments = [part.strip() for part in re.split(r"\s{2,}", value.strip()) if part.strip()]
    for segment in reversed(segments or [value]):
        pattern = r"(?<!\w)(?:[A-Z\u00c0-\u024f][\w&.-]*[ \t]+){0,5}(?:S[\u0130I]GORTA|Sigorta|Insurance|INSURANCE)(?!\w)(?:[ \t]+(?:A\.\s*[\u015eS]\.|A\.?[\u015eS]\.?|Anonim[ \t]+[\u015eS]irketi|Ltd\.?|Limited|Inc\.?))?"
        found = list(re.finditer(pattern, segment))
        if found:
            raw=found[-1].group().strip()
            # Do not absorb a preceding field label into the provider span.
            raw=re.sub(r"^(?:SIGORTA SAHIBI|S[\u0130I]GORTA SAH[\u0130I]B[\u0130I])\s+", "", raw)
            if _insurance_company_row(raw):return raw
    return value.strip() if _insurance_company_row(value.strip()) else None


def _insurance_text_candidates(chunks):
    """Exact short header/company lines and person rows with DOB/context support."""
    evidence = []
    for chunk in chunks:
        lines = chunk.text.splitlines()
        nonempty = 0
        previous = ""
        for number, line in enumerate(lines, 1):
            row = line.strip()
            if not row:
                continue
            nonempty += 1
            # Company mentions are useful in short insurance header lines, not
            # legal paragraphs or arbitrary mentions elsewhere on the page.
            company_rows=[part.strip() for part in re.split(r"\s{2,}",row) if part.strip()]
            company=next((_company_span(part) for part in reversed(company_rows) if _insurance_company_row(part)),None) if not LEGAL_REFERENCE.search(fold(row)) else None
            if company:
                company = company
                candidate = _record("insurer", "insurer", company, company, line, chunk.number, number)
                candidate["insurance_only"] = True
                evidence.append(candidate)
            matches = list(EVIDENCE_DATE_PATTERN.finditer(row))
            name = None
            if len(matches) == 1 and not row[matches[0].end():].strip():
                birthday = normalized_date(matches[0].group())
                prefix = row[:matches[0].start()].strip()
                if birthday and date.fromisoformat(birthday) <= date.today() and _person_row(prefix):
                    name = prefix
            elif not matches and _PERSON_CONTEXT.fullmatch(fold(previous)) and _person_row(row):
                name = row
            if name:
                candidate = _record("name", "holder_name", name, name, line, chunk.number, number)
                candidate["insurance_only"] = True
                evidence.append(candidate)
            previous = row
    return evidence



def administrative_candidates(chunks):
    """Narrow bilingual certificate/form evidence; quotes always come from source."""
    evidence = []
    months = {word: i for i, pair in enumerate((('january','ocak'),('february','subat'),('march','mart'),('april','nisan'),('may','mayis'),('june','haziran'),('july','temmuz'),('august','agustos'),('september','eylul'),('october','ekim'),('november','kasim'),('december','aralik')),1) for word in pair}
    months.update({'\u015fubat':2,'may\u0131s':5,'a\u011fustos':8,'eyl\u00fcl':9,'kas\u0131m':11,'aral\u0131k':12})
    pattern = re.compile(r'\b('+'|'.join(months)+r')\s+(\d{1,2}),?\s+(20\d{2})|\b(\d{1,2})\s+('+'|'.join(months)+r')\s+(20\d{2})',re.I)
    for chunk in chunks:
        lines=chunk.text.splitlines()
        if "personal information" in normalized(chunk.text[:250]):
            parts={}
            for n,row in enumerate(lines[:-1]):
                key=normalized(row).strip()
                if key in {"adi","soyadi"}:
                    value=lines[n+1].strip()
                    if value and len(value.split()) <= 3 and all(w.isalpha() for w in value.split()):
                        parts[key]=(value,n)
            if set(parts)=={"adi","soyadi"}:
                value=parts["adi"][0]+" "+parts["soyadi"][0]
                start=min(v[1] for v in parts.values());end=max(v[1] for v in parts.values())+2
                evidence.append(_record("name","applicant_name",value,value,"\n".join(lines[start:end]),chunk.number,start+1))
        for index,line in enumerate(lines):
            folded=normalized(line)
            def add(field,original,value,snippet=line):
                evidence.append(_record("date" if field.endswith("_date") else field,field,original,value,snippet,chunk.number,index+1))
            # Explicit Turkish person labels remain reliable even if the model
            # omits the field. Do not extend this to unlabelled uppercase rows.
            if re.match(r"^\s*ad(?:i)?\s*(?:/|ve)?\s*soyad(?:i)?\s*:",folded):
                raw=line.split(":",1)[1].strip()
                if _person_row(raw):
                    evidence.append(_record("name","holder_name",raw,raw,line,chunk.number,index+1))
            if re.search(r'\b(?:exchange student|degisim ogrencisi)',folded):
                add('exchange_student',line,'false' if re.search(r'not (?:an? )?exchange student|degisim ogrencisi degil',folded) else 'true')
            # Academic periods require explicit exchange context, never education tables.
            block='\n'.join(lines[max(0,index-2):index+2])
            if ('exchange' in normalized(block) or 'degisim' in normalized(block)):
                found=list(pattern.finditer(re.sub(r'(?i)between(?=[A-Z])','between ',line)))
                if len(found)==2:
                    for role,match in zip(('program_start_date','program_end_date'),found):
                        month,day,year=match.group(1,2,3) if match.group(1) else (match.group(5),match.group(4),match.group(6))
                        try: value=date(int(year),months[month.casefold()],int(day)).isoformat()
                        except ValueError: value=None
                        # Original string taken from the actual row, not the normalized parser.
                        original=match.group()
                        add(role,original,value,block)
            if 'short term residence permit registration form' in ' '.join(folded.split()) or 'ikamet izni kisa donem kayit formu' in ' '.join(folded.split()):
                add('permit_type',line,'short_term')
            if re.match(r'^\s*(?:isim\s*/\s*name|ad[i]?\s*/\s*name)\s*:?',folded) and index+1<len(lines):
                raw=lines[index+1].strip().lstrip(':').strip()
                if _person_row(raw):
                    evidence.append(_record('name','holder_name',raw,raw,line+'\n'+lines[index+1],chunk.number,index+1))
            if re.search(r'\bat sabanci university\b',folded):
                match=re.search(r'Sabanc[\u0131i] University',line,re.I)
                if match: add('institution',match.group(),match.group())
            if folded.strip() == "belge no" and index+1<len(lines) and "pasaport" in normalized(" ".join(lines[max(0,index-4):index])):
                value=lines[index+1].strip()
                if re.fullmatch(r"[A-Za-z0-9]{5,20}",value) and any(c.isdigit() for c in value):
                    add("passport_or_foreign_id",value,value,"\n".join(lines[max(0,index-3):index+2]))
            if folded.strip() == "number)" and index>0 and "registration" in normalized(lines[index-1]) and index+1<len(lines):
                value=lines[index+1].strip()
                if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{4,12}",value):
                    add("application_number",value,value,"\n".join(lines[index-1:index+2]))
            if "kayit numarasi" in folded or "registration number" in folded:
                block="\n".join(lines[index:index+3])
                identifiers=set(re.findall(r"\b\d{4}-\d{1,4}-\d{4,14}\b",block))
                if len(identifiers)==1:
                    value=next(iter(identifiers));add("application_number",value,value,block)
            # Header issue date on certificates, before DOB/enrollment labels.
            if index<4 and DATE_PATTERN.fullmatch(line.strip()):
                add('issue_date',line.strip(),normalized_date(line.strip()))
            # Known PDF reading order: start label, start value, end value, end label.
            if 'requested start date' in folded and index+4<len(lines):
                rows=lines[index+1:index+3]
                if 'requested end date' in normalized(' '.join(lines[index+3:index+6])):
                    matches=[EVIDENCE_DATE_PATTERN.fullmatch(row.strip()) for row in rows]
                    if all(matches):
                        for role,row in zip(('requested_start_date','requested_end_date'),rows):
                            add(role,row.strip(),source_date(row.strip(),'dmy'),'\n'.join(lines[index:index+6]))
    for item in evidence:
        item["administrative"] = True
    return evidence



# A tax-office payment receipt needs its own purpose signal. A taxpayer ID is
# an account identifier and never a payment reference or a fee code.
_TAX_OFFICE_RECEIPT = re.compile(
    r"(?:t\.c\.\s*)?(?:vergi dairesi (?:e[- ]?)?alindi(?:si)?|(?:vergi )?tahsilat alindisi|tax office (?:payment )?receipt)"
)
_CARD_FEE_PURPOSE = re.compile(
    r"\b(?:ikametgah\s+tezkeresi\s+defter\s+satis\s+bedeli|ikamet(?:\s+izin)?\s+(?:kart|belge)\s+bedeli"
    r"|ikametg?\.?\s*tezk\.?\s*(?:defter|dft)\.?\s*(?:satis|sat\.?)?\s*bed(?:eli|\.)?"
    r"|residence (?:permit )?(?:card|document) fee)\b"
)
_TAX_ASSIGNMENT = re.compile(
    r"(?:turkish |potential )?tax (?:id|identification number) (?:assignment(?: certificate)?|certificate)"
    r"|(?:potansiyel |yabancilar icin )?vergi kimlik numarasi (?:tahsis(?:i| belgesi)?|belgesi)"
    r"|(?:potansiyel|yabancilar icin) vergi kimlik numarasi"
)
_FEE_CODE_LABEL = r"vergi (?:kodu|turu|cinsi)|(?:odeme|tahsilat|gelir) kodu|payment code"
_LABELED_FEE_CODE = re.compile(rf"^(?:{_FEE_CODE_LABEL})\s*[:：]?\s*9207(?=$|[\s\-–—:])")
_FEE_CODE_ROW = re.compile(r"^9207(?:\s*[-–—:]\s*|\s+)(?=[a-z])")
_PASSPORT_LABEL = r"(?:passport|pasaport)(?:\s+(?:number|numarasi|no\.?|nr\.?))?"
_TAXPAYER_LABEL = r"(?:vergi (?:kimlik )?(?:numarasi|no\.?)|vkn|tax (?:id|identification number)|taxpayer id)"
_RECEIPT_FIELDS = {
    "payment_date": r"(?:odeme|tahsilat|tahsil) tarihi|payment date|collection date",
    "reference_number": r"(?:alindi|makbuz|dekont|tahsilat|islem) (?:no\.?|numarasi)|seri\s*(?:/|ve)?\s*sira no\.?|receipt (?:no\.?|number)|reference (?:no\.?|number)",
    "amount": r"toplam(?: tutar[i]?)?|(?:odenen|tahsil edilen) tutar[i]?|tutar[i]?|(?:paid )?amount|total(?: paid)?",
    "taxpayer_id": rf"(?:tckn\s*/\s*)?{_TAXPAYER_LABEL}",
    "holder_name": r"(?:mukellef(?:in)? )?ad[i]?\s*(?:/|ve)?\s*soyad[i]?(?:\s*/\s*unvan[i]?)?|taxpayer name|payer name",
}
_ASSIGNMENT_FIELDS = {
    "tax_identification_number": _TAXPAYER_LABEL,
    "holder_name": _RECEIPT_FIELDS["holder_name"] + r"|holder name|full name",
    "passport_number": _PASSPORT_LABEL,
    "issuing_authority": r"verg[iı] dairesi(?: adi)?|issuing authority|tax office|duzenleyen kurum",
    "issue_date": r"(?:tahsis|duzenleme|kayit) tarihi|issue date|(?:registration|assignment) date",
    "document_reference": r"belge (?:no\.?|numarasi)|basvuru (?:no\.?|numarasi)|document (?:reference|no\.?|number)|reference (?:no\.?|number)",
}


def _passport_token(text):
    return bool(re.fullmatch(r"[A-Za-z0-9]{5,20}", text) and any(char.isdigit() for char in text))


def _fee_code_line(lines, index):
    line = label_text(lines[index])
    if _LABELED_FEE_CODE.match(line) or _FEE_CODE_ROW.match(line):
        return True
    return index > 0 and line == "9207" and bool(
        re.fullmatch(rf"(?:{_FEE_CODE_LABEL})\s*[:：]?", label_text(lines[index - 1]))
    )


def tax_document_type(pages):
    """Return a supported tax profile, or an unsupported purpose for a tax-office receipt."""
    lines = [line for page in pages for line in page.text.splitlines()]
    if any(_TAX_OFFICE_RECEIPT.fullmatch(label_text(line).strip(" :-–")) for line in lines):
        return "card_fee_receipt" if (_CARD_FEE_PURPOSE.search(label_text(" ".join(lines)))
                                      or any(_fee_code_line(lines, i) for i in range(len(lines)))) else "unknown"
    if any(_TAX_ASSIGNMENT.fullmatch(label_text(line).strip(" :-–")) for line in lines):
        return "tax_id_certificate"
    return None


def _same_ocr_row(first, second):
    top, bottom = max(first[1], second[1]), min(first[3], second[3])
    return bottom - top >= min(first[3] - first[1], second[3] - second[1]) * 0.35


def _receipt_cell(page, lines, index, field, label):
    bounds = page.ocr_bounds
    if page.extraction_method != "rapidocr" or not bounds or len(bounds) != len(lines):
        return None
    anchor = bounds[index]
    right = [i for i, box in enumerate(bounds) if box[0] >= anchor[2] and _same_ocr_row(anchor, box)]
    if field == "amount" and label in {"tutar", "tutari"}:
        # The table column contains several charges. Keep the cell on the 9207
        # row below this heading and before the next column heading.
        edge = min((bounds[i][0] for i in right), default=anchor[0] + 4 * (anchor[2] - anchor[0]))
        by_code = [
            (i, j) for i, code in enumerate(bounds)
            if code[1] > anchor[1] and not _same_ocr_row(anchor, code) and _fee_code_line(lines, i)
            for j, value in enumerate(bounds) if anchor[0] <= value[0] < edge and _same_ocr_row(code, value)
            and _tax_value("amount", lines[j].strip()) is not None
        ]
        if len(by_code) == 1:
            code, cell = by_code[0]
            return lines[cell].strip(), "\n".join((lines[index], lines[code], lines[cell]))
    elif right:
        cell = min(right, key=lambda i: bounds[i][0])
        return lines[cell].strip(), lines[index] + "\n" + lines[cell]
    return None


def _tax_labeled_values(page, lines, labels, *, receipt):
    for index, line in enumerate(lines):
        parts = re.split(r"[:：]", line, maxsplit=1)
        label = label_text(parts[0])
        for field, pattern in labels.items():
            if not re.fullmatch(pattern, label):
                continue
            raw = parts[1].strip() if len(parts) == 2 else ""
            quote = line
            if not raw:
                cell = _receipt_cell(page, lines, index, field, label) if receipt else None
                if cell:
                    raw, quote = cell
                elif index + 1 < len(lines) and not re.search(r"[:：]", lines[index + 1]):
                    raw = lines[index + 1].strip()
                    quote += "\n" + lines[index + 1]
            if raw:
                yield field, raw, quote, index + 1


def _tax_value(field, raw):
    if field.endswith("_date"):
        text = re.sub(r"\s*[/.-]\s*", "/", raw)
        if re.fullmatch(r"\d{4}/\d{2}/\d{2}", text):
            return normalized_date(text.replace("/", "-"))
        return source_date(raw, "dmy")
    if field in {"reference_number", "document_reference"}:
        number = re.sub(r"\s*([./-])\s*", r"\1", raw)
        return raw if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9./-]{2,39}", number) and any(c.isdigit() for c in number) else None
    if field in {"taxpayer_id", "tax_identification_number"}:
        digits = "".join(raw.split())
        return digits if re.fullmatch(r"[0-9]{10,11}" if field == "taxpayer_id" else r"[0-9]{10}", digits) else None
    if field == "passport_number":
        return raw if _passport_token(raw) else None
    if field == "amount":
        amount = raw[1:-1].strip() if raw.startswith("#") and raw.endswith("#") else raw
        number = r"(?:\d{1,3}(?:[. ]\d{3})+(?:,\d{2})?|\d{1,3}(?:,\d{3})+(?:\.\d{2})?|\d+(?:[.,]\d{2})?)"
        return amount if re.fullmatch(rf"₺?\s*{number}(?:\s*(?:TL|TRY|₺|EUR|USD))?", amount, re.I) else None
    if field == "holder_name":
        return raw if _person_row(raw) else None
    return raw if any(char.isalpha() for char in raw) and not re.search(r"\[(?:unreadable|illegible|unclear)\]", raw, re.I) else None


def read_tax_document(pages, kind):
    from src.document_processing.schemas import DOCUMENT_SCHEMAS

    receipt = kind == "card_fee_receipt"
    schema = DOCUMENT_SCHEMAS[kind]
    candidates, receipt_headings = [], []
    for page in pages:
        lines = page.text.splitlines()
        method = {"multimodal_fallback": "openai_vision"}.get(page.extraction_method, page.extraction_method)

        def add(field, raw, normalized_value, quote, index):
            record = _record("name" if field == "holder_name" else field, field, raw, normalized_value,
                             quote, page.number, index)
            candidates.append({**record, "profile": schema.__name__, "method": method})

        labels = _RECEIPT_FIELDS if receipt else _ASSIGNMENT_FIELDS
        for field, raw, quote, index in _tax_labeled_values(page, lines, labels, receipt=receipt):
            add(field, raw, _tax_value(field, raw), quote, index)
        for index, line in enumerate(lines):
            folded = label_text(line)
            if receipt and _TAX_OFFICE_RECEIPT.fullmatch(folded.strip(" :-–")):
                receipt_headings.append((page.number, index + 1, line, method))
            if receipt and _fee_code_line(lines, index):
                quote = ("\n".join(lines[index - 1:index + 1])
                         if index and folded == "9207" else line)
                add("payment_code", "9207", "9207", quote, index + 1)
            if not receipt and folded.strip(" :-–") in {"gelir idaresi baskanligi", "revenue administration"}:
                add("issuing_authority", line.strip(), line.strip(), line, index + 1)
    # The amount on the 9207 line represents the fee even if the receipt
    # includes other charges or an unreadable printed total elsewhere.
    fee_amounts = [candidate for candidate in candidates if candidate["field"] == "amount"
                   and len(lines := candidate["snippet"].splitlines()) == 3 and _fee_code_line(lines, 1)]
    if fee_amounts:
        candidates = [candidate for candidate in candidates if candidate["field"] != "amount"] + fee_amounts
    # A fee category and a payer ID on their own do not show a transaction.
    low_confidence = {page.number: page.low_confidence_lines or () for page in pages if page.extraction_method == "rapidocr"}
    transaction = False
    for field in ("payment_date", "reference_number", "amount"):
        readings = [candidate for candidate in candidates if candidate["field"] == field]
        values = {candidate["normalized_value"] for candidate in readings}
        if len(values) == 1 and None not in values and any(
            not any(line in candidate["snippet"] for line in low_confidence.get(candidate["page"], ()))
            for candidate in readings
        ):
            transaction = True
            break
    if receipt and transaction:
        candidates.extend(
            {**_record("receipt_present", "receipt_present", quote, "true", quote, number, index),
             "profile": schema.__name__, "method": method}
            for number, index, quote, method in receipt_headings
        )
    return _labelled_document(schema, pages, candidates)


LATEST_ENTRY_LABEL = r"latest (?:(?:turkiye|turkey) )?entry date|son giris tarihi"


def preparation_candidates(chunks):
    """Conservative profiles: printed labels/headings, never visual signature inference."""
    output=[]
    for chunk in chunks:
        lines=chunk.text.splitlines()
        text=' '.join(normalized(chunk.text).split())
        profiles=[]
        tax_receipt=(any(word in text for word in ('tahsilat','odeme makbuzu','payment receipt','collection receipt')) and any(word in text for word in ('ikamet izin belge bedeli','ikamet belge bedeli','residence permit card','residence permit document fee')))

        if 'passport' in text or 'pasaport' in text: profiles.append('PassportIdentity')
        if any(title in text for title in ('turkiye entry stamp','turkey entry stamp','turkiye giris','turkey entry date')) or (('turkiye' in text or 'turkey' in text) and any(normalized(line).strip() in {'entry','entry stamp','giris'} for line in lines)): profiles.append('EntryStamp')
        if 'declaration control' in text or 'beyan kontrol' in text or 'beyan/kontrol' in text: profiles.append('DeclarationControl')
        if tax_receipt or any(title in text for title in ('card fee payment receipt','document fee payment receipt','ikamet belge bedeli','residence permit card fee receipt')): profiles.append('CardFeeReceipt')
        for profile in profiles:
            for index,line in enumerate(lines):
                folded=normalized(line).strip()
                def add(field,raw,value,quote=line):
                    c=_record('date' if field.endswith('_date') else field,field,raw,value,quote,chunk.number,index+1)
                    c['profile']=profile;output.append(c)
                if profile=='PassportIdentity':
                    if re.fullmatch(r'(?:passport(?: identity page)?|pasaport)',folded): add('identity_page_present',line,'true')
                    labels={'passport_number':r'(?:passport (?:number|no)|pasaport no)', 'nationality':r'(?:nationality|uyrugu|uyruk)'}
                elif profile=='EntryStamp':
                    if any(title in folded for title in ('turkiye entry stamp','turkey entry stamp','turkiye giris')) or folded in {'entry','entry stamp','giris'}: add('entry_stamp_present',line,'true')
                    labels={'latest_entry_date':rf'(?:{LATEST_ENTRY_LABEL}|(?:turkey |turkiye )?entry date|giris tarihi)'}
                elif profile=='DeclarationControl':
                    if 'declaration control' in folded or 'beyan kontrol' in folded or 'beyan/kontrol' in folded: add('document_present',line,'true')
                    labels={'signed':r'(?:signed|signature status|signed status|imzali)','passport_number':r'(?:passport number|pasaport no)','foreigner_id':r'(?:foreigner id|foreign id|yabanci kimlik no)','nationality':r'(?:nationality|uyrugu)','application_number':r'(?:application number|basvuru no)'}
                else:
                    if any(title in folded for title in ('card fee payment receipt','document fee payment receipt','ikamet belge bedeli','residence permit card fee receipt')) or (tax_receipt and any(word in folded for word in ('tahsilat','odeme makbuzu','payment receipt','collection receipt'))): add('receipt_present',line,'true')
                    labels={'payment_date':r'(?:payment date|collection date|tahsilat tarihi|odeme tarihi)','reference_number':r'(?:receipt (?:number|no)|reference (?:number|no)|makbuz no|dekont no|tahsilat no|alindi no)','amount':r'(?:amount|paid amount|tutar)'}
                for field,label in labels.items():
                    match=re.match(r'^'+label+r'\s*:\s*(.*)$',folded)
                    if not match: continue
                    raw=line.split(':',1)[1].strip();quote=line
                    if not raw and index+1<len(lines): raw=lines[index+1].strip();quote+='\n'+lines[index+1]
                    value=raw
                    if field.endswith('_date'): value=source_date(raw,date_order(chunk.text))
                    elif field=='signed': value={'yes':'true','signed':'true','evet':'true','no':'false','unsigned':'false','hayir':'false'}.get(normalized(raw))
                    elif field in {'passport_number','reference_number','foreigner_id','application_number'}:
                        value=raw if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9/-]{2,39}',raw) and any(c.isdigit() for c in raw) else None
                    elif field=='nationality': value=raw if re.fullmatch(r'[\w -]{2,50}',raw) and not any(c.isdigit() for c in raw) else None
                    elif field=='amount': value=raw if re.fullmatch(r'\d[\d., ]*(?:TRY|TL|EUR|USD)?',raw,re.I) else None
                    add(field,raw,value,quote)
        # Unlabelled stamp dates are observations only; multiple dates remain ambiguous.
        if 'EntryStamp' in profiles and not any(c.get('profile')=='EntryStamp' and c['field']=='latest_entry_date' and c['page_number']==chunk.number for c in output):
            for index,line in enumerate(lines):
                for match in EVIDENCE_DATE_PATTERN.finditer(line):
                    c=_record('date','latest_entry_date',match.group(),source_date(match.group(),date_order(chunk.text)),line,chunk.number,index+1)
                    c['profile']='EntryStamp';output.append(c)
    # A receipt heading alone (or a Tax ID) is not transaction evidence.
    output=[c for c in output if not (c.get('profile')=='CardFeeReceipt' and c['field']=='receipt_present' and not any(other.get('profile')=='CardFeeReceipt' and other['page_number']==c['page_number'] and other['field'] in {'payment_date','reference_number','amount'} and other['normalized_value'] is not None for other in output))]
    return output



# ISO 3166-1 alpha-3 snapshot for identity-code validation, not visa rules.
# References: ISO 3166 maintenance agency; ICAO Doc 9303 parts 3/4.
ISO_ALPHA3_CODES=frozenset("""AFG ALA ALB DZA ASM AND AGO AIA ATA ATG ARG ARM ABW AUS AUT AZE BHS BHR BGD BRB BLR BEL BLZ BEN BMU BTN BOL BES BIH BWA BVT BRA IOT BRN BGR BFA BDI CPV KHM CMR CAN CYM CAF TCD CHL CHN CXR CCK COL COM COG COD COK CRI CIV HRV CUB CUW CYP CZE DNK DJI DMA DOM ECU EGY SLV GNQ ERI EST SWZ ETH FLK FRO FJI FIN FRA GUF PYF ATF GAB GMB GEO DEU GHA GIB GRC GRL GRD GLP GUM GTM GGY GIN GNB GUY HTI HMD VAT HND HKG HUN ISL IND IDN IRN IRQ IRL IMN ISR ITA JAM JPN JEY JOR KAZ KEN KIR PRK KOR KWT KGZ LAO LVA LBN LSO LBR LBY LIE LTU LUX MAC MDG MWI MYS MDV MLI MLT MHL MTQ MRT MUS MYT MEX FSM MDA MCO MNG MNE MSR MAR MOZ MMR NAM NRU NPL NLD NCL NZL NIC NER NGA NIU NFK MKD MNP NOR OMN PAK PLW PSE PAN PNG PRY PER PHL PCN POL PRT PRI QAT REU ROU RUS RWA BLM SHN KNA LCA MAF SPM VCT WSM SMR STP SAU SEN SRB SYC SLE SGP SXM SVK SVN SLB SOM ZAF SGS SSD ESP LKA SDN SUR SJM SWE CHE SYR TWN TJK TZA THA TLS TGO TKL TON TTO TUN TUR TKM TCA TUV UGA UKR ARE GBR USA UMI URY UZB VUT VEN VNM VGB VIR WLF ESH YEM ZMB ZWE""".split())


def _mrz_nationality(raw,issuer):
    if raw in ISO_ALPHA3_CODES:return raw
    if raw=='D<<':return 'DEU'  # ICAO Germany representation.
    if raw=='UTO' and issuer=='UTO':return raw  # Preserve official ICAO specimen fixtures.
    corrected=raw.translate(str.maketrans({'1':'I','0':'O'}))
    return corrected if corrected!=raw and corrected in ISO_ALPHA3_CODES else None


def mrz_candidates(chunks):
    """TD3 checks first; OCR correction is confined to its nationality segment.

    Nationality is excluded from the composite checksum. It additionally needs
    an ISO code (or an explicit ICAO representation), never arbitrary OCR repair.
    """
    def check(text,digit):
        def number(c): return 0 if c=='<' else ord(c)-55 if 'A'<=c<='Z' else int(c)
        return digit.isdigit() and sum(number(c)*(7,3,1)[i%3] for i,c in enumerate(text))%10==int(digit)
    evidence=[]
    for chunk in chunks:
        lines=chunk.text.splitlines()
        for index in range(len(lines)-1):
            a,b=(re.sub(r'\s+','',line).upper() for line in lines[index:index+2])
            if len(a)!=44 or len(b)!=44 or not a.startswith('P<') or not re.fullmatch(r'[A-Z0-9<]+',a+b):continue
            if not (check(b[:9],b[9]) and check(b[13:19],b[19]) and check(b[21:27],b[27]) and check(b[:10]+b[13:20]+b[21:43],b[43])):continue
            try:expiry=date(2000+int(b[21:23]),int(b[23:25]),int(b[25:27])).isoformat()
            except ValueError:continue
            surname,separator,given=a[5:].partition('<<')
            name=' '.join((given+' '+surname).replace('<',' ').split())
            nationality=_mrz_nationality(b[10:13],a[2:5])
            values={'holder_name':(a[5:],name),'passport_number':(b[:9],b[:9].rstrip('<')),'nationality':(b[10:13],nationality),'expiry_date':(b[21:27],expiry),'identity_page_present':(a,'true')}
            if separator and a.endswith('<'):
                values.update(first_name=(given,' '.join(given.replace('<',' ').split())),last_name=(surname,' '.join(surname.replace('<',' ').split())))
            # Select a DOB century only when one plausible, non-future date exists.
            births=[]
            for century in (1900,2000):
                try:born=date(century+int(b[13:15]),int(b[15:17]),int(b[17:19]))
                except ValueError:continue
                if 0<=(date.today()-born).days<=110*366:births.append(born.isoformat())
            if len(births)==1:values['birth_date']=(b[13:19],births[0])
            for field,(raw,value) in values.items():
                c=_record('date' if field in {'expiry_date','birth_date'} else 'name' if field=='holder_name' else field,field,raw,value,'\n'.join(lines[index:index+2]),chunk.number,index+1)
                c['profile']='PassportIdentity';c['validated_mrz']=True
                if field=='nationality' and value and raw!=value:
                    c['mrz_correction']=f'nationality: MRZ source code {raw!r} normalized to {value!r} after TD3 structure/checks and country-code validation'
                evidence.append(c)
    return evidence


_ADDRESS_HEADINGS = {
    "dorm_certificate": r"(?:dorm(?:itory)? (?:residence )?certificate|(?:ogrenci )?yurt (?:ikamet |barinma )?belgesi|yurtta kaldigina dair belge)",
    "rental_agreement": r"(?:(?:notarized )?rental (?:agreement|contract)|(?:noter onayli )?kira (?:sozlesmesi|kontrati))",
    "notarized_commitment": r"(?:(?:notarized )?(?:(?:host )?undertaking|letter of commitment)|(?:noter onayli )?taahhutname)",
    "host_support": r"(?:host supporting (?:documents|evidence)|ev sahibi destekleyici belgeleri)",
    "utility_subscription": r"(?:(?:new )?utility subscription (?:agreement|contract)|(?:elektrik |su |dogalgaz |sabit telefon )?abonelik sozlesmesi)",
}
_ADDRESS_LABELS = {
    "declared_address": r"(?:(?:residence|rental|dormitory|property|host|supporting) address|address|adres(?:i)?|ikamet adresi|kiralananin adresi|yurt adresi)",
    "service_address": r"(?:service address|subscription address|address|adres(?:i)?|abonelik adresi)",
    "host_name": r"(?:host(?: name)?|guarantor(?: name)?|taahhut eden(?:in adi soyadi)?|ev sahibi adi soyadi)",
    "landlord_name": r"(?:landlord(?: name)?|lessor(?: name)?|kiraya veren(?:in adi soyadi)?)",
    "provider": r"(?:utility provider|provider|service provider|hizmet saglayici)",
}
_ADDRESS_HOLDER_LABELS = {
    "dorm_certificate": r"(?:student(?: name)?|resident(?: name)?|ogrenci(?:nin adi soyadi)?|ad soyad(?:i)?)",
    "rental_agreement": r"(?:tenant(?: name)?|lessee(?: name)?|kiraci(?:nin adi soyadi)?)",
    "notarized_commitment": r"(?:guest(?: name)?|student(?: name)?|misafir(?:in adi soyadi)?|ogrenci(?:nin adi soyadi)?)",
    "utility_subscription": r"(?:customer(?: name)?|subscriber(?: name)?|abone(?:nin adi soyadi)?|musteri adi soyadi)",
}


def address_evidence_type(text):
    """Recognize an accommodation document heading before any model classification."""
    for line in text.splitlines()[:18]:
        heading = " ".join(normalized(line).strip(" :–—-").split())
        for kind, pattern in _ADDRESS_HEADINGS.items():
            if re.fullmatch(pattern, heading):
                return kind
    return None


def read_address_evidence(pages, kind):
    """Read unambiguous labels with original-page provenance. Review is explicit."""
    from src.document_processing.schemas import ADDRESS_EVIDENCE_SCHEMAS, SourceEvidence

    schema = ADDRESS_EVIDENCE_SCHEMAS[kind]
    labels = {**_ADDRESS_LABELS, "holder_name": _ADDRESS_HOLDER_LABELS.get(kind, r"(?!)")}
    observations = {}
    fields = sorted(schema.observation_fields() & labels.keys())
    for page in pages:
        for number, line in enumerate(page.text.splitlines(), 1):
            label, separator, value = line.partition(":")
            if not separator or not value.strip():
                continue
            for field in fields:
                if re.fullmatch(labels[field], " ".join(normalized(label).split())):
                    item = SourceEvidence(field=field, original_value=value.strip(), normalized_value=value.strip(),
                                          snippet=line.strip(), page_number=page.number, line_number=number)
                    observations.setdefault(field, []).append(item)
    values, sources, ambiguous = {}, {}, []
    for field, items in observations.items():
        if len({" ".join(item.normalized_value.casefold().split()) for item in items}) != 1:
            ambiguous.append(field)
            continue
        sources[field] = items[0]
        values[field] = sources[field].normalized_value
    return schema(
        **values, ambiguous_fields=ambiguous, selected_pages=sorted({page.number for page in pages}),
        source_pages={field: item.page_number for field, item in sources.items()},
        source_snippets={field: item.snippet for field, item in sources.items()},
        source_evidence=list(sources.values()),
    )


def address_form_signal(text):
    """Heading or explicit address-declaration body, never a policy mention."""
    full=normalized(text)
    headings=any(re.fullmatch(r"(?:sabanci university )?(?:address\s*form|address declaration form|adres (?:beyan )?formu)"," ".join(normalized(line).split()).strip(" /-")) for line in text.splitlines()[:30])
    declaration=("adres bilgilerim" in full and "beyan ederim" in full and "address in" in full and "first name" in full and "date of birth" in full and "signature" in full)
    return headings or declaration


_ADDRESS_FORM_LABELS = {
    "nationality": r"nationality|citizenship|uyruk|uyrugu",
    "passport_number": _PASSPORT_LABEL,
    "first_name": r"first name|given name|name|ad|adi",
    "last_name": r"last name|last\s*/\s*family name|family name|surname|soyad|soyadi",
    "phone": r"(?:mobile|phone|telephone|telefon)(?: number| numarasi| no\.?)?",
    "declared_address": r"(?:(?:declared|residence|residential) address|address(?: in [\w ]+)?|adres|(?:ikamet|adres) bilgileri|ikamet adresi)",
    "birth_date": r"birth date|date of birth|dogum tarihi",
    "form_date": r"form date|date|tarih",
    "signature_present": r"signature present|signed|imzali",
}


def _address_form_field(label):
    # Compare whole labels and their bilingual counterparts.
    folded = label_text(label)
    if all(re.fullmatch(_PASSPORT_LABEL, part.strip()) for part in folded.split("/")):
        return "passport_number"
    for part in re.split(r"[()]", folded):
        part = part.strip()
        for field, labels in _ADDRESS_FORM_LABELS.items():
            if re.fullmatch(labels, part):
                return field
    return None


def _address_form_rows(page, text, method):
    lines = text.splitlines()
    order = date_order(text)
    for index, line in enumerate(lines):
        parts = re.split(r"[:：]", line, maxsplit=1)
        if len(parts) == 1:
            # Some scans read the filled box immediately before the standalone
            # bilingual passport caption. Bind only a complete isolated token.
            if (re.fullmatch(r"\(\s*(?:passport|pasaport)\s+(?:number|numarasi|no\.?)\s*\)", label_text(line))
                    and index > 0 and _passport_token(lines[index - 1].strip())
                    and (index < 2 or not re.search(r"\b(?:foreign(?:er)? id|yabanci kimlik|vergi kimlik|tax id|vkn|tckn)\b",
                                                  label_text(lines[index - 2])))):
                raw = lines[index - 1].strip()
                record = _record("passport_number", "passport_number", raw, raw,
                                 lines[index - 1] + "\n" + line, page.number, index)
                yield {**record, "profile": "AddressForm", "method": method}
            continue
        if not (field := _address_form_field(parts[0])):
            continue
        label, value = parts
        raw, quote = value.strip(), line
        if not raw and method != "pdf_form" and field != "signature_present":
            # Some bilingual scans place the printed given name above its label.
            previous = lines[index - 1].strip() if index else ""
            if (field == "first_name" and label_text(label).startswith(("ad ", "adi "))
                    and previous.isupper() and previous.isalpha()):
                raw, quote = previous, lines[index - 1] + "\n" + line
            else:
                limit = 3 if field == "declared_address" else 1
                following = []
                for candidate in lines[index + 1:index + 1 + limit]:
                    if not candidate.strip() or re.search(r"[:：]", candidate) or _address_form_field(candidate):
                        break
                    following.append(candidate)
                raw = " ".join(row.strip() for row in following)
                quote = "\n".join([line, *following])
        if not raw:
            continue
        value = raw
        if re.search(r"\[(?:unreadable|unclear|illegible)\]|\ufffd", raw, re.IGNORECASE):
            value = None
        elif field.endswith("_date"):
            value = source_date(raw, "dmy" if "tarih" in label_text(label) else order)
        elif field == "signature_present":
            value = {"yes": "true", "true": "true", "evet": "true",
                     "no": "false", "false": "false", "hayir": "false"}.get(normalized(raw))
        elif field == "passport_number" and not _passport_token(raw):
            value = None
        elif field == "phone" and not (re.fullmatch(r"[+()0-9 .-]{7,25}", raw)
                                       and 7 <= sum(char.isdigit() for char in raw) <= 18):
            value = None
        elif field in {"first_name", "last_name", "nationality"} and (
            len(raw.split()) > 5 or not all(c.isalpha() or c.isspace() or c in "'’-" for c in raw)
        ):
            value = None
        elif field == "declared_address" and not any(c.isalpha() for c in raw):
            value = None
        record = _record("date" if field.endswith("_date") else field, field, raw, value, quote, page.number, index + 1)
        yield {**record, "profile": "AddressForm", "method": method}


def read_address_form(pages):
    """Project all explicit fields, retaining differing or unreadable source evidence."""
    from src.document_processing.schemas import AddressForm

    if not any(address_form_signal(text) for page in pages for text in (page.text, page.native_text or "")):
        return AddressForm(validation_issues=["Address Form heading or declaration could not be verified. Review the original."])
    # Prefer independently readable PDF fields/text when they corroborate a
    # visual reading, while retaining conflicting candidates for review.
    candidates = []
    for page in pages:
        for label, value in page.form_fields:
            if _address_form_field(label):
                row = f"{label}: {' '.join(value.splitlines())}"
                candidates.extend(_address_form_rows(page, row, "pdf_form"))
        native_text = page.native_text if page.native_text is not None else (
            page.text if page.extraction_method == "native_text" else ""
        )
        if native_text:
            candidates.extend(_address_form_rows(page, native_text, "native_text"))
    for page in pages:
        if page.extraction_method in {"rapidocr", "multimodal_fallback"} and page.text != page.native_text:
            method = "openai_vision" if page.extraction_method == "multimodal_fallback" else "rapidocr"
            candidates.extend(_address_form_rows(page, page.text, method))
    return _labelled_document(AddressForm, pages, candidates)


def _labelled_document(schema, pages, candidates):
    payload, issues = {}, []
    ground_fields(payload, schema, candidates, issues, pages)
    field_methods = {}
    for candidate in candidates:
        field = candidate["field"]
        if (payload.get("source_pages", {}).get(field) == candidate["page_number"]
                and payload.get("source_snippets", {}).get(field) == candidate["snippet"]):
            field_methods.setdefault(field, candidate["method"])
    tracked = schema.observation_fields() if schema.__name__ != "AddressForm" else schema.observation_fields() - {"holder_name"}
    diagnostics = {
        "field_read_methods": field_methods,
        "fields_not_found_in_processed_source": sorted(tracked - {candidate["field"] for candidate in candidates}),
    }
    payload.update(validation_issues=issues, selected_pages=sorted(page.number for page in pages if page.text or page.form_fields),
                   extraction_diagnostics=diagnostics)
    return schema.model_validate(payload)


def address_candidates(chunks):
    """Explicit utility/form labels only; handwritten signatures are never inferred."""
    out=[]
    utility_labels={"holder_name":r"(?:holder name|account holder|customer name|ad[i]? soyad[i]?)", "service_address":r"(?:service address|supply address|hizmet adresi|kullanim yeri adresi|tesisat adresi|adres|address)", "provider":r"(?:provider|utility company|service provider|tedarikci|hizmet saglayici)"}
    form = address_form_signal("\n".join(c.text for c in chunks))
    for chunk in chunks:
        lines=chunk.text.splitlines()
        text=normalized(chunk.text)
        bill=any(w in text for w in ("utility bill","electricity bill","gas bill","water bill")) or "fatura" in text and any(w in text for w in ("elektrik","dogalgaz","su faturasi","internet"))
        if form:
            out.extend(_address_form_rows(chunk, chunk.text, chunk.extraction_method))
        if not bill:
            continue
        for index,line in enumerate(lines):
            row=line.strip()
            folded=normalized(row)
            for field,label in utility_labels.items():
                if not re.match(r"^(?:"+label+r")\s*[:\uff1a]",folded):
                    continue
                raw = re.split(r"[:\uff1a]", row, maxsplit=1)[1].strip()
                quote = line
                if not raw and index + 1 < len(lines):
                    raw = lines[index + 1].strip()
                    quote += "\n" + lines[index + 1]
                value = raw if raw and (field != "holder_name" or _person_row(raw)) else None
                record = _record("name" if field == "holder_name" else field, field, raw, value, quote, chunk.number, index + 1)
                out.append({**record, "profile": "UtilityBill"})
            if bill and re.match(r'^(?:billing date|invoice date|fatura (?:donemi/)?tarihi(?:/saati)?)\s*:',folded):
                dates=list(EVIDENCE_DATE_PATTERN.finditer(row))
                if len(dates)==1:
                    raw=dates[0].group();c=_record('date','billing_date',raw,source_date(raw,'dmy'),line,chunk.number,index+1);c['profile']='UtilityBill';out.append(c)
            # A bounded literal corporate header, never a provider inferred from a URL.
            if bill and index<16 and ":" not in row:
                parts=[r.strip() for r in re.split(r'\s{2,}',row) if r.strip()]
                company=parts[0] if parts else ''
                quote=line
                if index+1<len(lines) and re.search(r'elektrik|enerji|dogalgaz|water|energy|internet',normalized(company)):
                    next_row=re.split(r'\s{2,}',lines[index+1].strip())[0]
                    if re.search(r'a\.\s*s\.|ltd\.?|inc\.?',normalized(next_row)):company+=' '+next_row;quote+='\n'+lines[index+1]
                if len(company)<170 and re.search(r'elektrik|enerji|dogalgaz|water|energy|internet',normalized(company)) and re.search(r'a\.\s*s\.|ltd\.?|inc\.?',normalized(company)) and not re.search(r'kanun|gereg|uyari|must|should',normalized(company)):
                    c=_record('provider','provider',quote,company,quote,chunk.number,index+1);c['profile']='UtilityBill';out.append(c)
    return out


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
        line, quote = matching_lines[0] if matching_lines else (1, c.get("source_snippet",chunk.text))
        candidates.append(_record(c["kind"], c["kind"], c.get("original_value",c["value"]), c["value"], quote, c["page"], line))
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
                value = (_company_span(raw) if field == "insurer" else raw) if any(c.isalpha() for c in raw) else None
                if field == "insurer" and value is None and len(raw.split()) <= 6 and not re.search(r"\b(?:prim|primlerden|tazminat|iade|diger|uygulama|bulunmaktadir|ettirene)\b",fold(raw)) and not _TEXT_EXCLUSIONS.search(fold(raw)) and not LEGAL_REFERENCE.search(fold(raw)):
                    value=raw or None
                if field == "insurer" and value is None:continue
                candidates.append(_record(field, field, value or raw, value, quote, chunk.number, index))
    for chunk in chunks:
        for index, line in enumerate(chunk.text.splitlines()[:14], 1):
            row=line.strip()
            if len(row)<100 and len(row.split())<=8 and ":" not in row and not re.search(r"\b(?:student|studies|exchange|between|programme?|certificate|adres|address)\b",normalized(row)) and re.search(r"\bUniversity\b|\b[\u00dcU]niversitesi\b|UNIVERSITESI|[\u00dcU]N[\u0130I]VERS[\u0130I]TES[\u0130I]", row):
                candidates.append(_record("institution","institution",row,row,row,chunk.number,index))
    for chunk in chunks:
        for index,line in enumerate(chunk.text.splitlines(),1):
            if re.match(r"\s*(?:adres|address)\s*:",normalized(line)):
                found=re.search(r"([^\W\d_]+(?:[ \t]+[^\W\d_]+){0,4}[ \t]+\u00dcniversitesi)\b",line,re.I)
                if found:
                    value=found.group(1).strip()
                    candidates.append(_record("institution","institution",value,value,line,chunk.number,index))
    administrative = administrative_candidates(chunks)
    if any(c['field']=='requested_end_date' for c in administrative):
        candidates = [c for c in candidates if c.get('field') not in ('requested_start_date','requested_end_date')]
    candidates.extend(administrative)
    candidates.extend(preparation_candidates(chunks))
    candidates.extend(mrz_candidates(chunks))
    candidates.extend(address_candidates(chunks))
    candidates.extend(_insurance_text_candidates(chunks))
    category_aliases = {
        "private_health_insurance": ("private health insurance", "ozel saglik sigortasi"),
        "bilateral_social_security": ("bilateral social security", "ikili sosyal guvenlik"),
        "sgk_provision": ("sgk provision", "sgk provizyon"),
        "sgk_application": ("sgk application", "sgk basvuru"),
    }
    for chunk in chunks:
        for index, line in enumerate(chunk.text.splitlines(), 1):
            folded = label_text(line)
            if LEGAL_REFERENCE.search(folded):
                continue
            for category, aliases in category_aliases.items():
                if any(alias in folded for alias in aliases) and not re.search(r"\b(?:must|should|required|or|veya)\b|gereki", folded):
                    candidates.append(_record("insurance_evidence_type", "insurance_evidence_type", line, category, line, chunk.number, index))
            if (re.search(r"for (?:the )?residence permit|residence.permit purpose|ikamet izni (?:icin|amac)|ikamet amac", folded)
                    and not re.search(r"not |does not|must|should|required|degil|gecersiz", folded)):
                candidates.append(_record("residence_permit_purpose", "residence_permit_purpose", line, "true", line, chunk.number, index))
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
    candidates=[c for c in candidates if not c.get("profile") or c["profile"]==schema.__name__]
    if schema.__name__ in {"UtilityBill","AddressForm","TaxIDCertificate"}:
        choices=[c for c in candidates if c.get("profile")==schema.__name__ and c.get("field")==field]
        if field=="provider" and choices:
            key=lambda c:" ".join(normalized(c["normalized_value"] or "").split())
            longest=max(choices,key=lambda c:len(key(c)))
            if all(key(longest).endswith(key(c)) for c in choices):return [longest]
        return choices
    if schema.__name__ in {"PassportIdentity","EntryStamp","DeclarationControl","CardFeeReceipt","UtilityBill","AddressForm"} and field not in {"holder_name"}:
        choices=[c for c in candidates if c.get("field")==field]
        if schema.__name__=='PassportIdentity' and field=='nationality':
            mrz={c['normalized_value'] for c in choices if c.get('validated_mrz') and c['normalized_value'] is not None}
            if len(mrz)==1:
                aliases={'ITALIAN':'ITA','ITALIANA':'ITA','ITALIA':'ITA','ITALY':'ITA','GERMAN':'DEU','GERMANY':'DEU','DEUTSCH':'DEU','NORWEGIAN':'NOR','NORWAY':'NOR','FRENCH':'FRA','FRANCE':'FRA','BRITISH':'GBR'}
                reconciled=[]
                for c in choices:
                    raw=str(c.get('normalized_value') or c['original_value']).strip().upper()
                    clean=raw.translate(str.maketrans({'1':'I','0':'O'}))
                    country=aliases.get(clean,clean)
                    # A printed-label conflict is retained. A matching, controlled
                    # OCR/alias spelling corroborates the checked MRZ instead.
                    reconciled.append({**c,'normalized_value':country} if country in mrz else c)
                choices=reconciled
        return choices
    if schema.__name__ == "HealthInsurance":
        candidates = [c for c in candidates if not c.get("administrative")]
    if field in {"program_start_date", "program_end_date"}:
        return [c for c in candidates if c.get("field") == field]
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
    if field in {"exchange_student", "permit_type"}:
        return [c for c in candidates if c.get("field") == field]
    if field == "insurer":
        choices=[c for c in candidates if c["kind"]=="insurer"]
        def company_key(c):
            words=re.findall(r"[a-z0-9]+",fold(c["normalized_value"] or ""))
            stop=next((i+1 for i,w in enumerate(words) if w in {"sigorta","insurance"}),len(words))
            return tuple(words[:stop])
        if choices and len({company_key(c) for c in choices})==1:
            return [max(choices,key=lambda c: (bool(re.search(r"a\.?s\.?|anonim|ltd|inc",fold(c["normalized_value"] or ""))), len(c["normalized_value"] or "")))]
        return choices
    kind = "name" if field in ("holder_name", "applicant_name") else field
    return [c for c in candidates if c["kind"] == kind
            and (not c.get("insurance_only") or
                 (schema.__name__ == "HealthInsurance" and field in ("holder_name", "insurer")))]


def _equivalent(value, date_field=False):
    if date_field:
        return normalized_date(value)
    return " ".join(normalized(value).split()) if isinstance(value, str) else None


def ground_fields(payload, schema, candidates, issues, chunks=()):
    """Reconcile source evidence, keeping corrections separate from review issues.

    Insurance and narrow administrative labels receive deterministic corrections.
    Other enrollment/IKAMET fields retain model-uncertainty behavior; conflicts stay null for
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
        auto = insurance or schema.__name__ in {"PassportIdentity","EntryStamp","DeclarationControl","CardFeeReceipt","UtilityBill","AddressForm","TaxIDCertificate"} or field in {"exchange_student", "program_start_date", "program_end_date", "permit_type"} or any(c.get("administrative") for c in selected) or any(c.get("field")==field and c.get("kind")==field for c in selected) and field in {"institution", "issue_date", "requested_start_date", "requested_end_date"}
        known_kind = schema.__name__ in {"PassportIdentity","EntryStamp","DeclarationControl","CardFeeReceipt","UtilityBill","AddressForm","TaxIDCertificate"} or field in schema.DATES or field in {"exchange_student", "permit_type","holder_name", "applicant_name", "insurer", "institution", "insurance_evidence_type", "residence_permit_purpose", *IDENTIFIER_LABELS}
        if not known_kind:
            continue
        # Literal whole-line support for company/institution names interpreted
        # by the model, never for arbitrary numbers or legal-reference dates.
        if not selected and field == "institution" and isinstance(value, str):
            for chunk in chunks:
                for index, line in enumerate(chunk.text.splitlines(), 1):
                    if _equivalent(line) == _equivalent(value):
                        selected.append(_record(field, field, line, line.strip(), line, chunk.number, index))
        nonempty = [c for c in selected if c["original_value"]]
        comparison = (lambda v: _equivalent(v)) if field in {"holder_name","applicant_name","insurer","institution"} else (lambda v: v)
        values = {comparison(c["normalized_value"]) for c in nonempty if c["normalized_value"] is not None}
        unresolved = any(c["normalized_value"] is None for c in nonempty)
        unique = len(values) == 1 and not unresolved
        # For non-insurance, match a single source value without filling nulls.
        if value is None and not auto and not (unresolved or len(values) > 1):
            continue
        if field in ambiguous and not (auto and unique):
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
        chosen = next(c for c in nonempty if comparison(c["normalized_value"]) in values)
        source_value = chosen["normalized_value"]
        same = _equivalent(value, field in schema.DATES) == _equivalent(source_value, field in schema.DATES)
        if not auto and not same:
            payload[field] = None
            issues.append(f"{field}: value absent from reliable source candidates; discarded")
            continue
        payload[field] = (source_value == "true") if field in {"exchange_student","identity_page_present","entry_stamp_present","document_present","signed","receipt_present","residence_permit_purpose","signature_present"} else source_value
        snippets[field] = chosen["snippet"]
        if field not in sources:
            sources[field] = chosen["page_number"]
        if chosen.get('mrz_correction'):corrections.append(chosen['mrz_correction'])
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
