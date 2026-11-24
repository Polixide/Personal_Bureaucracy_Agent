"""Machine-readable zones (ICAO 9303): the lines of `<` characters on passports and ID cards.

Read by code, not by a model: every value is accepted only when its check digit is right, so a value
from an MRZ is as reliable as the scan. Two formats:
  TD3 (passports): 2 lines of 44 characters, starting with "P".
  TD1 (ID cards, e.g. the back of the Turkish T.C. Kimlik Kartı): 3 lines of 30 characters,
      starting with "I", "A" or "C".
OCR often adds spaces or reads `<` as `«`; `clean_line` repairs only those (never a letter or a digit).
"""
import re
from datetime import date

from pydantic import BaseModel

from src.document_processing.extractor import TextPage

WEIGHTS = (7, 3, 1)
MRZ_CHARS = re.compile(r"[A-Z0-9<]+")


class MrzRecord(BaseModel):
    """The checked values of one MRZ. `kind` is "passport" (TD3) or "id_card" (TD1)."""
    kind: str
    page: int
    lines: list[str]
    document_number: str
    nationality: str
    holder_name: str
    birth_date: date | None
    expiry_date: date | None
    personal_number: str = ""

    def snippet(self) -> str:
        """The MRZ lines exactly as they were read (the quote shown to the user)."""
        return "\n".join(self.lines)


def char_value(char: str) -> int:
    if char == "<":
        return 0
    if char.isdigit():
        return int(char)
    return ord(char) - 55


def check_digit_ok(text: str, digit: str) -> bool:
    """ICAO check digit: weights 7, 3, 1 repeated, sum modulo 10."""
    if not digit.isdigit():
        return False
    total = sum(char_value(char) * WEIGHTS[index % 3] for index, char in enumerate(text))
    return total % 10 == int(digit)


def clean_line(line: str) -> str:
    """Upper case, no spaces, and the OCR spellings '«' / '‹' of '<' turned back into '<'."""
    text = re.sub(r"\s+", "", line).upper()
    return text.replace("«", "<").replace("‹", "<")


def mrz_date(text: str, future: bool) -> date | None:
    """YYMMDD -> date. Expiry dates are in this century; birth dates in the past 100 years."""
    if not text.isdigit():
        return None
    year, month, day = int(text[:2]), int(text[2:4]), int(text[4:6])
    century = 2000 if future or year <= date.today().year % 100 else 1900
    try:
        return date(century + year, month, day)
    except ValueError:
        return None


def mrz_name(text: str) -> str:
    """'SURNAME<<GIVEN<NAMES<<<' -> 'GIVEN NAMES SURNAME'."""
    surname, _, given = text.partition("<<")
    return " ".join(f"{given} {surname}".replace("<", " ").split())


def parse_td3(first: str, second: str, page: int, raw: list[str]) -> MrzRecord | None:
    """Passport MRZ; None unless the document number, dates and composite check digits are right."""
    if len(first) != 44 or len(second) != 44 or not first.startswith("P"):
        return None
    if not (check_digit_ok(second[0:9], second[9]) and check_digit_ok(second[13:19], second[19])
            and check_digit_ok(second[21:27], second[27])
            and check_digit_ok(second[0:10] + second[13:20] + second[21:43], second[43])):
        return None
    return MrzRecord(kind="passport", page=page, lines=raw, document_number=second[0:9].rstrip("<"),
                     nationality=second[10:13].replace("<", ""), holder_name=mrz_name(first[5:]),
                     birth_date=mrz_date(second[13:19], future=False), expiry_date=mrz_date(second[21:27], future=True),
                     personal_number=second[28:42].rstrip("<"))


def parse_td1(first: str, second: str, third: str, page: int, raw: list[str]) -> MrzRecord | None:
    """ID-card MRZ; None unless the document number, dates and composite check digits are right."""
    if len(first) != 30 or len(second) != 30 or len(third) != 30 or first[0] not in "IAC":
        return None
    composite = first[5:30] + second[0:7] + second[8:15] + second[18:29]
    if not (check_digit_ok(first[5:14], first[14]) and check_digit_ok(second[0:6], second[6])
            and check_digit_ok(second[8:14], second[14]) and check_digit_ok(composite, second[29])):
        return None
    return MrzRecord(kind="id_card", page=page, lines=raw, document_number=first[5:14].rstrip("<"),
                     nationality=second[15:18].replace("<", ""), holder_name=mrz_name(third),
                     birth_date=mrz_date(second[0:6], future=False), expiry_date=mrz_date(second[8:14], future=True),
                     personal_number=first[15:30].strip("<"))


def find_mrz(pages: list[TextPage]) -> list[MrzRecord]:
    """Every valid MRZ in the pages (TD3 and TD1), in page order."""
    records = []
    for page in pages:
        raw = [line for line in page.text.splitlines() if line.strip()]
        lines = [clean_line(line) for line in raw]
        for index in range(len(lines)):
            window = lines[index:index + 3]
            if not all(MRZ_CHARS.fullmatch(line) for line in window[:2]):
                continue
            if len(window) >= 2:
                record = parse_td3(window[0], window[1], page.number, raw[index:index + 2])
                if record is not None:
                    records.append(record)
                    continue
            if len(window) == 3 and MRZ_CHARS.fullmatch(window[2]):
                record = parse_td1(window[0], window[1], window[2], page.number, raw[index:index + 3])
                if record is not None:
                    records.append(record)
    return records


def mrz_values(record: MrzRecord) -> dict[str, object]:
    """MRZ values under the standard document field names a pack can list in `extracts`."""
    values: dict[str, object] = {"holder_name": record.holder_name, "document_number": record.document_number,
                                 "nationality": record.nationality}
    if record.birth_date is not None:
        values["birth_date"] = record.birth_date
    if record.expiry_date is not None:
        values["expiry_date"] = record.expiry_date
    if record.personal_number:
        values["personal_number"] = record.personal_number
    return values
