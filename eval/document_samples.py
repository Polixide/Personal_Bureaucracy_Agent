"""Synthetic documents with known values, to measure how well uploads are read (eval/document_eval.py).

Every person, number and date here is invented ("SYNTHETIC EXAMPLE - NOT AN OFFICIAL DOCUMENT" is printed
on each page). Four kinds of files:
  - text PDFs (the text layer is read directly);
  - scans: the same pages rendered to an image, so only OCR can read them;
  - photos (PNG), the way a phone uploads a card;
  - several documents in one PDF, which must be split.
The public passport and entry-stamp images of data/reference_samples/ (SOURCES.md) are added when present.
"""
from pathlib import Path

import pymupdf
from pydantic import BaseModel

WATERMARK = "SYNTHETIC EXAMPLE - NOT AN OFFICIAL DOCUMENT"
REFERENCE_DIR = Path(__file__).resolve().parent.parent / "data" / "reference_samples"
SCAN_DPI = 150
FONT_SIZE = 10.5


class ExpectedDocument(BaseModel):
    doc_type: str
    facts: dict[str, str | bool]


class Sample(BaseModel):
    sample_id: str
    procedure_id: str
    kind: str                     # text, scan, photo, multi, reference
    file: str
    expected: list[ExpectedDocument]


# ------------------------------------------------------------------ MRZ --
def check_digit(text: str) -> str:
    weights = (7, 3, 1)
    total = 0
    for index, char in enumerate(text):
        value = 0 if char == "<" else int(char) if char.isdigit() else ord(char) - 55
        total += value * weights[index % 3]
    return str(total % 10)


def td1_lines(number: str, birth: str, sex: str, expiry: str, nationality: str, personal: str,
              surname: str, given: str) -> list[str]:
    """A valid 3 x 30 ID-card MRZ (ICAO 9303 TD1)."""
    first = f"I<{nationality}{number}{check_digit(number)}{personal.ljust(15, '<')}"
    second = f"{birth}{check_digit(birth)}{sex}{expiry}{check_digit(expiry)}{nationality}{'<' * 11}"
    composite = first[5:30] + second[0:7] + second[8:15] + second[18:29]
    second += check_digit(composite)
    third = f"{surname}<<{given.replace(' ', '<')}".ljust(30, "<")
    return [first, second, third]


# ---------------------------------------------------------------- pages --
STUDENT = [WATERMARK, "Student Enrollment Certificate", "", "Example University - Office of the Registrar",
           "This is to certify that the person named below is a registered student of Example University.",
           "Holder name: Alex Example", "Student ID: SYN-2026-0042", "Institution: Example University",
           "Programme: Exchange semester (Erasmus+), Faculty of Engineering", "Exchange student: Yes",
           "Programme start date: 15.09.2026", "Programme end date: 31.01.2027", "Issue date: 01.10.2026",
           "", "Registrar (signature and stamp)"]

CONDITIONS = [WATERMARK, "Health Insurance Policy", "General conditions of the policy", ""] + [
    f"Article {n}. The insurer covers medically necessary treatment in Türkiye, including emergency care, "
    f"hospital stays and outpatient treatment, within the limits stated in the schedule of benefits. Pre-existing "
    f"conditions are covered only when declared before the start of the policy. Claims must be reported within "
    f"thirty days of treatment with the original invoices and the medical report." for n in range(1, 13)]
SCHEDULE = [WATERMARK, "Policy schedule", "Insured person: Alex Example", "Passport number: SYN1234567",
            "Insurer: Example Insurance A.S.", "Policy number: POL-SYN-77881", "Coverage start date: 10.09.2026",
            "Coverage end date: 09.09.2027", "Territorial scope: Türkiye", "Premium paid in full."]
INSURANCE_ONE_PAGE = [WATERMARK, "Health Insurance Certificate", "Insured person: Alex Example",
                      "Passport number: SYN1234567", "Insurer: Example Insurance A.S.", "Policy number: POL-SYN-77881",
                      "Coverage start date: 10.09.2026", "Coverage end date: 09.09.2027"]
CARD_FEE = [WATERMARK, "Residence Permit Card Fee Payment Receipt", "Bank: Example Bank", "Payer: Alex Example",
            "Payment date: 20.10.2026", "Reference number: RCPT-SYN-5521", "Amount: 365,00 TL",
            "Description: ikamet belge bedeli"]

KIMLIK_FRONT = [WATERMARK, "TÜRKİYE CUMHURİYETİ KİMLİK KARTI", "REPUBLIC OF TÜRKİYE IDENTITY CARD",
                "T.C. Kimlik No / TR Identity No: 99999999990", "Soyadı / Surname: ÖRNEK", "Adı / Given Name(s): DENİZ",
                "Doğum Tarihi / Date of Birth: 14.03.1992", "Cinsiyeti / Gender: K/F", "Seri No / Document No: A99Z99999",
                "Son Geçerlilik / Valid Until: 14.03.2027"]
KIMLIK_BACK = [WATERMARK, "Anne Adı / Mother's Name: AYŞE", "Baba Adı / Father's Name: MEHMET",
               "Veren Makam / Issued By: T.C. İÇİŞLERİ BAKANLIĞI"] + td1_lines(
    "A99Z99999", "920314", "F", "270314", "TUR", "99999999990", "ORNEK", "DENIZ")
KIMLIK_FEE = [WATERMARK, "T.C. Hazine ve Maliye Bakanlığı - Tahsilat Makbuzu", "Kimlik kartı bedeli",
              "Ödeyen / Payer: DENİZ ÖRNEK", "Ödeme tarihi / Payment date: 25.11.2026", "Tutar / Amount: 445,00 TL",
              "Makbuz No: SYN-778812"]
NUFUS_CUZDANI = [WATERMARK, "TÜRKİYE CUMHURİYETİ NÜFUS CÜZDANI", "Soyadı: ÖRNEK", "Adı: DENİZ",
                 "Baba Adı: MEHMET", "Ana Adı: AYŞE", "Doğum Yeri: ANKARA", "Doğum Tarihi: 14.03.1992",
                 "Nüfusa Kayıtlı Olduğu İl: ANKARA", "Cilt No: 999", "Aile Sıra No: 99", "Sıra No: 9",
                 "Verildiği Yer: ÇANKAYA", "Veriliş Nedeni: DOĞUM"]


def write_lines(page, lines: list[str]) -> None:
    """Text block with a font that has Turkish letters (Helvetica has no 'ı' or 'ş')."""
    page.insert_textbox(page.rect + (50, 50, -50, -50), "\n".join(lines), fontsize=FONT_SIZE, fontname="F0",
                        fontfile=font_file())


def font_file() -> str:
    """A Windows / Linux / macOS TrueType font with Turkish letters."""
    for candidate in (r"C:\Windows\Fonts\arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):  # last: macOS 11+
        if Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("No TrueType font with Turkish letters found")


def text_pdf(path: Path, pages: list[list[str]]) -> Path:
    with pymupdf.open() as document:
        for lines in pages:
            write_lines(document.new_page(), lines)
        document.save(path)
    return path


def scanned(source: Path, path: Path) -> Path:
    """Every page rendered to an image: the file has no text layer, like a scanner's PDF."""
    with pymupdf.open(source) as original, pymupdf.open() as document:
        for page in original:
            image = page.get_pixmap(dpi=SCAN_DPI, colorspace=pymupdf.csGRAY)
            target = document.new_page(width=page.rect.width, height=page.rect.height)
            target.insert_image(target.rect, stream=image.tobytes("png"))
        document.save(path)
    return path


def photo(source: Path, path: Path) -> Path:
    """The pages of `source` stacked into one PNG, like a phone photo of a card's two sides."""
    with pymupdf.open(source) as original:
        images = [page.get_pixmap(dpi=SCAN_DPI, clip=page.rect) for page in original]
        width, height = images[0].width, sum(image.height for image in images)
        canvas = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
        canvas.clear_with(255)
        top = 0
        for image in images:
            image.set_origin(0, top)
            canvas.copy(image, image.irect)
            top += image.height
        canvas.save(path)
    return path


def combined(path: Path, parts: list[Path]) -> Path:
    with pymupdf.open() as document:
        for part in parts:
            with pymupdf.open(part) as source:
                document.insert_pdf(source)
        document.save(path)
    return path


# -------------------------------------------------------------- samples --
STUDENT_FACTS = {"holder_name": "Alex Example", "student_id": "SYN-2026-0042", "institution": "Example University",
                 "student_certificate_issue_date": "2026-10-01", "exchange_student": True,
                 "program_start_date": "2026-09-15", "program_end_date": "2027-01-31"}
INSURANCE_FACTS = {"holder_name": "Alex Example", "insurance_insurer": "Example Insurance A.S.",
                   "insurance_policy_number": "POL-SYN-77881", "insurance_start_date": "2026-09-10",
                   "insurance_end_date": "2027-09-09", "passport_or_foreign_id": "SYN1234567"}
CARD_FEE_FACTS = {"fee_receipt_present": True, "receipt_payment_date": "2026-10-20",
                  "receipt_reference_number": "RCPT-SYN-5521"}
KIMLIK_FACTS = {"holder_name": "DENİZ ÖRNEK", "birth_date": "1992-03-14", "card_expiry_date": "2027-03-14"}
KIMLIK_FEE_FACTS = {"fee_payment_date": "2026-11-25"}
NUFUS_FACTS = {"birth_date": "1992-03-14"}
NORWAY = {"passport_or_foreign_id": "CCC002251", "passport_expiry_date": "2030-04-15", "passport_id_page_present": True}
GERMANY = {"passport_or_foreign_id": "C01X0006H", "passport_expiry_date": "2017-10-31", "passport_id_page_present": True}


def expected(*pairs: tuple[str, dict]) -> list[ExpectedDocument]:
    return [ExpectedDocument(doc_type=doc_type, facts=facts) for doc_type, facts in pairs]


def synthetic_samples(folder: Path) -> list[Sample]:
    """Write the synthetic files into `folder` and describe what each one must give."""
    folder.mkdir(parents=True, exist_ok=True)
    f = folder
    student = text_pdf(f / "student_certificate.pdf", [STUDENT])
    policy = text_pdf(f / "insurance_policy_2_pages.pdf", [CONDITIONS, SCHEDULE])
    insurance = text_pdf(f / "insurance_certificate.pdf", [INSURANCE_ONE_PAGE])
    fee = text_pdf(f / "card_fee_receipt.pdf", [CARD_FEE])
    kimlik = text_pdf(f / "kimlik_card.pdf", [KIMLIK_FRONT, KIMLIK_BACK])
    kimlik_fee = text_pdf(f / "kimlik_fee_receipt.pdf", [KIMLIK_FEE])
    nufus = text_pdf(f / "nufus_cuzdani.pdf", [NUFUS_CUZDANI])
    samples = [
        Sample(sample_id="D01-student-text", procedure_id="ikamet_first", kind="text", file=str(student),
               expected=expected(("student_enrollment_certificate", STUDENT_FACTS))),
        Sample(sample_id="D02-policy-values-on-page-2", procedure_id="ikamet_first", kind="text", file=str(policy),
               expected=expected(("health_insurance_certificate", INSURANCE_FACTS))),
        Sample(sample_id="D03-insurance-scan", procedure_id="ikamet_first", kind="scan",
               file=str(scanned(insurance, f / "insurance_scan.pdf")),
               expected=expected(("health_insurance_certificate", INSURANCE_FACTS))),
        Sample(sample_id="D04-card-fee-text", procedure_id="ikamet_first", kind="text", file=str(fee),
               expected=expected(("card_fee_receipt", CARD_FEE_FACTS))),
        Sample(sample_id="D05-student-and-insurance-in-one-pdf", procedure_id="ikamet_first", kind="multi",
               file=str(combined(f / "student_plus_insurance.pdf", [student, insurance])),
               expected=expected(("student_enrollment_certificate", STUDENT_FACTS),
                                 ("health_insurance_certificate", INSURANCE_FACTS))),
        Sample(sample_id="D06-kimlik-card-text", procedure_id="tc_kimlik_renewal", kind="text", file=str(kimlik),
               expected=expected(("tc_kimlik_karti", KIMLIK_FACTS))),
        Sample(sample_id="D07-kimlik-card-photo", procedure_id="tc_kimlik_renewal", kind="photo",
               file=str(photo(kimlik, f / "kimlik_card_photo.png")),
               expected=expected(("tc_kimlik_karti", KIMLIK_FACTS))),
        Sample(sample_id="D08-kimlik-card-and-fee-in-one-pdf", procedure_id="tc_kimlik_renewal", kind="multi",
               file=str(combined(f / "kimlik_plus_fee.pdf", [kimlik, kimlik_fee])),
               expected=expected(("tc_kimlik_karti", KIMLIK_FACTS), ("id_card_fee_receipt", KIMLIK_FEE_FACTS))),
        Sample(sample_id="D09-nufus-and-fee-scanned-together", procedure_id="tc_kimlik_first", kind="multi",
               file=str(scanned(combined(f / "nufus_plus_fee_text.pdf", [nufus, kimlik_fee]), f / "nufus_plus_fee_scan.pdf")),
               expected=expected(("nufus_cuzdani", NUFUS_FACTS), ("id_card_fee_receipt", KIMLIK_FEE_FACTS))),
    ]
    return samples + reference_samples(folder)


def reference_samples(folder: Path) -> list[Sample]:
    """The public specimens of data/reference_samples/ (when present), alone and with a synthetic page."""
    norway, germany = REFERENCE_DIR / "passport_specimen_norway.pdf", REFERENCE_DIR / "passport_specimen_germany_2007.pdf"
    if not norway.is_file() or not germany.is_file():
        return []
    student = folder / "student_certificate.pdf"
    return [
        Sample(sample_id="D10-passport-norway-scan", procedure_id="ikamet_first", kind="reference", file=str(norway),
               expected=expected(("passport_identity_page", NORWAY))),
        Sample(sample_id="D11-passport-germany-scan", procedure_id="ikamet_first", kind="reference", file=str(germany),
               expected=expected(("passport_identity_page", GERMANY))),
        Sample(sample_id="D12-passport-and-student-in-one-pdf", procedure_id="ikamet_first", kind="multi",
               file=str(combined(folder / "passport_plus_student.pdf", [norway, student])),
               expected=expected(("passport_identity_page", NORWAY), ("student_enrollment_certificate", STUDENT_FACTS))),
        Sample(sample_id="D13-entry-stamp-photo", procedure_id="ikamet_first", kind="reference",
               file=str(REFERENCE_DIR / "entry_stamp_istanbul_airport.pdf"),
               expected=expected(("turkiye_entry_stamp", {"latest_entry_date": "2022-02-26", "entry_stamp_present": True}))),
    ]
