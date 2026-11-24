"""Machine-readable zones, splitting an upload into documents, and reading every page (no model, or a scripted one)."""
from datetime import date

from fakes import ScriptedBackend
from src.document_processing.extractor import TextPage
from src.document_processing.generic import extract_generic, page_batches
from src.document_processing.mrz import check_digit_ok, find_mrz
from src.document_processing.segment import heading, segment_pages
from src.pack import load_pack
from src.tools.extract_document import extract_file

IKAMET = load_pack("ikamet_first")
RENEWAL = load_pack("tc_kimlik_renewal")
FIRST = load_pack("tc_kimlik_first")

# ICAO 9303 specimens (public): a TD3 passport and a TD1 identity card.
PASSPORT_MRZ = "P<D<<MUSTERMANN<<ERIKA<<<<<<<<<<<<<<<<<<<<<<\nC01X0006H1D<<6408125F1710319<<<<<<<<<<<<<<<0"
ID_CARD_MRZ = "I<UTOD231458907<<<<<<<<<<<<<<<\n7408122F1204159UTO<<<<<<<<<<<6\nERIKSSON<<ANNA<MARIA<<<<<<<<<<"


# ----------------------------------------------------------------- MRZ --
def test_check_digits_follow_icao_9303():
    assert check_digit_ok("C01X0006H", "1") and not check_digit_ok("C01X0006H", "2")


def test_passport_and_id_card_zones_are_read_with_their_dates():
    [passport] = find_mrz([TextPage(1, "PASSPORT\n" + PASSPORT_MRZ)])
    [card] = find_mrz([TextPage(2, ID_CARD_MRZ)])
    assert (passport.kind, passport.document_number, passport.expiry_date) == ("passport", "C01X0006H", date(2017, 10, 31))
    assert passport.holder_name == "ERIKA MUSTERMANN"
    assert (card.kind, card.birth_date, card.expiry_date, card.page) == ("id_card", date(1974, 8, 12), date(2012, 4, 15), 2)
    assert card.holder_name == "ANNA MARIA ERIKSSON"


def test_a_zone_with_a_wrong_check_digit_is_ignored():
    broken = PASSPORT_MRZ.replace("6408125", "6408126")
    assert find_mrz([TextPage(1, broken)]) == []


# ------------------------------------------------------------ segments --
def test_a_label_with_a_value_is_not_a_title():
    page = TextPage(2, "Policy schedule\nInsured person: Alex Example\nPassport number: SYN1234567")
    assert "passport" not in heading(page)


def test_two_documents_in_one_file_are_split_where_the_second_title_is():
    pages = [TextPage(1, "Student Enrollment Certificate\nHolder name: Alex Example\nStudent ID: S1"),
             TextPage(2, "Health Insurance Certificate\nInsured person: Alex Example\nCoverage end date: 09.09.2027")]
    segments = segment_pages(pages, IKAMET)
    assert [(s.doc_type, s.pages, s.how) for s in segments] == [
        ("student_enrollment_certificate", [1], "title"), ("health_insurance_certificate", [2], "title")]


def test_pages_that_only_mention_another_document_continue_the_current_one():
    pages = [TextPage(1, "Health Insurance Policy\nGeneral conditions"),
             TextPage(2, "Article 3. You must show your passport and your student certificate at the hospital."),
             TextPage(3, "")]
    assert [(s.doc_type, s.pages) for s in segment_pages(pages, IKAMET)] == [("health_insurance_certificate", [1, 2, 3])]


def test_an_mrz_opens_a_document_without_a_title():
    pages = [TextPage(1, "Kimlik kartı bedeli\nÖdeme tarihi: 25.11.2026"), TextPage(2, "Anne adı: X\n" + ID_CARD_MRZ)]
    segments = segment_pages(pages, RENEWAL)
    assert [(s.doc_type, s.how) for s in segments] == [("id_card_fee_receipt", "title"), ("tc_kimlik_karti", "mrz")]


def test_a_tie_between_keywords_is_broken_by_the_model_among_the_tied_types_only():
    page = TextPage(1, "Belge: makbuz\nKayit: nufus cuzdani")   # one keyword of each document; labels are not titles
    backend = ScriptedBackend([{"doc_type": "nufus_cuzdani"}])
    [segment] = segment_pages([page], FIRST, backend)
    assert (segment.doc_type, segment.how) == ("nufus_cuzdani", "model")
    enum = backend.calls[0]["schema"]["properties"]["doc_type"]["enum"]
    assert set(enum) == {"nufus_cuzdani", "id_card_fee_receipt", "none"}


def test_nothing_recognised_is_one_segment_without_type():
    [segment] = segment_pages([TextPage(1, "Lorem ipsum dolor sit amet")], RENEWAL)
    assert segment.doc_type is None and segment.pages == [1]


# ------------------------------------------------------- reading pages --
def test_long_documents_are_read_in_batches_that_keep_page_numbers():
    pages = [TextPage(1, "a" * 5000), TextPage(2, "b" * 100)]
    batches = page_batches(pages, limit=3400)
    assert [[p.number for p in batch] for batch in batches] == [[1], [1, 2]]
    assert sum(len(p.text) for batch in batches for p in batch) == 5100


def test_values_on_the_second_page_are_found_and_only_missing_fields_are_asked_again():
    spec = RENEWAL.document("tc_kimlik_karti")
    pages = [TextPage(1, "Son Geçerlilik / Valid Until: 14.03.2027\n" + "x" * 3400), TextPage(2, "Adı: DENİZ ÖRNEK")]
    backend = ScriptedBackend([
        {"fields": [{"field": "expiry_date", "value": "14.03.2027", "page": 1, "quote": "Son Geçerlilik / Valid Until: 14.03.2027"}]},
        {"fields": [{"field": "holder_name", "value": "DENİZ ÖRNEK", "page": 2, "quote": "Adı: DENİZ ÖRNEK"}]}])
    observation = extract_generic(pages, spec, RENEWAL, backend)
    assert observation.grounded["expiry_date"].value == date(2027, 3, 14)
    assert observation.grounded["holder_name"].page == 2           # beyond the old 3,400-character cut
    second = backend.calls[1]["schema"]["$defs"]["ProposedField"]["properties"]["field"]["enum"]
    assert len(backend.calls) == 2 and second == ["birth_date", "holder_name"]


def test_mrz_dates_come_from_code_and_the_model_is_not_asked_for_them():
    spec = RENEWAL.document("tc_kimlik_karti")
    page = TextPage(1, "TÜRKİYE CUMHURİYETİ KİMLİK KARTI\nAdı Soyadı: ANNA MARIA ERIKSSON\n" + ID_CARD_MRZ)
    backend = ScriptedBackend([{"fields": [{"field": "holder_name", "value": "ANNA MARIA ERIKSSON", "page": 1,
                                            "quote": "Adı Soyadı: ANNA MARIA ERIKSSON"}]}])
    observation = extract_generic([page], spec, RENEWAL, backend)
    assert observation.grounded["birth_date"].value == date(1974, 8, 12)
    assert observation.grounded["expiry_date"].snippet.startswith("I<UTO")
    assert backend.calls[0]["schema"]["$defs"]["ProposedField"]["properties"]["field"]["enum"] == ["holder_name"]
    assert any("check digits" in note for note in observation.notes)


def test_a_card_is_named_and_read_by_code_alone_from_its_mrz(tmp_path):
    path = tmp_path / "card.txt"
    path.write_text("TÜRKİYE CUMHURİYETİ KİMLİK KARTI\n" + ID_CARD_MRZ, encoding="utf-8")
    result = extract_file(path, RENEWAL, backend=None)
    [card] = result.observations
    assert card.doc_type == "tc_kimlik_karti" and card.found_by == "mrz"     # the MRZ is checked before titles
    assert {f.field for f in card.facts} == {"birth_date", "card_expiry_date", "holder_name"}


def test_an_id_card_in_a_procedure_without_typed_documents_is_no_longer_unknown(tmp_path):
    """Before: 'detect automatically' only knew the seven İkamet documents, so a T.C. card was always unknown."""
    path = tmp_path / "card.txt"
    path.write_text("REPUBLIC OF TÜRKİYE IDENTITY CARD\nValid until: 14.03.2027", encoding="utf-8")
    backend = ScriptedBackend([{"fields": [{"field": "expiry_date", "value": "14.03.2027", "page": 1,
                                            "quote": "Valid until: 14.03.2027"}]}])
    [card] = extract_file(path, RENEWAL, backend=backend).observations
    assert card.doc_type == "tc_kimlik_karti"
    assert {f.field: f.value for f in card.facts} == {"card_expiry_date": date(2027, 3, 14)}


def test_unrecognised_pages_say_how_to_go_on(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("Lorem ipsum dolor sit amet", encoding="utf-8")
    [observation] = extract_file(path, RENEWAL, backend=None).observations
    assert observation.doc_type == "unknown" and "Choose its type" in observation.issues[0]


def test_only_selected_pages_are_read_again(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("Lorem ipsum", encoding="utf-8")
    assert "none of the pages" in extract_file(path, RENEWAL, only_pages=[4]).error
