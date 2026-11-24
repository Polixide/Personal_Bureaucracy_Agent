"""The vision model's checked answers (src/document_processing/vision.py) and the polite crawler (src/rag/crawl.py)."""
import json
from datetime import date

import pymupdf
import pytest

from fakes import ScriptedBackend
from src.document_processing.extractor import TextPage
from src.document_processing.schemas import UnknownDocument
from src.document_processing.vision import dates_in, read_image
from src.pack import load_pack
from src.rag.crawl import Seed, crawl, follows, load_corpus, parse_page, web_id
from src.rag.index import search_corpus
from src.rag.ingest import IngestError
from src.tools.extract_document import extract_file

IKAMET = load_pack("ikamet_first")
RENEWAL = load_pack("tc_kimlik_renewal")


class SeeingBackend(ScriptedBackend):
    """A scripted backend that also answers vision calls (from `seen`, in order)."""

    def __init__(self, replies=(), seen=()):
        super().__init__(replies)
        self.seen = list(seen)
        self.see_calls = []

    def see(self, prompt, images, schema, component):
        self.see_calls.append({"prompt": prompt, "images": images, "schema": schema})
        return json.dumps(self.seen.pop(0))


def picture_pdf(path, text_on_image=""):
    """A one-page scan: an image, no text layer (OCR is mocked by the tests)."""
    with pymupdf.open() as document:
        page = document.new_page(width=300, height=200)
        image = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 60, 40), False)
        image.clear_with(255)
        page.insert_image(page.rect, stream=image.tobytes("png"))
        document.save(path)
    return path


# -------------------------------------------------------------- vision --
def test_short_dates_on_stamps_are_understood():
    assert [day for day, _ in dates_in("ISTANBUL\n26.02.22 47")] == [date(2022, 2, 26)]
    assert [day for day, _ in dates_in("valid 2027-03-14")] == [date(2027, 3, 14)]


def test_a_date_the_ocr_text_also_shows_is_checked_and_a_name_always_waits(tmp_path):
    path = picture_pdf(tmp_path / "card.pdf")
    page = TextPage(1, "Valid until: 14.03.2027\nAdı: DENİZ", ocr=True)
    backend = SeeingBackend(seen=[{"doc_type": "this", "fields": [
        {"field": "expiry_date", "value": "14.03.2027"}, {"field": "holder_name", "value": "DENİZ"},
        {"field": "birth_date", "value": "01.01.1990"}]}])
    read = read_image(backend, RENEWAL, path, page, [RENEWAL.document("tc_kimlik_karti")],
                      known_type="tc_kimlik_karti")
    assert read.checked["expiry_date"].value == date(2027, 3, 14)
    assert read.checked["expiry_date"].snippet == "Valid until: 14.03.2027"
    assert set(read.to_confirm) == {"holder_name", "birth_date"}       # a word may be only part of a name
    assert "not found in its OCR text" in read.to_confirm["birth_date"].snippet
    assert backend.see_calls[0]["images"][0].startswith(b"\x89PNG")


def test_an_unrecognised_picture_is_named_by_the_vision_model_and_every_value_waits(tmp_path, monkeypatch):
    path = picture_pdf(tmp_path / "stamp.pdf")
    monkeypatch.setattr("src.tools.extract_document.extract_pages",
                        lambda p: [TextPage(1, "TURKIYE CUMHURIYETE\nISTANBUL\n26.02.22 47", ocr=True)])
    backend = SeeingBackend(seen=[{"doc_type": "turkiye_entry_stamp"},          # 1. which document
                                  {"doc_type": "this", "fields": [                # 2. its values only
                                      {"field": "latest_entry_date", "value": "26.02.22"},
                                      {"field": "entry_stamp_present", "value": "true"}]}])

    def typed_processor_finds_nothing(*args, **kwargs):
        return UnknownDocument(selected_pages=[1])

    result = extract_file(path, IKAMET, backend=backend, processor=typed_processor_finds_nothing)
    [stamp] = result.observations
    assert stamp.doc_type == "turkiye_entry_stamp" and stamp.found_by == "vision"
    facts = {fact.field: fact for fact in stamp.facts}
    assert facts["latest_entry_date"].value == date(2022, 2, 26)
    assert all(fact.requires_confirmation for fact in stamp.facts)
    naming, reading = backend.see_calls
    assert naming["schema"]["properties"]["doc_type"]["enum"][-1] == "none" and "fields" not in naming["schema"]["properties"]
    assert reading["schema"]["$defs"]["VisionValue"]["properties"]["field"]["enum"] == ["entry_stamp_present", "latest_entry_date"]


def test_without_a_vision_model_an_unrecognised_picture_stays_unknown(tmp_path, monkeypatch):
    path = picture_pdf(tmp_path / "stamp.pdf")
    monkeypatch.setattr("src.tools.extract_document.extract_pages",
                        lambda p: [TextPage(1, "ISTANBUL 26.02.22", ocr=True)])
    [observation] = extract_file(path, RENEWAL, backend=ScriptedBackend()).observations
    assert observation.doc_type == "unknown"


# --------------------------------------------------------------- crawl --
SITE = {
    "https://example.gov.tr/robots.txt": (b"User-agent: *\nDisallow: /private\n", "text"),
    "https://example.gov.tr/guide": (b"""<html><head><title>Guide</title></head><body>
        <nav><a href="/menu-page">Menu</a></nav>
        <main><h1>Student guide</h1><p>""" + b"Practical steps for students. " * 20 + b"""</p>
        <a href="/guide/tax-number">Tax number</a> <a href="/private/x">Private</a>
        <a href="https://other.org/page">Elsewhere</a> <a href="/guide/form.docx">Form</a></main></body></html>""", "html"),
    "https://example.gov.tr/guide/tax-number": (b"<html><body><main><h1>How to get a tax number</h1><p>"
                                                + b"Apply online with your passport number. " * 20 + b"</p></main></body></html>", "html"),
}


def fake_fetch(url):
    if url not in SITE:
        raise IngestError(f"Cannot fetch {url}")
    return SITE[url]


def test_links_in_menus_are_not_followed():
    page = parse_page(SITE["https://example.gov.tr/guide"][0].decode())
    assert "/menu-page" not in page.links and "/guide/tax-number" in page.links and page.h1 == "Student guide"


def test_only_pages_of_the_same_site_under_the_prefix_are_followed():
    seed = Seed(url="https://example.gov.tr/guide", topic="t", prefix="/guide")
    assert follows("https://example.gov.tr/guide/tax-number", seed)
    assert not follows("https://other.org/page", seed) and not follows("https://example.gov.tr/guide/form.docx", seed)
    assert web_id("https://example.gov.tr/guide/tax-number") == "WEB_EXAMPLE_GUIDE_TAX_NUMBER"


def test_crawl_obeys_robots_and_saves_readable_pages(tmp_path):
    seed = Seed(url="https://example.gov.tr/guide", topic="Students", prefix="/")
    report = crawl([seed], folder=tmp_path, fetch=fake_fetch, delay=0, today=date(2026, 11, 20))
    assert [page.title for page in report.saved] == ["Student guide", "How to get a tax number"]
    assert report.refused_by_robots == ["https://example.gov.tr/private/x"]
    corpus = load_corpus(tmp_path)
    assert {page.source_id for page in corpus} == {"WEB_EXAMPLE_GUIDE", "WEB_EXAMPLE_GUIDE_TAX_NUMBER"}
    text = (tmp_path / "corpus" / "WEB_EXAMPLE_GUIDE_TAX_NUMBER.txt").read_text(encoding="utf-8")
    assert text.startswith("SOURCE: How to get a tax number") and "RETRIEVED: 2026-11-20" in text


def test_the_corpus_is_searched_with_its_page_urls(tmp_path, store):
    crawl([Seed(url="https://example.gov.tr/guide", topic="Students", prefix="/")], folder=tmp_path,
          fetch=fake_fetch, delay=0, today=date(2026, 11, 20))
    hits = search_corpus(store, ScriptedBackend(), "tax number passport apply online", k=1, folder=tmp_path)
    assert hits[0].source_id == "WEB_EXAMPLE_GUIDE_TAX_NUMBER"
    assert hits[0].url == "https://example.gov.tr/guide/tax-number"


@pytest.mark.parametrize("robots", [b"<html><body>Not found</body></html>", None])
def test_a_site_without_robots_rules_can_be_read(tmp_path, robots):
    site = dict(SITE)
    if robots is None:
        site.pop("https://example.gov.tr/robots.txt")
    else:
        site["https://example.gov.tr/robots.txt"] = (robots, "html")
    report = crawl([Seed(url="https://example.gov.tr/guide/tax-number", topic="t")], folder=tmp_path,
                   fetch=lambda url: site[url] if url in site else (_ for _ in ()).throw(IngestError(url)), delay=0)
    assert len(report.saved) == 1
