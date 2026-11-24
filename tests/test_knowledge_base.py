"""src/rag/ingest.py, knowledge.py and drafts.py: the knowledge base of a pack, fully offline.

Pages come from a fake fetcher and model replies from ScriptedBackend, so nothing reaches the network or Ollama.
Each test works on a copy of the tc_kimlik_first pack in a temporary folder.
"""
import json
import shutil
from datetime import date

import pymupdf
import pytest

from config import PROCEDURES_DIR
from fakes import ScriptedBackend
from src.models import Source
from src.pack import load_pack
from src.rag.drafts import DRAFTS_FILE, draft_rules, quote_in_source, save_drafts
from src.rag.ingest import IngestError, html_to_text, pdf_to_text, stored_text
from src.rag.knowledge import add_source, check_sources, diff_preview, fetch_from_file, fetch_sources

PACK_ID = "tc_kimlik_first"
DAY = date(2026, 11, 19)
PAGE = """<html><head><title>t</title><script>var x = 1;</script></head>
<body class="no-sidebars better-responsive-menu">
<nav class="nav justify-content-end"><a>Ana Sayfa</a></nav>
<div class="menu-content"><a>Menu link</a></div>
<div class="row detail-content-container"><h1>T.C. Kimlik Kartı İşlemleri</h1>
<p>Son altı ay içinde çekilmiş biyometrik fotoğraf gereklidir.</p><ul><li>Fotoğraflı ve geçerli kimlik belgesi</li></ul>
</div><footer>Copyright</footer></body></html>"""


@pytest.fixture()
def root(tmp_path):
    """A procedures/ folder with a copy of the pack (its saved texts removed)."""
    target = tmp_path / "procedures"
    shutil.copytree(PROCEDURES_DIR / PACK_ID, target / PACK_ID)
    shutil.rmtree(target / PACK_ID / "sources_text")
    sources = json.loads((target / PACK_ID / "sources.json").read_text(encoding="utf-8"))
    for source in sources:
        source.pop("content_sha256", None)
    (target / PACK_ID / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
    return target


def fake_fetcher(pages: dict[str, str]):
    def fetch(url: str):
        if url not in pages:
            raise IngestError(f"Cannot fetch {url}: offline")
        return pages[url].encode("utf-8"), "html"
    return fetch


def all_pages(root, html=PAGE) -> dict[str, str]:
    return {source.url: html for source in load_pack(PACK_ID, root).sources}


# ------------------------------------------------------------ HTML / PDF --
def test_html_keeps_the_main_content_and_drops_menus_scripts_and_footer():
    text = html_to_text(PAGE)
    assert text.startswith("T.C. Kimlik Kartı İşlemleri")
    assert "biyometrik fotoğraf" in text and "Fotoğraflı ve geçerli" in text
    for noise in ("var x", "Ana Sayfa", "Menu link", "Copyright"):
        assert noise not in text


def test_a_small_guessed_container_falls_back_to_the_whole_page():
    page = "<body><div class='text'>Short</div><p>" + "Long official paragraph. " * 20 + "</p></body>"
    assert "Long official paragraph" in html_to_text(page)


def test_text_based_pdfs_are_read():
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Randevu Sikca Sorulan Sorular")
    assert "Randevu Sikca Sorulan Sorular" in pdf_to_text(document.tobytes())


def test_broken_pdfs_are_an_ingest_error():
    with pytest.raises(IngestError):
        pdf_to_text(b"not a pdf")


# -------------------------------------------------------- fetch and check --
def test_fetch_saves_text_with_header_and_records_hash_and_date(root):
    results = fetch_sources(PACK_ID, fetch=fake_fetcher(all_pages(root)), today=DAY, root=root)
    assert all(result.ok for result in results)
    pack = load_pack(PACK_ID, root)
    source = pack.sources[0]
    saved = (root / PACK_ID / "sources_text" / f"{source.source_id}.txt").read_text(encoding="utf-8")
    assert saved.startswith(f"SOURCE: {source.title}") and f"URL: {source.url}" in saved
    assert source.retrieved_at == "2026-11-19" and len(source.content_sha256) == 64
    assert stored_text(root / PACK_ID / "sources_text" / f"{source.source_id}.txt") == html_to_text(PAGE)


def test_a_failed_download_keeps_the_old_entry(root):
    pages = all_pages(root)
    first = load_pack(PACK_ID, root).sources[0]
    del pages[first.url]
    results = {r.source_id: r for r in fetch_sources(PACK_ID, fetch=fake_fetcher(pages), today=DAY, root=root)}
    assert not results[first.source_id].ok and "offline" in results[first.source_id].error
    assert load_pack(PACK_ID, root).source(first.source_id).content_sha256 == ""


def test_unknown_source_ids_are_reported(root):
    [result] = fetch_sources(PACK_ID, ["NOPE"], fetch=fake_fetcher({}), today=DAY, root=root)
    assert not result.ok and result.error == "not in sources.json"


def test_check_finds_changed_pages_and_the_rules_to_review(root):
    fetch_sources(PACK_ID, fetch=fake_fetcher(all_pages(root)), today=DAY, root=root)
    pack = load_pack(PACK_ID, root)
    changed = pack.source("NVI_TCKK_FAQ")
    pages = all_pages(root)
    pages[changed.url] = PAGE.replace("Son altı ay", "Son üç ay")
    checks = {c.source_id: c for c in check_sources(PACK_ID, fetch=fake_fetcher(pages), root=root)}
    assert checks["NVI_TCKK_FAQ"].status == "changed"
    assert checks["NVI_TCKK_FAQ"].rules_to_review == [r.rule_id for r in pack.rules if "NVI_TCKK_FAQ" in r.source_ids]
    assert any("Son üç ay" in line for line in checks["NVI_TCKK_FAQ"].preview)
    assert checks["NVI_TCKK_PROCEDURES"].status == "unchanged"


def test_check_reports_unreachable_and_never_saved_sources(root):
    fetch_sources(PACK_ID, ["NVI_TCKK_FAQ"], fetch=fake_fetcher(all_pages(root)), today=DAY, root=root)
    checks = {c.source_id: c.status for c in check_sources(PACK_ID, fetch=fake_fetcher({}), root=root)}
    assert checks == {"NVI_TCKK_PROCEDURES": "not_saved", "NVI_TCKK_FAQ": "unreachable",
                      "NVI_APPOINTMENT_FAQ": "not_saved"}


def test_diff_preview_lists_added_and_removed_lines():
    assert diff_preview("a\nb", "a\nc") == ["-b", "+c"]


# -------------------------------------------------------------- add source --
def test_a_local_document_can_be_added_with_its_official_url(root, tmp_path):
    copy = tmp_path / "talimat.txt"
    copy.write_text("Kayıp kimlik kartları için başvuru talimatı. " * 3, encoding="utf-8")
    source = Source(source_id="NVI_NEW_TALIMAT", title="Talimat", url="https://www.nvi.gov.tr/example.pdf",
                    retrieved_at="", authority="government_primary")
    result = add_source(PACK_ID, source, copy, today=DAY, root=root)
    assert result.ok
    saved = load_pack(PACK_ID, root).source("NVI_NEW_TALIMAT")
    assert saved.retrieved_at == "2026-11-19" and saved.content_sha256
    with pytest.raises(ValueError):
        add_source(PACK_ID, source, copy, today=DAY, root=root)


def test_an_existing_source_can_be_saved_from_a_local_copy(root, tmp_path):
    copy = tmp_path / "page.html"
    copy.write_text(PAGE, encoding="utf-8")
    assert fetch_from_file(PACK_ID, "NVI_TCKK_FAQ", copy, today=DAY, root=root).ok
    assert "biyometrik" in stored_text(root / PACK_ID / "sources_text" / "NVI_TCKK_FAQ.txt")


# ------------------------------------------------------------ rule drafts --
def drafting_pack(root):
    fetch_sources(PACK_ID, fetch=fake_fetcher(all_pages(root)), today=DAY, root=root)
    return load_pack(PACK_ID, root)


def test_drafts_keep_real_quotes_and_flag_invented_ones(root, store):
    pack = drafting_pack(root)
    reply = {"requirements": [
        {"description": "A recent biometric photo is needed.", "quote": "Son altı ay içinde çekilmiş biyometrik fotoğraf",
         "source_id": "NVI_TCKK_PROCEDURES", "fields": ["photo_date"], "severity": "blocking"},
        {"description": "Invented fee.", "quote": "Kimlik kartı ücreti 500 TL olarak ödenir",
         "source_id": "NVI_TCKK_FAQ", "fields": ["fee_payment_date"]},
        {"description": "Unknown field.", "quote": "Fotoğraflı ve geçerli kimlik belgesi",
         "source_id": "NVI_TCKK_PROCEDURES", "fields": ["shoe_size"]}]}
    drafts = draft_rules(store, pack, ScriptedBackend([reply]), topics=["biyometrik fotoğraf"], k=4)
    assert [d.quote_found for d in drafts] == [True, False, True]
    assert drafts[0].problems == [] and drafts[0].similar_rules == ["TCF008"]   # reads photo_date via photo_age_months
    assert drafts[1].problems == ["quote not found in the saved source text"]
    assert drafts[2].fields == [] and "unknown fields: shoe_size" in drafts[2].problems
    assert all(d.status == "unverified" for d in drafts)


def test_the_same_quote_from_two_topics_becomes_one_draft(root, store):
    pack = drafting_pack(root)
    proposal = {"description": "Photo.", "quote": "Son altı ay içinde çekilmiş biyometrik fotoğraf",
                "source_id": "NVI_TCKK_FAQ", "fields": ["photo_date"]}
    backend = ScriptedBackend([{"requirements": [proposal]}, {"requirements": [proposal]}])
    drafts = draft_rules(store, pack, backend, topics=["biyometrik fotoğraf", "kimlik belgesi"])
    assert len(drafts) == 1 and drafts[0].topics == ["biyometrik fotoğraf", "kimlik belgesi"]


def test_a_failed_model_call_skips_the_topic_and_rules_json_is_never_touched(root, store):
    pack = drafting_pack(root)
    rules_before = (pack.root / "rules.json").read_text(encoding="utf-8")
    drafts = draft_rules(store, pack, ScriptedBackend(["not json", "still not json"]), topics=["biyometrik fotoğraf"])
    path = save_drafts(pack, drafts)
    assert drafts == [] and path.name == DRAFTS_FILE and json.loads(path.read_text(encoding="utf-8")) == []
    assert (pack.root / "rules.json").read_text(encoding="utf-8") == rules_before


def test_quote_check_ignores_case_and_spacing_but_needs_a_real_passage(root):
    pack = drafting_pack(root)
    assert quote_in_source(pack, "NVI_TCKK_FAQ", "SON ALTI AY   içinde çekilmiş")
    assert not quote_in_source(pack, "NVI_TCKK_FAQ", "altı")            # too short to prove anything
    assert not quote_in_source(pack, "NVI_TCKK_FAQ", "Son on iki ay içinde")


def test_a_topic_with_no_similar_passage_is_skipped_without_a_model_call(root, store):
    pack = drafting_pack(root)
    backend = ScriptedBackend([])
    assert draft_rules(store, pack, backend, topics=["zzzz qqqq"]) == [] and backend.calls == []


def test_a_description_that_repeats_the_topic_and_a_list_of_every_field_are_flagged(root, store):
    """Seen live with qwen3:4b: the topic copied as description and the first six fields listed."""
    pack = drafting_pack(root)
    reply = {"requirements": [{"description": "Which photo do I need?", "quote": "Son altı ay içinde çekilmiş biyometrik",
                               "source_id": "NVI_TCKK_FAQ",
                               "fields": ["is_turkish_citizen", "applying_in_turkiye", "application_reason", "birth_date"]}]}
    [draft] = draft_rules(store, pack, ScriptedBackend([reply]), topics=["Which photo do I need? biyometrik fotoğraf"])
    assert draft.quote_found and len(draft.problems) == 1 and "4 fields named" in draft.problems[0]
    copied = {"requirements": [{**reply["requirements"][0], "description": "Which biyometrik fotoğraf?",
                                "fields": ["photo_date"]}]}
    [draft] = draft_rules(store, pack, ScriptedBackend([copied]), topics=["Which biyometrik fotoğraf"])
    assert draft.problems == ["the description repeats the topic: write the requirement in your own words"]
