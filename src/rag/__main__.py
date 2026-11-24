"""Knowledge base command line: python -m src.rag <command> <procedure_id> ...

  fetch <id> [SOURCE_ID ...]       download the official pages in sources.json and save their texts
  fetch <id> SOURCE_ID --file PATH use a saved copy instead (for sites that refuse automatic downloads)
  add <id> SOURCE_ID --url URL --title T --authority A [--organization O] [--file PATH]
                                   add a new official page or document (a local PDF/TXT with --file)
  check <id>                       download again and list the sources that changed and the rules to review
  index <id>                       embed every saved passage now (otherwise done on the first search)
  search <id> "question"           the passages the agent's search_sources tool would return
  draft-rules <id> [--topic T ...] propose rules from the texts, checked, into rule_drafts.json (for review)
  crawl                            web scraping for procedures without a pack: follow knowledge/seeds.json
                                   (robots.txt obeyed), save the pages in knowledge/corpus/, then embed them
  search-web "question"            the web-corpus passages the chat assistant would get

Embeddings use EMBED_MODEL and the chat model LLM_MODEL (config.py); vectors are cached in the Twin database.
"""
import argparse
import sys
from pathlib import Path
from time import perf_counter

from config import DATABASE_PATH, EMBED_MODEL
from src.llm import LLMError, OllamaBackend
from src.models import Source
from src.pack import load_pack
from src.rag.drafts import draft_rules, save_drafts
from src.rag.crawl import crawl, load_seeds
from src.rag.index import chunk_vectors, pack_chunks, search, search_corpus
from src.rag.knowledge import add_source, check_sources, fetch_from_file, fetch_sources
from src.twin.store import TwinStore


def parser() -> argparse.ArgumentParser:
    """The command line arguments of every command."""
    main = argparse.ArgumentParser(prog="python -m src.rag", description="Knowledge base of a Procedure Pack")
    commands = main.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("procedure_id")
    fetch.add_argument("source_ids", nargs="*")
    fetch.add_argument("--file", type=Path)
    add = commands.add_parser("add")
    add.add_argument("procedure_id")
    add.add_argument("source_id")
    for name in ("--url", "--title", "--authority"):
        add.add_argument(name, required=True)
    add.add_argument("--organization", default="")
    add.add_argument("--notes", default="")
    add.add_argument("--file", type=Path)
    for name in ("check", "index"):
        commands.add_parser(name).add_argument("procedure_id")
    find = commands.add_parser("search")
    find.add_argument("procedure_id")
    find.add_argument("query")
    draft = commands.add_parser("draft-rules")
    draft.add_argument("procedure_id")
    draft.add_argument("--topic", action="append")
    commands.add_parser("crawl")
    commands.add_parser("search-web").add_argument("query")
    return main


def print_fetch(results) -> int:
    """One line per source; exit code 1 when any download failed."""
    for result in results:
        if result.ok:
            print(f"OK   {result.source_id}: {result.chars} characters{' (changed)' if result.changed else ''}")
        else:
            print(f"FAIL {result.source_id}: {result.error}")
    return 0 if all(result.ok for result in results) else 1


def print_checks(checks) -> int:
    """Status per source, the rules to review and a short diff; exit code 1 when a page changed."""
    for check in checks:
        print(f"{check.status.upper():12} {check.source_id} {check.error}")
        if check.rules_to_review:
            print("             review rules: " + ", ".join(check.rules_to_review))
        for line in check.preview:
            print(f"             {line}")
    return 1 if any(check.status == "changed" for check in checks) else 0


def run_index(procedure_id: str) -> int:
    """Embed every passage of the pack now and report how long it took."""
    pack = load_pack(procedure_id)
    started = perf_counter()
    with TwinStore(DATABASE_PATH) as store:
        chunks = pack_chunks(pack)
        chunk_vectors(store, pack.id, OllamaBackend(), chunks)
    print(f"{len(chunks)} passages embedded with {EMBED_MODEL} in {perf_counter() - started:.1f} s")
    return 0


def run_search(procedure_id: str, query: str) -> int:
    with TwinStore(DATABASE_PATH) as store:
        for hit in search(store, load_pack(procedure_id), OllamaBackend(), query):
            print(f"[{hit.score:.3f}] {hit.chunk_id}: {' '.join(hit.text.split())[:220]}")
    return 0


def run_drafts(procedure_id: str, topics: list[str] | None) -> int:
    """Draft rules for the pack and say how many have no problem."""
    pack = load_pack(procedure_id)
    with TwinStore(DATABASE_PATH) as store:
        drafts = draft_rules(store, pack, OllamaBackend(), topics)
    path = save_drafts(pack, drafts)
    checked = sum(draft.quote_found and not draft.problems for draft in drafts)
    print(f"{len(drafts)} drafts ({checked} without problems) written to {path}")
    return 0


def run_crawl() -> int:
    """Crawl the seeds, report what was saved or refused, then embed the corpus."""
    started = perf_counter()
    report = crawl(load_seeds())
    for page in report.saved:
        print(f"SAVED  {page.source_id}: {page.title}")
    for url in report.refused_by_robots:
        print(f"ROBOTS {url} (robots.txt does not allow it)")
    for url in report.failed + report.too_short:
        print(f"SKIP   {url}")
    with TwinStore(DATABASE_PATH) as store:
        search_corpus(store, OllamaBackend(), "index")      # embeds every passage once
    print(f"{len(report.saved)} pages saved and embedded in {perf_counter() - started:.0f} s")
    return 0


def run_search_web(query: str) -> int:
    with TwinStore(DATABASE_PATH) as store:
        for hit in search_corpus(store, OllamaBackend(), query, k=5):
            print(f"[{hit.score:.3f}] {hit.source_id}: {' '.join(hit.text.split())[:200]}")
    return 0


def main(argv: list[str]) -> int:
    """Run one command; network and model errors are printed, not raised."""
    args = parser().parse_args(argv)
    try:
        if args.command == "fetch" and args.file:
            if len(args.source_ids) != 1:
                raise ValueError("--file needs exactly one SOURCE_ID")
            return print_fetch([fetch_from_file(args.procedure_id, args.source_ids[0], args.file)])
        if args.command == "fetch":
            return print_fetch(fetch_sources(args.procedure_id, args.source_ids or None))
        if args.command == "add":
            source = Source(source_id=args.source_id, title=args.title, url=args.url, retrieved_at="",
                            authority=args.authority, organization=args.organization, notes=args.notes)
            return print_fetch([add_source(args.procedure_id, source, args.file)])
        if args.command == "check":
            return print_checks(check_sources(args.procedure_id))
        if args.command == "index":
            return run_index(args.procedure_id)
        if args.command == "search":
            return run_search(args.procedure_id, args.query)
        if args.command == "crawl":
            return run_crawl()
        if args.command == "search-web":
            return run_search_web(args.query)
        return run_drafts(args.procedure_id, args.topic)
    except (LLMError, ValueError) as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
