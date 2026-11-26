"""Inspect, build or query the public corpus from the local checkout."""
import argparse
import json
import sys

from src.rag.retriever import configured_embedding, retrieval_chunks, retrieval_status, search_official_sources
from src.rag.semantic import EMBEDDING_ERRORS, build_index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "index", "search"))
    parser.add_argument("question", nargs="?")
    parser.add_argument("--scope", choices=("case", "public"), default="public")
    parser.add_argument("--model", help="Installed local Ollama embedding model")
    parser.add_argument("--rebuild", action="store_true", help="Regenerate vectors after replacing a model under the same tag")
    args = parser.parse_args(argv)
    model = configured_embedding(args.model)
    try:
        if args.action == "index":
            if not model:
                parser.error("Configure OLLAMA_EMBED_MODEL or give --model before building a vector index")
            retained_ids = {chunk.chunk_id for chunk in retrieval_chunks("public")} if args.scope == "case" else ()
            summary = build_index(retrieval_chunks(args.scope), model, rebuild=args.rebuild, retained_ids=retained_ids)
            print(json.dumps(summary, indent=2))
        elif args.action == "search":
            if not args.question:
                parser.error("Enter a public guidance question")
            details = {}
            hits = search_official_sources(args.question, scope=args.scope, details=details, embedding_model=model)
            print(json.dumps({"retrieval": details, "sources": [row.model_dump() for row in hits]}, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(retrieval_status(args.scope, embedding_model=model), indent=2, ensure_ascii=False))
    except EMBEDDING_ERRORS:
        print("The public index could not be built or read. Check corpus status and the installed Ollama model.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
