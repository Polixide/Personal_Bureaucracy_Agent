"""Read-only MCP server (stdio, JSON-RPC 2.0), pinned to handshake revision 2025-11-25.

Ported from Marco's demonstrator (feature/marco-milestone1, b0ee6c1) to the Procedure Packs and the
embedding RAG. It lets any MCP client (an IDE, another agent) ask the same official sources the agent uses:

  tools      find_procedure          which pack matches a goal (deterministic ranking, no model)
             search_official_sources passages of one pack's saved official texts (embedding search)
  resources  procedure://<id>        a pack's rules and sources as JSON

Read-only on purpose: nothing here changes a Digital Twin. The only write is the embedding cache
(`rag_vectors`), which the app shares. The app itself does not use MCP; it calls the tool registry.
Run: python -m src.integrations.mcp.server   (stdout carries JSON-RPC only)
"""
import json
import sys
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from config import DATABASE_PATH, RAG_TOP_K
from src.llm import Backend, LLMError, OllamaBackend, short_validation_error
from src.models import Pack
from src.pack import load_all_packs
from src.rag.index import search
from src.tools.search_procedures import rank
from src.twin.store import TwinStore

PROTOCOL_VERSION = "2025-11-25"
URI_PREFIX = "procedure://"
READ_ONLY = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}


class FindProcedureInput(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="The goal in the user's own words")


class SearchSourcesInput(BaseModel):
    procedure_id: str = Field(description="A procedure id, for example from find_procedure")
    query: str = Field(min_length=3, max_length=300, description="What to find in the official sources")
    k: int = Field(default=RAG_TOP_K, ge=1, le=5, description="How many passages")


class McpError(Exception):
    """A JSON-RPC error to return to the client (code and message)."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class McpServer:
    """Handles one JSON-RPC request at a time. Packs are loaded once; store and model only when needed."""

    def __init__(self, packs: dict[str, Pack], store_factory: Callable[[], TwinStore],
                 backend_factory: Callable[[], Backend]):
        self.packs = packs
        self.store_factory = store_factory
        self.backend_factory = backend_factory
        self._store: TwinStore | None = None
        self._backend: Backend | None = None

    def handle(self, request: Any) -> dict | None:
        """The response for one request, or None for a notification (no id)."""
        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or "method" not in request:
            return error_response(request.get("id") if isinstance(request, dict) else None, -32600,
                                  "Invalid JSON-RPC request")
        if "id" not in request:
            return None
        try:
            result = self.dispatch(request["method"], request.get("params") or {})
        except McpError as error:
            return error_response(request["id"], error.code, str(error))
        return {"jsonrpc": "2.0", "id": request["id"], "result": result}

    def dispatch(self, method: str, params: dict) -> dict:
        """Route a method name to its handler."""
        if method == "initialize":
            return {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {"name": "bureaucracy-official-sources", "version": "2.0.0"}}
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": tool_list()}
        if method == "resources/list":
            return {"resources": [resource_entry(pack) for pack in self.packs.values()]}
        if method == "resources/read":
            return self.read_resource(params.get("uri", ""))
        if method == "tools/call":
            return self.call_tool(params.get("name", ""), params.get("arguments") or {})
        raise McpError(-32601, f"Method not found: {method}")

    def read_resource(self, uri: str) -> dict:
        """A pack's rules and sources as one JSON text."""
        pack = self.packs.get(uri.removeprefix(URI_PREFIX)) if uri.startswith(URI_PREFIX) else None
        if pack is None:
            raise McpError(-32602, f"Unknown resource: {uri}")
        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(pack_summary(pack))}]}

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Validate the arguments, run the tool, and wrap the result. Tool failures are results, not errors."""
        if name not in TOOLS:
            raise McpError(-32602, f"Unknown tool {name!r}; this server only exposes read-only tools")
        try:
            args = TOOLS[name][0].model_validate(arguments)
        except ValidationError as error:
            return tool_result({"error": f"Invalid arguments: {short_validation_error(error)}"}, is_error=True)
        try:
            data = self.find_procedure(args) if name == "find_procedure" else self.search_sources(args)
        except (LLMError, McpError) as error:
            return tool_result({"error": str(error)}, is_error=True)
        return tool_result(data, is_error=False)

    def find_procedure(self, args: FindProcedureInput) -> dict:
        """Ranked packs for a goal; the same scoring as the agent's search_procedures, without its Twin writes."""
        matches = rank(self.packs, args.query)
        tie = len(matches) > 1 and matches[0].score == matches[1].score
        selected = matches[0].procedure_id if matches and not tie else None
        return {"matches": [m.model_dump() for m in matches], "selected": selected}

    def search_sources(self, args: SearchSourcesInput) -> dict:
        """Passages of one pack's official texts, best first, with their source ids for citation."""
        pack = self.packs.get(args.procedure_id)
        if pack is None:
            raise McpError(-32602, f"Unknown procedure {args.procedure_id!r}; call find_procedure first")
        hits = search(self.store(), pack, self.backend(), args.query, args.k)
        status = "verified" if pack.procedure.status == "verified" else "UNVERIFIED rules: say so when citing"
        return {"procedure_id": pack.id, "status": status, "hits": [hit.model_dump() for hit in hits]}

    def store(self) -> TwinStore:
        if self._store is None:
            self._store = self.store_factory()
        return self._store

    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = self.backend_factory()
        return self._backend


TOOLS: dict[str, tuple[type[BaseModel], str]] = {
    "find_procedure": (FindProcedureInput, "Which supported procedure matches a goal (deterministic, no model)."),
    "search_official_sources": (SearchSourcesInput, "Search one procedure's saved official texts and return "
                                                    "passages with source ids to cite. Sources explain; they never "
                                                    "decide whether a requirement is met."),
}


def tool_list() -> list[dict]:
    """MCP tool descriptors with JSON schemas from the Pydantic input models."""
    return [{"name": name, "description": text, "inputSchema": model.model_json_schema(), "annotations": READ_ONLY}
            for name, (model, text) in TOOLS.items()]


def resource_entry(pack: Pack) -> dict:
    return {"uri": URI_PREFIX + pack.id, "name": pack.procedure.title, "mimeType": "application/json",
            "description": f"Rules and official sources of {pack.id} ({pack.procedure.status})"}


def pack_summary(pack: Pack) -> dict:
    """What a client needs to cite a pack: its rules (with status and sources) and the source list."""
    return {"procedure_id": pack.id, "title": pack.procedure.title, "status": pack.procedure.status,
            "rules": [{"rule_id": r.rule_id, "description": r.description, "status": r.status,
                       "source_ids": r.source_ids} for r in pack.rules],
            "sources": [{"source_id": s.source_id, "title": s.title, "url": s.url} for s in pack.sources]}


def tool_result(data: dict, is_error: bool) -> dict:
    return {"content": [{"type": "text", "text": json.dumps(data, default=str)}], "isError": is_error}


def error_response(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def default_server() -> McpServer:
    """Packs from procedures/, the shared Twin database (embedding cache only) and the local Ollama models."""
    return McpServer(load_all_packs(), lambda: TwinStore(DATABASE_PATH), OllamaBackend)


def serve(server: McpServer, lines, write: Callable[[str], None]) -> None:
    """Read one JSON request per line, write one JSON response per line."""
    for line in lines:
        if not line.strip():
            continue
        try:
            response = server.handle(json.loads(line))
        except json.JSONDecodeError:
            response = error_response(None, -32700, "Invalid JSON input")
        if response is not None:
            write(json.dumps(response))


def main() -> None:
    serve(default_server(), sys.stdin, lambda text: print(text, flush=True))


if __name__ == "__main__":
    main()
