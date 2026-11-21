"""Small read-only MCP stdio demonstrator, pinned to handshake revision 2025-11-25.

Normal application execution uses the internal registry and has no MCP dependency.
Run: python -m src.integrations.mcp.server
stdout is JSON-RPC only. No external mutation or independent Digital Twin exists.
"""
import json
import sys
from src.tools.registry import get_registry, execute_tool
from src.tools.schemas import ToolCall

PROTOCOL_VERSION = "2025-11-25"
RESOURCE_URI = "ikamet://official-guidance"


def handle(request):
    if not isinstance(request,dict) or request.get("jsonrpc")!="2.0":
        return {"jsonrpc":"2.0","id":request.get("id") if isinstance(request,dict) else None,"error":{"code":-32600,"message":"Invalid JSON-RPC request"}}
    request_id, method = request.get("id"), request.get("method")
    if request_id is None:
        return None
    result = None
    if method=="initialize":
        result = {"protocolVersion":PROTOCOL_VERSION,"capabilities":{"tools":{},"resources":{}},"serverInfo":{"name":"ikamet-local-guidance","version":"1.0.0"}}
    elif method=="ping":
        result = {}
    elif method=="tools/list":
        spec = get_registry()["search_official_sources"]
        result = {"tools":[{"name":spec.name,"description":spec.description,"inputSchema":spec.input_schema,
            "annotations":{"readOnlyHint":True,"destructiveHint":False,"openWorldHint":False}}]}
    elif method=="resources/list":
        result = {"resources":[{"uri":RESOURCE_URI,"name":"Official Ikamet guidance","mimeType":"application/json","description":"Authoritative project guidance with source citations"}]}
    elif method=="resources/read":
        if request.get("params",{}).get("uri") != RESOURCE_URI:
            return {"jsonrpc":"2.0","id":request_id,"error":{"code":-32602,"message":"Unsupported resource URI"}}
        from src.rag.retriever import search_official_sources
        sources = search_official_sources("residence permit exchange student", limit=5)
        result = {"contents":[{"uri":RESOURCE_URI,"mimeType":"application/json","text":json.dumps([source.model_dump(mode="json") for source in sources])}]}
    elif method=="tools/call":
        params = request.get("params",{})
        if params.get("name") != "search_official_sources":
            return {"jsonrpc":"2.0","id":request_id,"error":{"code":-32602,"message":"Only read-only guidance search is exposed"}}
        observation = execute_tool({}, ToolCall(name="search_official_sources",arguments=params.get("arguments",{})))
        result = {"content":[{"type":"text","text":json.dumps(observation.data)}],"isError":observation.status!="SUCCEEDED"}
    else:
        return {"jsonrpc":"2.0","id":request_id,"error":{"code":-32601,"message":"Method not found for pinned protocol"}}
    return {"jsonrpc":"2.0","id":request_id,"result":result}


def main():
    for line in sys.stdin:
        try:
            response = handle(json.loads(line))
        except (ValueError,TypeError,KeyError):
            response = {"jsonrpc":"2.0","id":None,"error":{"code":-32700,"message":"Invalid JSON input"}}
        except Exception:
            response = {"jsonrpc":"2.0","id":None,"error":{"code":-32603,"message":"Local guidance capability unavailable"}}
        if response is not None:
            print(json.dumps(response),flush=True)

if __name__=="__main__":main()
