"""Test MCP server - proves it works with any MCP client."""
import urllib.request
import json

BASE = "https://verisearch-production.up.railway.app/mcp/messages"

def mcp_call(method, params=None, msg_id=1):
    body = {"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}}
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE, data=data, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read().decode())

print("=== 1. Initialize MCP ===")
r = mcp_call("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}})
print(f"  Server: {r['result']['serverInfo']}")

print("\n=== 2. List Tools ===")
r = mcp_call("tools/list", msg_id=2)
for tool in r["result"]["tools"]:
    print(f"  - {tool['name']}: {tool['description'][:70]}")

print("\n=== 3. Call 'search' tool ===")
r = mcp_call("tools/call", {"name": "search", "arguments": {"query": "artificial intelligence", "mode": "hybrid", "num_results": 3}}, msg_id=3)
print(f"  {r['result']['content'][0]['text'][:200]}")

print("\n=== 4. Call 'answer' tool ===")
r = mcp_call("tools/call", {"name": "answer", "arguments": {"query": "What is AI?"}}, msg_id=4)
print(f"  {r['result']['content'][0]['text'][:300]}")

print("\n=== MCP Server fully operational! ===")
