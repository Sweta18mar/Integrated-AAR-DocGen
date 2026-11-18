"""A deterministic OpenAI-compatible server that plays the agent, for tests and offline demos.

It follows the same JSON protocol a real model is asked to follow, so the whole
workflow (search → write each section → finish, vendor detection, slides) runs
without an API key. Every request body is recorded in `REQUESTS`.

Run standalone:  python tests/fake_llm.py 8099
"""

import hashlib
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REQUESTS: list[dict] = []


def _agent_reply(messages: list[dict]) -> dict:
    task = messages[1]["content"]
    required = re.findall(r"^### Section: (.+)$", task, re.M)
    revising = "This is a REVISION" in task
    tool_results = [m["content"] for m in messages if m["role"] == "user" and m["content"].startswith("Result of")]
    actions = []
    for m in messages:
        if m["role"] == "assistant":
            try:
                actions.append(json.loads(m["content"]))
            except json.JSONDecodeError:
                pass
    written = [a["args"]["heading"] for a in actions if a.get("tool") == "write_section"]

    if "TEST_BAD_FIRST_REPLY" in task and len(messages) == 2:
        return {"raw": "I will now search the documents."}
    if not revising and not any(r.startswith("Result of search") for r in tool_results):
        return {"thought": "Find relevant material.", "tool": "search", "args": {"query": required[0], "k": 3}}
    if "get_vendors" in messages[0]["content"] and not any(a.get("tool") == "get_vendors" for a in actions):
        return {"thought": "Check bidders.", "tool": "get_vendors", "args": {}}
    todo = required[:1] if revising else [h for h in required if h not in written]
    todo = [h for h in todo if h not in written]
    if todo:
        prefix = "Revised per reviewer feedback. " if revising else ""
        return {"thought": "Write the next section.", "tool": "write_section",
                "args": {"heading": todo[0], "content": f"{prefix}Content for {todo[0]} based on the documents [S1]."}}
    return {"thought": "All done.", "tool": "finish", "args": {"note": "All sections drafted."}}


def _reply(body: dict) -> str:
    messages = body["messages"]
    system = messages[0]["content"] if messages[0]["role"] == "system" else ""
    user = "\n".join(m["content"] for m in messages if m["role"] == "user")
    if system.startswith("VENDOR_DETECTION"):
        names = sorted(set(re.findall(r"\b([A-Z][a-zA-Z]+ (?:Corp|Inc|Ltd|LLC|Systems))\b", user)))
        return json.dumps({"vendors": [{"name": n, "evidence": "named as a bidder"} for n in names]})
    if system.startswith("SLIDE_GENERATOR"):
        headings = re.findall(r"^## (.+)$", user, re.M)
        return "```json\n" + json.dumps({"slides": [
            {"title": h, "bullets": [f"Key point about {h} [S1]", "Second point"], "notes": f"Notes for {h}"}
            for h in headings]}) + "\n```"
    action = _agent_reply(messages)
    return action["raw"] if "raw" in action else json.dumps(action)


def _vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255 for b in digest[:16]]


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append(body)
        if self.path.endswith("/embeddings"):
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            payload = {"object": "list", "model": body["model"], "usage": {"prompt_tokens": 0, "total_tokens": 0},
                       "data": [{"object": "embedding", "index": i, "embedding": _vector(t)} for i, t in enumerate(inputs)]}
        else:
            payload = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                       "choices": [{"index": 0, "finish_reason": "stop",
                                    "message": {"role": "assistant", "content": _reply(body)}}]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Routed-Via", "fake/agent-1")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def start(port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    print(f"Fake model server on http://127.0.0.1:{port}/v1")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
