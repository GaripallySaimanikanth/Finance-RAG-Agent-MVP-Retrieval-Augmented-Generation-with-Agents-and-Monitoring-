#!/usr/bin/env python3
"""
Stage 4: Minimal API & usage tracking stub using Python's stdlib.
Endpoints (mock):
  GET /health -> 200
  POST /ask -> accepts {"query": "..."}, returns mocked response and increments usage
  GET /usage -> returns per-key counts

This is a stub. Replace with FastAPI or similar for production.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

USAGE = {}


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            return self._json(200, {"ok": True})
        if path == "/usage":
            return self._json(200, USAGE)
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/ask":
            key = self.headers.get("X-API-Key", "anon")
            ln = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(ln)
            try:
                obj = json.loads(body.decode("utf-8"))
            except Exception:
                return self._json(400, {"error": "invalid json"})
            query = obj.get("query", "")
            USAGE[key] = USAGE.get(key, 0) + 1
            # In a full implementation, call the RAG pipeline here.
            return self._json(200, {"answer": f"stubbed answer for: {query}", "usage": USAGE[key]})
        return self._json(404, {"error": "not found"})


def run(port=8080):
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"API stub listening on :{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()

