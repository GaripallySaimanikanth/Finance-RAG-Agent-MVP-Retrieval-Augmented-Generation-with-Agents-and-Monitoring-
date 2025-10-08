#!/usr/bin/env python3
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]

# Import pipeline pieces
import sys
sys.path.append(str(ROOT / "src"))
from data.loader import load_and_clean
from rag.vector_store import DocumentStore
from rag.pipeline import RagPipeline
from llm.local import LocalAnswerComposer
from eval.metrics import evaluate_answer


class RagApp:
    def __init__(self):
        raw_dir = ROOT / "data" / "raw"
        clean_dir = ROOT / "data" / "clean"
        os.makedirs(clean_dir, exist_ok=True)
        self.docs = load_and_clean(raw_dir, clean_dir)
        self.store = DocumentStore(chunk_size=600, chunk_overlap=80)
        self.store.fit([d["content"] for d in self.docs], meta=[{"source": d["path"]} for d in self.docs])
        self.rag = RagPipeline(store=self.store, answerer=LocalAnswerComposer())

    def answer(self, query: str) -> dict:
        out = self.rag.answer(query, top_k=4)
        metrics = evaluate_answer(out["answer"], [c["text"] for c in out["contexts"]])
        out["metrics"] = metrics
        return out


APP = RagApp()


class Handler(BaseHTTPRequestHandler):
    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def _json(self, code, obj):
        self.send_response(code)
        self._set_cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode("utf-8"))

    def _serve_static(self, rel_path: str):
        # prevent path traversal
        safe = Path("public") / rel_path.strip("/")
        safe = safe.resolve()
        if not str(safe).startswith(str((ROOT / "public").resolve())):
            return self._json(403, {"error": "forbidden"})
        if safe.is_dir():
            safe = safe / "index.html"
        if not safe.exists():
            return self._json(404, {"error": "not found"})
        ctype = "text/html" if safe.suffix == ".html" else "text/css" if safe.suffix == ".css" else "application/javascript" if safe.suffix == ".js" else "text/plain"
        self.send_response(200)
        self._set_cors()
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(safe.read_bytes())

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            return self._json(200, {"ok": True})
        # Serve dashboard/static
        if path == "/":
            return self._serve_static("index.html")
        if path.startswith("/public/"):
            return self._serve_static(path[len("/public/"):])
        # Fallback to serving root public path
        return self._serve_static(path)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/ask":
            ln = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(ln)
            try:
                obj = json.loads(body.decode("utf-8"))
                query = obj.get("query", "").strip()
            except Exception:
                return self._json(400, {"error": "invalid json"})
            if not query:
                return self._json(400, {"error": "empty query"})
            out = APP.answer(query)
            return self._json(200, out)
        return self._json(404, {"error": "not found"})


def run(port: int = 4000, host: str = "0.0.0.0"):
    try:
        server = HTTPServer((host, port), Handler)
    except OSError as exc:
        if port != 0:
            print(f"[warn] Port {port} unavailable ({exc}); retrying with an ephemeral port.")
            server = HTTPServer((host, 0), Handler)
        else:
            raise

    bound_port = server.server_port
    print(f"Finance RAG server listening on {host}:{bound_port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[info] Shutting down server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 4000)))
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    args = ap.parse_args()
    run(args.port, args.host)
