# SPDX-License-Identifier: Apache-2.0
"""A stand-in inference server for tests: a real process with real HTTP, but no model.

Modes:
  ok           503 on /health while "loading", then 200; serves /v1/models and chat completions
  crash        prints a llama.cpp-style out-of-memory line and exits 1
  hang         never becomes healthy
  ignore-term  ignores SIGTERM (tests SIGKILL escalation)
  exit-later   becomes healthy, then exits with code 3 after a moment
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", default="ok")
    parser.add_argument("--load-seconds", type=float, default=0.2)
    parser.add_argument("--model-name", default="fake")
    parser.add_argument("--child-pid-file", default=None)
    parser.add_argument("--require-key", default=None, help="Bearer token required on /v1/*")
    parser.add_argument("--mount", default=None, help="Ignored; stands in for a Docker mount.")
    args = parser.parse_args()

    print(f"fake engine pid={os.getpid()} mode={args.mode} port={args.port}", flush=True)
    print(f"api key set: {bool(os.environ.get('LLAMA_API_KEY'))}", flush=True)
    if args.mode == "crash":
        print("ggml_metal: failed to allocate buffer of size 9999 MiB", flush=True)
        sys.exit(1)
    if args.mode == "ignore-term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if args.child_pid_file:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
        Path(args.child_pid_file).write_text(str(child.pid))

    ready_at = time.monotonic() + args.load_seconds

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:  # silence access logs
            return None

        def _json(self, status: int, body: object) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _authorized(self) -> bool:
            expected = args.require_key or os.environ.get("LLAMA_API_KEY")
            if not expected:
                return True
            if self.headers.get("Authorization") == f"Bearer {expected}":
                return True
            self._json(401, {"error": "bad key"})
            return False

        def do_GET(self) -> None:
            if self.path == "/metrics":
                if not self._authorized():
                    return
                payload = b"fake_requests_total 1\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path == "/health":
                loading = args.mode == "hang" or time.monotonic() < ready_at
                self._json(503 if loading else 200, {"status": "loading" if loading else "ok"})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": args.model_name}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)  # the prompt is never printed or stored
            if not self._authorized():
                return
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "not found"})
                return
            request = json.loads(raw or b"{}")
            echo = {
                "model": request.get("model"),
                "saw_client_credentials": "x-api-key" in self.headers
                or "Origin" in self.headers
                or "Cookie" in self.headers,
            }
            if request.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for word in ("Hello", " from", " the fake engine."):
                    delta = {"choices": [{"delta": {"content": word}}], **echo}
                    self.wfile.write(f"data: {json.dumps(delta)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                return
            message = {"role": "assistant", "content": "Hello from the fake engine."}
            self._json(200, {"choices": [{"index": 0, "message": message}], **echo})

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    if args.mode == "exit-later":
        threading.Timer(args.load_seconds + 0.5, lambda: os._exit(3)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
