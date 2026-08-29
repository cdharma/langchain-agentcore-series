#!/usr/bin/env python3
"""Demo hub server: static files + a same-origin proxy to the local AgentCore agent.

chat.html can't call the agent directly: the `agentcore dev` server (port 8082)
doesn't answer CORS preflights, so the browser would block cross-origin calls.
This serves the hub AND forwards POST /agent -> localhost:8082/invocations,
adding the X-Agentcore-Local header the dev server requires.

Run:  python3 serve.py          # hub on http://localhost:8321
      AGENT_PORT=8080 python3 serve.py   # if `agentcore dev` bound elsewhere
"""
import json
import os
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

AGENT = f"http://localhost:{os.environ.get('AGENT_PORT', '8082')}"


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/agent/ping":
            return self._forward("GET", f"{AGENT}/ping")
        if self.path == "/agent/meta":  # who are we fronting? (label set at server start)
            data = json.dumps({"label": os.environ.get("AGENT_LABEL", "WeatherAgent (local dev)")}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def do_POST(self):
        if self.path != "/agent":
            self.send_error(404)
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        headers = {"Content-Type": "application/json", "X-Agentcore-Local": "true"}
        sid = self.headers.get("X-Session-Id")
        if sid:  # becomes context.session_id -> the agent's thread_id (v0.3 trick)
            headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] = sid
        self._forward("POST", f"{AGENT}/invocations", body, headers)

    def _forward(self, method, url, body=None, headers=None):
        req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data, status = r.read(), r.status
        except urllib.error.HTTPError as e:
            data, status = e.read(), e.code
        except OSError as e:
            data = json.dumps({"error": f"local agent not reachable at {AGENT} — is `agentcore dev` running? ({e})"}).encode()
            status = 502
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print(f"demo hub: http://localhost:8321  (proxying /agent -> {AGENT})")
    ThreadingHTTPServer(("127.0.0.1", 8321), Handler).serve_forever()
