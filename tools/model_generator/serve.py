#!/usr/bin/env python3
"""Serve the hexapod model-generator web tool and handle saving.

    .venv/bin/python tools/model_generator/serve.py          # opens the tool

The page (index.html) is a live three.js preview + control panel. Hitting
"save" POSTs the current spec to /save, which writes
    output/<name>.yaml   (the source-of-truth spec)
    output/<name>.xml    (the MJCF, via generate.py)
so the browser never touches the disk itself. Serving over http (not file://)
is needed so the page can POST.
"""

import http.server
import json
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "output"
PORT = 8001


def load_spec(name: str) -> dict:
    """Read output/<name>.yaml, merged with defaults so every field is present."""
    name = (name or "hexapod").strip() or "hexapod"
    path = OUT / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path.relative_to(ROOT)} not found (save it first)")
    return {"config": generate.load_config(path), "path": str(path.relative_to(ROOT))}


def save_spec(cfg: dict) -> dict:
    """Write <name>.yaml then generate <name>.xml; return the paths."""
    name = (cfg.get("name") or "hexapod_custom").strip() or "hexapod_custom"
    cfg["name"] = name
    OUT.mkdir(parents=True, exist_ok=True)
    yaml_path = OUT / f"{name}.yaml"
    xml_path = OUT / f"{name}.xml"
    yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    generate.write_model(generate.load_config(yaml_path), xml_path)
    return {
        "yaml": str(yaml_path.relative_to(ROOT)),
        "xml": str(xml_path.relative_to(ROOT)),
    }


def main() -> None:
    open_browser = "--no-open" not in sys.argv[1:]
    viewer_path = "/tools/model_generator/index.html"
    url = f"http://localhost:{PORT}{viewer_path}"

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(ROOT), **k)

        def log_message(self, *a):  # quieter console
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(302)
                self.send_header("Location", viewer_path)
                self.end_headers()
                return
            if urlparse(self.path).path == "/load":
                name = parse_qs(urlparse(self.path).query).get("name", [""])[0]
                try:
                    body = json.dumps({"ok": True, **load_spec(name)}).encode()
                    print(f"loaded {json.loads(body)['path']}")
                except Exception as e:
                    body = json.dumps({"ok": False, "error": str(e)}).encode()
                    print(f"load failed: {e}")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def do_POST(self):
            if self.path != "/save":
                self.send_error(404)
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
                cfg = json.loads(self.rfile.read(n) or b"{}")
                result = save_spec(cfg)
                body = json.dumps({"ok": True, **result}).encode()
                print(f"saved {result['yaml']} + {result['xml']}")
            except Exception as e:  # surface the error back to the page
                body = json.dumps({"ok": False, "error": str(e)}).encode()
                print(f"save failed: {e}")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    socketserver.TCPServer.allow_reuse_address = True  # avoid TIME_WAIT on quick restarts
    try:
        httpd = socketserver.TCPServer(("", PORT), Handler)
    except OSError as e:
        if e.errno == 48:  # EADDRINUSE
            sys.exit(f"port {PORT} is already in use. Free it with: lsof -ti tcp:{PORT} | xargs kill")
        raise
    with httpd:
        print(f"serving {ROOT} at http://localhost:{PORT}")
        print(f"open {url}\n(Ctrl-C to stop)")
        if open_browser:
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
