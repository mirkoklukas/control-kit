#!/usr/bin/env python3
"""Serve the repo root and open the MJCF viewer in a browser.

    python tools/model_viz/serve.py                      # opens models/hexapod.xml
    python tools/model_viz/serve.py models/cartpole.xml  # opens a specific model

Serving over http (instead of opening the .html as a file://) lets the viewer
fetch the model and its assets without the browser's local-file restrictions.
"""

import http.server
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
PORT = 8000


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--no-open"]
    open_browser = "--no-open" not in sys.argv[1:]
    model = args[0] if args else "models/hexapod.xml"
    model = "/" + model.lstrip("/")  # make it root-relative for the URL

    viewer_path = f"/tools/model_viz/mjcf_viewer.html?model={quote(model)}"

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(ROOT), **k)

        def do_GET(self):
            if self.path == "/":  # land straight on the viewer
                self.send_response(302)
                self.send_header("Location", viewer_path)
                self.end_headers()
                return
            super().do_GET()

    handler = Handler
    url = f"http://localhost:{PORT}{viewer_path}"

    with socketserver.TCPServer(("", PORT), handler) as httpd:
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
