"""Serve replay.html with auto-load of the most recent log files.

Opens browser to http://localhost:8090/replay.html?autoload=latest
The server injects a small script that auto-loads the newest JSONL + arena files.
"""
import http.server
import json
import os
import re
import threading
import webbrowser

PORT = 8090
BASE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(BASE, "logs")


def find_latest_replay():
    """Find the most recent JSONL file and its companion arena files."""
    jsonl_files = []
    for f in os.listdir(LOGS):
        if f.endswith(".jsonl"):
            path = os.path.join(LOGS, f)
            jsonl_files.append((os.path.getmtime(path), f))

    if not jsonl_files:
        return None

    jsonl_files.sort(reverse=True)
    latest = jsonl_files[0][1]
    base = latest.replace(".jsonl", "")

    result = {"jsonl": f"logs/{latest}"}
    arena_json = f"{base}_arena.json"
    arena_png = f"{base}_arena.png"
    if os.path.exists(os.path.join(LOGS, arena_json)):
        result["arena_json"] = f"logs/{arena_json}"
    if os.path.exists(os.path.join(LOGS, arena_png)):
        result["arena_png"] = f"logs/{arena_png}"

    return result


class ReplayHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE, **kwargs)

    def do_GET(self):
        if self.path == "/api/latest":
            # Return paths to the latest replay files
            latest = find_latest_replay()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(latest or {}).encode())
            return

        # For replay.html with ?autoload, inject auto-load script
        if self.path.startswith("/replay.html") and "autoload" in self.path:
            replay_path = os.path.join(BASE, "replay.html")
            with open(replay_path, "r", encoding="utf-8") as f:
                html = f.read()

            # Inject auto-load script before </body>
            autoload_script = """
<script>
(async function autoLoad() {
  try {
    const resp = await fetch('/api/latest');
    const info = await resp.json();
    if (!info.jsonl) { console.log('No replay files found'); return; }

    const files = [];

    // Load JSONL
    const jsonlResp = await fetch('/' + info.jsonl);
    const jsonlBlob = await jsonlResp.blob();
    files.push(new File([jsonlBlob], info.jsonl.split('/').pop(), {type: 'application/octet-stream'}));

    // Load arena JSON
    if (info.arena_json) {
      const ajResp = await fetch('/' + info.arena_json);
      const ajBlob = await ajResp.blob();
      files.push(new File([ajBlob], info.arena_json.split('/').pop(), {type: 'application/json'}));
    }

    // Load arena PNG
    if (info.arena_png) {
      const apResp = await fetch('/' + info.arena_png);
      const apBlob = await apResp.blob();
      files.push(new File([apBlob], info.arena_png.split('/').pop(), {type: 'image/png'}));
    }

    console.log('Auto-loading', files.length, 'files:', files.map(f => f.name));
    loadFiles(files);
  } catch (e) {
    console.error('Auto-load failed:', e);
  }
})();
</script>
"""
            html = html.replace("</body>", autoload_script + "</body>")

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html.encode())))
            self.end_headers()
            self.wfile.write(html.encode())
            return

        return super().do_GET()

    def log_message(self, format, *args):
        # Quiet logging — only show errors
        if args and "404" in str(args[0]):
            super().log_message(format, *args)


if __name__ == "__main__":
    latest = find_latest_replay()
    if latest:
        print(f"Latest replay: {latest['jsonl']}")
    else:
        print("No replay files found in logs/")

    server = http.server.HTTPServer(("", PORT), ReplayHandler)
    url = f"http://localhost:{PORT}/replay.html?autoload=latest"
    print(f"\nServing at {url}")
    print("Press Ctrl+C to stop\n")

    # Open browser after a short delay
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()
