#!/usr/bin/env python3
# webapp_static.py — super basic localhost viewer for last_scan.json

import os, json
from flask import Flask, render_template, send_from_directory

JSON_PATH = os.getenv("SCAN_JSON_PATH", "data/last_scan.json")

app = Flask(__name__)

@app.route("/")
def index():
    if not os.path.exists(JSON_PATH):
        return f"""
        <h2>No data yet</h2>
        <p>Run the scanner once:</p>
        <pre>export IB_PORT=4001  # or your TWS/Gateway port
python scan_to_json.py</pre>
        """, 200
    with open(JSON_PATH) as f:
        data = json.load(f)
    return render_template("index.html", data=data)

@app.route("/data/last_scan.json")
def raw_json():
    d = os.path.dirname(JSON_PATH) or "."
    return send_from_directory(d, os.path.basename(JSON_PATH))

@app.route("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    app.run("127.0.0.1", 5173, debug=True)