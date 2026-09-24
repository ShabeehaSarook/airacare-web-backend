"""Flask backend for the Airacare cross-platform web app."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

from src.config import TARGET_CLASSES
from src.web_detection_service import WebDetectionConfig, WebDetectionService


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_DIR = PROJECT_ROOT / "web_client"

service = WebDetectionService(WebDetectionConfig())
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.get("/api/health")
def health():
    service.load()
    return jsonify(
        {
            "ok": True,
            "targetClasses": TARGET_CLASSES,
            "modelPath": str(service.model_path) if service.model_path else None,
            "usingPretrainedFallback": service.using_pretrained_fallback,
        }
    )


@app.route("/api/detect", methods=["POST", "OPTIONS"])
def detect():
    if request.method == "OPTIONS":
        return ("", 204)
    payload: dict[str, Any] = request.get_json(silent=True) or {}
    image = payload.get("image")
    if not image:
        return jsonify({"ok": False, "error": "Missing image"}), 400

    try:
        result = service.detect_data_url(
            image,
            {
                "latitude": payload.get("latitude"),
                "longitude": payload.get("longitude"),
                "bearingDegrees": payload.get("bearingDegrees"),
                "vehicleSpeedKmh": payload.get("vehicleSpeedKmh"),
                "deviceType": payload.get("deviceType"),
            },
        )
    except Exception as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    return jsonify(result)


@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")


@app.get("/<path:path>")
def static_files(path: str):
    target = WEB_DIR / path
    if target.exists() and target.is_file():
        return send_from_directory(WEB_DIR, path)
    return send_from_directory(WEB_DIR, "index.html")


if __name__ == "__main__":
    service.load()
    port = int(os.getenv("PORT", os.getenv("AIRACARE_WEB_PORT", "8000")))
    app.run(host="0.0.0.0", port=port, threaded=True)
