"""Flask backend for the Airacare cross-platform web app."""

from __future__ import annotations

from pathlib import Path
import logging
import os
import time
import traceback
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

from src.config import TARGET_CLASSES
from src.web_detection_service import WebDetectionConfig, WebDetectionService


PROJECT_ROOT = Path(__file__).resolve().parent
WEB_DIR = PROJECT_ROOT / "web_client"

service = WebDetectionService(WebDetectionConfig())
app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("airacare-web-backend")

ALLOWED_ORIGINS = {
    "https://airacare-animal-safety.web.app",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
}


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS or (origin and origin.startswith("http://localhost:")):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    elif not origin:
        response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.get("/api/health")
def health():
    logger.info("health request received url=%s", request.url)
    model_error = None
    try:
        service.load()
    except Exception as error:
        model_error = str(error)
        logger.error("health model load failed: %s\n%s", error, traceback.format_exc())
    return jsonify(
        {
            "status": "ok",
            "service": "airacare-web-backend",
            "build": "render-free-stability-192",
            "ok": True,
            "modelReady": model_error is None,
            "modelError": model_error,
            "targetClasses": TARGET_CLASSES,
            "modelPath": str(service.model_path) if service.model_path else None,
            "inferenceImageSize": service.config.imgsz,
            "usingPretrainedFallback": service.using_pretrained_fallback,
        }
    )


@app.route("/api/detect", methods=["POST", "OPTIONS"])
def detect():
    if request.method == "OPTIONS":
        return ("", 204)
    logger.info("detect request received url=%s", request.url)
    payload: dict[str, Any] = request.get_json(silent=True) or {}
    image = payload.get("image")
    if not image:
        logger.warning("detect request missing image field")
        return jsonify({"ok": False, "error": "Missing image"}), 400

    try:
        logger.info("image received bytes_or_chars=%s", len(image) if isinstance(image, str) else "not-string")
        logger.info("inference started")
        started_at = time.perf_counter()
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
        processing_time_ms = int((time.perf_counter() - started_at) * 1000)
        result["processingTimeMs"] = processing_time_ms
        logger.info(
            "inference completed detections=%s processing_ms=%s",
            len(result.get("detections", [])),
            processing_time_ms,
        )
    except Exception as error:
        logger.error("detect failed: %s\n%s", error, traceback.format_exc())
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
