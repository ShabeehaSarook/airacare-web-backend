"""Flask backend for the Airacare cross-platform web app."""

from __future__ import annotations

from pathlib import Path
import base64
import logging
import os
import time
import traceback
from typing import Any

import cv2
from flask import Flask, jsonify, request, send_from_directory
import numpy as np

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


@app.route("/api/remove-background", methods=["POST", "OPTIONS"])
def remove_background():
    if request.method == "OPTIONS":
        return ("", 204)
    logger.info("background removal request received url=%s", request.url)
    payload: dict[str, Any] = request.get_json(silent=True) or {}
    image = payload.get("image")
    label = str(payload.get("label") or "").lower()
    if label not in {"dog", "cat"}:
        return jsonify({"ok": False, "error": "Background removal is only available for dog/cat"}), 400
    if not image:
        return jsonify({"ok": False, "error": "Missing image"}), 400

    try:
        started_at = time.perf_counter()
        frame = _decode_data_url(image)
        png_data_url, background_removed, method = _remove_background_grabcut(frame)
        processing_time_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "background removal completed label=%s method=%s removed=%s processing_ms=%s",
            label,
            method,
            background_removed,
            processing_time_ms,
        )
        return jsonify(
            {
                "ok": True,
                "image": png_data_url,
                "backgroundRemoved": background_removed,
                "method": method,
                "processingTimeMs": processing_time_ms,
            }
        )
    except Exception as error:
        logger.error("background removal failed: %s\n%s", error, traceback.format_exc())
        return jsonify({"ok": False, "error": str(error)}), 400


def _decode_data_url(image_data_url: str) -> np.ndarray:
    if "," in image_data_url:
        image_data_url = image_data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data_url)
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("Could not decode image")
    return frame


def _remove_background_grabcut(frame: np.ndarray) -> tuple[str, bool, str]:
    height, width = frame.shape[:2]
    if width < 8 or height < 8:
        return _encode_png_with_alpha(frame, np.full((height, width), 255, dtype=np.uint8)), False, "fallback_too_small"

    process_frame = frame
    process_height, process_width = height, width
    max_segmentation_side = int(os.getenv("AIRACARE_BG_REMOVE_MAX_SIDE", "512"))
    scale = min(1.0, max_segmentation_side / float(max(width, height)))
    if scale < 1.0:
        process_width = max(8, int(round(width * scale)))
        process_height = max(8, int(round(height * scale)))
        process_frame = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_AREA)

    border_x = max(2, int(process_width * 0.06))
    border_y = max(2, int(process_height * 0.06))
    rect = (
        border_x,
        border_y,
        max(1, process_width - border_x * 2),
        max(1, process_height - border_y * 2),
    )
    mask = np.zeros((process_height, process_width), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(process_frame, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
    foreground_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    foreground_ratio = float(np.count_nonzero(foreground_mask)) / float(max(1, process_width * process_height))
    if foreground_ratio < 0.04 or foreground_ratio > 0.96:
        return _encode_png_with_alpha(frame, np.full((height, width), 255, dtype=np.uint8)), False, "fallback_grabcut_uncertain"

    kernel_size = max(3, int(round(min(process_width, process_height) * 0.025)) | 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    foreground_mask = cv2.GaussianBlur(foreground_mask, (5, 5), 0)
    if foreground_mask.shape[:2] != (height, width):
        foreground_mask = cv2.resize(foreground_mask, (width, height), interpolation=cv2.INTER_LINEAR)
    return _encode_png_with_alpha(frame, foreground_mask), True, "opencv_grabcut"


def _encode_png_with_alpha(frame: np.ndarray, alpha: np.ndarray) -> str:
    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    ok, encoded = cv2.imencode(".png", rgba)
    if not ok:
        raise ValueError("Could not encode transparent PNG")
    return "data:image/png;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


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
