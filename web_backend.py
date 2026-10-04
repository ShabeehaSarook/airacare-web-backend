"""Flask backend for the Airacare cross-platform web app."""

from __future__ import annotations

from pathlib import Path
import base64
import importlib.util
from io import BytesIO
import logging
import os
import time
import traceback
import uuid
from urllib.parse import unquote, urlparse
from typing import Any

import cv2
from flask import Flask, jsonify, request, send_from_directory
import numpy as np

from src.android_tflite_detector import _threshold_for_label as android_tflite_threshold_for_label
from src.config import TARGET_CLASSES
from src.web_detection_service import (
    CLASS_CONFIDENCE_THRESHOLDS,
    MIN_CONSECUTIVE_FRAMES,
    WebDetectionConfig,
    WebDetectionService,
    resolve_web_model_backend,
)


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

MAX_CAPTURE_UPLOAD_BYTES = int(os.getenv("AIRACARE_CAPTURE_UPLOAD_MAX_BYTES", str(5 * 1024 * 1024)))
CLOUDINARY_FOLDER = os.getenv("AIRACARE_CLOUDINARY_FOLDER", "airacare/captured_pets")
BACKGROUND_REMOVAL_ENGINE = os.getenv("AIRACARE_BACKGROUND_REMOVAL_ENGINE", "opencv_only")
REMBG_MODEL_NAME = os.getenv("AIRACARE_REMBG_MODEL", "u2netp")
_rembg_session = None


def _cloudinary_configured() -> bool:
    if not _cloudinary_sdk_available():
        return False
    if os.getenv("CLOUDINARY_URL"):
        return True
    required = ("CLOUDINARY_CLOUD_NAME", "CLOUDINARY_API_KEY", "CLOUDINARY_API_SECRET")
    return all(os.getenv(name) for name in required)


def _cloudinary_sdk_available() -> bool:
    try:
        import cloudinary  # noqa: F401
        import cloudinary.uploader  # noqa: F401
    except ImportError:
        return False
    return True


def _configure_cloudinary():
    import cloudinary

    cloudinary_url = os.getenv("CLOUDINARY_URL")
    if cloudinary_url:
        parsed = urlparse(cloudinary_url)
        if parsed.scheme != "cloudinary" or not parsed.username or not parsed.password or not parsed.hostname:
            raise ValueError("CLOUDINARY_URL is not in the expected cloudinary://API_KEY:API_SECRET@CLOUD_NAME format")
        cloudinary.config(
            cloud_name=parsed.hostname,
            api_key=unquote(parsed.username),
            api_secret=unquote(parsed.password),
            secure=True,
        )
    else:
        cloudinary.config(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key=os.getenv("CLOUDINARY_API_KEY"),
            api_secret=os.getenv("CLOUDINARY_API_SECRET"),
            secure=True,
        )


def _capture_storage_status() -> str:
    if not _cloudinary_sdk_available():
        return "cloudinary_sdk_missing"
    if _cloudinary_configured():
        return "cloudinary_configured"
    return "cloudinary_not_configured"


def _rembg_status() -> str:
    if BACKGROUND_REMOVAL_ENGINE == "opencv_only":
        return "disabled_opencv_only"
    if importlib.util.find_spec("rembg") is None:
        return "rembg_not_installed_opencv_fallback"
    return f"rembg_available_model_{REMBG_MODEL_NAME}"


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
    loaded_classes = [
        service.model_names[class_id]
        for class_id in sorted(service.model_names)
    ] if service.model_names else TARGET_CLASSES
    return jsonify(
        {
            "status": "ok",
            "service": "airacare-web-backend",
            "build": "render-tflite-threshold-fix-2026-10-04",
            "ok": True,
            "modelReady": model_error is None,
            "modelError": model_error,
            "targetClasses": loaded_classes,
            "modelPath": str(service.model_path) if service.model_path else None,
            "modelNames": service.model_names,
            "modelBackend": resolve_web_model_backend(),
            "inferenceImageSize": service.config.imgsz,
            "usingPretrainedFallback": service.using_pretrained_fallback,
            "captureImageStorage": _capture_storage_status(),
            "captureImageUploadMaxBytes": MAX_CAPTURE_UPLOAD_BYTES,
            "backgroundRemovalEngine": BACKGROUND_REMOVAL_ENGINE,
            "backgroundRemovalAi": _rembg_status(),
            "minConfirmationFrames": MIN_CONSECUTIVE_FRAMES,
            "classConfidenceThresholds": CLASS_CONFIDENCE_THRESHOLDS,
            "tfliteParserThresholds": {
                label: android_tflite_threshold_for_label(label)
                for label in ("person", "dog", "cat", "horse", "cow", "deer", "goat", "elephant")
            },
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
                "frameId": payload.get("frameId"),
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
        image_bytes = _decode_data_url_bytes(image)
        frame = _decode_data_url(image)
        png_data_url, background_removed, method = _remove_background_best_effort(frame, image_bytes)
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


@app.route("/api/captured-pets/upload", methods=["POST", "OPTIONS"])
def upload_captured_pet():
    if request.method == "OPTIONS":
        return ("", 204)
    logger.info("captured pet upload request received url=%s", request.url)
    if not _cloudinary_configured():
        logger.error("captured pet upload failed: Cloudinary is not configured")
        return jsonify(
            {
                "ok": False,
                "error": "Image storage is not configured. Add Cloudinary environment variables to the backend.",
            }
        ), 503

    uploaded = request.files.get("image")
    label = str(request.form.get("label") or "").lower()
    owner_id = str(request.form.get("ownerId") or "anonymous")
    if label not in {"dog", "cat"}:
        return jsonify({"ok": False, "error": "Capture upload is only available for dog/cat"}), 400
    if not uploaded:
        return jsonify({"ok": False, "error": "Missing image file"}), 400
    if uploaded.content_type != "image/png":
        return jsonify({"ok": False, "error": "Captured pet image must be a PNG"}), 400

    image_bytes = uploaded.read(MAX_CAPTURE_UPLOAD_BYTES + 1)
    if not image_bytes:
        return jsonify({"ok": False, "error": "Captured pet image is empty"}), 400
    if len(image_bytes) > MAX_CAPTURE_UPLOAD_BYTES:
        return jsonify({"ok": False, "error": "Captured pet image is too large"}), 413

    try:
        _configure_cloudinary()
        import cloudinary.uploader

        decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.size == 0:
            return jsonify({"ok": False, "error": "Captured pet PNG could not be decoded"}), 400
        if decoded.ndim < 3 or decoded.shape[2] != 4:
            return jsonify({"ok": False, "error": "Captured pet PNG must preserve transparency"}), 400

        safe_owner = _safe_identifier(owner_id)
        public_id = f"{safe_owner}_{label}_{int(time.time())}_{uuid.uuid4().hex[:12]}"
        logger.info(
            "uploading captured pet to Cloudinary label=%s owner=%s bytes=%s public_id=%s",
            label,
            safe_owner,
            len(image_bytes),
            public_id,
        )
        result = cloudinary.uploader.upload(
            BytesIO(image_bytes),
            folder=CLOUDINARY_FOLDER,
            public_id=public_id,
            resource_type="image",
            format="png",
            overwrite=False,
        )
        image_url = result.get("secure_url")
        image_public_id = result.get("public_id")
        if not image_url or not image_public_id:
            raise ValueError("Cloudinary did not return an image URL")
        logger.info("captured pet upload completed public_id=%s url=%s", image_public_id, image_url)
        return jsonify(
            {
                "ok": True,
                "imageUrl": image_url,
                "imagePublicId": image_public_id,
                "storageProvider": "cloudinary",
            }
        )
    except Exception as error:
        logger.error("captured pet upload failed: %s\n%s", error, traceback.format_exc())
        return jsonify({"ok": False, "error": str(error)}), 500


@app.route("/api/captured-pets/delete", methods=["POST", "OPTIONS"])
def delete_captured_pet_image():
    if request.method == "OPTIONS":
        return ("", 204)
    if not _cloudinary_configured():
        return jsonify({"ok": False, "error": "Image storage is not configured"}), 503
    payload: dict[str, Any] = request.get_json(silent=True) or {}
    image_public_id = str(payload.get("imagePublicId") or "").strip()
    if not image_public_id:
        return jsonify({"ok": False, "error": "Missing imagePublicId"}), 400
    try:
        _configure_cloudinary()
        import cloudinary.uploader

        result = cloudinary.uploader.destroy(image_public_id, resource_type="image")
        logger.info("captured pet image delete public_id=%s result=%s", image_public_id, result)
        return jsonify({"ok": True, "result": result})
    except Exception as error:
        logger.error("captured pet image delete failed: %s\n%s", error, traceback.format_exc())
        return jsonify({"ok": False, "error": str(error)}), 500


def _decode_data_url(image_data_url: str) -> np.ndarray:
    image_bytes = _decode_data_url_bytes(image_data_url)
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise ValueError("Could not decode image")
    return frame


def _decode_data_url_bytes(image_data_url: str) -> bytes:
    if "," in image_data_url:
        image_data_url = image_data_url.split(",", 1)[1]
    return base64.b64decode(image_data_url)


def _safe_identifier(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in value)
    cleaned = cleaned.strip("_")
    return cleaned[:80] or "anonymous"


def _remove_background_best_effort(frame: np.ndarray, image_bytes: bytes) -> tuple[str, bool, str]:
    if BACKGROUND_REMOVAL_ENGINE != "opencv_only":
        try:
            return _remove_background_rembg(image_bytes)
        except Exception as error:
            logger.warning("rembg background removal failed, falling back to OpenCV: %s", error)
    return _remove_background_grabcut(frame)


def _remove_background_rembg(image_bytes: bytes) -> tuple[str, bool, str]:
    global _rembg_session
    if importlib.util.find_spec("rembg") is None:
        raise RuntimeError("rembg is not installed")

    from rembg import new_session, remove

    if _rembg_session is None:
        logger.info("loading rembg session model=%s", REMBG_MODEL_NAME)
        _rembg_session = new_session(REMBG_MODEL_NAME)

    output_bytes = remove(
        image_bytes,
        session=_rembg_session,
        force_return_bytes=True,
        alpha_matting=False,
    )
    decoded = cv2.imdecode(np.frombuffer(output_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.size == 0:
        raise ValueError("rembg returned an undecodable image")
    if decoded.ndim < 3 or decoded.shape[2] != 4:
        raise ValueError("rembg did not return an RGBA/transparent image")
    alpha = decoded[:, :, 3]
    alpha_ratio = float(np.count_nonzero(alpha)) / float(max(1, alpha.shape[0] * alpha.shape[1]))
    if alpha_ratio < 0.03 or alpha_ratio > 0.98:
        raise ValueError(f"rembg alpha mask is uncertain ratio={alpha_ratio:.3f}")
    return "data:image/png;base64," + base64.b64encode(output_bytes).decode("ascii"), True, f"rembg_{REMBG_MODEL_NAME}"


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

    rect, mask = _build_grabcut_trimap(process_width, process_height)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(process_frame, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)
    foreground_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    foreground_mask = _refine_foreground_mask(foreground_mask)

    foreground_ratio = float(np.count_nonzero(foreground_mask)) / float(max(1, process_width * process_height))
    if foreground_ratio < 0.04 or foreground_ratio > 0.96:
        return _encode_png_with_alpha(frame, np.full((height, width), 255, dtype=np.uint8)), False, "fallback_grabcut_uncertain"

    if foreground_mask.shape[:2] != (height, width):
        foreground_mask = cv2.resize(foreground_mask, (width, height), interpolation=cv2.INTER_LINEAR)
    foreground_mask = cv2.GaussianBlur(foreground_mask, (5, 5), 0)
    return _encode_png_with_alpha(frame, foreground_mask), True, "opencv_grabcut_trimap_components"


def _build_grabcut_trimap(width: int, height: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    border_x = max(2, int(width * 0.08))
    border_y = max(2, int(height * 0.08))
    rect = (
        border_x,
        border_y,
        max(1, width - border_x * 2),
        max(1, height - border_y * 2),
    )
    mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    mask[:border_y, :] = cv2.GC_BGD
    mask[-border_y:, :] = cv2.GC_BGD
    mask[:, :border_x] = cv2.GC_BGD
    mask[:, -border_x:] = cv2.GC_BGD

    core_x1 = int(width * 0.18)
    core_x2 = int(width * 0.82)
    core_y1 = int(height * 0.12)
    core_y2 = int(height * 0.88)
    mask[core_y1:core_y2, core_x1:core_x2] = cv2.GC_PR_FGD

    ellipse_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(
        ellipse_mask,
        (width // 2, height // 2),
        (max(1, int(width * 0.34)), max(1, int(height * 0.42))),
        0,
        0,
        360,
        255,
        -1,
    )
    mask[ellipse_mask == 255] = cv2.GC_PR_FGD
    inner_x1 = int(width * 0.30)
    inner_x2 = int(width * 0.70)
    inner_y1 = int(height * 0.22)
    inner_y2 = int(height * 0.78)
    mask[inner_y1:inner_y2, inner_x1:inner_x2] = cv2.GC_FGD
    return rect, mask


def _refine_foreground_mask(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    small_kernel_size = max(3, int(round(min(width, height) * 0.012)) | 1)
    large_kernel_size = max(3, int(round(min(width, height) * 0.025)) | 1)
    small_kernel = np.ones((small_kernel_size, small_kernel_size), dtype=np.uint8)
    large_kernel = np.ones((large_kernel_size, large_kernel_size), dtype=np.uint8)

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, large_kernel, iterations=1)
    cleaned = _keep_foreground_components(cleaned)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, small_kernel, iterations=1)
    return cleaned


def _keep_foreground_components(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return mask

    min_area = max(12, int(width * height * 0.006))
    center_x1 = width * 0.18
    center_x2 = width * 0.82
    center_y1 = height * 0.10
    center_y2 = height * 0.92
    keep = np.zeros(component_count, dtype=bool)
    candidates: list[tuple[int, int]] = []
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = centroids[component_id]
        touches_center = center_x1 <= cx <= center_x2 and center_y1 <= cy <= center_y2
        if touches_center:
            keep[component_id] = True
            candidates.append((area, component_id))

    if not candidates:
        largest_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep[largest_id] = True
    elif len(candidates) > 3:
        for _, component_id in sorted(candidates, reverse=True)[3:]:
            keep[component_id] = False

    return np.where(keep[labels], 255, 0).astype(np.uint8)


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
