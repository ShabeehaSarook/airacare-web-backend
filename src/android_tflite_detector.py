"""LiteRT detector for the Android Airacare TFLite model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANDROID_TFLITE_MODEL_PATH = PROJECT_ROOT / "models" / "airacare_custom.tflite"
ANDROID_TFLITE_LABELS_PATH = PROJECT_ROOT / "models" / "airacare_custom_labels.txt"

PERSON_CONFIDENCE = 0.25
SMALL_ANIMAL_CONFIDENCE = 0.50
LARGE_ANIMAL_CONFIDENCE = 0.55
DEFAULT_CONFIDENCE = 0.50
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 30
MIN_BOX_SIZE_PIXELS = 2.0
MAX_WEAK_BOX_AREA_RATIO = 0.70
MAX_WEAK_BOX_DIMENSION_RATIO = 0.92
HUGE_BOX_CONFIDENCE = 0.80


@dataclass
class _Candidate:
    class_id: int
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    image_width: int
    image_height: int


class AndroidTfliteDetector:
    """Small adapter that exposes a YOLO-like predict() method."""

    def __init__(
        self,
        model_path: Path = ANDROID_TFLITE_MODEL_PATH,
        labels_path: Path = ANDROID_TFLITE_LABELS_PATH,
        num_threads: int = 1,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.labels = _read_labels(self.labels_path)
        self.names = {index: label for index, label in enumerate(self.labels)}
        self.interpreter = Interpreter(model_path=str(self.model_path), num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        shape = self.input_detail["shape"].tolist()
        if len(shape) != 4:
            raise RuntimeError(f"Unsupported Android TFLite input shape: {shape}")
        self.input_uses_nchw = shape[1] == 3 and shape[3] != 3
        self.input_height = int(shape[2] if self.input_uses_nchw else shape[1])
        self.input_width = int(shape[3] if self.input_uses_nchw else shape[2])

    def predict(
        self,
        source: Any,
        conf: float | None = None,
        imgsz: int | None = None,
        iou: float | None = None,
        device: str | None = None,
        verbose: bool = False,
    ) -> list[Any]:
        frame = source if isinstance(source, np.ndarray) else cv2.imread(str(source))
        if frame is None or frame.size == 0:
            raise ValueError("Could not load frame for Android TFLite inference")
        frame_height, frame_width = frame.shape[:2]
        input_tensor = self._preprocess(frame)
        self.interpreter.set_tensor(self.input_detail["index"], input_tensor)
        self.interpreter.invoke()
        detections = self._parse_outputs(frame_width, frame_height, float(iou or IOU_THRESHOLD))
        boxes = [_Box(candidate) for candidate in detections]
        return [SimpleNamespace(boxes=boxes)]

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        letterboxed = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        source_height, source_width = frame.shape[:2]
        gain = min(self.input_width / float(source_width), self.input_height / float(source_height))
        scaled_width = max(1, int(round(source_width * gain)))
        scaled_height = max(1, int(round(source_height * gain)))
        resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        left = int(round((self.input_width - scaled_width) / 2.0))
        top = int(round((self.input_height - scaled_height) / 2.0))
        letterboxed[top : top + scaled_height, left : left + scaled_width] = rgb
        tensor = letterboxed.astype(np.float32) / 255.0
        if self.input_uses_nchw:
            tensor = np.transpose(tensor, (2, 0, 1))
        return np.expand_dims(tensor, axis=0).astype(self.input_detail["dtype"])

    def _parse_outputs(self, source_width: int, source_height: int, iou_threshold: float) -> list[_Candidate]:
        best: list[_Candidate] = []
        for output_detail in self.output_details:
            output = self.interpreter.get_tensor(output_detail["index"])
            parsed = self._parse_output(output.reshape(output_detail["shape"]), source_width, source_height)
            if len(parsed) > len(best):
                best = parsed
        return _nms(best, iou_threshold)

    def _parse_output(self, output: np.ndarray, source_width: int, source_height: int) -> list[_Candidate]:
        squeezed = np.squeeze(output, axis=0) if output.ndim == 3 and output.shape[0] == 1 else output
        if squeezed.ndim != 2:
            return []
        if squeezed.shape[0] >= 6 and squeezed.shape[1] > squeezed.shape[0]:
            return self._parse_channels_first(squeezed, source_width, source_height)
        if squeezed.shape[1] >= 6:
            return self._parse_boxes_first(squeezed, source_width, source_height)
        return []

    def _parse_channels_first(self, data: np.ndarray, source_width: int, source_height: int) -> list[_Candidate]:
        channels, boxes = data.shape
        class_start = channels - len(self.labels)
        if class_start < 4 or class_start >= channels:
            return []
        objectness_index = 4 if class_start > 4 else None
        candidates: list[_Candidate] = []
        for box_index in range(boxes):
            cx, cy, width, height = (float(data[row, box_index]) for row in range(4))
            objectness = _normalize_score(float(data[objectness_index, box_index])) if objectness_index is not None else 1.0
            class_scores = [_normalize_score(float(data[class_start + idx, box_index])) for idx in range(len(self.labels))]
            class_id = int(np.argmax(class_scores))
            confidence = objectness * class_scores[class_id]
            if confidence >= _threshold_for_label(self.labels[class_id]):
                candidates.append(
                    self._candidate_from_center(cx, cy, width, height, class_id, confidence, source_width, source_height)
                )
        return candidates

    def _parse_boxes_first(self, data: np.ndarray, source_width: int, source_height: int) -> list[_Candidate]:
        row_len = data.shape[1]
        candidates: list[_Candidate] = []
        if row_len == 6:
            for row in data:
                confidence = _normalize_score(float(row[4]))
                class_id = int(row[5])
                if class_id in self.names and confidence >= _threshold_for_label(self.labels[class_id]):
                    candidates.append(
                        self._candidate_from_corners(
                            float(row[0]), float(row[1]), float(row[2]), float(row[3]), class_id, confidence, source_width, source_height
                        )
                    )
            return candidates
        class_start = row_len - len(self.labels)
        if class_start < 4 or class_start >= row_len:
            return []
        objectness_index = 4 if class_start > 4 else None
        for row in data:
            objectness = _normalize_score(float(row[objectness_index])) if objectness_index is not None else 1.0
            class_scores = [_normalize_score(float(row[class_start + idx])) for idx in range(len(self.labels))]
            class_id = int(np.argmax(class_scores))
            confidence = objectness * class_scores[class_id]
            if confidence >= _threshold_for_label(self.labels[class_id]):
                candidates.append(
                    self._candidate_from_center(
                        float(row[0]), float(row[1]), float(row[2]), float(row[3]), class_id, confidence, source_width, source_height
                    )
                )
        return candidates

    def _candidate_from_center(
        self,
        cx: float,
        cy: float,
        width: float,
        height: float,
        class_id: int,
        confidence: float,
        source_width: int,
        source_height: int,
    ) -> _Candidate:
        normalized = max(cx, cy, width, height) <= 1.5
        scale_x = float(self.input_width) if normalized else 1.0
        scale_y = float(self.input_height) if normalized else 1.0
        scaled_cx = cx * scale_x
        scaled_cy = cy * scale_y
        scaled_w = width * scale_x
        scaled_h = height * scale_y
        gain, pad_x, pad_y = self._letterbox_geometry(source_width, source_height)
        left = ((scaled_cx - scaled_w / 2.0) - pad_x) / gain
        top = ((scaled_cy - scaled_h / 2.0) - pad_y) / gain
        right = ((scaled_cx + scaled_w / 2.0) - pad_x) / gain
        bottom = ((scaled_cy + scaled_h / 2.0) - pad_y) / gain
        return self._candidate_from_box(left, top, right, bottom, class_id, confidence, source_width, source_height)

    def _candidate_from_corners(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        class_id: int,
        confidence: float,
        source_width: int,
        source_height: int,
    ) -> _Candidate:
        normalized = max(left, top, right, bottom) <= 1.5
        scale_x = float(self.input_width) if normalized else 1.0
        scale_y = float(self.input_height) if normalized else 1.0
        gain, pad_x, pad_y = self._letterbox_geometry(source_width, source_height)
        return self._candidate_from_box(
            (left * scale_x - pad_x) / gain,
            (top * scale_y - pad_y) / gain,
            (right * scale_x - pad_x) / gain,
            (bottom * scale_y - pad_y) / gain,
            class_id,
            confidence,
            source_width,
            source_height,
        )

    def _candidate_from_box(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        class_id: int,
        confidence: float,
        source_width: int,
        source_height: int,
    ) -> _Candidate:
        x1 = float(np.clip(min(left, right), 0, source_width))
        y1 = float(np.clip(min(top, bottom), 0, source_height))
        x2 = float(np.clip(max(left, right), 0, source_width))
        y2 = float(np.clip(max(top, bottom), 0, source_height))
        return _Candidate(class_id, self.labels[class_id], float(confidence), (x1, y1, x2, y2), source_width, source_height)

    def _letterbox_geometry(self, source_width: int, source_height: int) -> tuple[float, float, float]:
        gain = min(self.input_width / float(source_width), self.input_height / float(source_height))
        pad_x = (self.input_width - source_width * gain) / 2.0
        pad_y = (self.input_height - source_height * gain) / 2.0
        return gain, pad_x, pad_y


class _Box:
    def __init__(self, candidate: _Candidate) -> None:
        self.cls = np.array([candidate.class_id], dtype=np.float32)
        self.conf = np.array([candidate.confidence], dtype=np.float32)
        self.xyxy = np.array([candidate.xyxy], dtype=np.float32)


def _read_labels(path: Path) -> list[str]:
    return [line.strip().lstrip("\ufeff") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize_score(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if 0.0 <= value <= 1.0:
        return float(value)
    return float(1.0 / (1.0 + np.exp(-value)))


def _threshold_for_label(label: str) -> float:
    label = label.lower()
    if label == "person":
        return PERSON_CONFIDENCE
    if label in {"dog", "cat"}:
        return SMALL_ANIMAL_CONFIDENCE
    if label in {"horse", "cow", "deer", "goat", "elephant"}:
        return LARGE_ANIMAL_CONFIDENCE
    return DEFAULT_CONFIDENCE


def _nms(candidates: list[_Candidate], iou_threshold: float) -> list[_Candidate]:
    kept: list[_Candidate] = []
    for candidate in sorted((item for item in candidates if _usable_box(item)), key=lambda item: item.confidence, reverse=True):
        if len(kept) >= MAX_DETECTIONS:
            break
        if any(existing.class_id == candidate.class_id and _iou(existing.xyxy, candidate.xyxy) > iou_threshold for existing in kept):
            continue
        kept.append(candidate)
    return kept


def _usable_box(candidate: _Candidate) -> bool:
    left, top, right, bottom = candidate.xyxy
    width = max(0.0, right - left)
    height = max(0.0, bottom - top)
    if width <= MIN_BOX_SIZE_PIXELS or height <= MIN_BOX_SIZE_PIXELS:
        return False
    frame_area = max(1.0, float(candidate.image_width * candidate.image_height))
    area_ratio = (width * height) / frame_area
    width_ratio = width / max(1.0, float(candidate.image_width))
    height_ratio = height / max(1.0, float(candidate.image_height))
    if area_ratio > MAX_WEAK_BOX_AREA_RATIO and candidate.confidence < HUGE_BOX_CONFIDENCE:
        return False
    if (width_ratio > MAX_WEAK_BOX_DIMENSION_RATIO or height_ratio > MAX_WEAK_BOX_DIMENSION_RATIO) and candidate.confidence < HUGE_BOX_CONFIDENCE:
        return False
    return True


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1.0, area_a + area_b - intersection)
