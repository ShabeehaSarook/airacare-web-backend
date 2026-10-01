import math
import tempfile
from unittest import mock
import time
import unittest
from pathlib import Path

import numpy as np

from src.config import ANIMAL_CLASS_SET, TARGET_CLASSES
from src.detector import frame_mean_brightness, is_normal_airacare_display_class, is_risk_relevant_class, normalize_network_camera_url, resize_frame_to_max_width, sanitize_camera_source, webcam_backend_candidates
from src.distance import DistanceHistory, estimate_distance, load_calibration
from src.relative_motion import RelativeMotionAnalyzer
from src.path_analysis import PATH_IN, PathAnalyzer
from src.risk import HIGH_RISK, LOW_RISK, MEDIUM_RISK, UNKNOWN, evaluate_risk
from src.train_model import BATCH_SIZE, parse_batch
from src.tracker import TrackHistory
from src.warning import CAUTION, DANGER, WarningCandidate, WarningManager, select_primary_threat
import src.calibrate_distance as calibrate_distance
from src.validate_distance_accuracy import calculate_error
from src.android_tflite_detector import ANDROID_TFLITE_LABELS_PATH
from src.web_detection_service import (
    MIN_CONSECUTIVE_FRAMES,
    _analyze_frame_quality,
    _confidence_threshold_for_class,
    _validate_and_clamp_bbox,
)


class AiracareCoreTests(unittest.TestCase):
    def test_target_class_configuration(self):
        self.assertEqual(TARGET_CLASSES, ["person", "dog", "cat", "horse", "cow", "deer", "goat"])
        self.assertNotIn("person", ANIMAL_CLASS_SET)
        self.assertIn("goat", ANIMAL_CLASS_SET)

    def test_android_tflite_label_order_is_explicit(self):
        labels = ANDROID_TFLITE_LABELS_PATH.read_text(encoding="utf-8").splitlines()
        self.assertEqual(labels[:7], ["person", "dog", "cat", "horse", "cow", "deer", "goat"])

    def test_web_class_specific_thresholds(self):
        self.assertGreaterEqual(_confidence_threshold_for_class("cat"), 0.75)
        self.assertGreaterEqual(_confidence_threshold_for_class("dog"), 0.50)
        self.assertGreaterEqual(_confidence_threshold_for_class("goat"), 0.60)

    def test_frame_quality_rejects_covered_camera(self):
        covered = np.zeros((160, 160, 3), dtype=np.uint8)
        quality = _analyze_frame_quality(covered)
        self.assertFalse(quality["ok"])

    def test_frame_quality_accepts_textured_frame(self):
        rng = np.random.default_rng(42)
        frame = rng.integers(40, 220, size=(160, 160, 3), dtype=np.uint8)
        quality = _analyze_frame_quality(frame)
        self.assertTrue(quality["ok"])

    def test_bbox_validation_rejects_invalid_and_tiny_boxes(self):
        invalid = _validate_and_clamp_bbox((30, 20, 10, 50), 320, 240)
        self.assertFalse(invalid["ok"])
        tiny = _validate_and_clamp_bbox((10, 10, 12, 12), 320, 240)
        self.assertFalse(tiny["ok"])
        valid = _validate_and_clamp_bbox((40, 30, 180, 180), 320, 240)
        self.assertTrue(valid["ok"])

    def test_web_temporal_confirmation_default(self):
        self.assertGreaterEqual(MIN_CONSECUTIVE_FRAMES, 2)

    def test_camera_source_url_handling(self):
        self.assertEqual(sanitize_camera_source("0"), 0)
        self.assertEqual(normalize_network_camera_url("http://192.168.1.230:8080"), "http://192.168.1.230:8080/video")
        self.assertEqual(normalize_network_camera_url("http://192.168.1.230:8080/video"), "http://192.168.1.230:8080/video")
        with self.assertRaises(ValueError):
            sanitize_camera_source("[http://192.168.1.230:8080/video](http://192.168.1.230:8080/video)")
        self.assertEqual(calibrate_distance.sanitize_camera_source("http://192.168.1.230:8080"), "http://192.168.1.230:8080/video")
        with self.assertRaises(ValueError):
            calibrate_distance.sanitize_camera_source("http://PHONE_IP:8080/video")
        with self.assertRaises(ValueError):
            calibrate_distance.sanitize_camera_source("[http://192.168.1.230:8080/video](http://192.168.1.230:8080/video)")



    def test_webcam_backend_candidates_and_brightness(self):
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        white = np.full((10, 10, 3), 255, dtype=np.uint8)
        self.assertEqual(frame_mean_brightness(black), 0.0)
        self.assertGreater(frame_mean_brightness(white), 250.0)
        with mock.patch("src.detector.os.name", "nt"):
            self.assertEqual([name for name, _ in webcam_backend_candidates()], ["DSHOW", "MSMF", "DEFAULT"])


    def test_resize_frame_to_max_width(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        resized = resize_frame_to_max_width(frame, 640)
        self.assertEqual(resized.shape[:2], (360, 640))
        same = resize_frame_to_max_width(resized, 640)
        self.assertIs(same, resized)
    def test_person_visible_and_not_animal(self):
        self.assertTrue(is_normal_airacare_display_class("person"))
        self.assertTrue(is_normal_airacare_display_class("dog"))
        self.assertTrue(is_risk_relevant_class("person"))
        self.assertTrue(is_risk_relevant_class("dog"))
        self.assertNotIn("person", ANIMAL_CLASS_SET)

    def test_person_tracking_path_and_risk(self):
        tracks = TrackHistory(stale_seconds=10)
        state = tracks.update(12, 0, "person", 0.86, (280.0, 250.0, 360.0, 620.0), (320.0, 520.0))
        self.assertEqual(state.class_name, "person")
        self.assertEqual(tracks.active_count(), 1)

        path_zone = PathAnalyzer().classify((320.0, 520.0), 640, 720)
        self.assertEqual(path_zone, PATH_IN)

        risk = evaluate_risk(path_zone, 7.0, None, None, "UNKNOWN", 0.86, 30.0)
        self.assertEqual(risk.level, HIGH_RISK)


    def test_calibration_model_selection_prefers_pretrained(self):
        self.assertEqual(calibrate_distance.select_calibration_model_path("pretrained"), calibrate_distance.FALLBACK_MODEL_PATH)
        self.assertEqual(calibrate_distance.select_calibration_model_path("custom"), calibrate_distance.CUSTOM_MODEL_PATH)

    def test_calibration_auto_capture_stability(self):
        self.assertIsNone(calibrate_distance.stable_auto_capture_height([300.0, 301.0], 3, 12.0))
        self.assertIsNone(calibrate_distance.stable_auto_capture_height([300.0, 340.0, 301.0], 3, 12.0))
        self.assertEqual(calibrate_distance.stable_auto_capture_height([300.0, 304.0, 302.0], 3, 12.0), 302.0)

    def test_calibration_loading_valid_and_invalid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            valid = Path(temp_dir) / "valid.json"
            valid.write_text('{"focal_length_pixels": 900, "calibration_resolution": [1280, 720]}', encoding="utf-8")
            self.assertEqual(load_calibration(valid)["focal_length_pixels"], 900)

            invalid = Path(temp_dir) / "invalid.json"
            invalid.write_text('{"focal_length_pixels": 0}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_calibration(invalid)

    def test_distance_and_smoothing_are_per_track(self):
        self.assertAlmostEqual(estimate_distance("dog", 90, 900), 6.0)
        self.assertIsNone(estimate_distance("dog", 0, 900))
        history = DistanceHistory(history_limit=3, stale_seconds=10)
        self.assertEqual(history.update(1, 6.0), 6.0)
        self.assertEqual(history.update(1, 9.0), 7.5)
        self.assertEqual(history.update(2, 20.0), 20.0)

    def test_distance_accuracy_error_calculation(self):
        absolute_error, percentage_error = calculate_error(4.0, 5.0)
        self.assertEqual(absolute_error, 1.0)
        self.assertEqual(percentage_error, 25.0)
    def test_relative_motion_and_ttc(self):
        analyzer = RelativeMotionAnalyzer(history_limit=5, min_samples=3, closing_threshold_mps=0.5)
        self.assertEqual(analyzer.update(1, 10.0, 1.0).state, "UNKNOWN")
        self.assertEqual(analyzer.update(1, 9.0, 2.0).state, "UNKNOWN")
        closing = analyzer.update(1, 8.0, 3.0)
        self.assertEqual(closing.state, "CLOSING")
        self.assertGreater(closing.closing_speed_mps, 0.5)
        self.assertIsNotNone(closing.ttc_seconds)

        opening_analyzer = RelativeMotionAnalyzer(history_limit=5, min_samples=3, closing_threshold_mps=0.5)
        opening_analyzer.update(2, 5.0, 1.0)
        opening_analyzer.update(2, 6.0, 2.0)
        opening = opening_analyzer.update(2, 7.0, 3.0)
        self.assertEqual(opening.state, "OPENING")
        self.assertIsNone(opening.ttc_seconds)

    def test_path_and_ttc_based_risk(self):
        low = evaluate_risk("OUTSIDE_PATH", 30.0, -1.0, None, "OPENING", 0.8, 20)
        self.assertEqual(low.level, LOW_RISK)

        medium = evaluate_risk("NEAR_PATH", 8.0, 1.0, None, "CLOSING", 0.8, 20)
        self.assertIn(medium.level, {MEDIUM_RISK, HIGH_RISK})

        high_without_ttc = evaluate_risk("IN_PATH", 7.0, None, None, "UNKNOWN", 0.8, 30)
        self.assertEqual(high_without_ttc.level, HIGH_RISK)

        high_with_ttc = evaluate_risk("IN_PATH", 8.0, 3.0, 2.0, "CLOSING", 0.8, 30)
        self.assertEqual(high_with_ttc.level, HIGH_RISK)

        unknown = evaluate_risk("IN_PATH", 8.0, 3.0, 2.0, "CLOSING", 0.1, 30)
        self.assertEqual(unknown.level, UNKNOWN)

    def test_training_batch_parsing(self):
        self.assertEqual(BATCH_SIZE, 4)
        self.assertEqual(parse_batch(4), 4)
        self.assertEqual(parse_batch("8"), 8)
        with self.assertRaises(ValueError):
            parse_batch("auto")
        with self.assertRaises(ValueError):
            parse_batch(0)
    def test_warning_priority_and_cooldown(self):
        candidates = [
            WarningCandidate(2, "cat", HIGH_RISK, "IN_PATH", 8.0, 3.0),
            WarningCandidate(1, "dog", HIGH_RISK, "IN_PATH", 12.0, 2.0),
        ]
        self.assertEqual(select_primary_threat(candidates).track_id, 1)

        manager = WarningManager(warnings_enabled=True, audio_enabled=False)
        first = manager.update(candidates, timestamp=100.0)
        self.assertEqual(first.level, DANGER)
        self.assertFalse(first.audio_triggered)
        second = manager.update(candidates, timestamp=100.2)
        self.assertEqual(second.level, DANGER)
        self.assertFalse(second.audio_triggered)


if __name__ == "__main__":
    unittest.main()









