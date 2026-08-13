# Airacare Animal Detector

## Project Goal

Airacare Animal Detector will eventually use a live camera to detect these classes in real time:

- Person
- Dog
- Cat
- Horse
- Cow
- Deer
- Goat

The final system should display a bounding box, object class, and confidence percentage for each detected object.

This repository currently implements Phase 15: optional sensor integration and sensor fusion, while preserving camera-only operation.

## Phase 1 Objective

Phase 1 verified:

- Python environment
- OpenCV installation
- Ultralytics YOLO installation
- Webcam access
- Real-time camera display

## Phase 2 Objective

Phase 2 runs pretrained Ultralytics YOLO object detection on live webcam video.

The application:

- Opens the default webcam
- Runs a pretrained YOLO detection model on each frame
- Draws bounding boxes around detected objects
- Displays class names and confidence percentages
- Separates final target classes from other pretrained detections
- Shows approximate FPS
- Exits cleanly when `Q` is pressed

Phase 2 only tests pretrained model capabilities. It does not train a custom model and does not implement tracking, distance estimation, collision-risk calculation, GPS, external sensors, or mobile app features.

## Phase 3 Objective

Phase 3 prepares the custom 7-class dataset that will later be annotated and used for model fine-tuning.

Required classes:

- person
- dog
- cat
- horse
- cow
- deer
- goat

Dataset diversity is important. Each class should include close, medium, and far objects; front, side, and rear views; partial occlusion; road-side scenes; animals crossing roads; vegetation and urban backgrounds; daylight, evening, nighttime, light rain, and wet road conditions.

Phase 3 does not train the AI model and does not create YOLO annotation files.

Run the dataset inspector:

```bash
python -m src.dataset_inspector
```

The inspector writes a report to:

```text
dataset/reports/dataset_report.txt
```

## Phase 4 Objective

Phase 4 prepares the Airacare dataset for YOLO object-detection training. Bounding boxes are created manually with a YOLO-compatible annotation tool, then the dataset is split and validated.

Phase 4 does not train the YOLO model and does not implement tracking, distance estimation, collision-risk calculation, sensors, or mobile app features.

### YOLO Label Format

Each image must have a matching `.txt` label file with the same filename stem:

```text
image_name.jpg
image_name.txt
```

Each annotation line must use normalized YOLO object-detection format:

```text
class_id center_x center_y width height
```

All coordinate values must be between 0 and 1. Width and height must be greater than 0.

Class IDs:

```text
0 person
1 dog
2 cat
3 horse
4 cow
5 deer
6 goat
```

Every visible target object needs its own bounding box. For example, an image with two dogs, one person, and one cow must have four annotation lines.

### Annotation Tools

Use a manual annotation tool that can export YOLO labels, such as:

- CVAT
- Label Studio
- Roboflow
- LabelImg
- another YOLO-compatible annotation tool

Place exported annotated source files here before splitting:

```text
dataset/annotated/images/
dataset/annotated/labels/
```

### Train/Val/Test Split

The prepared YOLO dataset structure is:

```text
dataset/
+-- images/
|   +-- train/
|   +-- val/
|   +-- test/
+-- labels/
|   +-- train/
|   +-- val/
|   +-- test/
+-- reports/
+-- data.yaml
+-- README.md
```

Default split:

- 70% train
- 20% validation
- 10% test

Preview the split without writing files:

```bash
python -m src.split_dataset --dry-run
```

Split annotated image/label pairs:

```bash
python -m src.split_dataset --source-images dataset/annotated/images --source-labels dataset/annotated/labels --output dataset
```

The split utility copies by default, keeps each image with its matching label, skips images with missing labels, uses a reproducible random seed, and refuses to overwrite output files unless `--allow-overwrite` is provided.

### Annotation Validation

Validate the YOLO dataset:

```bash
python -m src.validate_annotations
```

The validator reports missing label files, labels with missing images, invalid class IDs, coordinates outside 0 to 1, invalid box sizes, malformed lines, empty files, duplicate label lines, total image and annotation counts, and bounding-box counts per class. It only reports problems and does not modify labels.

### Visual Annotation Preview

Preview a few annotated images:

```bash
python -m src.preview_annotations --split train --count 10 --shuffle
```

The preview utility converts normalized YOLO boxes back to pixel coordinates, draws boxes, and displays class names for manual quality checks.


## Phase 5 - Custom Model Training

Phase 5 fine-tunes a pretrained Ultralytics YOLO detection model for exactly these Airacare target classes:

- Person
- Dog
- Cat
- Horse
- Cow
- Deer
- Goat

The default base model is `yolo11n.pt`, configured in `src/train_model.py` as `BASE_MODEL`. It is a small pretrained YOLO detection model suitable for an initial real-time-oriented fine-tuning run.

Phase 5 uses the annotated YOLO dataset from Phase 4:

```text
dataset/data.yaml
dataset/images/train/
dataset/images/val/
dataset/labels/train/
dataset/labels/val/
```

Run preflight checks without training:

```bash
python -m src.train_model --check-only
```

Start training:

```bash
python -m src.train_model
```

Useful options:

```bash
python -m src.train_model --base-model yolo11n.pt --epochs 100 --imgsz 640 --batch auto --device auto
```

Resume an interrupted run only after confirming the previous checkpoint exists:

```bash
python -m src.train_model --resume
```

Expected training outputs:

- `runs/airacare/phase5_training/weights/best.pt`: checkpoint with the best validation performance.
- `runs/airacare/phase5_training/weights/last.pt`: checkpoint from the final training epoch.
- Training plots and logs under `runs/airacare/phase5_training/`.
- Validation metrics including Precision, Recall, mAP50, and mAP50-95 where Ultralytics reports them.
- `models/airacare_animal_detector_best.pt`: copied best model for later integration.
- `results/phase5_training_report.txt`: Phase 5 summary report.

The training script validates `dataset/data.yaml`, checks class names, verifies train/validation images and labels, runs the Phase 4 annotation validator, and stops before training if serious dataset errors are found.

Do not replace the Phase 2 live-camera model with this custom model yet. That integration belongs to a later phase after model evaluation.

## Phase 6 - Model Evaluation

Phase 6 evaluates the trained Airacare YOLO model before any live-camera integration. The goal is to decide whether the custom model is reliable enough for Phase 7 real-time use.

Evaluate this trained model by default:

```text
models/airacare_animal_detector_best.pt
```

Run preflight checks without evaluation:

```bash
python -m src.evaluate_model --check-only
```

Run evaluation:

```bash
python -m src.evaluate_model
```

Review saved prediction images:

```bash
python -m src.review_predictions
```

Important metrics:

- Precision: how often model detections are correct. Low precision usually means too many false positives.
- Recall: how many real objects the model finds. Low recall usually means too many false negatives.
- mAP50: mean average precision using a 0.50 IoU match threshold.
- mAP50-95: stricter mean average precision averaged over IoU thresholds from 0.50 to 0.95.

Phase 6 outputs:

- `results/phase6_evaluation/`: Ultralytics evaluation files such as confusion matrix and metric plots where supported.
- `results/phase6_predictions/`: test images copied as prediction previews with bounding boxes, class names, and confidence scores drawn on them.
- `results/phase6_evaluation_report.txt`: summary with overall metrics, per-class metrics, inference speed, confidence-threshold counts, false-positive/false-negative review, image-size-based near/medium/far proxy analysis, and readiness recommendation.

The confusion matrix helps identify class mistakes such as dog predicted as cat, goat predicted as deer, cow predicted as horse, missed animals, and background false detections.

False positive means the model detected an object that was not actually present. False negative means the model failed to detect a real annotated object.

Inference speed is measured on the actual device used by the script. CPU results are CPU-only results; they should not be treated as GPU performance.

Lighting-condition analysis requires metadata or manual grouping for daytime, evening, nighttime, rainy, and wet-road images. The evaluator documents this requirement instead of inventing labels.

Readiness is not automatically assumed. The report says `READY FOR PHASE 7` only when actual metrics meet the current readiness checks; otherwise it says `MODEL NEEDS IMPROVEMENT`.

Do not integrate the custom model into the live-camera application yet. That belongs to Phase 7.

## Phase 7 - Final Real-Time Detection

Phase 7 uses the custom-trained Airacare YOLO model with a live camera. It does not use the original pretrained YOLO model for detection.

Default custom model path:

```text
models/airacare_animal_detector_best.pt
```

Supported classes:

- Person
- Dog
- Cat
- Horse
- Cow
- Deer
- Goat

Run the live detector:

```bash
python main.py
```

Windows virtual environment example:

```powershell
venv\Scripts\python.exe main.py
```

Expected output:

- Live camera feed
- Bounding boxes around detected target objects
- Class labels such as `PERSON 97%`, `DOG 91%`, `COW 89%`
- Confidence scores
- Compact current detection counts
- Approximate FPS
- Approximate inference time per frame



### Low-Latency Phone Camera Mode

Phone IP camera streams can lag if old MJPEG frames build up while YOLO is processing. The detector now uses a background latest-frame capture thread for phone streams. It keeps only the newest frame and drops stale frames, so YOLO works on the current camera view instead of processing an old queue.

Recommended IP Webcam settings on the phone:

- Start with `1280x720`.
- If it is still delayed, use `640x480`.
- Do not use 4K for real-time detection; it increases Wi-Fi bandwidth, decoding time, CPU usage, YOLO time, and latency.

Phone camera with YOLO, optimized for CPU responsiveness:

```powershell
python main.py --camera-source "http://10.195.215.13:8080/video" --confidence 0.50 --imgsz 416
```

If detection is still slow, try a smaller inference size:

```powershell
python main.py --camera-source "http://10.195.215.13:8080/video" --confidence 0.50 --imgsz 320
```

Phone stream only, no YOLO. Use this to test if lag comes from the phone/Wi-Fi stream:

```powershell
python main.py --camera-source "http://10.195.215.13:8080/video" --no-detection
```

If `--no-detection` is already delayed, fix the phone stream first by lowering phone camera resolution or improving Wi-Fi. If `--no-detection` is smooth but detection mode is slow, YOLO inference is the bottleneck.
### Phone Camera

You can use a phone as the camera source if the phone exposes a video stream URL.

Recommended simple setup:

1. Connect the phone and laptop to the same Wi-Fi network.
2. Install/open an IP camera app on the phone, such as an app that provides an HTTP video stream.
3. Start the phone camera server.
4. Copy the stream URL shown by the app. Common examples look like:

```text
http://192.168.1.10:8080/video
rtsp://192.168.1.10:8554/live
```

Run with the phone stream:

```powershell
venv\Scripts\python.exe main.py --camera-source "http://192.168.1.10:8080/video"
```

Default laptop webcam:

```powershell
venv\Scripts\python.exe main.py --camera-source 0
```

You can also set an environment variable:

```powershell
$env:AIRACARE_CAMERA_SOURCE="http://192.168.1.10:8080/video"
venv\Scripts\python.exe main.py
```

If the phone stream does not open, confirm the phone and laptop are on the same network, the phone camera app is running, and the URL opens in a browser or VLC first.
Keyboard controls:

- `Q`: quit and close the camera window cleanly.
- `S`: save the current annotated frame to `results/phase7_live_captures/`.

Default Phase 7 settings are in `src/detector.py`:

```python
CUSTOM_MODEL_PATH = PROJECT_ROOT / "models" / "airacare_animal_detector_best.pt"
CONFIDENCE_THRESHOLD = 0.50
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
INFERENCE_IMAGE_SIZE = 640
```

If inference is too slow, reduce `INFERENCE_IMAGE_SIZE` deliberately, for example from `640` to `512` or `416`. The application does not silently reduce image size.

Before opening the camera, the detector checks that the custom model exists, loads with Ultralytics YOLO, and validates that it exposes exactly the 7 Airacare classes.
Phase 7 intentionally does not implement tracking, distance estimation, collision-risk calculation, driver warnings, GPS, radar, thermal sensors, or mobile application integration.

## Phase 08 - Object Tracking + Movement Direction

Phase 08 adds persistent object IDs and simple image-plane movement labels on top of live YOLO detection.

For each visible target object, the display shows:

```text
PERSON ID:1 94% RIGHT
DOG ID:4 87% STATIONARY
```

Tracking uses Ultralytics tracking with `bytetrack.yaml` by default. The same physical object should keep approximately the same ID while it remains visible.

Supported movement labels are:

```text
STATIONARY
LEFT
RIGHT
UP
DOWN
UP-LEFT
UP-RIGHT
DOWN-LEFT
DOWN-RIGHT
```

Movement is calculated only in the image plane by comparing recent bounding-box centers. It does not estimate real-world distance, speed, collision risk, or whether an object is moving toward a vehicle.

Run laptop webcam tracking:

```powershell
python main.py --camera-source 0 --tracking --tracker bytetrack.yaml
```

Run phone camera tracking:

```powershell
python main.py --camera-source "http://10.195.215.13:8080/video" --confidence 0.25 --imgsz 640 --iou 0.45 --tracking --tracker bytetrack.yaml
```

Run without tracking for comparison:

```powershell
python main.py --camera-source 0 --no-tracking
```

Debug every tracked box:

```powershell
python main.py --camera-source 0 --debug-detections
```

Phone stream only, no YOLO/tracking:

```powershell
python main.py --camera-source "http://10.195.215.13:8080/video" --no-detection
```

Press `S` to save the current tracking frame to:

```text
results/phase8_tracking_captures/
```

Press `Q` to quit cleanly.
## Dataset Target

Initial target:

- 500+ usable images per class
- 3500+ usable images total

See `dataset/README.md` for full collection and annotation guidelines.

## Target Classes

The final required classes are:

- Person
- Dog
- Cat
- Horse
- Cow
- Deer
- Goat

The Phase 2 pretrained model is trained on COCO-style object classes. It can directly detect several target classes, but custom training will be required later for target classes that are unsupported or not accurate enough.

With the configured pretrained model, these target classes are expected to be supported:

- Person
- Dog
- Cat
- Horse
- Cow

These target classes are not expected to be included in the pretrained model class list and will need custom dataset work in later phases:

- Deer
- Goat

## Requirements

- Python 3.10 or 3.11
- Webcam
- pip

## Installation

Create a virtual environment.

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Phase 2 Detection

```bash
python main.py
```

A webcam window titled `Airacare Animal Detector - Phase 2` should open and display live video with detected objects, bounding boxes, class names, and confidence percentages.

The first run may download the configured pretrained YOLO model weights into the `models/` folder.

## Stopping

Press `Q` to close the camera window.

## Configuration

Shared target classes are configured in `src/config.py`.

The main Phase 2 detection settings are in `src/detector.py`:

```python
MODEL_NAME = "models/yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.50
CAMERA_INDEX = 0
```

## Future Phases

Phase 7: Final real-time detection for:

- Person
- Dog
- Cat
- Horse
- Cow
- Deer
- Goat












## Phase 09 - Monocular Distance Estimation

Phase 09 adds approximate distance labels to the existing Phase 08 real-time detection and tracking system.

The displayed distance is an estimate based on camera calibration and object size assumptions. It is not exact physical distance, not radar-level accuracy, and not a collision warning.

Supported target classes remain:

- Person
- Dog
- Cat
- Horse
- Cow
- Deer
- Goat

Distance formula:

```text
distance_meters = (known_real_height_meters * focal_length_pixels) / bounding_box_height_pixels
```

Approximate class-height assumptions are stored in `src/config.py`:

- person: 1.70 m
- dog: 0.60 m
- cat: 0.30 m
- horse: 1.60 m
- cow: 1.50 m
- deer: 1.20 m
- goat: 0.75 m

These values are representative only. A small dog and a large dog do not have the same height, so animal distance estimates can have significant error.

### Calibrate A Camera

Calibrate each camera separately. Laptop webcam and phone camera can have different focal length, field of view, resolution, and zoom.

Example for a phone camera:

```powershell
python -m src.calibrate_distance --known-height 1.70 --known-distance 3.0 --pixel-height 520 --camera-source phone --resolution 1280 720 --output config/phone_camera_calibration.json
```

Example for a laptop webcam:

```powershell
python -m src.calibrate_distance --known-height 1.70 --known-distance 3.0 --pixel-height 480 --camera-source laptop --resolution 1280 720 --output config/laptop_camera_calibration.json
```

Calibration procedure:

1. Place a known object, such as a person, exactly 3 metres from the camera.
2. Keep the camera stationary.
3. Make sure the full object is visible.
4. Run detection and note the bounding-box height in pixels, or use a saved frame to measure it.
5. Run `python -m src.calibrate_distance` with known height, known distance, and pixel height.
6. Do not change phone camera zoom after calibration.

The calibration script writes a report template to `results/phase9_distance_report.txt`. Fill in real test measurements at known distances such as 2 m, 3 m, 5 m, and 8 m. Do not invent accuracy numbers.

### Run With Estimated Distance

Laptop webcam:

```powershell
python main.py --camera-source 0 --confidence 0.25 --imgsz 640 --iou 0.45 --calibration config/laptop_camera_calibration.json
```

Android IP Webcam:

```powershell
python main.py --camera-source "http://PHONE_IP:8080/video" --confidence 0.25 --imgsz 640 --iou 0.45 --calibration config/phone_camera_calibration.json
```

Disable distance if you only want detection and tracking:

```powershell
python main.py --camera-source 0 --no-distance
```

Print distance calculation details:

```powershell
python main.py --camera-source 0 --debug-distance
```

Expected label format:

```text
PERSON ID:1 95% RIGHT ~8.4m
DOG ID:3 89% LEFT ~14.7m
```

If calibration is missing or a box is too small, detection and tracking continue and the label shows `DIST:N/A`.

### Resolution And Zoom Notes

Calibration is resolution dependent. If runtime resolution differs from calibration resolution, the app scales the focal length using the frame height and prints a warning. If the aspect ratio changes substantially, distance accuracy may be reduced.

For phone camera testing, start with 1280x720 or 640x480 in the IP Webcam app. Do not use 4K for real-time detection because it increases Wi-Fi bandwidth, decoding time, CPU usage, and latency.

### Future Distance Improvements

Future phases may improve distance accuracy using stereo cameras, monocular depth-estimation neural networks, LiDAR, radar, calibrated road-plane geometry, or sensor fusion. These are not implemented in Phase 09.

## Phase 11 - Vehicle Speed + Relative Closing Analysis

Phase 11 adds manual vehicle speed metadata, per-track distance-over-time history, estimated relative closing speed, and experimental estimated Time-to-Collision.

This phase does not create risk levels, driver alerts, braking commands, GPS integration, or production automotive safety decisions.

Vehicle speed and relative closing speed are different:

- Vehicle speed: how fast the vehicle is moving, supplied manually for testing in Phase 11.
- Relative closing speed: how quickly the measured camera distance to a tracked object is reducing.

Manual speed conversion:

```text
speed_mps = speed_kmh / 3.6
```

Relative closing speed:

```text
closing_speed = (previous_distance - current_distance) / dt
```

Estimated TTC:

```text
EST.TTC = current_distance / closing_speed
```

TTC is shown only when distance is valid, enough track history exists, and closing speed is positive. Otherwise the app shows `TTC:N/A`.

Run laptop webcam with manual speed:

```powershell
python main.py --camera-source 0 --vehicle-speed-kmh 30 --calibration config/laptop_camera_calibration.json
```

Run Android IP Webcam with manual speed:

```powershell
python main.py --camera-source "http://PHONE_IP:8080/video" --vehicle-speed-kmh 30 --calibration config/phone_camera_calibration.json
```

Debug motion calculations:

```powershell
python main.py --camera-source 0 --vehicle-speed-kmh 20 --debug-motion
```

Path configuration can be supplied with:

```powershell
python main.py --camera-source 0 --path-config config/path_config.json
```

If `config/path_config.json` does not exist, the app uses a simple default image-zone classifier. Copy `config/path_config.example.json` to `config/path_config.json` and tune the polygons for your camera view when needed.

Example display:

```text
DOG ID:3 91% IN_PATH
~15.0m CLOSING EST.CLOSE:4.5m/s
EST.TTC:3.3s
```

Controlled testing should be done with a stationary camera first: have a person walk toward the camera, then away from it. Do not test while driving.

## Phase 12 - Hazard Risk Classification

Phase 12 combines existing detection, tracking, distance, path position, vehicle speed, closing speed, and estimated TTC into a transparent rule-based risk level.

Supported risk levels:

- UNKNOWN
- LOW_RISK
- MEDIUM_RISK
- HIGH_RISK

This is a prototype estimated risk classifier. It does not provide guaranteed collision prediction, certified automotive safety behavior, driver audio warnings, or automatic braking.

The risk engine is implemented in `src/risk.py`. Thresholds are configured in `src/config.py` under `RISK_CONFIG`.

Main factors:

- Path zone: `IN_PATH`, `NEAR_PATH`, `OUTSIDE_PATH`
- Estimated distance
- Estimated TTC
- Relative closing speed
- Vehicle speed
- Detection confidence

Run webcam with risk classification:

```powershell
python main.py --camera-source 0 --vehicle-speed-kmh 30 --calibration config/laptop_camera_calibration.json
```

Run Android IP Webcam with risk classification:

```powershell
python main.py --camera-source "http://PHONE_IP:8080/video" --vehicle-speed-kmh 30 --calibration config/phone_camera_calibration.json
```

Debug risk decisions:

```powershell
python main.py --camera-source 0 --vehicle-speed-kmh 30 --debug-risk
```

Example display:

```text
DOG ID:3 91% IN_PATH
~12.4m CLOSING EST.CLOSE:4.7m/s
EST.TTC:2.6s RISK:HIGH_RISK
```

The highest active track risk is shown once as `Scene Risk`. Press `S` to save an annotated frame under `results/phase12_risk_captures/`.

Use controlled tests only. Do not test risk behavior by driving toward people or animals.

## Phase 13 - Driver Visual + Audio Warning System

Phase 13 uses the Phase 12 risk result to show a driver warning banner and optionally play a local audio tone.

Warning mapping:

- `UNKNOWN` -> `NONE`
- `LOW_RISK` -> `NONE`
- `MEDIUM_RISK` -> `CAUTION`
- `HIGH_RISK` -> `DANGER`

The warning manager chooses the most important threat by risk level first, then lowest valid TTC, then shortest valid distance. All object boxes and risk labels remain visible.

Run webcam with warnings:

```powershell
python main.py --camera-source 0 --vehicle-speed-kmh 30 --calibration config/laptop_camera_calibration.json
```

Run phone camera with warnings:

```powershell
python main.py --camera-source "http://PHONE_IP:8080/video" --vehicle-speed-kmh 30 --calibration config/phone_camera_calibration.json
```

Disable audio but keep visual warnings:

```powershell
python main.py --camera-source 0 --no-audio-warning
```

Disable warnings while keeping detection/risk calculations:

```powershell
python main.py --camera-source 0 --no-warnings
```

Debug warning priority and cooldown:

```powershell
python main.py --camera-source 0 --debug-warning
```

Test warnings without YOLO:

```powershell
python -m src.test_warning --risk high --class dog
```

Use audio-disabled simulation during presentations or quiet testing:

```powershell
python -m src.test_warning --risk high --class dog --no-audio
```

Press `S` in the live detector to save annotated warning frames under `results/phase13_warning_captures/`.

Phase 13 does not implement automatic braking, steering control, or vehicle control. Use controlled tests only; do not test by driving toward people or animals.

## Phase 14 - Night / Rain / Low-Visibility Optimization

Phase 14 adds lightweight visibility analysis and optional image enhancement before YOLO inference.

The system estimates:

- Brightness
- Contrast
- Blur score
- Visibility class: `DAY`, `LOW_LIGHT`, or `VERY_DARK`

Enhancement modes:

- `off`: use the original frame for YOLO.
- `auto`: enhance only `LOW_LIGHT` and `VERY_DARK` frames.
- `lowlight`: force low-light enhancement on every frame.

Run webcam with automatic enhancement:

```powershell
python main.py --camera-source 0 --enhancement auto --debug-visibility
```

Run phone camera with automatic enhancement:

```powershell
python main.py --camera-source "http://PHONE_IP:8080/video" --enhancement auto --debug-visibility
```

Show original and enhanced frames side by side:

```powershell
python main.py --camera-source 0 --enhancement auto --show-enhancement
```

Disable enhancement:

```powershell
python main.py --camera-source 0 --enhancement off
```

Compare original vs enhanced detections on saved images:

```powershell
python -m src.evaluate_enhancement --input dataset/images/test --enhancement auto --save-failures
```

Phase 14 uses OpenCV CLAHE on the luminance channel, gamma correction, and optional denoising/sharpening configured in `config/visibility_config.json`. It does not add thermal imaging, infrared, radar, LiDAR, or guaranteed night detection.

Press `S` in live mode to save:

- annotated frame
- original frame
- enhanced / YOLO-input frame

Saved files are written to `results/phase14_enhanced_captures/`.

Failure examples can be collected in `results/phase14_failures/`. Use the comparison report at `results/phase14_visibility_report.txt` to document actual low-light/rain/glare results. Do not invent night or rain metrics unless the dataset has real labels or metadata.

Recommended dataset improvements:

- night road scenes
- low-light animals
- rain and wet-road scenes
- glare/headlight scenes
- shadows
- noisy phone-camera video
- distant animals
- partially occluded animals

RGB low-light enhancement can improve visible contrast, but it cannot recover information that the camera did not capture. Camera hardware, exposure, focus, lens quality, sensor size, and frame rate strongly affect night performance.

## Phase 15 - Optional Sensor Integration + Sensor Fusion

Phase 15 prepares Airacare to combine RGB camera/YOLO tracks with optional external sensors. External sensors are not required; the application still works in camera-only mode.

Supported prototype modes:

- `CAMERA_ONLY`: RGB camera, YOLO, tracking, monocular distance, path, TTC, risk, and warnings.
- `CAMERA + MOCK_RADAR`: simulated radar-style range and relative speed for safe development.
- `CAMERA + RADAR`: placeholder for future real radar hardware integration.
- `CAMERA + DEPTH`: placeholder for future depth sensor integration.
- `CAMERA + THERMAL`: placeholder for future thermal camera integration.

RGB YOLO remains responsible for semantic class detection: person, dog, cat, horse, cow, deer, goat. Radar-style sensors can improve physical distance and relative speed, but they do not identify dog vs cow vs person by themselves.

Run camera-only:

```bash
python main.py --camera-source 0 --sensor none
```

Run phone camera with camera-only mode:

```bash
python main.py --camera-source "http://PHONE_IP:8080/video" --sensor none
```

Run with mock radar fusion:

```bash
python main.py --camera-source 0 --sensor mock --sensor-data test_data/radar_sample.csv --debug-sensors
```

Useful Phase 15 options:

```bash
--sensor none|mock|radar|depth|thermal
--sensor-data test_data/radar_sample.csv
--sensor-calibration config/sensor_calibration.json
--sensor-port COM5
--sensor-baud 115200
--thermal-source <source>
--debug-sensors
```

Fusion behavior:

- Camera tracks are still the main object identities.
- A camera track horizontal center is converted to approximate bearing using the configured camera horizontal field of view.
- Sensor measurements are associated only when timestamp and angle are close enough.
- If a valid associated radar/depth/mock distance exists, it is preferred over camera-only distance.
- If valid sensor relative speed exists, it is preferred over camera-derived closing speed.
- If sensor data is missing, stale, ambiguous, or not configured, Airacare falls back to camera-only values.

Important limitations:

- Mock sensor data is simulated and is not real hardware validation.
- Real radar/depth/thermal support needs hardware documentation and calibration.
- RGB low-light enhancement is not thermal imaging.
- Sensor fusion values are prototype estimates and should not be treated as certified automotive safety measurements.
- Phase 15 does not implement automatic braking or vehicle control.


## Distance, TTC, Risk, and Warning Setup

If the live UI shows `Distance: OFF`, `DIST:N/A`, `EST.CLOSE:N/A`, `TTC:N/A`, and `RISK:UNKNOWN`, distance is either disabled with `--no-distance` or no valid calibration file was loaded.

Distance estimation requires camera calibration. Do not use the same calibration for every camera unless you actually calibrated them the same way. Laptop webcam and phone IP Webcam should have separate files.

Numeric calibration example:

```bash
python -m src.calibrate_distance --known-height 1.70 --known-distance 3.0 --pixel-height 500 --camera-source laptop_webcam --resolution 1280 720 --output config/webcam_calibration.json
```

Camera-assisted laptop calibration:

```bash
python -m src.calibrate_distance --camera-source 0 --known-height 1.70 --known-distance 3.0 --output config/webcam_calibration.json
```

Camera-assisted phone calibration:

```bash
python -m src.calibrate_distance --camera-source "http://PHONE_IP:8080/video" --known-height 1.70 --known-distance 3.0 --output config/phone_calibration.json
```

In the calibration window, stand at the measured distance with the full person visible, then press `C` to save the displayed bounding-box height. Press `Q` to quit without saving.

Run laptop webcam with distance enabled:

```bash
python main.py --camera-source 0 --calibration config/webcam_calibration.json --confidence 0.25 --imgsz 416 --iou 0.45 --vehicle-speed-kmh 30
```

Run phone camera with distance enabled:

```bash
python main.py --camera-source "http://PHONE_IP:8080/video" --calibration config/phone_calibration.json --confidence 0.25 --imgsz 416 --iou 0.45 --vehicle-speed-kmh 30
```

Disable distance only when you intentionally want detection/tracking without distance, TTC, risk, or warnings:

```bash
python main.py --camera-source 0 --no-distance
```

Distance formula:

```text
distance_m = known_real_object_height_m * focal_length_pixels / bounding_box_height_pixels
```

The result is approximate. Object sizes vary, bounding boxes jitter, and camera zoom/resolution changes affect accuracy.
