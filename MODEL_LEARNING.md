# Airacare Model Learning Notes

## Project Purpose

Airacare Animal Detector uses YOLO object detection to identify objects from a live camera feed.

The final target classes are:

```text
0 person
1 dog
2 cat
3 horse
4 cow
5 deer
6 goat
```

## Current Model Behavior

The project first tries to load the custom trained Airacare model:

```text
models/airacare_animal_detector_best.pt
```

If that file is missing, the system uses the pretrained YOLO demo model:

```text
models/yolo11n.pt
```

The pretrained fallback model can detect:

```text
person
dog
cat
horse
cow
```

The pretrained model does not properly support:

```text
deer
goat
```

Deer and goat require the custom trained model.

## How Live Detection Works

The live detector is implemented in:

```text
src/detector.py
```

The entry point is:

```text
main.py
```

When the project runs:

```text
camera frame
    -> YOLO model
    -> detection boxes
    -> class names
    -> confidence scores
    -> display boxes on screen
```

Example output on the video window:

```text
PERSON 94%
DOG 88%
CAT 76%
```

## Phone Camera Mode

The phone camera is connected through an IP Webcam URL:

```text
http://PHONE_IP:8080/video
```

Example:

```powershell
python main.py --camera-source "http://10.195.215.13:8080/video"
```

The system detects whether the camera source is:

- laptop webcam index, such as `0`
- phone/IP camera URL, such as `http://10.195.215.13:8080/video`

## Low-Latency Frame Capture

Phone camera streams can lag because many old frames can build up while YOLO is processing.

To reduce lag, the project uses a latest-frame design:

```text
phone camera stream
    -> background capture thread
    -> keep only newest frame
    -> discard old frames
    -> YOLO processes latest frame
```

This is implemented by:

```text
LatestFrameCapture
```

in:

```text
src/detector.py
```

This avoids processing old frames later.

## YOLO Prediction Code

The detector runs prediction like this:

```python
results = model.predict(
    source=frame,
    conf=confidence_threshold,
    iou=iou_threshold,
    imgsz=inference_image_size,
    device=device,
    verbose=False,
)
```

Important settings:

```text
confidence_threshold = 0.25
iou_threshold = 0.45
inference_image_size = 640
```

## Multiple Object Detection

The code loops through every bounding box returned by YOLO:

```python
for detected_box in boxes:
    confidence = float(detected_box.conf[0])
    class_id = int(detected_box.cls[0])
    class_name = model_names.get(class_id)
    coordinates = detected_box.xyxy[0].tolist()
```

This means multiple objects of the same class can be displayed.

Example:

```text
PERSON 94%
PERSON 87%
DOG 91%
```

The system does not remove a detection just because another object has the same class name.

## Confidence Threshold

Confidence controls how sure YOLO must be before showing a detection.

Default:

```text
0.25
```

Lower confidence:

```text
more detections
more false positives
```

Higher confidence:

```text
fewer detections
more missed objects
```

Example:

```powershell
python main.py --camera-source 0 --confidence 0.25
```

## IoU Threshold

IoU affects non-maximum suppression.

Default:

```text
0.45
```

This helps YOLO decide whether two overlapping boxes are separate objects or duplicate boxes.

For two close people, avoid using an extremely low IoU value because it may suppress one person.

Example:

```powershell
python main.py --camera-source "http://10.195.215.13:8080/video" --iou 0.45
```

## Inference Image Size

YOLO internally resizes the image for inference.

Default:

```text
640
```

Better accuracy but slower:

```text
--imgsz 640
```

Faster but may miss small/far objects:

```text
--imgsz 416
--imgsz 320
```

Example for faster phone camera detection:

```powershell
python main.py --camera-source "http://10.195.215.13:8080/video" --confidence 0.25 --imgsz 416 --iou 0.45
```

## Debug Detections

To print every detected box:

```powershell
python main.py --camera-source "http://10.195.215.13:8080/video" --confidence 0.25 --imgsz 640 --iou 0.45 --debug-detections
```

Example debug output:

```text
Detected:
person confidence=0.94 box=(120.1, 80.5, 400.3, 710.8)
person confidence=0.87 box=(520.4, 90.2, 780.0, 720.0)
dog confidence=0.76 box=(300.2, 500.1, 610.7, 890.4)
```

## No-Detection Mode

This mode shows phone video only and does not run YOLO:

```powershell
python main.py --camera-source "http://10.195.215.13:8080/video" --no-detection
```

Use this to test whether lag comes from:

```text
phone/Wi-Fi stream
```

or:

```text
YOLO inference
```

If no-detection mode is smooth but detection mode is slow, the CPU/YOLO inference is the bottleneck.

## Custom Model Training

The custom model training script is:

```text
src/train_model.py
```

It trains using:

```text
dataset/data.yaml
dataset/images/train
dataset/images/val
dataset/labels/train
dataset/labels/val
```

Training command:

```powershell
python -m src.train_model
```

After training, the best model should be copied to:

```text
models/airacare_animal_detector_best.pt
```

Then the live detector will use the custom 7-class model instead of the pretrained fallback.

## Custom Model Evaluation

The evaluation script is:

```text
src/evaluate_model.py
```

Evaluation command:

```powershell
python -m src.evaluate_model
```

It reports:

```text
Precision
Recall
mAP50
mAP50-95
Inference speed
False positives
False negatives
```

## Important Limitation

Right now, because the custom trained model is missing, the system uses:

```text
models/yolo11n.pt
```

This is enough for a demo with:

```text
person
dog
cat
horse
cow
```

But for full Airacare detection:

```text
person
dog
cat
horse
cow
deer
goat
```

you must train the custom model using annotated data.

