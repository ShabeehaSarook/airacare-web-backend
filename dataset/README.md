# Airacare Custom Dataset

## Purpose

This folder stores the custom image dataset for the Airacare Animal Detector. Images collected during Phase 3 are manually annotated during Phase 4 and prepared as a YOLO object-detection dataset for later training.

Phase 4 prepares annotations and validates dataset structure only. It does not train a model.

## Folder Structure

```text
dataset/
+-- raw/
|   +-- person/
|   +-- dog/
|   +-- cat/
|   +-- horse/
|   +-- cow/
|   +-- deer/
|   +-- goat/
+-- cleaned/
+-- annotated/
|   +-- images/
|   +-- labels/
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

## Class Names

Use only these class folders for raw images:

0. person
1. dog
2. cat
3. horse
4. cow
5. deer
6. goat

Do not place incorrectly labeled classes in these folders. For example, an image containing only sheep should not be placed in `goat/`.

## YOLO Class IDs

Use exactly these class IDs in every YOLO label file:

```text
0 person
1 dog
2 cat
3 horse
4 cow
5 deer
6 goat
```

## Image Collection Guidelines

Each class should include varied examples:

- close-distance objects
- medium-distance objects
- far-distance objects
- front views
- side views
- rear views
- partially occluded objects
- multiple objects in one scene
- road-side scenes
- animals crossing roads
- vegetation backgrounds
- urban backgrounds where appropriate
- daylight
- evening
- nighttime
- light rain
- wet road conditions
- different camera angles
- different object sizes

## Dataset Size Target

Initial target:

- 500+ usable images per class
- 3500+ usable images total

Quality matters more than raw count. Remove or replace images that are blurry, irrelevant, duplicated, too small, or incorrectly categorized before annotation.

## Road-Scene Priority

Prioritize road-scene images because the final system is an animal road-hazard detector. Useful images include animals near roads, crossing roads, standing beside roads, partially hidden by vegetation near roads, and captured from vehicle-like camera angles.

## Lighting and Weather Diversity

Collect examples from daylight, evening, nighttime, light rain, and wet road conditions. This helps later training handle realistic driving environments.

## Duplicate Avoidance

Avoid exact duplicate images and large sets of nearly identical frames. Keep varied examples instead of many copies from the same moment or camera angle.

## Manual Annotation Workflow

Create bounding boxes manually with a YOLO-compatible annotation tool such as CVAT, Label Studio, Roboflow, LabelImg, or another equivalent tool.

Recommended workflow:

1. Start from cleaned Phase 3 images.
2. Import images into the annotation tool.
3. Configure exactly the 7 approved classes listed above.
4. Draw one bounding box for every visible target object.
5. Export annotations in YOLO object-detection text format.
6. Place exported images in `dataset/annotated/images/`.
7. Place exported labels in `dataset/annotated/labels/`.
8. Run the split utility to copy image/label pairs into `images/train`, `images/val`, `images/test`, `labels/train`, `labels/val`, and `labels/test`.

For every image, there must be a matching `.txt` file with the same stem:

```text
image_name.jpg
image_name.txt
```

Each label line must use normalized YOLO format:

```text
class_id center_x center_y width height
```

All coordinates must be between 0 and 1. Width and height must be greater than 0.

If one image contains two dogs, one person, and one cow, the label file must contain four lines.

## Annotation Guidelines

1. Draw boxes tightly around visible objects.
2. Annotate every visible target object.
3. Use only the approved 7 classes.
4. Do not confuse deer and goat.
5. Do not label unrelated animals.
6. Include partially occluded animals if identifiable.
7. Include small or distant animals when identifiable.
8. Do not label objects that cannot be confidently identified.
9. Do not create oversized boxes containing lots of background.
10. Do not create very tight boxes that cut off important parts of the object.

## Train/Validation/Test Split

Default split:

- 70% train
- 20% validation
- 10% test

Run from the project root after manual annotation export:

```bash
python -m src.split_dataset --source-images dataset/annotated/images --source-labels dataset/annotated/labels --output dataset
```

Use `--dry-run` first to preview counts without copying files:

```bash
python -m src.split_dataset --dry-run
```

The split utility keeps each image with its matching label, skips images with missing labels, uses a configurable seed, and refuses to overwrite existing output files unless `--allow-overwrite` is provided.

## Annotation Validation

Run the validator from the project root:

```bash
python -m src.validate_annotations
```

The validator reports missing labels, orphan labels, invalid class IDs, out-of-range coordinates, malformed lines, empty annotation files, duplicate label lines, total counts, and bounding-box counts per class. It does not modify any label files.

## Visual Annotation Preview

After validation, inspect a sample visually:

```bash
python -m src.preview_annotations --split train --count 10 --shuffle
```

The preview tool draws YOLO boxes on images and shows class names so the annotation quality can be checked before Phase 5 training.

## Current Phase

Phase 4 is dataset annotation preparation and validation. Do not start YOLO training until annotations validate cleanly and visual previews confirm box placement.

## Dataset Inspection

Run the report-only dataset inspector from the project root:

```bash
python -m src.dataset_inspector
```

The report is written to:

```text
dataset/reports/dataset_report.txt
```
