# Version 2 implementation and conservative defaults

This document describes the reviewed v2 behavior. Constants in the named modules are the source of truth. Historical development scores are described in [VALIDATION.md](VALIDATION.md); they must not be read as validation of the revised default release.

## Default contract

- Analyze every selected image with all four vision experts; `empty_frame_gate` is false.
- Keep a single canonical person roster so per-person attributes and combined people counts agree.
- Count backpack objects regardless of whether experimental bag-size estimation is enabled.
- Keep primary age unknown. Native PP-LCNet age and presentation outputs remain comparison evidence, not demographic conclusions.
- Report apparent facing from single images. Motion direction requires explicitly enabled event tracking.
- `events`, `geometry_age`, and `large_bags` default to false. `near_fraction` defaults to zero, so `people_near == people_total`.
- The Ollama age expert defaults to off and is not required or installed by the package.
- Write one image CSV, a run JSON, and a default-on contact sheet with up to 20 diverse examples into a new timestamped export folder. `--events` adds event and camera-day CSVs.
- Use CUDA when an actual kernel succeeds, otherwise CPU; runtime CUDA failures also fall back to CPU. Paddle attributes use CPU in the default installation.

CLI opt-ins are `--events`, `--geometry-age`, `--large-bags`, `--empty-frame-gate`, nonzero `--near-fraction`, and `--age-model`. The Boolean switches have matching `--no-...` forms. Enabling an experiment does not establish its reliability.

## Models and inference

| Component | Standard / fast input | Purpose |
|---|---|---|
| YOLOE-26s segmentation | 1280 / 960 | Prompted people/object boxes and instance masks |
| YOLO26n-pose | 1280 / 960 | Person boxes and 17 COCO keypoints |
| YOLO26n | 960 / 640 | Independent COCO object/person detections |
| MegaDetector V6 compact | 960 / 640 | Generic camera-trap animal/person/vehicle evidence and optional gate; no dog-species corroboration |
| PP-LCNet x1.0 pedestrian attributes | Person crops | Native age bins, front/back/side, backpack scores and presentation proxy |

`vision.py` decodes with EXIF orientation. Large JPEG/MPO files can use half-resolution draft decoding; every stored box, mask and keypoint is rescaled into the original EXIF-corrected image coordinates. The detector models use FP32, confidence floor 0.12, and at most 300 detections. YOLOE, YOLO26n and pose use their one-to-many heads with NMS IoU 0.7.

Stable detection indices connect boxes to their original masks and keypoints even when malformed/non-finite boxes are removed. Pose re-matching preserves those indices. This prevents valid geometry from being attached to a different person after filtering.

The YOLOE prompts are prepared once during setup:

```text
person, bicycle, motorcycle, dog, backpack, baby stroller,
all-terrain vehicle, utility terrain vehicle, car, truck, bus,
golf cart, tractor, kick scooter, suitcase, duffel bag
```

MobileCLIP2-B computes prompt embeddings only at setup. Its research-only license is documented in [MODEL_SOURCES.md](../MODEL_SOURCES.md); it is not loaded for image inference.

The optional empty-frame gate runs YOLO26n and MegaDetector first and opens on any detection at confidence >= 0.20 outside the camera data strip. A closed gate skips YOLOE and pose, creates no candidates, and records skipped expert counts as blank, not zero. It may hide a person or object missed by both early experts, so the default is full analysis.

PyTorch's requested thread count is restored after Paddle setup/model warm-up and before each image. This addresses a measured CPU slowdown caused by another runtime changing the process thread setting.

## Candidate roster and object rules

`roster.py` clusters person boxes from YOLOE, pose and YOLO26n at confidence >= 0.12, using IoU >= 0.45 and at most one box from each expert per candidate. Boxes at least half inside the camera info strip are excluded. Representative boxes prefer YOLOE, then pose, then YOLO26n.

A person is accepted with YOLOE confidence >= 0.18, pose or YOLO26n alone >= 0.45, or at least two of these experts each >= 0.12. These are heuristic thresholds, not calibrated probabilities. MegaDetector does not set the combined people count. Its native count is retained for comparison.

`fusion.py` considers object detections at confidence >= 0.25, clusters matching labels across models at IoU >= 0.3, and uses category-specific lone-expert thresholds. See the [CSV schema](CSV_SCHEMA.md#main-counts) for the table. `objects_v2_1` requires dog-specific YOLOE and YOLO26n agreement, or a single dog expert at confidence >= 0.90. Generic MegaDetector animal boxes do not corroborate dog species; the previous rule promoted ibex in three review frames. Raw animal evidence remains available.

Vehicle de-duplication preserves separate ATV/UTV/golf-cart and road-vehicle/motorcycle groups. Final competing ATV/motorcycle labels use confidence at IoU >= 0.4; counted ATV/UTV overrides overlapping other-vehicle labels at IoU >= 0.4. Remaining near-identical motorcycle/other-vehicle labels compete at IoU >= 0.8. `cars_trucks_buses` is derived from final accepted other vehicles with a car/truck/bus label, so it remains a true subset.

Strict-majority count votes and means remain diagnostic comparisons. They do not replace the canonical roster or verified-object totals, and agreement between related model families is not independent validation.

## Per-person evidence and abstention

PP-LCNet inference retains native scores and labels. Age and presentation outputs are unvalidated and never silently become the default combined age/gender result. Native bins under18,18_60,over60 differ from the optional geometry definition of a clearly short pre-teen child.

Body orientation combines native PP-LCNet and pose front/back/side estimates: agreeing known labels are retained; one usable expert can supply the label; conflicting known labels produce unknown. Front maps to apparent toward, back to apparent away, and side maps to left/right only with compatible pose-profile evidence.

Backpack association is spatial evidence only. `associated_backpack` is separate from `carrying_backpack`; an association cannot turn an abstaining classifier into a positive carrying claim. A negative classifier contradicted by association remains unknown. The source is exported as `backpack_source`, including `spatial_support_only_unverified` where appropriate. Backpack object counts remain separate.

Large-bag size is disabled by default: image counts are None/blank, per-person size is null, and provenance reports disabled. `--large-bags` enables `bags.py` geometry rules, which compare associated bag boxes with shoulders/hips or full-body height, plus suitcase/duffel detections. Counts de-duplicate bags, not people. Historical large-bag presence F1 was only 0.31 against development AI labels.

## Pipeline and memory

1. `VisionEngine.analyze` reads metadata, decodes, runs detectors and pose, builds candidates, and computes appearance descriptors. `AttributeEngine` scores candidate crops.
2. `postprocess.prepare` performs per-image fusion immediately after inference or cache loading, assigns camera identity and compacts data. Raw results remain in the disk cache; rejected candidates and unnecessary geometry are removed from the run's memory.
3. Optional Ollama requests use their own answer cache and are made only when requested.
4. `postprocess.postprocess` calculates geometry/near-zone evidence and any explicitly enabled age/event experiments across the current inventory.
5. `export.py` emits the CSV schema, run metadata and contact sheet. Event files are conditional on `--events`.

Only inference and optional VLM answers are cached. Post-processing is recomputed every run, so changes to experimental options and fusion thresholds do not require model inference. The inference fingerprint covers inference source modules, candidate construction/policy, model files, package versions, device request, profile and gate setting. Threads are recorded but do not invalidate the fingerprint. Only complete successful results are reused; partial errors are retried. Cache validation rejects malformed boxes, confidences, indices and schema. Metadata is re-read even on a content-cache hit, so current filenames and capture-time fallbacks take effect.

Rows are keyed by relative path. Byte-identical files in different folders share inference but are separate appearances. Snapshot exports are not an append-only visitor database; adding or removing files changes the inventory and can change enabled cross-image estimates.

## Optional age geometry and near zone

`age_geometry.py` measures standing height only with sufficient head, torso, hip and ankle evidence and a standing/full-body gate. Calibration groups use camera id, EXIF make/model and original dimensions. The robust height fit needs at least 20 complete people and may abstain if unreliable.

With `--geometry-age`, a complete person's calibrated height ratio <= 0.82 can classify child and >= 0.88 adult; intermediate or implausible ratios abstain. Relative-height fallback may classify clearly shorter children (ratio <= 0.75), but **relative height alone never establishes adult**: equally tall children must remain unknown without a verified adult reference. Calibration-based ages remain experimental too.

Without the flag, every primary person age is unknown, with `age_method=disabled`. A requested per-person VLM response can override this: adult/child is recorded as a VLM estimate, teen becomes unknown, and unclear retains the existing abstention or enabled geometry estimate. Whole-photo VLM counts remain separate and never rewrite the canonical people count.

`near_fraction=0` includes everyone and avoids camera calibration by default. A nonzero fraction explicitly enables experimental filtering using expected standing height from calibration or same-depth peers. The earlier 0.07 choice was tuned on the owner's same 100 labels; the final saved development export had MAE 0.73 and bias +0.45 there, not an independent test result.

## Optional events and motion

`--events` groups photos by camera and capture-time gaps (default 60 seconds), splitting on camera metadata/image-dimension changes. Missing or suspicious timestamps remain visible as review evidence. Keep separate folders for different placements, including the same physical camera after moving it.

`events.py` associates people using appearance, size and position, with limits on match cost, appearance difference, size ratio and physically plausible displacement. Missing appearance evidence is flagged; such associations cannot establish motion direction. Ambiguity flags and reasons are preserved in `persons_json` and event review fields.

Only sufficiently supported multi-image tracks supply motion. Lateral motion is foot-point displacement in body-height units; radial motion uses change in apparent height. Small changes can be stationary only over a sufficient time interval; otherwise direction is unclear. Facing is a separately labelled fallback and does not prove movement.

`people_unique` counts tracks and can merge different people or split one person. `people_max_frame` is the largest accepted image count. They are two estimates, **not confidence bounds**. Objects including backpacks use the maximum image count within each event. Disabled large-bag columns stay blank through event and camera-day summaries.

## Exports and reproducibility

One dated folder contains the images CSV, run JSON and optional/default-on contact sheet. With events enabled, it also contains the events and camera-days CSVs. Filenames share `export_YYYY-MM-DD_HH-MM-SS_microseconds`; older runs are never overwritten.

The main CSV keeps native detector counts, votes, means, count sources, native attribute age/orientation/presentation aggregates, pose aggregates, and per-person native records. Primary ages remain unknown unless a requested experiment provides an estimate. `age_status`, `large_bags_status` and enabled flags explain unavailable outputs. Full field names and units are documented in [CSV_SCHEMA.md](CSV_SCHEMA.md).

UTF-8 BOM CSVs preserve non-English filenames. Formula-like text is escaped. Large evidence cells may exceed spreadsheet display limits; standard CSV readers preserve their data. Blank means unsupported, disabled, unavailable or abstained; zero means no accepted detection/assignment. Primary age zeros must be interpreted together with `age_unknown` and `age_status`.

`trailcam.evaluate` uses only the standard library, creates manual-label templates, and scores counts/presence with blank labels skipped and missing predictions reported. It includes backpacks independently of large bags, supports v1 prefixes, event labels and maxima derived from fully labelled frames. The script does not establish annotation quality or repair a contaminated test split.

## Evidence limits

The development labels for 157 photos and 28 series came from two AI annotators plus AI adjudication. The owner's manual 100-photo CSV is a separate reference. Threshold sweeps inspected the root subset too, so no untouched held-out claim is justified. Historical direction 56% is aggregate bucket overlap, not person-matched accuracy. Tracker direction was scored only at dominant-group level. See [VALIDATION.md](VALIDATION.md) for exact definitions, results and the workflow for a new human-labelled test set.

Historical CPU timing was 147.1 seconds image analysis / 157.3 seconds complete run for 157 photos; the earlier GPU revision took 43.2/ 52.2 seconds. Those runs enabled experimental settings and the gate; the revised conservative default has not been fully benchmarked on that set. Fresh Windows execution checks (see the README) additionally covered 100 GPU sequence photos and 20 CPU photos with the gate off; timings preceded the final dog post-processing correction. Unit/regression tests and those runs verify software operation, not deployment accuracy.
