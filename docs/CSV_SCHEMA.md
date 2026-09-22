# CSV schema and interpretation

Each export contains one row per selected input image. By default the CSV is a snapshot of the current input folder, including cached results. `--export-new-only` includes only images actually analyzed during that run. An empty selection produces a header-only CSV.

Files are comma-separated UTF-8 with a BOM, with standard CSV quoting. Read them using a CSV parser, not by splitting lines on commas. JSON evidence is stored inside quoted CSV cells.

## Missing values, zero, and unknown

- An empty numeric cell means unavailable, unsupported, or no accepted vote. It is not zero.
- Zero in a supported object-count column means that expert accepted no detections of the category. It does not prove the object was absent.
- `unknown` and `unclear` preserve abstentions. Their count columns must not be discarded when calculating classification coverage.
- `combined_adults` and `combined_children` are always zero in the current policy because no age classifications are accepted. `combined_age_unknown` equals the canonical person count. These zeroes are not evidence that no adults or children were present.
- Boolean CSV values use `True`/`False`; booleans inside JSON use `true`/`false`. JSON `null` represents an unavailable or abstained value.
- Strings that could be interpreted as spreadsheet formulas are prefixed with an apostrophe on CSV export. This can affect displayed filenames starting with characters such as `=` or `+`.

## Identity, status, and timing columns

| Column | Meaning |
|---|---|
| `run_id` | Local timestamp shared by the CSV, run JSON, and contact-sheet filename. |
| `image_id` | `img_` plus the first 16 hexadecimal characters of the image SHA256; fallback ordinal when hashing failed. Identical bytes can have the same ID on multiple filename rows. |
| `relative_path` | Image path relative to the selected input directory, using `/` separators. |
| `sha256` | Full image-content hash. A duplicate filename row with identical bytes can reuse inference. |
| `status` | `ok`, `partial_error` for failed person-attribute predictions, or `error` for failed image analysis. |
| `error` | Error text, if applicable. Successful images have an empty value. |
| `cache_hit` | Whether this row reused a successful cached result. |
| `analyzed_at_utc` | UTC timestamp of the original inference, preserved on cache hits. |
| `width`, `height` | Dimensions after applying EXIF orientation. All evidence coordinates use this image space. |
| `vision_device` | Final detector/pose engine device, such as `cpu` or `cuda:0`. Detailed expert evidence records each expert's device if fallback occurred during an image. |
| `attribute_device` | Attribute-classifier device, independent of the vision device. CPU is the default installation. |
| `analysis_seconds` | Original per-image vision and attribute processing time, excluding model initialization and final export rendering; preserved on cache hits. |
| `current_run_seconds` | Time spent on this file during the current run, including hashing and cache lookup. For the first uncached file it can include lazy model initialization. |
| `configuration_hash` | Fingerprint of code, model files, runtime versions, requested device, and thread configuration used for caching. |
| `needs_review` | Broad review flag. Current policy flags all images with canonical people because age remains unvalidated, plus relevant direction/orientation/count disagreements. It is not an error or accuracy probability. |

Do not sum cached rows' `analysis_seconds` to estimate work performed during the new run. The matching `_run.json` contains current-run elapsed time, cache totals, model initialization time, and error totals. Bootstrap package installation occurs before this analysis timer; model verification/preparation invoked during analysis can be included in the run total.

For image-level `error` rows, detector/count/evidence fields are blank. `partial_error` rows retain usable detector and per-person results, with failed attributes represented as unknown. Errors are retried on the next run rather than cached as successful output.

## Object-count families

For each category below, the schema includes all four column prefixes:

```text
combined_<category>
yoloe_<category>
yolo26n_<category>
megadetector_<category>
```

| Category | Definition | YOLOE | YOLO26n | MegaDetector |
|---|---|---|---|---|
| `people_total` | Person detections | Yes | Yes | Yes |
| `bicycles` | Bicycle detections | Yes | Yes | Unsupported |
| `strollers` | Baby stroller detections | Yes | Unsupported | Unsupported |
| `motorcycles` | Motorcycle detections | Yes | Yes | Unsupported |
| `atv_utv` | ATV, UTV, or golf cart | Yes | Unsupported | Unsupported |
| `other_vehicles` | Car, truck, bus, or tractor | Yes | Unsupported | Unsupported |
| `dogs` | Dog detections | Yes | Yes | Unsupported; its animal class is generic |
| `backpacks` | Backpack objects | Yes | Yes | Unsupported |
| `cars_trucks_buses` | Car, truck, or bus | Yes | Yes | Unsupported |
| `kick_scooters` | Kick scooter detections | Yes | Unsupported | Unsupported |

`cars_trucks_buses` is a subset of `other_vehicles`; do not add those two columns together. Suitcase and duffel-bag detections are available in JSON evidence, without a dedicated scalar count. MegaDetector generic animal/vehicle totals also remain in JSON evidence. No physical bag-size classification is produced.

**`combined_people_total` is the number of YOLOE canonical person records.** It deliberately does not switch to a different detector's person count, because orientation, age-unknown, and carrying-backpack totals must describe the same people. Separate voting columns make disagreements visible.

Other `combined_` counts use an accepted strict majority, or YOLOE's explicitly provisional fallback when no majority exists. A fallback is not a verified count.

## Vote and comparison families

Each of the ten categories also has:

| Column pattern | Meaning |
|---|---|
| `vote_<category>` | Integer strict-majority count; empty if no majority. At least two agreeing experts and strictly more than half of eligible experts are required. |
| `mean_<category>` | Arithmetic mean of available eligible model counts, rounded to six decimals. May be fractional. |
| `vote_status_<category>` | `majority`, `single_expert`, `no_majority`, or `no_eligible_output`. |
| `vote_model_count_<category>` | Number of eligible supported model outputs. |
| `vote_spread_<category>` | Maximum minus minimum eligible count. A single expert has spread zero, which does not imply agreement. |
| `combined_source_<category>` | `yoloe_canonical_person_roster`, `strict_majority`, or `yoloe_provisional_fallback`. |

For people, YOLOE, YOLO26n, and MegaDetector are eligible. For other categories, only YOLOE and YOLO26n are considered, and unsupported values are excluded. Pose is exported separately and does not cast a person-count vote. YOLOE segmentation masks are not an extra voter.

Examples:

| Model counts | Vote | Mean | Status |
|---|---|---|---|
| People: YOLOE 4, nano 3, MegaDetector 3 | 3 | 3.333333 | `majority`; combined people remains YOLOE's 4 |
| Backpacks: YOLOE 2, nano 1 | Empty | 1.5 | `no_majority`; combined uses provisional YOLOE 2 |
| Strollers: YOLOE 1; others unsupported | Empty | 1 | `single_expert`; combined uses YOLOE 1 |

## Combined person summaries

| Columns | Meaning |
|---|---|
| `combined_adults`, `combined_children`, `combined_age_unknown` | Accepted adult/child/unknown ages over the canonical person roster. Current policy puts every person in unknown. |
| `age_status` | Currently `abstained_unvalidated_classifier`. |
| `presentation_status` | Currently `abstained_unvalidated_native_binary_proxy`. |
| `orientation_conflicts` | People for whom attribute and pose experts both supplied known, conflicting orientations; combined orientation abstains. |
| `combined_direction_left`, `combined_direction_right`, `combined_direction_toward`, `combined_direction_away`, `combined_direction_unclear` | Combined apparent-facing counts, relative to the image/camera. These do not measure travel. |
| `combined_orientation_front`, `combined_orientation_back`, `combined_orientation_side`, `combined_orientation_unknown` | Combined body-orientation counts. |
| `combined_carrying_backpack_yes`, `combined_carrying_backpack_no`, `combined_carrying_backpack_unknown` | Person-level backpack classifier/spatial-evidence combination. Distinct from detected backpack objects. |

For successful rows, each complete person-level partition sums to `combined_people_total`:

```text
adults + children + age_unknown
left + right + toward + away + unclear
front + back + side + unknown
carrying_backpack_yes + carrying_backpack_no + carrying_backpack_unknown
```

Front becomes toward, back becomes away. Side becomes left/right only with compatible pose-profile evidence. If the two known orientation predictions conflict, the combined orientation and direction become unknown/unclear. A single known expert may supply the orientation; per-person JSON records its source.

## Native attribute summaries

These are review outputs from PP-LCNet, not accepted demographic facts.

| Column family | Labels |
|---|---|
| `attribute_age_<label>` | `under18`, `18_60`, `over60`, `unknown` |
| `attribute_orientation_<label>` | `front`, `back`, `side`, `unknown` |
| `attribute_presentation_proxy_<label>` | `feminine`, `masculine`, `unclear` |

The age and orientation groups use crop-size, score, and score-margin gates. Age labels describe the classifier's native appearance bins; `under18` includes teenagers and does not define a validated child count. The presentation proxy derives from an unvalidated native binary training label and does not establish gender identity or sex.

Each complete native summary partition covers the canonical person roster, with failed or insufficient-detail crops assigned unknown/unclear.

## Pose summaries

| Column family | Meaning |
|---|---|
| `pose_people_total` | Number of independent pose detections, which can differ from YOLOE's person count. |
| `pose_direction_<label>` | Pose evidence on the canonical YOLOE person roster: `left`, `right`, `toward`, `away`, `unclear`. |
| `pose_orientation_<label>` | Pose evidence on the canonical roster: `front`, `back`, `side`, `unknown`. |

The direction and orientation families include unmatched or ambiguous canonical people as unknown/unclear, so they sum to **`combined_people_total`**, not necessarily `pose_people_total`. Independent pose detections and keypoints remain available in `expert_evidence_json`.

## JSON evidence cells

### `persons_json`

A list with one entry for each canonical YOLOE person, preserving:

- `person_id`: `person_0001`, etc.; local to this image, not an identity or track across photographs.
- `source_detection_index`, `xyxy`/`bbox_xyxy`, and detector `confidence`.
- `mask_backpack`, `mask_backpack_status`, and `bag_associations`. A spatial-support flag alone is not a carrying label. No detected association remains `null`, not false.
- `pose`: matched pose index, match IoU/margin, matched/ambiguous/unmatched status, orientation, facing direction, evidence score, reason codes, and geometric features.
- `attributes`: native scores, gated age/orientation, backpack presence, presentation proxy, crop dimensions, detail-gate status, abstention reasons, runtime device, and per-person error information.
- `combined`: orientation and source, apparent direction, carrying-backpack result and source, `age: "unknown"`, and `presentation: "unclear"`.

Relevant native scores are `handbag`, `shoulder_bag`, `backpack`, `age_under18`, `age_18_60`, `age_over60`, `native_female`, `front`, `side`, and `back`. These are independent sigmoid model scores, not calibrated class probabilities. They need not sum to one.

### `expert_evidence_json`

An object keyed by `yolo26n`, `yoloe`, `megadetector`, and `pose`. Each expert includes its count dictionary, detections, device, checkpoint name, inference time, and model timing details. Detections preserve class labels, confidences, and pixel boxes.

YOLOE detections additionally contain `mask_polygon_xy` and mask-area data. Polygons describe exterior or merged contours rounded to 0.1 pixel; holes and full-resolution binary masks are not retained. Backpack associations preserve candidate scores, containment/proximity measures, and mask-overlap ratios. Suppressed competing motor-vehicle detections remain recorded separately.

Pose detections contain 17 `[x, y, confidence]` keypoints in COCO order and their uncalibrated orientation-rule evidence. The order is nose; left/right eye; left/right ear; left/right shoulder; left/right elbow; left/right wrist; left/right hip; left/right knee; left/right ankle. Left/right keypoint labels refer to the model's anatomical sides; direction labels refer to the image.

### `attribute_runtime_json`

Attribute runtime metadata: model, requested and actual device, GPU fallback reasons, image processing time, policy notes, and any failed-person count. Per-person records retain their own devices when a GPU failure caused a mid-image CPU fallback.

### `timings_json`

Detailed original image timings such as decode, individual detector/pose expert processing, total vision time, and attribute time. Like `analysis_seconds`, these are cached with the original prediction rather than replaced by cache-lookup times.

### `notes`

A compact interpretation reminder repeated on each row: direction is facing rather than travel, primary age classification abstains, presentation is not gender identity, blanks differ from zero, and people are image appearances rather than unique visitors.

## Reading full evidence in Python

Python's default CSV field-size limit may be smaller than a crowded image's evidence cell. Increase it before loading:

```python
import csv
import json

csv.field_size_limit(100_000_000)
with open("exports/export_DATE.csv", encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        if row["status"] not in {"ok", "partial_error"}:
            continue
        persons = json.loads(row["persons_json"])
        print(row["relative_path"], row["combined_people_total"], len(persons))
```

The run JSON records the actual input-directory context, full configuration fingerprint inputs, and contact-sheet selection. Keep it beside its CSV when comparing different runs or moving exports to another computer.
