# CSV schema and interpretation

Each default run writes one images CSV into its own `export_YYYY-MM-DD_HH-MM-SS_microseconds` subfolder of the configured output directory, together with a run JSON and a contact sheet enabled by default. `--events` additionally writes the experimental events and camera-days CSVs:

| File | One row per | Use it for |
|---|---|---|
| `export_<run>.csv` (images CSV) | image | Per-photo appearances, evidence, and review |
| `export_<run>_events.csv` (events CSV) | event | Experimental estimates of tracks within an event (`--events` only) |
| `export_<run>_camera_days.csv` (camera-days CSV) | camera and capture date | Experimental event totals (`--events` only) |

By default the images CSV is a snapshot of the current input folder, including cached results. `--export-new-only` limits the images CSV to images analyzed during that run; when enabled, the events and camera-days CSVs cover every image currently in the folder. An empty selection produces header-only CSVs.

Files are comma-separated UTF-8 with a BOM, with standard CSV quoting. Read them using a CSV parser, not by splitting lines on commas. JSON evidence is stored inside quoted CSV cells.

## Missing values, zero, and unknown

- An **empty cell** means unavailable, unsupported, or not assessable. It is never zero.
- **Zero** in a count column means the photo (or event) was assessed and nothing was accepted. It does not prove absence: a small or hidden person or object can be missed.
- `unknown` and `unclear` keep abstentions visible. Their count columns must not be discarded when calculating coverage.
- For `status = error` rows, every count, per-model, vote, and JSON column is empty.
- Boolean CSV values use `True`/`False`; booleans inside JSON use `true`/`false`. JSON `null` represents an unavailable or abstained value.
- Strings that could be interpreted as spreadsheet formulas are prefixed with an apostrophe on CSV export. This can affect displayed filenames starting with characters such as `=`, `+`, `-`, or `@` (after leading spaces), or with a tab, carriage return, or line feed.

## Images CSV

One row per image. **Counts are appearances in that photograph.** A person photographed in three consecutive photos appears in three rows; use the optional events CSV to review experimental de-duplication.

### Identity, time, camera, and event

| Column | Meaning |
|---|---|
| `run_id` | Local timestamp shared by the export subfolder and all files of this run. |
| `image_id` | `img_` plus the first 16 hexadecimal characters of the image SHA256; `img_` plus a six-digit ordinal when hashing failed. Identical bytes under different filenames share an `image_id` (and cached inference), but each file is a separate row and a separate appearance with its own capture time, camera, and event. |
| `relative_path` | Image path relative to the selected input directory, using `/` separators. The unique key of a row; the events CSV lists images by this path. |
| `sha256` | Full image-content hash. |
| `camera_id` | Camera the image was assigned to: the parent folder (`.` for images directly in the input folder), the `camera` group of `--camera-pattern`, or `all` with `--camera-id single`. Empty for `error` rows. |
| `capture_time` | Camera-clock capture time, `YYYY-MM-DDTHH:MM:SS` with optional `.ffffff`; no time zone. This and the next three columns are read from the file on every run, also on cache hits. |
| `time_source` | `exif_original` (EXIF DateTimeOriginal), `exif_digitized`, `exif_datetime` (EXIF DateTime), or `file_mtime` (file modification time, local). Empty when no time was found. |
| `clock_suspect` | `True` when the capture time is before 2015, more than one day after the file modification time, or within ten minutes after midnight on 1 January. Informational; events still use the time. |
| `sequence_number` | Last run of digits in the filename stem (`IMAG0571` → 571), used to order photos with equal or missing times. Empty when the name has no digits. |
| `event_id` | Event this image belongs to; matches the events CSV. Empty for `error` rows and when `--events` is off. |
| `event_frame` | Position in capture order, starting at 1; blank when events are disabled. |
| `event_image_count` | Number of images in the event; blank when events are disabled. |

### Status, runtime, and review

| Column | Meaning |
|---|---|
| `status` | `ok`; `partial_error` when the attribute model failed for some people (counts are kept, their attributes are unknown); `error` when image analysis or post-processing failed. |
| `error` | Error text, if applicable. Post-processing failures read `Post-processing failed: <step>: <error type>: <message>`, where the step is `fusion` (per-image rules) or `geometry` (body measurement). A row that could not be written reads `Export failed: ...`; that image's detections still count in its event. |
| `cache_hit` | Whether this row reused a cached inference result. |
| `analyzed_at_utc` | UTC timestamp of the original inference, preserved on cache hits. |
| `width`, `height` | Original image dimensions after EXIF orientation. All coordinates use this pixel space, also when a large JPEG was decoded at reduced resolution. |
| `profile` | `standard` or `fast`: the model input sizes used for inference. |
| `vision_device` | Final detector/pose device, such as `cpu` or `cuda:0`. |
| `attribute_device` | Device used by the PP-LCNet attribute model; CPU in the default installation. |
| `analysis_seconds` | Original per-image inference time (models and attributes), excluding model start-up; preserved on cache hits. |
| `current_run_seconds` | Time spent on this file during the current run, including hashing and cache lookup. For the first uncached file it can include model start-up. |
| `configuration_hash` | Fingerprint of the inference code, model files, runtime package versions, requested device, profile, and empty-frame-gate setting. Post-processing settings are not included. |
| `gated_empty` | `True` when the empty-frame gate skipped YOLOE and pose because YOLO26n and MegaDetector found nothing at confidence 0.20 or more. Such rows report no people or objects, no candidates, and empty `yoloe_*` columns. |
| `data_strip_bottom` | Height in pixels of the camera info strip detected at the bottom edge; 0 when none. Detections at least half inside the strip are ignored (a top strip is handled the same way and kept in the cache). |
| `needs_review` | `True` when the two orientation models conflicted for someone, a per-model count vote had no majority, an object was found by one model below its threshold (`unverified_*`), or a large-bag reading was uncertain. A review hint, not an error or accuracy probability. |

Do not sum cached rows' `analysis_seconds` to estimate work performed during the new run. The run JSON contains current-run elapsed time, cache totals, model start-up time, and error totals.

### Main counts

| Column | Meaning |
|---|---|
| `people_total` | Accepted people: YOLOE confidence ≥ 0.18, or pose or YOLO26n alone ≥ 0.45, or at least two of these three models ≥ 0.12 on the same person. Every visible person, including distant background people. |
| `people_near` | Accepted people in the near zone: an adult standing at the person's foot position would be at least `--near-fraction` (default 0, all people) of the image height tall. The expected height comes from the camera calibration when it is `ok` and the foot is inside its fitted range; otherwise it is the tallest measured height (the pose-based standing height, else the box height) among the people standing at the same depth or farther away (feet no more than 0.02 × image height lower in the image), including the person — so anyone closer to the camera than a near person is near too. At most `people_total`; equal to it with `--near-fraction 0`. |
| `people_class`, `people_near_class` | Class of `people_total` and of `people_near`: `0`, `1`, `2`, `3–4`, `5–10`, or `>10` (see [Count classes](#count-classes)). |
| `adults`, `children`, `age_unknown` | Accepted age assignments per person; they sum to `people_total`. By default adults and children are zero and every person is age_unknown: no age assignments were made. Geometry is opt-in with `--geometry-age`, and with `--age-mode mosaic` (or `photo+mosaic`) the age expert's answer replaces it for people it called adult or child; a `teen` answer becomes `unknown`, an `unclear` answer keeps unknown, or the geometry estimate only when enabled. |
| `dir_left`, `dir_right`, `dir_toward`, `dir_away`, `dir_stationary`, `dir_unclear` | Image-relative direction per person; they sum to `people_total`. Motion across the event's photos when the person was matched in another photo, otherwise apparent facing. Motion is only attempted with `--events`. |
| `direction_from_motion` | People whose direction came from motion (including motion results `stationary` and `unclear`). |
| `direction_from_facing` | People whose direction came from facing. The remaining people have no direction evidence and are in `dir_unclear`. |
| `bicycles`, `strollers`, `motorcycles`, `atv_utv`, `other_vehicles`, `dogs` | Verified objects (found by two models, or by one model above its threshold). |
| `large_bags` | Large bags, not people: bags associated with a person and classified as large, plus suitcases and duffel bags (confidence ≥ 0.35) not associated with anyone. Overlapping bag boxes (IoU ≥ 0.5) are one bag, also when loose luggage overlaps a carried large bag. Experimental and blank unless `--large-bags` is enabled. |
| `large_bags_uncertain` | Associated bags that could not be classified as large or daypack (bags, not people); blank when `--large-bags` is off. |
| `backpacks` | Verified backpack objects. Different from people carrying a backpack and from large bags. |
| `cars_trucks_buses` | Verified cars, trucks, and buses; a subset of `other_vehicles`. Do not add the two columns. |
| `kick_scooters` | Verified kick scooters. |

Model and threshold per object category:

| Category | Models (labels) | One model alone at | Additional rule |
|---|---|---|---|
| `bicycles` | YOLOE, YOLO26n (`bicycle`) | 0.25 | |
| `strollers` | YOLOE (`baby stroller`) | 0.25 | |
| `motorcycles` | YOLOE, YOLO26n (`motorcycle`) | 0.50 | Dropped when overlapping a bicycle box (IoU ≥ 0.4). |
| `atv_utv` | YOLOE (`all-terrain vehicle`, `utility terrain vehicle`, `golf cart`) | 0.35 | |
| `other_vehicles` | YOLOE (`car`, `truck`, `bus`, `tractor`), YOLO26n (`car`, `truck`, `bus`) | 0.50 | Merged into an overlapping counted ATV/UTV (IoU ≥ 0.4). |
| `dogs` | YOLOE, YOLO26n (`dog` only) | 0.90 | Dropped when overlapping a YOLO26n or MegaDetector person box (IoU ≥ 0.7, person confidence ≥ 0.4). |
| `backpacks` | YOLOE, YOLO26n (`backpack`) | 0.25 | |
| `cars_trucks_buses` | YOLOE, YOLO26n (`car`, `truck`, `bus`) | 0.50 | Same merge rule as `other_vehicles`. |
| `kick_scooters` | YOLOE (`kick scooter`) | 0.25 | |

Dog corroboration requires dog-specific predictions from both YOLOE and YOLO26n; a single dog prediction needs confidence at least 0.90. MegaDetector `animal` never confirms the species, although its raw detections remain in evidence.

Detections from different models overlapping at IoU ≥ 0.3 are one object. Only detections at confidence ≥ 0.25, outside the info strip, are considered.

For successful rows, these partitions each sum to `people_total`:

```text
adults + children + age_unknown
dir_left + dir_right + dir_toward + dir_away + dir_stationary + dir_unclear
facing_left + facing_right + facing_toward + facing_away + facing_unclear
orientation_front + orientation_back + orientation_side + orientation_unknown
carrying_backpack_yes + carrying_backpack_no + carrying_backpack_unknown
```

`people_near` is at most `people_total`. The age expert's `vlm_adults + vlm_teens + vlm_children + vlm_age_unclear` is the model's own count of people and need not equal `people_total`.

### Count classes

Class columns turn a people count into a group-size class, `export.count_class`:

| Count | 0 | 1 | 2 | 3–4 | 5–10 | 11 or more |
|---|---|---|---|---|---|---|
| Class | `0` | `1` | `2` | `3–4` | `5–10` | `>10` |

An empty count gives an empty class. Classes summarize estimated group size; they do not make an uncertain count accurate. Historical class scores were development comparisons on reused labels, including AI-generated series labels; see the README. Use counts, not classes, for sums. The range labels use an en dash (`3–4`, `5–10`, not a hyphen) so that spreadsheet programs do not turn them into dates.

### Diagnostics

| Column | Meaning |
|---|---|
| `unverified_bicycles`, `unverified_strollers`, `unverified_motorcycles`, `unverified_atv_utv`, `unverified_other_vehicles`, `unverified_dogs`, `unverified_backpacks` | Objects found by one model at confidence ≥ 0.25 but below that model's threshold, without corroboration. Not included in the counts; worth reviewing. |
| `candidates_total` | Candidate people: groups of overlapping person boxes from YOLOE, pose, and YOLO26n at confidence ≥ 0.12. 0 when the empty-frame gate closed, because no candidates are built then. |
| `candidates_rejected` | Candidates that failed the acceptance rule. `candidates_total − candidates_rejected = people_total`. |
| `age_by_camera_calibration` | People whose adult/child label came from the camera's height calibration. |
| `age_by_relative_height` | People whose accepted child label came from relative height; relative height alone never establishes adult. |
| `camera_calibration_status` | Height calibration of this image's camera group (camera id, camera make and model, image size): `ok`; `insufficient` (fewer than 20 complete, standing people); `unreliable` (the fit was too poor to use). A group with no accepted people is `insufficient` when calibration is requested. `disabled` when both geometry age and near-zone filtering are off. Empty for `error` rows. |

### Optional age expert

`age_model` and the `vlm_*` columns are filled only when the run used `--age-model` and an answer for this photo exists (from the server or its cache; see the [README](../README.md#optional-age-expert-local-vision-language-model)); otherwise they are empty. `age_by_vlm` is empty unless per-person (`mosaic`) answers exist.

| Column | Meaning |
|---|---|
| `age_model` | Ollama model whose answer is used for this image, for example `gemma4:26b`. Empty when the expert was off, or when no answer exists for this photo (server unavailable and nothing cached, or the request failed; failed requests are retried on the next run). |
| `vlm_adults`, `vlm_teens`, `vlm_children`, `vlm_age_unclear` | `--age-mode photo` or `photo+mosaic`: the model's count of adults, teenagers, clearly pre-teen children (roughly under 12, including carried babies), and people it could not judge, over the whole photo. Not tied to detected people; they do not change `people_total`, `adults`, or `children`. |
| `age_by_vlm` | People whose `adult`/`child` label came from the age expert (`--age-mode mosaic` or `photo+mosaic`; included in `adults`/`children`). Empty in `photo` mode and without the expert; 0 when a per-person answer exists but judged nobody adult or child. |

### Facing, orientation, and carrying a backpack

| Column | Meaning |
|---|---|
| `facing_left`, `facing_right`, `facing_toward`, `facing_away`, `facing_unclear` | Apparent facing in this photo for every person, regardless of motion. Front → toward, back → away, side → left/right only with compatible pose-profile evidence. Facing is not travel direction. |
| `orientation_front`, `orientation_back`, `orientation_side`, `orientation_unknown` | Combined body orientation from PP-LCNet and pose keypoints. |
| `orientation_conflicts` | People for whom both models gave a known but different orientation; their orientation is unknown. |
| `carrying_backpack_yes`, `carrying_backpack_no`, `carrying_backpack_unknown` | Per person: PP-LCNet evidence accepts yes/no only above its confidence and crop-detail gates. An associated backpack detection is spatial support, not proof of carrying; classifier abstention stays unknown, and a negative classifier contradicted by association stays unknown. See `associated_backpack` and `backpack_source` in `persons_json`. |

### Native attribute and pose outputs

These fields are retained for manual comparison, independently of the primary unknown age defaults:

| Column | Meaning |
|---|---|
| `attribute_age_under18`, `attribute_age_18_60`, `attribute_age_over60`, `attribute_age_unknown` | PP-LCNet native age-bin labels across accepted people, including its confidence/detail abstentions. These bins are not the package's adult/child definitions and do not set primary ages. |
| `attribute_orientation_front`, `attribute_orientation_back`, `attribute_orientation_side`, `attribute_orientation_unknown` | PP-LCNet body-orientation labels across accepted people. |
| `attribute_presentation_proxy_feminine`, `attribute_presentation_proxy_masculine`, `attribute_presentation_proxy_unclear` | Unvalidated presentation proxy from a native binary classifier. Not gender identity or biological sex, and not used in combined decisions. |
| `pose_people_total`, `pose_direction_<d>`, `pose_orientation_<o>` | Pose model's own person count, apparent-facing buckets, and orientation buckets. Pose detections are not the canonical combined person roster. |
| `combined_source_<category>` | Rule/source used for the combined category count. |
| `age_status`, `large_bags_status` | Whether the corresponding output abstains, is disabled, or uses an enabled experimental method. |
| `events_enabled`, `geometry_age_enabled`, `large_bags_enabled` | Explicit per-run experiment switches; false by default. |

### Per-model counts

For each category below, the schema has three columns:

```text
yoloe_<category>
yolo26n_<category>
megadetector_<category>
```

| Category | YOLOE | YOLO26n | MegaDetector |
|---|---|---|---|
| `people_total` | Yes | Yes | Yes |
| `bicycles` | Yes | Yes | Unsupported |
| `strollers` | Yes | Unsupported | Unsupported |
| `motorcycles` | Yes | Yes | Unsupported |
| `atv_utv` | Yes | Unsupported | Unsupported |
| `other_vehicles` | Yes | Unsupported | Unsupported |
| `dogs` | Yes | Yes | Unsupported; its animal class is generic |
| `backpacks` | Yes | Yes | Unsupported |
| `cars_trucks_buses` | Yes | Yes | Unsupported |
| `kick_scooters` | Yes | Unsupported | Unsupported |

These are each model's own detections at confidence ≥ 0.25 outside the info strip, after removing near-identical boxes (IoU ≥ 0.8) that compete for vehicle labels within one group: all-terrain vehicle, utility terrain vehicle, and golf cart; or car, truck, bus, tractor, and motorcycle. A utility-vehicle box is never removed in favour of a truck box. Unsupported cells are empty. They are comparison values, not the main counts: for example `yoloe_people_total` counts YOLOE persons at ≥ 0.25, while `people_total` also accepts YOLOE persons from 0.18 and people found by two models. When `gated_empty` is `True`, YOLOE did not run and every `yoloe_*` cell is empty, not zero. MegaDetector's generic animal and vehicle totals are in `expert_evidence_json`.

### Vote and comparison columns

Each of the ten categories also has:

| Column pattern | Meaning |
|---|---|
| `vote_<category>` | Integer strict-majority count; empty if no majority. At least two agreeing models and strictly more than half of eligible models are required. |
| `mean_<category>` | Arithmetic mean of available eligible model counts, rounded to six decimals. May be fractional. |
| `vote_status_<category>` | `majority`, `single_expert`, `no_majority`, or `no_eligible_output`. |
| `vote_model_count_<category>` | Number of eligible supported model outputs. |
| `vote_spread_<category>` | Maximum minus minimum eligible count. A single model has spread zero, which does not imply agreement. |

For people, YOLOE, YOLO26n, and MegaDetector are eligible; for other categories, YOLOE and YOLO26n. A model skipped by the empty-frame gate is not eligible. The votes are kept for comparison with version 1 and to flag disagreement (`needs_review`); they do not set the main counts.

### JSON evidence cells

#### `persons_json`

A list with one entry per accepted person (its length equals `people_total`):

| Key | Meaning |
|---|---|
| `id` | Candidate id (`cand_0001`, ...), local to this image. |
| `box` | `[x1, y1, x2, y2]` in pixels, rounded. |
| `conf` | Highest confidence among the models that found the person. |
| `experts` | Confidence per model that found the person, for example `{"yoloe": 0.62, "pose": 0.4}`. |
| `age`, `age_method` | `adult`/`child`/`unknown`; `camera_calibration`, `relative_height`, `vlm` (the optional age expert in `mosaic` mode), or `disabled`/`none`. |
| `height_ratio` | Measured height divided by the expected (calibrated) or reference height; `null` when not measured. Kept from the geometry also when the age came from the age expert. |
| `near`, `near_source` | Whether the person is in the near zone (counted in `people_near`); the reference height used: `camera_calibration` or `height_at_similar_depth`; with near fraction zero everyone is near regardless of that reference. |
| `height_px`, `complete` | Measured standing height in pixels; whether the measurement was complete enough for age. |
| `direction`, `direction_source` | Final direction and `motion`, `facing`, or `disabled`/`none`. |
| `lateral`, `radial` | Motion only: sideways foot movement in body heights; logarithm of the height change (positive = approaching). |
| `facing`, `orientation` | Apparent facing and body orientation in this photo. |
| `backpack` | Carrying evidence: `true`, `false`, or `null` (unknown). Spatial association alone does not establish carrying. |
| `associated_backpack`, `backpack_source` | Separate spatial support and the rule/source used for carrying evidence. |
| `attributes` | Full native PP-LCNet output record, including scores, native labels, detail gates and presentation proxy. Unvalidated age/presentation estimates remain expert evidence only. |
| `pose` | Native pose `orientation`, `facing_direction`, and confidence. |
| `track_ambiguous`, `track_review_reasons` | Experimental association review flags; only meaningful with events enabled. |
| `large_bag` | `true` (large), `false` (daypack only), or `null` (no bag seen, or uncertain). |
| `track` | Track id within the event (`<event_id>__t001`, ...). The same track id in several rows is one matched person; it is not an identity across events. |

#### `objects_json`

An object keyed by category (only categories with at least one detection). Each object has `field`, `status`, `confidence`, `xyxy`, `experts` (models that found it), and `label`. `status` is `corroborated` or `single_expert_confident` (both counted), `unverified` (counted in `unverified_*`), `vetoed` (dropped by an overlap rule), or a vehicle-label merge status such as `merged_into_atv_utv`. Vehicle reconciliation runs before `cars_trucks_buses` is derived from final accepted road-vehicle labels.

#### `expert_evidence_json`

An object keyed by `yolo26n`, `yoloe`, `megadetector`, and `pose`. Each model entry has its count dictionary (all `null` when the gate skipped the model), device, checkpoint, input size (`imgsz`), inference time and speed details, `skipped: "empty_frame_gate"` when the gate skipped it, and its detections at confidence ≥ 0.25 as compact lists `[label, confidence, x1, y1, x2, y2]`. Instance-mask polygons, pose keypoints, and detections below 0.25 are kept in the inference cache but not in the CSV, to keep cells small.

#### `attribute_runtime_json`

Attribute-model metadata: model, requested and actual device, GPU fallback reasons, processing time, policy notes, and any failed-person count.

#### `timings_json`

Original per-image timings: `decode_seconds`, `yolo26n_seconds`, `megadetector_seconds`, `yoloe_seconds` and `pose_seconds` (both absent when the gate skipped those models), `vision_total_seconds`, and `attributes_seconds`. Preserved on cache hits.

#### `notes`

A compact interpretation reminder repeated on each row.

### Complete column order

```text
run_id image_id relative_path sha256 camera_id capture_time time_source
clock_suspect sequence_number event_id event_frame event_image_count status error
cache_hit analyzed_at_utc width height profile vision_device attribute_device
analysis_seconds current_run_seconds configuration_hash gated_empty data_strip_bottom needs_review people_total
people_near people_class people_near_class adults children age_unknown dir_left
dir_right dir_toward dir_away dir_stationary dir_unclear direction_from_motion direction_from_facing
bicycles strollers motorcycles atv_utv other_vehicles dogs backpacks
large_bags large_bags_uncertain cars_trucks_buses kick_scooters unverified_bicycles unverified_strollers unverified_motorcycles
unverified_atv_utv unverified_other_vehicles unverified_dogs unverified_backpacks age_model vlm_adults vlm_teens
vlm_children vlm_age_unclear age_by_vlm candidates_total candidates_rejected age_by_camera_calibration age_by_relative_height
camera_calibration_status facing_left facing_right facing_toward facing_away facing_unclear orientation_front
orientation_back orientation_side orientation_unknown orientation_conflicts carrying_backpack_yes carrying_backpack_no carrying_backpack_unknown
yoloe_people_total yoloe_bicycles yoloe_strollers yoloe_motorcycles yoloe_atv_utv yoloe_other_vehicles yoloe_dogs
yoloe_backpacks yoloe_cars_trucks_buses yoloe_kick_scooters yolo26n_people_total yolo26n_bicycles yolo26n_strollers yolo26n_motorcycles
yolo26n_atv_utv yolo26n_other_vehicles yolo26n_dogs yolo26n_backpacks yolo26n_cars_trucks_buses yolo26n_kick_scooters megadetector_people_total
megadetector_bicycles megadetector_strollers megadetector_motorcycles megadetector_atv_utv megadetector_other_vehicles megadetector_dogs megadetector_backpacks
megadetector_cars_trucks_buses megadetector_kick_scooters vote_people_total mean_people_total vote_status_people_total vote_model_count_people_total vote_spread_people_total
vote_bicycles mean_bicycles vote_status_bicycles vote_model_count_bicycles vote_spread_bicycles vote_strollers mean_strollers
vote_status_strollers vote_model_count_strollers vote_spread_strollers vote_motorcycles mean_motorcycles vote_status_motorcycles vote_model_count_motorcycles
vote_spread_motorcycles vote_atv_utv mean_atv_utv vote_status_atv_utv vote_model_count_atv_utv vote_spread_atv_utv vote_other_vehicles
mean_other_vehicles vote_status_other_vehicles vote_model_count_other_vehicles vote_spread_other_vehicles vote_dogs mean_dogs vote_status_dogs
vote_model_count_dogs vote_spread_dogs vote_backpacks mean_backpacks vote_status_backpacks vote_model_count_backpacks vote_spread_backpacks
vote_cars_trucks_buses mean_cars_trucks_buses vote_status_cars_trucks_buses vote_model_count_cars_trucks_buses vote_spread_cars_trucks_buses vote_kick_scooters mean_kick_scooters
vote_status_kick_scooters vote_model_count_kick_scooters vote_spread_kick_scooters age_status large_bags_status events_enabled geometry_age_enabled
large_bags_enabled attribute_age_under18 attribute_age_18_60 attribute_age_over60 attribute_age_unknown attribute_orientation_front attribute_orientation_back
attribute_orientation_side attribute_orientation_unknown attribute_presentation_proxy_feminine attribute_presentation_proxy_masculine attribute_presentation_proxy_unclear pose_people_total pose_direction_left
pose_direction_right pose_direction_toward pose_direction_away pose_direction_unclear pose_orientation_front pose_orientation_back pose_orientation_side
pose_orientation_unknown combined_source_people_total combined_source_bicycles combined_source_strollers combined_source_motorcycles combined_source_atv_utv combined_source_other_vehicles
combined_source_dogs combined_source_backpacks combined_source_cars_trucks_buses combined_source_kick_scooters persons_json objects_json expert_evidence_json
attribute_runtime_json timings_json notes
```

## Events CSV

Written only with `--events`. Backpacks are summarized as the maximum accepted object count in a frame, just like the other objects.

One row per event: a run of images from one camera in which consecutive capture times are at most `--event-gap` seconds apart (default 60). Images with status `error` are not part of any event.

| Column | Meaning |
|---|---|
| `run_id` | As in the images CSV. |
| `event_id` | `<camera>__<first capture time>`: sanitized camera id (`root` for `.`) and the recorded time of the earliest image as `YYYY-MM-DDTHH-MM-SS` (the first filename when no image has a time). The id depends only on the event itself, so an event with the same images has the same id in every export; adding images to the event, or next to it within the event gap, can change it. When two events would share an id, the later ones in camera and time order get `__02`, `__03`, and so on. |
| `camera_id` | Camera of all images in the event. |
| `start`, `end` | Recorded capture time of the earliest and latest image (ISO format). Empty when no image has a time. |
| `duration_seconds` | `end − start`; 0 for a single image. |
| `image_count` | Number of images in the event; blank when events are disabled. |
| `people_unique` | Number of tracks: people matched across the event's images by appearance, size, and position, each counted once. The best estimate of different people, with about 25% error per visit on the validation series (see below); undercounts when similar-looking people are merged, overcounts when one person is split. |
| `people_max_frame` | Largest `people_total` of any single image in the event (MaxN). A comparison estimate; may undercount dispersed groups or overcount false detections. It is not a guaranteed lower bound. |
| `people_near_max_frame` | Largest `people_near` of any single image in the event. |
| `people_near_unique` | Tracks with at least one observation in the near zone. At most `people_unique`. |
| `people_unique_class`, `people_near_unique_class` | Class of `people_unique` and of `people_near_unique` (see [Count classes](#count-classes)). |
| `adults`, `children`, `age_unknown` | Per track: child when at least one observation is a child and none is an adult; adult in the reverse case; otherwise unknown. They sum to `people_unique`. With `--age-mode mosaic`, the observations carry the age expert's per-person answers. |
| `vlm_adults_max_frame`, `vlm_teens_max_frame`, `vlm_children_max_frame`, `vlm_unclear_max_frame` | Optional age expert, `photo` mode: the largest `vlm_adults`, `vlm_teens`, `vlm_children`, and `vlm_age_unclear` of any image in the event, each taken separately (they can come from different images). An estimate per visit, not a guaranteed lower bound. Empty when the expert answered for no image of the event. |
| `dir_left`, `dir_right`, `dir_toward`, `dir_away`, `dir_stationary`, `dir_unclear` | One direction per track; they sum to `people_unique`. |
| `direction_from_motion`, `direction_from_facing` | Tracks whose direction came from motion (two or more observations) or from facing (usually one observation). |
| `bicycles`, `strollers`, `motorcycles`, `atv_utv`, `other_vehicles`, `dogs`, `backpacks`, `large_bags`, `large_bags_uncertain` | Maximum over the event's images of the images CSV value. |
| `needs_review` | `True` when `people_unique` differs from `people_max_frame` by more than half of `people_max_frame`, when any track is ambiguous, or when an event of two or more images uses file-date or missing capture times. |
| `review_reasons` | Semicolon-separated: `people_unique_vs_max_frame`; `competing_candidate` (a rival match was almost as good as the chosen one); `near_gate_candidate` (a person started a new track although a match only just failed the limits); `ambiguous_track`; `times_from_file_dates` (at least one image's time came from the file modification date or is missing; copied files often share modification dates, so the event may combine separate visits). |
| `images` | JSON list of the event's image `relative_path` values in capture order. When the list would exceed 32,000 characters (spreadsheet cell limit), the cell holds `{"first": ..., "last": ..., "count": ..., "truncated": true}` instead. Join to the images CSV by `event_id` to get every image. |
| `notes` | Interpretation reminder. |

Tracks link people only between images of the same event, one image to the next, with a missing image in between allowed. Byte-identical copies of a photo in different folders are separate images, each in its own camera's events. A person returning after the event gap starts a new event and is counted again.

Historical development runs on 28 AI-labelled series had unique-count MAE 5.04 and MaxN MAE 6.36, with mean relative errors of 24.6% and 25.3%. Those labels were used during association experiments and do not validate the revised rules on new cameras. Review both estimates and flagged associations; do not interpret their range as confidence bounds. The revised tracker splits events when camera metadata/image dimensions change, rejects implausible motion links, and flags missing appearance evidence instead of treating it as reliable motion.

## Camera-days CSV

Written only with `--events`. Disabled large-bag size estimates remain blank rather than zero.

One row per camera and capture date, adding up the events whose `start` falls on that date (camera clock). An event crossing midnight counts on its start date; events without a capture time use the date `unknown`.

| Column | Meaning |
|---|---|
| `run_id` | As in the images CSV. |
| `camera_id`, `date` | Camera and `YYYY-MM-DD` date of the event start. |
| `events` | Number of events. |
| `images` | Number of images in those events. |
| `people_max_frame` | Sum of the events' `people_max_frame` (MaxN). A daily sum of model estimates, not a guaranteed lower bound. |
| `people_unique` | Sum of the events' `people_unique`. The best estimate, with the event values' error (about 25% per visit on the validation series). |
| `people_near_max_frame`, `people_near_unique` | Sums of the events' near-zone counts: the same two estimates for people passing the camera. |
| `adults`, `children`, `age_unknown` | Sums of the events' track-based age counts. |
| `dir_left`, `dir_right`, `dir_toward`, `dir_away`, `dir_stationary`, `dir_unclear` | Sums of the events' track directions. |
| `bicycles`, `strollers`, `motorcycles`, `atv_utv`, `other_vehicles`, `dogs`, `backpacks`, `large_bags` | Sums of the events' per-event maxima: an object seen in two separate events is counted twice. |
| `events_needing_review` | Number of those events with `needs_review = True`. |

## Run JSON

`<run>_run.json` records `version` (`2.0.0`), input folder, output file names, `event_count`, `camera_calibrations` (per camera group, including groups without people: status, number of people, fitted coefficients, residual spread, inliers, and reasons), `postprocess_settings` (camera-id mode and pattern, event gap, near fraction and experimental enabled flags), image, cache, and error totals (`images_with_errors` includes `partial_error` rows; `images_with_postprocess_errors` counts post-processing failures), model start-up and total elapsed time, the configuration fingerprint and its inputs, and the contact-sheet selection. Keep it beside its CSVs when comparing runs or moving exports to another computer.

## Changes from version 1

| Version 1 column(s) | Version 2 |
|---|---|
| `combined_people_total` (YOLOE roster at confidence 0.25) | `people_total` (multi-model roster; see above) |
| `combined_<category>` | `<category>` (verified objects) |
| `combined_adults`, `combined_children`, `combined_age_unknown` (always unknown) | `adults`, `children`, `age_unknown` (still unknown by default; geometry/per-person age experts require opt-in) |
| `combined_direction_<d>` (apparent facing) | `facing_<d>`. The new `dir_<d>` columns are travel direction (motion, else facing) and add `stationary`. |
| `combined_orientation_<o>`, `combined_carrying_backpack_<b>` | `orientation_<o>`, `carrying_backpack_<b>` |
| `combined_source_<category>`, `age_status` | Retained; also see `objects_json` statuses and `persons_json` |
| `presentation_status` | Native presentation remains explicitly unvalidated; no combined gender classification is produced |
| `attribute_age_*`, `attribute_orientation_*`, `attribute_presentation_proxy_*` | Retained in the CSV; full native records also remain in `persons_json.attributes` |
| `pose_people_total`, `pose_direction_*`, `pose_orientation_*` | Retained alongside each accepted person's pose evidence |
| `yoloe_*`, `yolo26n_*`, `megadetector_*`, `vote_*`, `mean_*`, `vote_status_*`, `vote_model_count_*`, `vote_spread_*` | Unchanged meaning (YOLOE now runs at 1280 pixels in the standard profile; `yoloe_*` is empty when the empty-frame gate skipped YOLOE) |
| `needs_review` | Narrower: no longer set for every image with people |
| — | New: camera, time, and event columns, `profile`, `gated_empty`, `data_strip_bottom`, `people_near`, `people_class`, `people_near_class`, `direction_from_*`, `large_bags*`, `unverified_*`, `candidates_*`, `age_by_*`, `camera_calibration_status`, the optional age expert's `age_model` and `vlm_*` columns, `objects_json`; events and camera-days CSVs |

To score a version 1 export with `python -m trailcam.evaluate`, pass `--prefix combined_`.

## Reading the CSVs in Python

The event-reading part of the example requires an export made with `--events`.

Python's default CSV field-size limit may be smaller than a crowded image's evidence cell. Increase it before loading, and replace `DATE` with the timestamp of your chosen run:

```python
import csv
import json

csv.field_size_limit(100_000_000)
folder = "exports/export_DATE/"
with open(folder + "export_DATE.csv", encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        if row["status"] not in {"ok", "partial_error"}:
            continue
        persons = json.loads(row["persons_json"])
        print(row["relative_path"], row["event_id"], row["people_total"], row["people_near"], len(persons))

with open(folder + "export_DATE_events.csv", encoding="utf-8-sig", newline="") as handle:
    events = list(csv.DictReader(handle))
print("MaxN total:", sum(int(e["people_max_frame"]) for e in events))
print("Near-zone estimates:", sum(int(e["people_near_max_frame"]) for e in events),
      "to", sum(int(e["people_near_unique"]) for e in events))
for event in events:
    images = json.loads(event["images"])  # relative paths, or a dict when truncated
    if isinstance(images, dict):
        print(event["event_id"], "has", images["count"], "images; join by event_id for the full list")

with open(folder + "export_DATE_camera_days.csv", encoding="utf-8-sig", newline="") as handle:
    for day in csv.DictReader(handle):
        print(day["camera_id"], day["date"], "MaxN:", day["people_max_frame"])
```
