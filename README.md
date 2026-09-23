# Trail Camera Expert Export

Analyze a selected folder of trail-camera photographs locally. Each default run writes **one CSV**, a run manifest, and a contact sheet of up to **20 diverse images**, together in a separate dated export folder. Object detectors, masks, pose keypoints and a small attribute classifier count people, bicycles, strollers, motorcycles, ATVs/UTVs, other vehicles, dogs and **backpacks**. Individual expert outputs are retained alongside combined results for comparison.

**No Gemma, Ollama, account, or image upload is required.** An available NVIDIA GPU is used automatically, with CPU fallback. All four vision models run by default. Apparent facing is inferred from each image; it is not proof of travel direction. Primary adult/child ages remain **unknown** by default because the lightweight age methods are not sufficiently validated.

Event tracking, geometry-based age and large-bag size are experimental **opt-in** features. An optional Ollama age expert and the empty-frame speed gate are also off by default. This is an analysis and manual-validation tool, not verified visitor or demographic statistics. See [Validation evidence](#validation-evidence).

## Quick start on Windows

1. Install **64-bit Python 3.12** or 3.11. If Python is missing, this command installs the recommended version:

   ```powershell
   winget install -e --id Python.Python.3.12
   ```

   Alternatively, use the [official Python installer](https://www.python.org/downloads/windows/). Reopen the terminal after installation.

2. Download the repository using **Code → Download ZIP**, then extract it, or clone it:

   ```powershell
   git clone https://github.com/victorchina-ops/trailcam-expert-export.git
   cd trailcam-expert-export
   ```

3. Put your photographs in the repository's `images` folder, **one subfolder per camera placement** (for example `images\North gate\` and `images\Spring trail\`). Subfolders are included by default. See [Cameras, events, and de-duplication](#cameras-events-and-de-duplication).

4. Double-click **`run_analysis.cmd`**, or run it from PowerShell to keep progress and error messages visible:

   ```powershell
   .\run_analysis.cmd
   ```

The first run creates a local `.venv`, installs the tested dependencies, downloads and verifies the model weights, and analyzes the images. It does not modify your global Python packages. The first setup needs an internet connection and can take considerably longer than image analysis. Models require approximately 312 MB of downloads; Python dependencies, particularly GPU PyTorch, need additional disk space.

Subsequent runs keep compatible installed packages, reuse the downloaded models, and analyze only new or changed image content. **Add more images and run the same file again.** Each run creates a separate dated export folder without overwriting older exports.

Windows with Python 3.12 is the tested platform. Python 3.11 is accepted by the installer. The Python entry point can also be invoked on other supported platforms, but those operating systems have not been validated end to end for this package.

**Upgrading from version 1:** the first version 2 run analyzes every image again, because the inference cache format changed. Several CSV columns were renamed or replaced; see [changes from version 1](docs/CSV_SCHEMA.md#changes-from-version-1). Existing exports are left in place.

## What each run writes

By default, each run creates its own dated subfolder inside `exports`, keeping that run's files together:

```text
exports/
  export_2026-09-22_14-30-15_123456/
    export_2026-09-22_14-30-15_123456.csv               one row per image
    export_2026-09-22_14-30-15_123456_run.json
    export_2026-09-22_14-30-15_123456_contactsheet.jpg
```

The folder and filenames share a local date/time timestamp that includes microseconds, so multiple runs on the same day get separate folders. Inference timestamps inside the CSV use UTC.

- **Images CSV:** one row for every image currently in the selected input folder, including cached results. Counts are **appearances in that photograph**: a person photographed three times in a burst appears in three rows. `people_total` counts everyone visible and `people_near` the people in the near zone (see [Counting people](#counting-people-everyone-visible-or-the-near-zone)); `people_class` and `people_near_class` give them as classes. Each row also carries the camera id, capture time, event id, per-model counts, per-person details, evidence, and, when the optional age expert is on, its age-group counts (`vlm_*`).
- **Events CSV, only with `--events`:** one row per event (a run of photos from one camera with short gaps between them). Use this experimental file to review possible de-duplication. `people_max_frame` is the largest number of people in any single photo of the event (MaxN); `people_unique` counts people matched across the photos (the best estimate, with about 25% error per visit on the validation series). Near-zone versions (`people_near_max_frame`, `people_near_unique`) and classes (`people_unique_class`, `people_near_unique_class`) are included. Object columns are the maximum over the event's photos. The `images` column lists the event's photos by relative path.
- **Camera-days CSV, only with `--events`:** event totals added up per camera and capture date. Its `people_max_frame` (the sum of the events' MaxN) is the sum of event MaxN estimates; `people_unique` and the near-zone counts are added up the same way.
- **Run JSON:** configuration and model hashes, image/error/cache totals, elapsed time, number of events, the per-camera height calibrations used for adult/child estimates and the near zone, post-processing settings (including the near fraction), and the contact-sheet panel-to-filename mapping.
- **Contact sheet:** YOLOE boxes (confidence at least 0.25) with translucent instance masks, pose keypoints, and a white box with a short label for each accepted person: number, age (`A` adult, `C` child, `?` unknown), direction (`L`/`R`/`T`/`A`/`S` = left/right/toward/away/stationary) followed by `m` (from motion) or `f` (from facing), and `B` for a large bag. Selection favors new/changed images and varied objects, group sizes, directions, children, large bags, multi-photo events, and model disagreements. This is a review sample, not a random sample for estimating accuracy.

The CSVs use UTF-8 with a BOM for spreadsheet compatibility. Large JSON evidence cells can exceed Excel's 32,767-character cell limit; the CSV retains the full text, so use Python or another CSV reader when inspecting detailed evidence. See [CSV schema and interpretation](docs/CSV_SCHEMA.md) for every column.

## Select an input folder and options

Drag a folder onto `run_analysis.cmd`, or pass a quoted folder path:

```powershell
.\run_analysis.cmd "D:\Trail Photos\Camera 1"
```

Use explicit input and output options when desired:

```powershell
.\run_analysis.cmd --input "D:\Trail Photos" --output "D:\Trail Exports"
```

`--output` selects the parent export directory; the program creates a new dated subfolder inside it for each run. In the example above, all files for one run appear under `D:\Trail Exports\export_YYYY-MM-DD_HH-MM-SS_microseconds\`. The `output_dir` setting below has the same meaning.

Relative paths are resolved from the repository folder. Input, output, cache, and model folders must be distinct. Generated output/model/cache folders are excluded from recursive image discovery.

| Option | Behavior |
|---|---|
| `--no-contact-sheet` | Skip contact-sheet export for this run. Enabled by default. |
| `--contact-sheet-size 20` | Maximum review panels; accepts 1–100. |
| `--device cpu` | Force CPU inference, including the attribute model. |
| `--device auto` | Use a working NVIDIA GPU when available; otherwise CPU. Default. |
| `--device cuda` | Try CUDA with the same safe CPU fallback if unavailable. |
| `--threads 8` | CPU inference thread setting. Default 8. |
| `--profile standard` | Model input sizes: `standard` (default) or `fast` for slow CPUs. See [Profiles, GPU, and CPU](#profiles-gpu-and-cpu). |
| `--empty-frame-gate` | Opt in to skipping YOLOE and pose when the fast detectors find nothing. Off by default; may miss objects. `--no-empty-frame-gate` restores full analysis. |
| `--geometry-age` | Enable experimental height-based age estimates. Off by default; `--no-geometry-age` restores abstention. |
| `--large-bags` | Enable experimental bag-size estimates. Off by default; `--no-large-bags` leaves size columns blank. Backpacks remain detected. |
| `--events` | Enable experimental tracking, motion direction, and events/camera-days CSVs. Off by default; `--no-events` exports per-image results only. |
| `--camera-id folder` | How photos are assigned to cameras: `folder` (parent folder, default), `regex` (with `--camera-pattern`), or `single` (all photos are one camera). |
| `--camera-pattern "..."` | Regular expression with a named group `camera`, searched in each photo's relative path. Used only with `--camera-id regex`. |
| `--event-gap 60` | Seconds, a positive number. A longer gap between consecutive photos of one camera starts a new event. Default 60. |
| `--near-fraction 0.07` | Near zone for `people_near`: a person is near when an adult standing at their position would be at least this fraction of the image height tall. From 0 up to (not including) 1; 0 counts everyone, so `people_near` equals `people_total`. Default 0 (all visible people). A nonzero fraction is experimental. See [Counting people](#counting-people-everyone-visible-or-the-near-zone). |
| `--age-model gemma4:26b` | Switch on the optional age expert with this local Ollama vision model. Off by default. See [Optional age expert](#optional-age-expert-local-vision-language-model). |
| `--age-mode photo` | With `--age-model`: `photo` (age-group counts per photo; default), `mosaic` (an age for each detected person), or `photo+mosaic` (both). |
| `--ollama-host URL` | Address of the Ollama server used by the age expert. Default `http://localhost:11434` (this computer). |
| `--no-recursive` | Analyze only files directly in the input folder. |
| `--export-new-only` | The images CSV includes only images analyzed in this run. Events and camera-days CSVs still cover the whole folder. |
| `--force` | Reanalyze all selected images instead of reusing inference results. |
| `--offline` | Prohibit dependency/model downloads; requires completed setup. |
| `--models "D:\Models"` | Use another model-cache directory. |
| `--cache "D:\Trail Cache"` | Use another inference-cache directory. |

For example, disable the contact sheet and export only newly analyzed images:

```powershell
.\run_analysis.cmd --no-contact-sheet --export-new-only
```

`--export-new-only` affects the images CSV. If a contact sheet is enabled, it still samples the current folder inventory and may include cached images to fill the requested panel count.

Changing `--device`, `--profile`, or `--empty-frame-gate` analyzes the images again, because cached results are tied to those settings. Changing `--camera-id`, `--camera-pattern`, `--event-gap`, `--near-fraction`, `--events`, `--geometry-age`, or `--large-bags` does not: camera ids, events, the near zone, and adult/child estimates are recomputed from cached results on every run. The age expert's answers are cached separately, per model and mode, so switching `--age-model` on or off never re-runs the detectors, and a later run with the same model asks only about new photos.

Set persistent defaults in `settings.json`:

```json
{
  "input_dir": "images",
  "output_dir": "exports",
  "device": "auto",
  "threads": 8,
  "recursive": true,
  "contact_sheet": true,
  "contact_sheet_size": 20,
  "profile": "standard",
  "empty_frame_gate": false,
  "geometry_age": false,
  "large_bags": false,
  "events": false,
  "camera_id_mode": "folder",
  "camera_id_pattern": null,
  "event_gap_seconds": 60,
  "near_fraction": 0,
  "age_model": null,
  "age_mode": "photo",
  "ollama_host": "http://localhost:11434"
}
```

Command-line flags override these defaults. For example, `"contact_sheet": false` disables the contact sheet by default and `--contact-sheet` re-enables it for one run; `"empty_frame_gate": true` has the command-line counterpart `--empty-frame-gate`. The `events`, `geometry_age`, and `large_bags` switches also default to false. `"age_model": null` keeps the age expert off; `"age_model": "gemma4:26b"` switches it on for every run.

Values in `settings.json` are checked like command-line values before any image is analyzed: `device`, `profile`, `camera_id_mode`, and `age_mode` must be one of the listed choices, `threads` and `contact_sheet_size` whole numbers (`8`, not `"8"`), the switches `true` or `false`, `event_gap_seconds` a finite positive number of seconds, and `near_fraction` a number from 0 up to (not including) 1. An invalid value stops the run with a message naming the setting.

A regular expression is easiest to set in `settings.json`, because the Windows command shell treats `<` and `>` as redirection characters. In `settings.json`, write backslashes twice (`\\d`), or use `[0-9]` instead. When passing a pattern to `run_analysis.cmd` from PowerShell, wrap it in both kinds of quotes: `--camera-pattern '"^(?P<camera>[^/]+)/"'`.

Supported filename extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, and `.webp`, case-insensitively. The pipeline analyzes still images; for a multi-frame image file, only the initially opened frame is analyzed. Symlinked files and symlinked subfolders are skipped. Images are decoded with EXIF orientation applied before detecting or cropping; 16-bit and floating-point greyscale images are rescaled to 8 bits. Filenames in any language are supported; characters the console cannot display are shown as escape codes in the progress messages, and the CSVs keep the real names.

## Counting people: everyone visible or the near zone

What counts as "a person in the photo" changes the numbers more than the choice of model. The camera owner's labels count **the group passing the camera**; the independent validation labels count **every visible person**. On the same 100 photos the two differed on 33 photos, almost always because of distant background people: for example, 3 passers-by in front of a far plaza with about 20 people. The export therefore reports both:

| Column | Counts | Use it for |
|---|---|---|
| `people_total` | Every accepted person anywhere in the photo, including distant and background people | How many people are in view, for example crowding at a viewpoint or plaza |
| `people_near` | Accepted people in the **near zone** | People passing the camera: trail use and visitor counts |

By default `--near-fraction 0` makes `people_near` equal `people_total`. To experiment with near-zone filtering, use `--near-fraction 0.07`: a person is near when **an adult standing where that person's feet are would be at least 7% of the image height tall**. When filtering is enabled, the expected standing height comes from the camera's height calibration (see [Adults and children](#adults-and-children)) when it is `ok` and the feet are inside the part of the picture it was fitted on, so a seated person or a child close to the camera still counts as near. Otherwise it is the tallest height measured among the people standing at the same depth or farther away (feet at about the same image row or higher, within 2% of the image height), including the person themselves, so a child walking in front of a near adult is near too. The run JSON flags a camera with `check_near_fraction` when it has at least 10 people but none in the near zone. In `persons_json`, `near` says whether each person was near and `near_source` which reference was used (`camera_calibration` or `height_at_similar_depth`).

Which count to use:

- For an explicitly defined near-zone experiment, compare **`people_near`** per photo and `people_near_max_frame` / `people_near_unique` per event and camera-day. Against the owner's labels it was clearly closer than `people_total`: mean absolute error 0.73 against 0.90 (version 1: 0.89) in the final saved development export. This is an in-sample comparison, not a guarantee for new cameras.
- Use `people_total` when everyone in view matters.
- Decide before the analysis, keep the same `--near-fraction` for the whole study, and state the definition with the results. `--near-fraction 0` makes `people_near` equal to `people_total`.

The experimental 7% setting was chosen because it matched the owner's labels best on those same 100 photos, so the figures above are optimistic. The zone is a fraction of the image height: a camera mounted higher or with a wider lens shows people smaller at the same distance, so the same fraction reaches a different distance. Check it on a labelled sample from your own cameras ([docs/VALIDATION.md](docs/VALIDATION.md#near-zone-counts-classes-and-the-age-expert)). Without an `ok` calibration, a partly hidden person standing apart from others is judged by the visible part of the body and may be left out of the near zone. Changing the near fraction does not require re-analysis.

### Count classes

Every people count also comes as a group-size class: `0`, `1`, `2`, `3–4`, `5–10`, or `>10`. The images CSV has `people_class` (of `people_total`) and `people_near_class`; the events CSV has `people_unique_class` and `people_near_unique_class`. A class is empty when its count is empty.

Counts of groups are often off by a person or two, which usually leaves the class unchanged. Per photo, `people_near_class` matched the class of the owner's label for **79%** of the 100 photos and was within one class for **98%** (`people_total`: 72%; version 1: 68%); most misses were neighbouring classes (2 against 3–4, or 3–4 against 5–10). Per visit, `people_unique_class` matched for **82%** of the 28 validation series, against 71% for MaxN. The classes widen with group size because counting errors grow with group size. Use classes to describe group sizes, for example the share of visits by groups of 3–4; use the counts for totals, because classes cannot be added up (the camera-days CSV therefore has counts only).

## Cameras, events, and de-duplication

**Experimental and off by default.** Enable `--events` to write events and camera-days CSVs. Event columns in the image CSV are blank otherwise. The following workflow applies when enabled.

### Camera ids

Every photo is assigned a camera id. Events never combine photos from different cameras, and the height calibration used for adult/child estimates and the near zone is fitted per camera.

| Mode | Camera id | Example: `North gate/2025-05 card/IMG_0042.JPG` |
|---|---|---|
| `folder` (default) | The photo's parent folder, relative to the input folder; `.` for photos directly in the input folder | `North gate/2025-05 card` |
| `regex` | The named group `camera` of the first match of `--camera-pattern` in the relative path; the folder when the pattern does not match | With `^(?P<camera>[^/]+)/`: `North gate` |
| `single` | `all` for every photo | `all` |

Relative paths use `/` separators, also on Windows. Keep **one folder per camera placement**. If a camera was moved or re-aimed, put the later photos in a new folder (or choose a pattern that separates them), because the height calibration describes one view. Photos from different camera models or image sizes in one folder never share a calibration.

### Capture times

The capture time comes from the photo's EXIF data (`DateTimeOriginal` with sub-seconds, then `DateTimeDigitized`, then `DateTime`), otherwise from the file's modification time. The `time_source` column says which was used. Times are the camera's own clock without a time zone. `clock_suspect` is `True` for a year before 2015, a time more than one day after the file's modification time, or a time within ten minutes after midnight on 1 January (a typical reset default); it is informational only. Photo-management tools that remove EXIF data force the less reliable file times. Capture metadata is read from each file on every run, also for cached images, so a renamed file or a corrected file date takes effect without reanalysis.

Copying files often gives them the same modification time. An event of several photos in which any photo has a file time (or no time) is therefore marked `needs_review` with the reason `times_from_file_dates`: it may combine separate visits.

### Events

For each camera, photos are sorted by capture time (then by the number in the filename, for example `IMAG0571` → 571). A new event starts when the gap to the previous photo is longer than `--event-gap` (default 60 seconds). If a photo has no usable time, neighbouring photos stay together only when their filename numbers differ by at most 3.

Within an event, people in consecutive photos are matched when their clothing colours (upper and lower body), apparent size, and position are sufficiently compatible. Physically implausible jumps are rejected. Camera metadata or image-size changes split events. Missing appearance evidence is flagged and cannot establish motion direction. The matching is solved as an assignment problem (Hungarian method, using SciPy, which is installed with the detector package; a greedy matcher is used if SciPy is missing). A person missing from one photo in between can still be matched to the next photo in which they appear. Each chain of matched detections is a **track**. A track only prevents double counting within one event; it is not an identification of a person, and a person who passes again later is counted in the later event again.

### MaxN versus `people_unique`

The events CSV reports two people counts, each also for the near zone (`people_near_max_frame`, `people_near_unique`) and as a class:

- **`people_max_frame` (MaxN)** is the largest number of accepted people in any single photo of the event. It is a comparison estimate, not a guaranteed lower bound: false detections or duplicate boxes can inflate it. It undercounts when a group passes in single file and different members are visible in different photos: in one validation series, no photo showed more than 8 people, but 27 different people passed.
- **`people_unique`** is the number of tracks. It includes people who never appear together in one photo, but matching can fail in both directions: one person becomes several tracks (occlusion, night infrared photos with little colour, large jumps between photos), or two similar-looking people at the same spot in consecutive photos are merged into one. `people_near_unique` counts the tracks with at least one observation in the near zone.

**Historical development experiment, AI-generated labels.** 28 series of 3–4 consecutive photos (100 photos, 3–19 seconds apart, taken without burst mode) were labelled with the number of different people in each series. Both counts had about **25% error per visit**. `people_unique` was closer on average (mean absolute error 5.0 people against 6.4 for MaxN) and put the visit in the right class more often (82% against 71%); adding up the photos instead would have overcounted by 128%. Aggregate detections were 1,094 against 1,123 AI-labelled person-photos, which can hide offsetting errors. Many visible errors involved **linking** people between photos 5–16 seconds apart, mostly by merging similar-looking hikers walking in a line past the same spot, sometimes by splitting one person into two tracks. Looser or stricter matching limits and a different colour descriptor did not help, and a large vision-language model shown all photos of a series at once was not clearly better (22% error, class right for 82%).

Review `people_max_frame` and `people_unique` together, preferably as classes, and check events flagged `needs_review`. These are two model estimates, not statistical bounds: false detections can inflate MaxN and tracking can merge different people. Burst mode may help matching by shortening observation gaps, but this has not been validated with the package.

`needs_review` is `True` for events where the two counts differ by more than half of MaxN, where a person could plausibly have been matched to someone else, or where capture times came from file dates; `review_reasons` says which. The camera-days CSV adds up both counts per day: its `people_max_frame` is the **sum of event MaxN estimates** and `people_unique` the best estimate; the near-zone counts and the track-based ages and directions are added up the same way.

Event ids look like `North_gate_2025-05_card__2025-05-03T07-30-15`: the camera id and the first capture time of the event (the first filename when no photo has a time). An id depends only on its own event, so an event whose photos are unchanged keeps the same id in later exports, and adding photos elsewhere never renumbers it. Adding photos inside or right next to an event can change its membership and therefore its id. In the rare case that two events would get the same id (for example two untimed events starting with the same filename), the later ones get `__02`, `__03`, and so on. Use the images, events, and camera-days CSVs of one export together.

## Direction: motion versus facing

The package reports two different quantities. Both are relative to the image, not compass directions: `left` and `right` mean toward the image's left or right edge, `toward` and `away` mean approaching or moving away from the camera.

- **Motion** (`direction_source` = `motion`): when a person is matched in at least two photos of an event, the direction comes from the change between their first and last observation: the sideways shift of the foot point, measured in body heights, and the change in apparent height (growing means approaching). `toward` or `away` requires a height change of about 15% or more that dominates the sideways shift; otherwise `left` or `right` requires a shift of at least a quarter of the body height; a person who moved less than that over at least 4 seconds is `stationary`; anything else is `unclear`.
- **Facing** (`direction_source` = `facing`): a person seen in only one photo gets the apparent body facing, combined from the PP-LCNet attribute model and pose keypoints: front → `toward`, back → `away`, side → `left` or `right` only when the pose profile agrees; otherwise `unclear` (`direction_source` = `none`). Facing is not travel: a person can look or stand sideways while walking toward the camera.

In the images CSV, the `dir_*` columns use motion where available and facing otherwise; `direction_from_motion` and `direction_from_facing` say how many people each source covered. The `facing_*` columns keep the facing estimate for every person. In the events CSV, each track has one direction. On the development photos, facing counts overlapped AI-labelled direction counts by **56%**: the sum of the smaller count in each direction bucket, divided by assigned facing counts. This is aggregate count overlap, **not per-person accuracy**. On the 28 validation series, the dominant direction of travel of the passing group (the direction most tracks took) matched the labels in **20 of 28** series (see [Validation evidence](#validation-evidence)). Burst photos (see [Recommended camera settings](#recommended-camera-settings)) give more people a direction from motion.

## Adults and children

By default, `adults = 0`, `children = 0`, and `age_unknown = people_total`: zero means no accepted age assignments, not an absence of adults or children. Native PP-LCNet scores and labels remain available for comparison and do not set primary ages. Enable `--geometry-age` to try the experimental standing-height rules below. Height measurements can still support the near-zone estimate with age classification off. An optional age expert is described [below](#optional-age-expert-local-vision-language-model).

1. **Measurement.** For each accepted person with pose keypoints, the standing height in pixels is measured from the crown (head keypoints, extended upward) to the ankles. A measurement counts as complete only when head, shoulder, hip and ankle keypoints are visible, the torso is upright, the legs are extended (ankles at least 1.2 torso lengths below the hips, so seated and crouching people are excluded), and the person does not touch the image border or the camera's info strip.
2. **Camera calibration.** For each camera (camera id, camera make and model, and image size), the expected adult height is fitted as a function of where the feet are in the image. The fit is robust to a minority of children and seated people and needs **at least 20 complete people** in that camera's photos in the current export. `camera_calibration_status` shows `disabled` with both geometry age and near-zone filtering off, otherwise `ok`, `insufficient`, or `unreliable`; the run JSON has the details. With an `ok` calibration, a person whose height is **at most 0.82** of the expected height is a child, **at least 0.88** an adult, and anyone in between is `unknown`.
3. **Fallback within the photo.** Without a usable calibration, a person is compared with the tallest other complete person at a similar image depth: at most 0.75 of that height can be a child. Relative height alone never establishes an adult; equally tall children remain unknown without a verified adult reference.
4. Everyone else is `unknown`.

What this can and cannot tell you:

- **"Child" means short stature, roughly children up to 10–11 years old.** Teenagers are usually as tall as adults and are reported as adults or unknown. The output cannot count teenagers.
- Short adults can be classified as children and tall children as adults. Historical AI-labelled adult-only photos had about 1% of measured people below the child threshold; these reused development labels do not establish the false-positive rate on new data.
- **Most people cannot be measured.** On the validation photos, 58% of people had hidden feet or no usable keypoints and received age `unknown`. `adults` and `children` describe only the measured people and are not population totals; `age_unknown` is often the largest group.
- Detecting that a photo contains children was the **weakest output** in validation (F1 0.38; the optional age expert reached 0.73 on the same photos). Treat geometry adult/child results as experimental, and validate them on your own cameras before using them for age structure.
- The calibration is refitted on every export from all photos currently in the folder, so estimates for older photos can change when photos are added.
- In the events CSV a track is a child when any of its observations is a child and none is an adult, an adult in the reverse case, and unknown otherwise.

Labels describe apparent standing height. They are not identity claims or actual ages.

### Optional age expert (local vision-language model)

Geometry cannot measure most people, so it misses most photos that contain children (24 of 37 on the validation photos). A vision-language model looking at the whole photo judges age from body proportions, size relative to nearby adults, and context (held by the hand, carried, in a stroller), and found children far more reliably on the validation photos. It is **optional and off by default**. Everything else works without it; primary age remains unknown unless `--geometry-age` or a per-person age expert is enabled. These optional estimates still require manual validation.

| Children present in a photo (F1), development experiment | AI-generated labels, 157 photos | Owner's labels, People_100 | Child-count error (MAE) |
|---|---|---|---|
| Height geometry (opt-in experiment) | 0.38 | 0.27 | 1.07 |
| Age expert, `--age-mode photo` | **0.73** | **0.75** | **0.33** |
| Age expert, `--age-mode mosaic` | 0.68 | 0.69 | 0.51 |

The age expert was tested with **Gemma 4 26B** (`gemma4:26b`, a 17 GB download) through [Ollama](https://ollama.com) on an NVIDIA RTX 3090 (24 GB), at about **3–4 seconds per request**. It needs a GPU with enough memory for the model; Ollama can also run a model partly or fully on the CPU, but that was not tested and is expected to be much slower. Leave it off on computers without such a GPU.

To set it up once:

1. Install Ollama for Windows from [ollama.com/download](https://ollama.com/download) (or `winget install -e --id Ollama.Ollama`). It runs a local server at `http://localhost:11434`.
2. Download the model:

   ```powershell
   ollama pull gemma4:26b
   ```

3. Run the analysis with the expert switched on (or set `"age_model": "gemma4:26b"` in `settings.json`):

   ```powershell
   .\run_analysis.cmd --age-model gemma4:26b
   ```

If the Ollama server is not running or the model has not been downloaded, the run prints `Age model ... is not available at ...; using cached answers where present and geometry otherwise.` and continues: photos answered in an earlier run keep their cached answers, the others keep unknown ages, or geometry estimates only when `--geometry-age` is enabled. If the server stops answering during a run (three failed or invalid answers in a row), the run prints a message and continues the same way; the run JSON reports `age_model_failures`. The model name may be given without a tag (`gemma4` means `gemma4:latest`).

Choose the mode with `--age-mode`:

| Mode | What the model is asked | Output | Use it when |
|---|---|---|---|
| `photo` (default) | One request per photo: how many adults, teenagers, children, and unclear people are visible | `vlm_adults`, `vlm_teens`, `vlm_children`, `vlm_age_unclear` per photo; the largest of each per event (`vlm_*_max_frame`) | You need the number of children and adults. Per-photo totals; best of the tested age-only prompts on development labels. |
| `mosaic` | Crops of the detected people, up to 16 numbered panels per request: an age group for each | An age for each detected person, replacing the geometry estimate in `adults`/`children`, `persons_json`, and the tracks of the events CSV | Ages must belong to individual people and be counted once per visit |
| `photo+mosaic` | Both | Both | You want both; twice the requests |

- In `photo` mode the model counts the people itself, so the `vlm_*` columns can add up to a different number than `people_total`; they do not change `people_total`, `adults`, or `children`. It is asked about every photo in which the detectors found something; photos closed by the empty-frame gate are skipped. Per-event maxima are model estimates and are not guaranteed lower bounds for a visit.
- In `mosaic` mode, a `teen` answer becomes age `unknown` (a child is clearly pre-teen, as in the labels) and an `unclear` answer keeps unknown, or the geometry estimate when explicitly enabled. `age_method` is `vlm` for these people and `age_by_vlm` counts them. At most the 40 largest people of a photo (by box height) are asked about. Separate crop images per request did not work: the model answered for only one to three of them, hence the single tiled image.
- On the tested GPU, `photo` mode adds roughly an hour per 1,000 photos (3–4 seconds each). Answers are cached in the cache folder (`.cache\vlm_age\` by default) per photo, model, mode, prompt version and inference configuration (for `mosaic`, also the exact people asked about), so later runs ask only about new photos, and a re-analysis never attaches cached ages to other people. `--force` asks again. A failed request is not cached and is retried on the next run; such a photo has empty `vlm_*` and `age_model` columns.
- Photos are sent only to the Ollama server named by `--ollama-host`, which is this computer by default. Pointing it at another computer sends the photos there.
- Ollama and the models it downloads are separate software with their own licenses. This package does not install or distribute them; check the model's terms before use.
- 157 photos from a few sites are a small sample, and the model still makes mistakes (see the table above). Check a labelled sample from your own cameras before reporting age structure.

## Large bags (experimental)

**Off by default.** Enable `--large-bags` to fill `large_bags` and `large_bags_uncertain`; those columns are blank otherwise. Backpack object counts and native backpack attribute scores remain available independently.

A **large bag** is a trekking or frame backpack, a suitcase, or a large duffel bag; ordinary daypacks are not large. The estimate uses geometry:

- YOLOE detects backpacks, suitcases, and duffel bags (confidence at least 0.25). A bag is associated with a person when at least half of it lies inside that person's box and it is clearly closer to that person than to anyone else. This spatial support suggests, but does not prove, that the person carries it.
- A backpack is measured against the body using the shoulder and hip keypoints (torso length T, shoulder line to hip line). It is **large** when it is at least 1.4 T tall and its top rises at least 0.25 T above the shoulders, or when the whole body is visible and the bag is at least half of the person's height. It is a **daypack** when it is at most 1.15 T tall and its top rises no more than 0.15 T above the shoulders; without torso keypoints, when it is at most 30% of the person's height. Anything else is **uncertain**.
- A suitcase or duffel bag associated with a person (confidence at least 0.35) is large unless its longest side is shorter than 0.8 T (then uncertain). A suitcase or duffel bag not associated with anyone (confidence at least 0.35) also counts as a large bag.

`large_bags` counts **bags, not people**: a person with a trekking pack who also pulls a suitcase adds two. Overlapping bag boxes (IoU at least 0.5) are one bag, so a suitcase detected twice, or loose luggage overlapping a carried large bag, is counted once. `large_bags_uncertain` counts associated bags that could not be classified. On the validation photos, large bags were reported correctly in 5 of the 11 photos that had them, and wrongly in 16 photos without them (F1 0.31): presence predictions have many false positives; count error can be positive or negative. Review the flagged photos before using the numbers.

## How detection works

The package runs up to four vision models on each new image (two when the empty-frame gate closes), followed by one lightweight attribute-model pass per candidate person. It does not require training on your images.

| Model | Input size (standard / fast) | Role |
|---|---|---|
| **YOLOE-26s segmentation** with fixed trail prompts | 1280 / 960 px | Main people detector; object boxes and instance masks; bags. Uses the one-to-many detection head with non-maximum suppression (IoU 0.7), which keeps overlapping people in groups apart. |
| **YOLO26n pose** | 1280 / 960 px | Second people detector; 17 body keypoints used for height, bags, and facing. |
| **YOLO26n** | 960 / 640 px | Independent detector that corroborates people, bicycles, motorcycles, dogs, backpacks, and cars/trucks/buses; also opens the empty-frame gate. |
| **MegaDetector V6 compact** | 960 / 640 px | Camera-trap detector. Supplies generic animal/person/vehicle evidence and the optional empty-frame gate. Its `animal` label cannot confirm dog species. It does not set combined people counts. |
| **PP-LCNet x1.0 pedestrian attributes** | person crops | Front/back/side body orientation and backpack presence for each candidate person. Runs on CPU with the default installation. |

Processing has three stages:

1. **Inference (cached).** Decode the image (with EXIF orientation; JPEGs whose long side is at least 3000 pixels are decoded at half resolution, which is still larger than the model inputs), read the capture metadata, find a camera info strip at the top or bottom edge, and run YOLO26n and MegaDetector. Unless the empty-frame gate closes, run YOLOE and pose. Every detection down to confidence **0.12** is kept. Person boxes from YOLOE, pose, and YOLO26n that overlap are grouped into candidate people, and each candidate gets a colour descriptor and attribute scores. When the gate closes, no candidates are built.
2. **Per-image post-processing (every run).** Accept people, verify objects, optionally estimate bag size, and estimate facing. This runs right after each image is analyzed or read from the cache. Details that later steps do not need (rejected candidates, most instance masks, detections below confidence 0.25) are then dropped from memory, so folders with thousands of photos do not accumulate raw detections; the inference cache on disk keeps everything. When the optional age expert is on, it is asked about the image at this point, or its cached answer is read.
3. **Across images (every run).** Assign camera ids, measure body geometry, fit height calibrations for the near zone, optionally classify age with `--geometry-age`, and optionally form events/tracks and measure motion with `--events`. The colour descriptors are dropped once people have been matched.

Only stage 1 (and the optional age expert's answers) is cached. Post-processing thresholds can be changed without re-analyzing images; changes to inference thresholds or candidate construction invalidate the inference cache. Coordinates everywhere refer to pixels of the original, EXIF-rotated image.

### People

A candidate person is accepted when:

- YOLOE detected it with confidence at least **0.18**, or
- pose or YOLO26n alone detected it with confidence at least **0.45**, or
- at least two of YOLOE, pose, and YOLO26n detected it, each with confidence at least **0.12**.

Boxes lying at least half inside the camera info strip are ignored. `people_total` is the number of accepted people; `candidates_total` and `candidates_rejected` show how many candidates were considered and rejected. The per-model count columns and majority votes of version 1 are still exported for comparison, but they do not change `people_total`.

### Objects

Boxes of the same category from different models that overlap (IoU at least 0.3) are treated as one object. An object is counted when at least two models found it, or when a model that may count alone found it with at least the listed confidence. Detections below 0.25 are not considered. Objects found by one model below its threshold are reported in the `unverified_*` columns and do not enter the counts.

| Export category | Models and labels | Alone at | Additional rule |
|---|---|---|---|
| `bicycles` | YOLOE, YOLO26n: `bicycle` | 0.25 | |
| `strollers` | YOLOE: `baby stroller` | 0.25 | |
| `motorcycles` | YOLOE, YOLO26n: `motorcycle` | 0.50 | Dropped when the box overlaps a bicycle (IoU ≥ 0.4). |
| `atv_utv` | YOLOE: `all-terrain vehicle`, `utility terrain vehicle`, `golf cart` | 0.35 | |
| `other_vehicles` | YOLOE: `car`, `truck`, `bus`, `tractor`; YOLO26n: `car`, `truck`, `bus` | 0.50 | Merged into the ATV/UTV when it overlaps a counted ATV/UTV (IoU ≥ 0.4). |
| `dogs` | YOLOE, YOLO26n: `dog` | 0.90 | Dropped when it overlaps a person box from YOLO26n or MegaDetector (IoU ≥ 0.7). |
| `backpacks` | YOLOE, YOLO26n: `backpack` | 0.25 | |
| `cars_trucks_buses` | YOLOE, YOLO26n: `car`, `truck`, `bus` | 0.50 | Same merge rule as `other_vehicles`. |
| `kick_scooters` | YOLOE: `kick scooter` | 0.25 | |

Dog corroboration requires species-specific `dog` detections from both YOLOE and YOLO26n, or one dog expert at confidence at least 0.90. Generic MegaDetector `animal` evidence cannot confirm a dog: the former rule promoted ibex to dogs in three review frames. Raw animal evidence remains exported. Separately, a small utility cart that YOLOE labels both "utility terrain vehicle" and "truck" is counted once, as an ATV/UTV. When one model draws near-identical boxes (IoU at least 0.8) with competing vehicle labels, only one is kept, but only within the ATV/UTV/golf-cart labels or within the car/truck/bus/tractor/motorcycle labels, so a "utility terrain vehicle" label is never discarded in favour of "truck". MegaDetector's generic `vehicle` label is not used, because it fired on bicycles and strollers. Competing ATV/motorcycle labels are resolved by confidence at IoU 0.4; remaining near-identical motorcycle/other-vehicle labels compete at IoU 0.8. `cars_trucks_buses` is derived from final accepted other vehicles with car/truck/bus labels, so it remains a subset of `other_vehicles`; do not add the two columns. `backpacks` counts backpack objects; carrying a backpack and large bags are separate outputs.

YOLOE uses these fixed prompts:

```text
person, bicycle, motorcycle, dog, backpack, baby stroller,
all-terrain vehicle, utility terrain vehicle, car, truck, bus,
golf cart, tractor, kick scooter, suitcase, duffel bag
```

Setup computes their text embeddings once using the MobileCLIP2-B text encoder and saves a prepared checkpoint. Image inference uses the prepared checkpoint without loading the text encoder.

The models are not equally validated experts, and related model families can make correlated mistakes. The rules use fixed thresholds, not learned reliability weights; the thresholds were tuned on labelled photos (see [Validation evidence](#validation-evidence)).

## Profiles, GPU, and CPU

- `--device auto` (default) uses a working NVIDIA GPU for the detectors, pose model, and colour descriptors; the GPU is checked by running a real CUDA computation. Without one, or if CUDA fails during a run, the work continues on CPU. A GPU out-of-memory error is first retried once after freeing cached GPU memory. The PP-LCNet attribute model uses the CPU with the default installation.
- `--profile standard` (default) runs YOLOE and pose at 1280 pixels and YOLO26n and MegaDetector at 960 pixels. `--profile fast` uses 960 and 640 pixels for slower CPUs; its accuracy has not been independently validated.
- The **empty-frame gate** (opt-in with `--empty-frame-gate`, default off) runs YOLO26n and MegaDetector first. If neither finds anything at confidence 0.20 or more outside the info strip, YOLOE and pose are skipped and `gated_empty` is `True`; such photos report no people or objects, and the YOLOE per-model columns are empty (not run) rather than zero. This saves time on photos triggered by wind or light. A person missed by both fast detectors is then missed entirely; the default full-analysis setting avoids this early rejection.
- GPU and CPU arithmetic can yield threshold-sensitive differences. Historical development runs differed in people count by one on one of 157 photos; that does not guarantee identical outputs for every field or the revised release.

## Incremental processing and repeated exports

The cache keys successful inference results by the image's SHA256 content hash and a configuration fingerprint covering the inference code, model content, runtime package versions, the requested device, the profile, and the empty-frame-gate setting. Replacing an image with changed bytes triggers reanalysis. Use `--force` when deliberately repeating inference. Post-processing (acceptance thresholds, object rules, bags, ages, near zone, events) is recomputed from the cached raw detections on every run, so changing it never requires reanalysis. The optional age expert's answers have their own cache (see [Optional age expert](#optional-age-expert-local-vision-language-model)).

Each normal export is a snapshot of the **currently selected folder**, not an append-only visitor database. Previously analyzed images still present in that folder are included from cache. Removed images are absent from the next snapshot, while old export files remain unchanged. Because height calibrations and events are computed over the whole current folder, adding photos can change the age estimates of photos analyzed earlier, and the membership and id of events next to the new photos. Events whose photos are unchanged keep their ids. Do not combine rows from different exports.

**Counts in the images CSV represent appearances in photographs, not unique people, visits, or trail passages.** Enable `--events` to review experimental de-duplication; it does not establish visitor identities. Summing overlapping snapshots or repeated photos double-counts. Identical image bytes under different filenames share cached inference and one `image_id`, but each file is a separate appearance: it gets its own row, capture time and camera id, and, when enabled, its own place in events, which identify photos by relative path. A copy of a photo in a second folder is therefore counted in that folder's camera as well.

Failed and partially failed results are exported with error information and retried on the next run. Successful results already cached before an interrupted run can be reused. If a run is forcibly closed, remove `.cache/run.lock` only after confirming that no analysis process is still running.

## GPU, CPU, installation, and offline use

On a fresh setup, the installer checks NVIDIA driver capability and chooses a supported CUDA PyTorch build when possible; otherwise it installs CPU PyTorch. Runtime checks execute a real CUDA kernel. Unsupported hardware, unavailable drivers, or CUDA execution failures cause fallback to CPU. AMD GPUs and Apple Metal are not accelerated by this implementation.

Paddle's CPU package is the default for the small attribute classifier. A compatible preinstalled `paddlepaddle-gpu` build inside this package's environment is recognized and tried, with CPU fallback. Actual detector and attribute devices are recorded separately; GPU detection does not imply every component runs on the GPU.

To install dependencies and prepare models without analyzing any images:

```powershell
.\setup.cmd
```

To repair or reselect the PyTorch build after adding a GPU or updating its driver:

```powershell
.\setup.cmd --repair-torch
```

For a CPU-only install, use `--cpu-install` on the first setup. To replace an already installed Torch build with CPU wheels, use:

```powershell
.\setup.cmd --cpu-install --repair-torch
```

After setup, offline operation is available:

```powershell
.\run_analysis.cmd --offline
```

An offline first installation additionally requires compatible Python package wheels and model files to be transferred to the target computer. A `.venv` directory is not a portable installation to copy between computers. For an ordinary connected second computer, clone/download the code and let its launcher create a fresh environment.

The Python launcher equivalent is `python bootstrap.py`; after setup, the environment's Python can call `python -m trailcam` directly. Direct module invocation assumes dependencies are installed. On Windows, the explicit command is:

```powershell
.\.venv\Scripts\python.exe -m trailcam --help
```

### Analysis time

Fresh execution checks on Windows/Python 3.12 used installed models, full analysis (empty-frame gate off), primary age abstention, bag size off, and near fraction zero. Both completed with **zero processing errors** and a **20-image contact sheet**:

| Run | Photos | Image analysis | Model loading | Complete analysis/export |
|---|---:|---:|---:|---:|
| RTX 3090, `--events`, sequences in 28 folders | 100 | 31.36 s | 5.14 s | 44.08 s |
| i9-11900K, 8 CPU threads, default events off | 20 selected photos | 21.57 s | 5.32 s | 31.40 s |

The total includes model loading and the contact sheet, but excludes dependency installation, model downloads and launcher setup. These runs used different workloads, so their totals do not establish a GPU speedup ratio. They were measured immediately before the final dog-corroboration post-processing correction; final outputs are regenerated from raw cached inference. They verify execution, not accuracy. Hardware, image size, people counts and cache state affect time. Use each run's JSON for elapsed time; cached rows retain their original `analysis_seconds`.

## Validation evidence

Development comparisons cover **157 photos** (100 `People_100` photos from five sites and 57 root photos), plus **28 series containing 100 frames** for tracking. The full-photo and series labels were made by **two AI annotators and a third AI adjudicator**, with 76 photo and 11 series disagreements adjudicated. They are not human ground truth. The owner's manual `People_100` labels are a separate reference and count the passing group, excluding some distant background people.

Age and bag threshold searches examined all 157 AI-labelled photos; association experiments examined the 28 series. The root photos are **not an untouched held-out test set**. The near-zone fraction was chosen on the same manual labels used to score it. These are development comparisons, not accuracy estimates for new cameras or the revised conservative defaults. Photos and labels remain private.

| Saved comparison | Result |
|---|---|
| People count, People_100 AI labels | Mean absolute error (MAE) 1.59 in v1 → 1.19 in development v2; exact counts 59% → 61% |
| People count, all 157 AI labels | Development v2 MAE 1.51; root subset MAE 2.07, bias +0.95 |
| People count, owner's manual labels | v1 MAE 0.89; v2 `people_total` 0.90; final saved `people_near` 0.73 at fraction 0.07, bias +0.45 |
| Dogs present, People_100 AI labels | Historical true/false/missed positive photos: 3/5/2 → 5/1/0; predates the corrected species-specific corroboration rule |
| Bicycles present, People_100 AI labels | 11/0/2 → 12/0/1 |
| Strollers present, People_100 AI labels | 2/0/1 → 3/0/0; both positive root photos were missed |
| Geometry children present, all 157 AI labels | F1 0.38: 13 true positive, 18 false positive, 24 missed photos |
| Large bags present, all 157 AI labels | F1 0.31: 5 true positive, 16 false positive, 6 missed photos |
| Facing versus AI-labelled travel direction | 56% aggregate direction-bucket overlap, **not per-person accuracy** |
| Event tracking, 28 AI-labelled series | Unique-count MAE 5.04 versus MaxN 6.36; mean relative error 24.6% versus 25.3%; class agreement 82% versus 71% |
| Series direction | Dominant direction matched in 20/28; per-person motion accuracy was not established |

F1 here measures category presence per photo, not detection-box accuracy. Similar aggregate totals can hide offsetting errors. Motorcycle/ATV accuracy, new-camera generalization, and real burst sequences about one second apart remain unvalidated. There is no manually validated backpack benchmark yet; the evaluator includes backpacks so one can be built.

The optional age-expert experiments are described [above](#optional-age-expert-local-vision-language-model) and in [docs/VALIDATION.md](docs/VALIDATION.md). Label fresh camera/event groups manually and keep an untouched test subset before choosing production thresholds. Use `python -m trailcam.evaluate` to measure errors and missing predictions.

## Recommended camera settings

These recommendations come from camera documentation and field practice elsewhere; they have not been tested with this package.

- **Turn on burst mode**, for example Browning *Rapid Fire* or *Multi-shot* with 3 photos, or Boly *3 Photo*. Several photos of the same passage make direction from motion and de-duplication possible. On the validation series, whose photos were 3–19 seconds apart, linking people between photos was the main source of error (about 25% per visit); photos about a second apart are expected to be the most effective improvement.
- **Set a capture delay of 5–10 seconds** between triggers. Keep `--event-gap` longer than the delay plus the burst, so one passing group stays in one event.
- **Synchronize the camera clock at every card swap.** Events and camera-day totals depend on the camera's clock.
- **Keep one folder per camera placement** (or use `--camera-pattern`), and start a new folder when a camera is moved or re-aimed, so the height calibration describes one view.
- Copy photos with their EXIF data intact.

## Limitations

- **Crowds:** counts of groups of 8 or more people are approximate (error of several people per photo); on the root development subset, large groups were overcounted by about two.
- **Counting definition:** `people_total` and `people_near` answer different questions (see [Counting people](#counting-people-everyone-visible-or-the-near-zone)). The optional 7% near zone was tuned on the owner's labels of the `People_100` photos and depends on camera height and lens.
- **De-duplication:** on photos 5–16 seconds apart, `people_unique` had about 25% error per visit (similar hikers in a line merged, some people split), and MaxN had a similar relative error. Both are estimates, not bounds. Compare classes and review flagged matches; burst mode is an unvalidated possible improvement. Night infrared photos carry little colour for matching people between photos.
- **Adults and children:** unknown by default; geometry is opt-in; most people cannot be measured; teenagers are not identified; children-present F1 was 0.38. Experimental. The optional age expert (F1 0.73) needs a suitable GPU and was tested on few sites.
- **Large bags:** F1 0.31 with many false positives. Experimental.
- **Direction:** facing is not travel direction (56% aggregate overlap with AI direction labels, not per-person accuracy). Direction from motion was checked only as the group's dominant direction (right in 20 of 28 series).
- **Motorcycles and ATVs/UTVs:** not validated; small utility carts are counted as ATV/UTV.
- **Events depend on capture times.** Wrong or reset camera clocks, missing EXIF data (flagged `times_from_file_dates` in multi-photo events), or a gap setting that does not match the camera produce wrong events. The same person returning after the event gap is counted again.
- **Empty-frame gate:** when explicitly enabled, a person missed by both fast detectors is not counted. Full analysis is the default.
- **Camera info strip:** only dark info bars at the top or bottom edge are recognized.
- Validation covered 157 photos and 28 photo series from a few sites. Review the contact sheet and label a sample of your own photos before relying on counts or demographic estimates.

## Troubleshooting

| Symptom | Action |
|---|---|
| Python is missing or the wrong version is selected | Install 64-bit Python 3.12 or 3.11. The Windows launcher checks the existing environment, then the Python launcher, then `python`. |
| Download or pip installation fails | Check internet/proxy access and available disk space, then rerun. Existing compatible packages and verified model files are reused. |
| Model checksum fails | A download may be incomplete or altered. Rerun setup online; `--offline` cannot repair missing or invalid model files. |
| An expected GPU is not used | Check the recorded device and fallback messages. Update the NVIDIA driver if needed, then run `setup.cmd --repair-torch`. CPU inference remains available. |
| The run stops with a message such as "Invalid device" or "threads must be an integer" | A value in `settings.json` has the wrong type or an unknown choice. Correct it (for example `"threads": 8`, not `"8"`); see [Select an input folder and options](#select-an-input-folder-and-options). |
| `--camera-pattern` is rejected or "The system cannot find the file specified" appears | The pattern needs a named group `(?P<camera>...)`. Put it in `settings.json`, or wrap it as `'"..."'` in PowerShell (see [Select an input folder and options](#select-an-input-folder-and-options)). |
| Every event contains only one photo | Check `capture_time` and `time_source` in the images CSV. Without burst mode, or with a capture delay longer than `--event-gap`, events have one photo each. |
| Many events are flagged `times_from_file_dates` | The photos have no EXIF capture time, so file modification times were used, and copying often makes them identical. Copy the photos with their EXIF data intact, or treat those events with care. |
| Almost every age is `unknown` | Expected by default: primary age classification is off. `--geometry-age` enables the experimental height rule, which still abstains for hidden or distant people. Check `camera_calibration_status`: `insufficient` means the camera has fewer than 20 fully visible standing people in the folder. On a computer with a suitable GPU, consider the [optional age expert](#optional-age-expert-local-vision-language-model). |
| `Age model ... is not available at ...; using cached answers where present and geometry otherwise.` | The Ollama server is not running, the address in `--ollama-host` is wrong, or the model is not downloaded. Start Ollama, check that `ollama list` shows the model given to `--age-model` (a name without a tag means `:latest`), or run `ollama pull gemma4:26b`. The run continues with cached answers and otherwise unknown ages, or geometry estimates only when `--geometry-age` is enabled. |
| The age expert is very slow | Most likely the model does not fit in GPU memory and runs partly on the CPU. `ollama ps` shows the split; use a GPU with more memory, or leave `--age-model` off. |
| `people_near` is much smaller than `people_total` | Many people are far from the camera, as intended. If people close to the camera are left out, lower `--near-fraction` (or use `0` to count everyone) and check against a few labelled photos. |
| Analysis is slow on CPU | Use `--profile fast`, consider the optional gate only after checking missed detections, and check `--threads`. |
| Analysis reports another run is active | Wait for it to finish. After a confirmed crash, remove the stale `.cache/run.lock`. |
| Some images fail | Inspect `status` and `error` in the images CSV. Other images continue; failed images are retried next time. An error starting with `Post-processing failed:` names the step (`fusion` or `geometry`) and the Python error; the cached detections are kept, and post-processing is attempted again on every run. |
| Contact-sheet export fails | The CSVs are saved first. Check the run JSON's `contact_sheet_error`; use `--no-contact-sheet` if necessary. |
| Window closes before messages can be read | Open PowerShell in the repository and run `.\run_analysis.cmd` there. |

## Development, validation, privacy, and licenses

Run the regression tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover cache/export behavior, inference plumbing, the post-processing rules, events, geometry, and the evaluation script. They do not establish accuracy on a deployment's camera images. To measure accuracy on your own photos, label a sample and score an export with `python -m trailcam.evaluate`; see [docs/VALIDATION.md](docs/VALIDATION.md). Design notes are in [docs/DESIGN_V2.md](docs/DESIGN_V2.md).

The public repository contains code and documentation. Model weights are downloaded from their upstream sources. Your images, model cache, Python environment, inference cache, exports, and contact sheets are excluded from version control by the supplied `.gitignore`. Inference runs locally; the package does not upload photographs. With the optional age expert switched on, photos are sent to the Ollama server named by `--ollama-host`, which is this computer by default. Generated CSVs contain filenames and visual evidence, and contact sheets contain the photographs themselves, so share outputs deliberately.

The project is distributed under [AGPL-3.0](LICENSE). Model weights keep their own licenses: the MegaDetector V6 weights are AGPL-3.0, and the MobileCLIP2-B text encoder, used once during setup to prepare the YOLOE prompts, is released by Apple under a research-only license. Ollama and any model used as the optional age expert are not part of this package and have their own licenses. Check that your use is compatible. See [third-party notices](THIRD_PARTY_NOTICES.md) and [model sources, hashes, and upstream documentation](MODEL_SOURCES.md).
