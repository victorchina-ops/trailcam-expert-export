# Validating exports against manual labels

Development experiments used 157 photos and 28 series with AI-generated labels, plus a separate manual reference for 100 photos (see [Validation evidence](../README.md#validation-evidence)). This is not independent human validation of the revised defaults. Your cameras, mounting heights, seasons, and visitors can differ. Before relying on the counts, label a sample of your own photos and score an export with the included evaluation script. The script uses only the Python standard library, loads no model, and runs in seconds.

The default run performs full per-image analysis, keeps primary ages unknown, and leaves optional large-bag size/event outputs unassessed. To evaluate an experiment, enable its flag explicitly (`--geometry-age`, `--large-bags`, or `--events`); record those settings with the labels. `--near-fraction 0` is the default and includes everyone; a nonzero near-zone fraction is experimental.

The workflow is:

1. Run the analysis on a folder that contains the photos to be labelled.
2. Create a blank labels file from that export.
3. Label the photos (or events) following the [definitions below](#labelling-definitions).
4. Score the export against the labels.

Commands below use the package environment on Windows; replace `DATE` with the timestamp of your export.

## Choose the photos

- Include every camera and site you will report on, day and night (infrared) photos, and the full range of group sizes, including large groups.
- Include some photos without people. They measure false detections.
- To check events and de-duplication, label **complete events**: every photo of the chosen events (see [Multi-photo events](#labelling-multi-photo-events)).
- Choose photos by a rule decided in advance (for example every tenth event per camera), not by looking at the model output. The contact sheet is a review sample, not a random sample.
- If you adjust any settings while looking at some labelled photos, keep other labelled photos aside that you do not look at, and report accuracy on those.

## Create a labels template

```powershell
.\.venv\Scripts\python.exe -m trailcam.evaluate --template "exports\export_DATE\export_DATE.csv" --out labels.csv
```

This writes one row per exported image with the key `relative_path`, one empty column per label field, and a `notes` column. Add `--force` to overwrite an existing file. The template refuses to overwrite the export itself.

Label columns (whole numbers per photo):

| Column | Meaning |
|---|---|
| `people_total` | Every visible person (see definitions) |
| `adults`, `children`, `age_unclear` | Age buckets; they should add up to `people_total` |
| `dir_left`, `dir_right`, `dir_toward`, `dir_away`, `dir_stationary`, `dir_unclear` | Direction per person; they should add up to `people_total` |
| `bicycles`, `strollers`, `motorcycles`, `atv_utv`, `other_vehicles`, `dogs` | Objects visible in the photo |
| `backpacks` | All visible backpack objects, regardless of size or whether carried |
| `large_bags` | Large bags visible in the photo; optional `--large-bags` estimate |

**Leave a cell empty** when you did not label that field for that photo; it is skipped. **Enter 0** when you checked and the item is absent. You may add columns of your own (for example `people_min`, `people_max`, `large_bags_uncertain`, `light`); the script ignores columns it does not know.

Photos are matched to export rows by `relative_path`, then by the same path ignoring case, then by a unique filename or path ending. A label naming a different folder never matches, because trail cameras reuse filenames. Do not rename or move photos between the export and the scoring.

Save the file as **CSV UTF-8** when editing in a spreadsheet program.

## Score an export

```powershell
.\.venv\Scripts\python.exe -m trailcam.evaluate --labels labels.csv --export "exports\export_DATE\export_DATE.csv" --json report.json
```

| Option | Meaning |
|---|---|
| `--labels FILE` | Labels: image rows (with `relative_path`, `filename`, or `file`) or event rows (with `event_id`). |
| `--export FILE` | Images CSV to score image labels against. Required with image labels. |
| `--events FILE` | Events CSV. Scored against event labels, or against event labels derived from fully labelled events in the image labels. |
| `--event-labels FILE` | Event labels, when `--labels` holds image labels. |
| `--prefix TEXT` | Prediction column prefix in the images CSV. None for version 2; `combined_` for version 1 exports. |
| `--json FILE` | Also write the full report, including the 20 largest people-count errors and any invalid label cells. |
| `--template FILE --out FILE [--force]` | Write a blank labels template for an images or events CSV (see above). |

The printed report has one line per field:

| Column | Meaning |
|---|---|
| `n` | Photos (or events) with both a label and a prediction |
| `nopred` | Labelled photos whose prediction cell was empty or invalid (a coverage gap, never scored as 0) |
| `MAE` | Mean absolute count error |
| `bias` | Mean of predicted minus labelled count; negative = undercount |
| `exact` | Share of photos with exactly the right count |
| `within1` | People counts only: share within ±1 |
| `prec`, `recall`, `F1`, `tp/fp/fn` | Presence (count > 0) at photo level: correct, false, and missed photos |

`large_bags_with_uncertain` scores the labelled `large_bags` against the export's `large_bags + large_bags_uncertain`. The `group age` line compares the three age buckets per photo (bucket MAE, per-photo sum of absolute errors, share of exact photos) and gives the F1 of "photo contains children". The `group direction` line does the same for the six directions. Rows of `error` images match their labels but have no predictions; they are reported as `matched error rows`.

The script exits with code 2 and an error message for unreadable files, repeated columns, or mismatched inputs (for example event labels scored against an images CSV).

## Near-zone counts, classes, and the age expert

The script scores `people_total` (and `people_unique` at event level), not `people_near`, the class columns, or the optional age expert's `vlm_*` columns. To check them, add your own label columns (for example `people_near`, labelled with the [near-zone convention](#people-near-the-camera)) and compare them with a few lines of Python, run from the repository folder so that `trailcam` can be imported:

```python
import csv
from trailcam.export import count_class

csv.field_size_limit(100_000_000)

def rows(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return {row["relative_path"]: row for row in csv.DictReader(handle)}

labels = rows("labels.csv")
export = rows(r"exports\export_DATE\export_DATE.csv")
label_column, export_column = "people_near", "people_near"  # or "children", "vlm_children"
pairs = [(int(label[label_column]), int(export[path][export_column]))
         for path, label in labels.items()
         if (label.get(label_column) or "").strip() and (export.get(path, {}).get(export_column) or "").strip()]
print("photos:", len(pairs))
print("MAE:", sum(abs(p - t) for t, p in pairs) / len(pairs))
print("class right:", sum(count_class(t) == count_class(p) for t, p in pairs) / len(pairs))
```

The same comparison works for `vlm_children` against a `children` label, and, with the events CSV and event labels keyed by `event_id`, for `people_near_unique` or `vlm_children_max_frame`. Keep `--near-fraction` fixed while scoring; if you adjust it on labelled photos, report accuracy on other photos you did not look at.

## Labelling definitions

The rules for people, ages, direction, and large bags are the ones used for the published validation labels. The vehicle and dog categories follow the export categories. Label what is visible in the photo; do not guess.

### People

- `people_total` counts **every visible person**, including distant and background people, people partly hidden behind others or vegetation when the visible part clearly belongs to a separate person, and **carried people** (babies and children carried in arms, carriers, or on shoulders).
- Count each person once per photo. If the count is genuinely uncertain, record your best count in `people_total` and, if useful, the range in your own `people_min` and `people_max` columns.

### People near the camera

If you report `people_near`, also label your own `people_near` column: count **the group passing the camera**, that is, people on the trail or path near the camera, including carried people, and leave out distant background people (for example a crowd at a far plaza, or hikers on a distant slope). This is the convention of the camera owner's labels, on which the default near fraction was chosen. Decide on it before labelling: on the published validation photos, the two conventions gave different counts for a third of the photos.

### Adults and children

- `adults`: people with a clearly adult build and enough visible detail to be sure.
- `children`: **clearly a child**, by size, body proportions, and face, roughly **under 12 years**. Carried babies and toddlers are children.
- `age_unclear`: everyone else, including **teenagers**, people too distant, too small, or too hidden to judge, and anyone you cannot place confidently. Do not guess from context (for example "a family, so the small person is a child") unless the person is clearly a child.
- `adults + children + age_unclear = people_total`.

The package's "child" is short standing stature and corresponds roughly to children up to 10–11 years; see the [README](../README.md#adults-and-children). The optional age expert is asked for the same definition (clearly pre-teen, roughly under 12, including toddlers and carried babies) and reports teenagers separately in `vlm_teens`; compare `vlm_children` with `children` and `vlm_teens + vlm_age_unclear` with `age_unclear`.

### Direction

Direction is **image-relative travel direction**, one per person:

- `dir_left`, `dir_right`: moving toward the image's left or right edge.
- `dir_toward`, `dir_away`: approaching or moving away from the camera.
- **Diagonal movement:** use the dominant component. Choose left or right only when the movement is clearly sideways; a camera that looks down along a path shortens movement in depth, so judge carefully.
- `dir_stationary`: standing still, seated, or with feet planted, not walking.
- `dir_unclear`: no gait, stride, or body cue shows the direction. Do not infer a person's direction from the rest of the group.
- Judge travel from stride, feet, wheels, and body posture; a slightly turned head alone is not direction.
- `dir_left + dir_right + dir_toward + dir_away + dir_stationary + dir_unclear = people_total`.

The package reports facing for people seen in one photo and motion for people matched across photos; see [Direction](../README.md#direction-motion-versus-facing).

### Vehicles and dogs

- `bicycles`: each bicycle, ridden, pushed, or parked.
- `strollers`: baby strollers and prams.
- `motorcycles`: motorcycles and dirt bikes.
- `atv_utv`: off-road or utility vehicles: quads, side-by-sides, golf carts, and small electric utility carts (such as Goupil).
- `other_vehicles`: cars, trucks, buses, and tractors.
- `dogs`: each dog. Record other animals in your own column if needed; they are not scored.

### Backpacks

Count each visible backpack once, including daypacks and trekking backpacks, whether worn, carried by hand, or set down. Do not count handbags, shoulder bags, suitcases or duffels as backpacks. This is an object count, distinct from the number of people carrying a backpack and from optional large-bag size. Leave the label blank if the photo does not let you judge it.

### Large bags

- **Large**: a trekking or frame backpack that is bulky and reaches from **above the shoulders down to about the hips** (hip-belt line), often with a top lid or a strapped mat or pot; a suitcase or rolling bag; a large duffel bag. Count large bags whether carried or set down.
- **Not large**: daypacks and small backpacks, handbags, shoulder bags, and shopping bags.
- **Uncertain** (a medium bag, a pack whose size or top cannot be seen, an unworn backpack that could be either): leave it out of `large_bags`; record it in your own `large_bags_uncertain` column if useful.
- Count each large bag once. Count bags, not people: a person with a trekking pack and a suitcase has two. The export's `large_bags` counts the same way.

## Labelling multi-photo events

This workflow requires an export made with `--events`. The default run writes no events or camera-days CSV. Add `--events "exports\export_DATE\export_DATE_events.csv"` to the evaluator command when scoring event output. Event backpacks, like the other objects, use the maximum count in any frame.

Burst photos show the same people several times. There are two ways to label them.

**1. Label every photo of an event as image rows** (recommended; uses the same template). Label each photo on its own, counting the people visible in that photo. For direction, you may look at the neighbouring photos of the same event to see where a person is moving, because the export's `dir_*` columns use that movement when the person was matched across photos. When you pass `--events` together with image labels, the script derives event labels for every event whose photos are **all** labelled (it takes the events from the export's `event_id` column):

- objects and `large_bags`: the maximum over the photos, as in the events CSV;
- `people_max_frame`: the largest photo `people_total`, scored against the export's `people_max_frame`;
- `people_unique`: checked against the range from the largest photo count to the sum of all photo counts. The report states how many events fell within, below, or above that range. A value above the range can reflect false detections or failed de-duplication; below it indicates undercounting relative to the busiest labelled frame. This is a broad consistency check, not a score for identity matching.

Ages and directions are not derived at event level, because they need to know which observations are the same person. Events with a missing photo label are skipped and reported as partly labelled.

**2. Label events directly.** Make an event template from the events CSV:

```powershell
.\.venv\Scripts\python.exe -m trailcam.evaluate --template "exports\export_DATE\export_DATE_events.csv" --out event_labels.csv
```

The template has `event_id`, the event's `camera_id`, `start`, `end`, `image_count`, and `images` for orientation, and the label columns `people_unique`, `adults`, `children`, `age_unclear`, the six directions, the objects, and `large_bags`. `images` lists the relative paths of the event's photos in capture order; for a very long event it holds only the first and last path and the count (`"truncated": true`), so join to the images CSV by `event_id` to find every photo. Look at all photos of the event, then enter:

- `people_unique`: the number of **different** people across all photos of the event (the published series labels were made this way);
- ages and directions: one per different person, using the definitions above; for direction, use the movement across the photos;
- objects and `large_bags`: the **largest number visible in any one photo** (the events CSV definition).

Score with:

```powershell
.\.venv\Scripts\python.exe -m trailcam.evaluate --labels event_labels.csv --events "exports\export_DATE\export_DATE_events.csv"
```

or together with image labels by adding `--event-labels event_labels.csv` to the image command. An event id is made from the camera and the event's first capture time (or first filename), so it stays the same in later exports as long as the event keeps the same photos, even when photos are added to other cameras or other days. Adding or removing photos inside an event, or within the event gap of it, can change its membership and its id, and changing `--event-gap` or `--camera-id` regroups the photos; **score event labels against an export whose events match the template**, ideally the one the template was made from. Labelled event ids missing from the export are reported as unmatched. If you think the export split one passage into two events or merged two passages, label the event as defined by the export and explain it in `notes`.

## How the development comparisons were made

The historical experiments used 157 photos (`People_100`: 100 photos from five sites; `root`: 57 photos, many with large crowds). **Two AI annotators labelled each photo, and a third AI annotator adjudicated 76 disagreements.** The 28 series were labelled with the same AI process, with 11 adjudications. These are AI-generated reference labels, not human ground truth. The original separate People_100 labels are the owner's manual reference.

Age and bag threshold sweeps scored all 157 images, including `root`, and association experiments scored the 28 series. The root images are therefore **not an untouched held-out test set**. Both threshold tuning and evaluation on these labels can overstate generalization. AI annotators can also share systematic errors. The photos and labels are private and are not included in the repository.

Published tables describe saved development configurations with optional heuristics enabled. Revised defaults leave geometry age, large bags, events, the empty-frame gate and the Ollama expert off. Those tables do not measure the revised default release on fresh data. Choose new camera/event groups for manual validation and keep tuning and final-test groups separate.

Three further references were used:

- **The camera owner's labels** for the 100 `People_100` photos (kept as a spreadsheet and an identical CSV). They count the group passing the camera and leave out distant background crowds (see [People near the camera](#people-near-the-camera)), and describe ages as free text. They differ from the AI-generated people counts on 33 of the 100 photos, almost always because of distant background people. The experimental near fraction (0.07) was chosen as the best match to these labels, so `people_near` figures measured on them are optimistic. For children present, a photo counts as containing children when the age text mentions a child, kid, son, boy, girl, toddler, or baby; photos described only as teenagers were left out.
- **28 photo series** (`People_Series_100`): 3–4 consecutive trigger photos per series, 100 photos in all, 3–19 seconds apart, from the same sites, taken without burst mode. Two independent AI annotators labelled, per series, the number of different people (everyone visible and near zone), their ages, and each person's direction of travel from the movement across the photos; disagreements were adjudicated (11 series). With the default 60-second event gap, the export formed exactly one event per series. The matching settings were searched on these series (limits, weights, time allowance, and a Lab colour-moment descriptor instead of the HSV histograms); nothing was clearly better, so the defaults were kept and the published figures use them.
- **The optional age expert** (Gemma 4 26B through Ollama) on the 157 photos. Children present (F1) against the AI-generated labels and against the owner's labels (the 100 `People_100` photos only), and the mean absolute error of the child count against the AI-generated labels:

  | How the model was asked | F1, AI-generated | F1, owner's | Child-count MAE |
  |---|---|---|---|
  | Height geometry, no model (opt-in experiment) | 0.38 | 0.27 | 1.07 |
  | Whole photo, age prompt (`--age-mode photo`) | 0.73 | 0.75 | 0.33 |
  | Whole photo, an earlier prompt asking for all fields at once (not in the package) | 0.77 | 0.81 | 0.45 |
  | Numbered boxes drawn on the detected people (`marks`; module only) | 0.63 | 0.69 | 0.62 |
  | Numbered crops tiled into one image (`--age-mode mosaic`) | 0.68 | 0.69 | 0.51 |
  | `marks` or `mosaic` finding a child | 0.67 | 0.76 | 0.59 |
  | Separate crop images, several per request (`crops`; module only) | failed: the model answered for only 1–3 of the crops | | |

  The whole-photo age prompt performed better than the tested per-person prompts on these development labels; this was not a held-out comparison. Per-person modes are useful only when ages must follow tracked people through a visit. On the 28 series, children present per series reached F1 0.90 with geometry and 0.92 with the model shown all photos of a series at once; children in those series were mostly large and close to the camera.

The historical series results against AI-generated labels (different people per visit) are in the [README](../README.md#validation-evidence): `people_unique` MAE 5.0 people, relative error 25%, class right for 82%; `people_max_frame` MAE 6.4, 25%, 71%; the sum of the photos' counts 128% too high; the model shown all photos at once 22%, MAE 5.9, 82%. The dominant direction of travel of the group was right in 20 of 28 series (the model: 16 of 28). Aggregate detections totalled 1,094 person-photos against 1,123 AI-labelled person-photos; these totals can hide offsetting errors, and the per-visit error comes from linking people across photos 5–16 seconds apart. Burst sequences about a second apart remain unvalidated.

The historical 56% facing result is aggregate overlap: for each photo and direction, take the smaller of the predicted and labelled bucket counts, sum those overlaps, and divide by the number of assigned facing labels. There are no person-matched direction annotations in that result, so it cannot establish per-person accuracy. The evaluator reports explicit bucket error rather than calling this an accuracy percentage.

## Execution verification of the revised defaults

Windows/Python 3.12 checks with installed models completed without processing errors: 100 sequence photos in 28 folders on an RTX 3090 with `--events` took 31.3623 seconds of image analysis, 5.1439 seconds model loading, and 44.075 seconds total; 20 selected photos on an i9-11900K with eight CPU threads and events off took 21.5742, 5.3153, and 31.3988 seconds respectively. Both used full analysis, primary age abstention, bag size off, near fraction zero, and a 20-image contact sheet. Total includes model loading and export but excludes dependency/model installation and launcher setup. Different workloads prevent a direct GPU speedup comparison.

These are execution checks, not accuracy evaluations. Timings were recorded immediately before the final `objects_v2_1` dog rule correction; final outputs are regenerated from raw inference. The corrected rule requires dog-specific YOLOE/YOLO26n corroboration or a single dog confidence of at least 0.90, because generic MegaDetector animal support had promoted ibex in three review frames. Earlier dog accuracy tables describe the historical rule, not the corrected release.
