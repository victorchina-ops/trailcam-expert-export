# Model sources and reproducibility

The repository contains source code only. The first run downloads the following
public upstream weights into `models/`; later runs reuse them. Images are processed
locally. The installer checks SHA256 before loading any downloaded checkpoint.
The URL and hash manifest is also executable data in `trailcam/model_setup.py`.

| Model | Role | Upstream download | SHA256 |
|---|---|---|---|
| YOLO26n | Independent COCO detector: corroborates people and objects; empty-frame gate | [Ultralytics v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt) | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| YOLOE-26s segmentation | Main people detector (one-to-many head), trail object boxes, masks, and bags | [Ultralytics v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt) | `48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654` |
| YOLO26n pose | Second people detector; 17 COCO body keypoints for standing height, bag geometry, and facing | [Ultralytics v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt) | `eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9` |
| MegaDetector V6 compact | Camera-trap detector: optional empty-frame gate and generic animal/person/vehicle evidence (not dog-species corroboration or combined people counts); weights AGPL-3.0 | [Microsoft / Zenodo](https://zenodo.org/records/15398270/files/MDV6-yolov10-c.pt?download=1) | `21ee78a2d4887128e2a4920937d3295b493f44d788d13e1a635378c16dd74ef7` |
| PP-LCNet x1.0 pedestrian attribute | Person-crop front/back/side orientation and backpack presence | [Paddle official inference archive](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_pedestrian_attribute_infer.tar) | `1b405e1c09484b3f4f18b435815e8473b9bf5d6c8aa8af871a667c0c9ca44f1c` |
| MobileCLIP2-B text encoder | One-time YOLOE category embedding generation during setup; Apple research-only model license | [Ultralytics v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts) | `35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f` |

The first model download is approximately 312 MB, most of which is the text
encoder. Python dependencies require additional disk space, especially CUDA
PyTorch. No model training or external API account is required.

## Model licenses

The weights are not part of this repository and keep their upstream licenses.
Two of them need particular attention:

- **MegaDetector V6 compact** (`MDV6-yolov10-c.pt`): the MegaDetector model zoo
  lists this YOLOv10-based variant under **AGPL-3.0**. It is used at every
  analysis for the optional empty-frame gate and generic animal/person/vehicle evidence. Its generic animal label cannot corroborate a dog species prediction.
- **MobileCLIP2-B text encoder** (`mobileclip2_b.ts`): released by Apple under
  the Apple Machine Learning Research Model (AMLR) license, which permits
  **research use only**. It runs once during setup to compute the text
  embeddings of the fixed YOLOE prompts; the prepared checkpoint
  `yoloe-26s-trail-prompts.pt` contains those embeddings. Analysis does not
  load the encoder again.

The Ultralytics models are available under AGPL-3.0 or a commercial Ultralytics
license. The PP-LCNet archive is distributed through PaddleX, whose repository
uses Apache-2.0; the archive itself contains no separate weight license. Check that your use (for example commercial or non-research
use) is compatible with every model's terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). This summary is not legal
advice and grants no rights beyond the upstream licenses.

## Fixed YOLOE prompts

The categories, in checkpoint order, are:

`person`, `bicycle`, `motorcycle`, `dog`, `backpack`, `baby stroller`,
`all-terrain vehicle`, `utility terrain vehicle`, `car`, `truck`, `bus`,
`golf cart`, `tractor`, `kick scooter`, `suitcase`, `duffel bag`.

The first setup run uses `YOLOE.set_classes()` on CPU, then saves
`yoloe-26s-trail-prompts.pt` with the embeddings already included. Its adjacent
JSON records the base hash, exact prompts, package version, and generated hash.
Subsequent inference does not need to load the text encoder. The prepared file
is generated locally, so its file hash can differ between machines while the
upstream base and prompts remain identical.

Tokenizer source is pinned to
[`ultralytics/CLIP` commit b0c7af36](https://github.com/ultralytics/CLIP/tree/b0c7af36eb99a5e103713e1792fc642f78059c39).
The archive is installed by pip without requiring Git on the target computer.

## Attribute archive contents

Only these three regular files are extracted, checked separately, and used:

| File | SHA256 |
|---|---|
| `inference.json` | `1dc2805fdc5174cd627bc59ea4a9ab2cf6d5d3da6a3e568434dd442447fe87c0` |
| `inference.pdiparams` | `3a388425abc9187f485d5bfe6ad486a87e4895a1b33df098050683b13be8cea4` |
| `inference.yml` | `29b27d7f302391c70e42b486b6e2dba50984317c4815daddc340f0771d527243` |

The lightweight Paddle inference API is used directly; PaddleX is not installed.
The resize and normalization match the official model configuration. Direct
inference was checked against saved PaddleX results; score differences were
within the five-decimal rounding used by that wrapper.

## Runtime and hardware

The tested versions are Python 3.12, PyTorch 2.11.0, torchvision 0.26.0,
Ultralytics 8.4.37, PaddlePaddle 3.3.0, NumPy 2.3.5, and Pillow 12.3.0.
SciPy (1.18.1 in the tested environment) is installed as an Ultralytics
dependency and provides the Hungarian assignment used to match people across
the photos of an event; without SciPy a greedy matcher is used.
Python 3.11 is also accepted by setup; the full integration run used Python 3.12.

On a fresh install, setup chooses the official CUDA 12.8 or 12.6 PyTorch wheels
when the NVIDIA driver reports support; otherwise it installs CPU wheels.
Actual CUDA kernel execution is checked before announcing GPU availability.
The vision engine also handles unavailable CUDA by falling back to CPU. Already
compatible packages are kept. After adding/updating GPU hardware, use
`python bootstrap.py --setup-only --repair-torch` to reselect the Torch build.

Paddle's CPU package is the portable default for the small crop classifier.
An already installed, compatible `paddlepaddle-gpu` package is retained and can
be used by the attribute engine if its GPU runtime is usable. Detector, mask,
pose, and attribute device fields record the devices actually used.

The `standard` profile runs YOLOE and pose at 1280 pixels and YOLO26n and
MegaDetector at 960 pixels; `fast` uses 960 and 640 pixels. YOLOE, YOLO26n,
and pose use their one-to-many detection heads with non-maximum suppression
(IoU 0.7) instead of the end-to-end heads the checkpoints default to. On the
157 development photos, GPU (RTX 3090) and CPU results were identical except
for one photo that differed by one person.

For an offline computer, first run setup online on a computer with the same
platform. Cache the Python packages for that platform separately and copy
`models/` along with the source. Python virtual environments are not portable
between computers. Once dependencies and verified models are present, pass
`--offline` to prohibit setup/model downloads.

## Primary documentation

- [Ultralytics YOLOE prompting and segmentation](https://docs.ultralytics.com/models/yoloe/)
- [Ultralytics pose models and keypoints](https://docs.ultralytics.com/tasks/pose/)
- [MegaDetector model zoo, variants and licenses](https://microsoft.github.io/MegaDetector/model_zoo/)
- [Official pedestrian attribute model](https://paddlepaddle.github.io/PaddleX/3.7/en/module_usage/tutorials/cv_modules/pedestrian_attribute_recognition.html)
- [PyTorch installation commands for pinned versions](https://pytorch.org/get-started/previous-versions/)

These models estimate visible content. Agreement is not a calibrated confidence
measure. Primary adult/child assignments remain unknown by default. Standing-height
geometry and a separately enabled local age expert are experimental opt-ins; cross-photo matching of people is a
de-duplication aid, not identification. See the README for output semantics,
development evidence and limitations.
