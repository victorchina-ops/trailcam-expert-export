# Model sources and reproducibility

The repository contains source code only. The first run downloads the following
public upstream weights into `models/`; later runs reuse them. Images are processed
locally. The installer checks SHA256 before loading any downloaded checkpoint.
The URL and hash manifest is also executable data in `trailcam/model_setup.py`.

| Model | Role | Upstream download | SHA256 |
|---|---|---|---|
| YOLO26n | Independent COCO object count baseline | [Ultralytics v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt) | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| YOLOE-26s segmentation | Canonical people, trail object boxes and masks | [Ultralytics v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26s-seg.pt) | `48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654` |
| YOLO26n pose | 17 COCO body keypoints; conservative orientation heuristics | [Ultralytics v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt) | `eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9` |
| MegaDetector V6 compact | Independent trail-camera person/animal/vehicle detector | [Microsoft / Zenodo](https://zenodo.org/records/15398270/files/MDV6-yolov10-c.pt?download=1) | `21ee78a2d4887128e2a4920937d3295b493f44d788d13e1a635378c16dd74ef7` |
| PP-LCNet x1.0 pedestrian attribute | Person crop orientation, backpack and experimental native attribute scores | [Paddle official inference archive](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_pedestrian_attribute_infer.tar) | `1b405e1c09484b3f4f18b435815e8473b9bf5d6c8aa8af871a667c0c9ca44f1c` |
| MobileCLIP2-B text encoder | One-time YOLOE category embedding generation | [Ultralytics v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts) | `35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f` |

The first model download is approximately 312 MB, most of which is the text
encoder. Python dependencies require additional disk space, especially CUDA
PyTorch. No model training or external API account is required.

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
measure, and this package is not a validated adult/child classifier or movement
tracker. See the README for output semantics and limitations.
