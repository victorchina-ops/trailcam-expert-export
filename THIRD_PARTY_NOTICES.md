# Third-party software and model notices

The package source is distributed under GNU AGPL version 3; see [LICENSE](LICENSE).
Third-party dependencies and weights retain their own licenses. They are fetched
from upstream repositories, not included as binary files in this repository.

| Component | Upstream / license information |
|---|---|
| Ultralytics YOLO, YOLOE and pose software/models | [Ultralytics](https://github.com/ultralytics/ultralytics), [AGPL-3.0 and commercial licensing](https://www.ultralytics.com/license) |
| Ultralytics CLIP tokenizer fork | [Pinned source and license](https://github.com/ultralytics/CLIP/tree/b0c7af36eb99a5e103713e1792fc642f78059c39), derived from OpenAI CLIP; the fork's license is AGPL-3.0 |
| MegaDetector V6 compact YOLOv10 model | [Microsoft model zoo and model-specific licensing](https://microsoft.github.io/MegaDetector/model_zoo/); this YOLO variant is listed under AGPL-3.0 |
| PaddlePaddle inference runtime | [Apache-2.0 license](https://github.com/PaddlePaddle/Paddle/blob/develop/LICENSE) |
| PP-LCNet pedestrian attribute model | [PaddleX official model distribution](https://paddlepaddle.github.io/PaddleX/3.7/en/module_usage/tutorials/cv_modules/pedestrian_attribute_recognition.html); PaddleX repository uses Apache-2.0; the downloaded inference archive contains no separate weight license |
| MobileCLIP text encoder | [Apple MobileCLIP](https://github.com/apple/ml-mobileclip), [model license](https://github.com/apple/ml-mobileclip/blob/main/LICENSE_MODELS); software license and model terms are separate |
| PyTorch and torchvision | [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE), [torchvision license](https://github.com/pytorch/vision/blob/main/LICENSE) |
| NumPy | [NumPy BSD license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| Pillow | [Pillow license](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| OpenCV Python bindings | [OpenCV Python license](https://github.com/opencv/opencv-python/blob/master/LICENSE.txt) and its bundled third-party notices |

The complete dependency tree also includes the dependencies installed by these
packages. Their notices remain in the installed distributions. See
`python -m pip list` and `python -m pip show PACKAGE` inside `.venv` to inspect
the actual environment. The source license does not replace upstream model or
dependency terms.
