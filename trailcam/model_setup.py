"""Download verified upstream weights and prepare the reusable YOLOE prompts.

No image data is sent anywhere. Downloads are atomic and a checksum mismatch
fails closed; a previously cached valid model never needs network access.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

LABELS = [
    "person", "bicycle", "motorcycle", "dog", "backpack", "baby stroller",
    "all-terrain vehicle", "utility terrain vehicle", "car", "truck", "bus",
    "golf cart", "tractor", "kick scooter", "suitcase", "duffel bag",
]
ASSET_ROOT = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
ATTRIBUTE_NAME = "PP-LCNet_x1_0_pedestrian_attribute_infer"
ASSETS = {
    "nano": {
        "filename": "yolo26n.pt", "url": ASSET_ROOT + "yolo26n.pt",
        "sha256": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    },
    "pose": {
        "filename": "yolo26n-pose.pt", "url": ASSET_ROOT + "yolo26n-pose.pt",
        "sha256": "eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9",
    },
    "megadetector": {
        "filename": "MDV6-yolov10-c.pt",
        "url": "https://zenodo.org/records/15398270/files/MDV6-yolov10-c.pt?download=1",
        "sha256": "21ee78a2d4887128e2a4920937d3295b493f44d788d13e1a635378c16dd74ef7",
    },
    "yoloe_base": {
        "filename": "yoloe-26s-seg.pt", "url": ASSET_ROOT + "yoloe-26s-seg.pt",
        "sha256": "48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654",
    },
    "text_encoder": {
        "filename": "mobileclip2_b.ts", "url": ASSET_ROOT + "mobileclip2_b.ts",
        "sha256": "35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f",
    },
    "attribute_archive": {
        "filename": ATTRIBUTE_NAME + ".tar",
        "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/" + ATTRIBUTE_NAME + ".tar",
        "sha256": "1b405e1c09484b3f4f18b435815e8473b9bf5d6c8aa8af871a667c0c9ca44f1c",
    },
}
ATTRIBUTE_FILES = {
    "inference.json": "1dc2805fdc5174cd627bc59ea4a9ab2cf6d5d3da6a3e568434dd442447fe87c0",
    "inference.pdiparams": "3a388425abc9187f485d5bfe6ad486a87e4895a1b33df098050683b13be8cea4",
    "inference.yml": "29b27d7f302391c70e42b486b6e2dba50984317c4815daddc340f0771d527243",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def download_verified(url: str, destination: Path, expected_sha256: str,
                      *, offline: bool = False, attempts: int = 3) -> Path:
    """Reuse only a correct cache file, and never install an incomplete download."""
    destination = Path(destination)
    if destination.is_file() and sha256(destination) == expected_sha256:
        return destination
    if offline:
        raise RuntimeError(f"Model missing or checksum mismatch in offline mode: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(attempts):
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".part")
        try:
            print(f"Downloading {destination.name} ({attempt + 1}/{attempts})...", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "trailcam-export/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            actual = sha256(temporary)
            if actual != expected_sha256:
                raise RuntimeError(f"SHA256 mismatch for {destination.name}: expected {expected_sha256}, got {actual}")
            os.replace(temporary, destination)
            return destination
        except (OSError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 4))
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download verified model {destination.name}: {last_error}") from last_error


def _asset(models_dir: Path, key: str, offline: bool) -> Path:
    item = ASSETS[key]
    return download_verified(item["url"], models_dir / item["filename"], item["sha256"], offline=offline)


def ensure_attributes(models_dir: Path, offline: bool = False) -> Path:
    target = models_dir / ATTRIBUTE_NAME
    if all((target / name).is_file() and sha256(target / name) == digest
           for name, digest in ATTRIBUTE_FILES.items()):
        return target
    archive = _asset(models_dir, "attribute_archive", offline)
    target.mkdir(parents=True, exist_ok=True)
    # Extract only the three expected regular files, never arbitrary archive paths.
    with tempfile.TemporaryDirectory(prefix="attributes-", dir=models_dir) as temporary_dir:
        staging = Path(temporary_dir)
        with tarfile.open(archive, "r:*") as source:
            for name, digest in ATTRIBUTE_FILES.items():
                matches = [member for member in source.getmembers()
                           if member.isfile() and Path(member.name).name == name]
                if len(matches) != 1:
                    raise RuntimeError(f"Expected one regular {name} in the attribute archive")
                member = matches[0]
                if member.size > 30_000_000:
                    raise RuntimeError(f"Unexpectedly large attribute file: {name}")
                output = staging / name
                stream = source.extractfile(member)
                if stream is None:
                    raise RuntimeError(f"Could not read {name} from attribute archive")
                with stream, output.open("wb") as handle:
                    shutil.copyfileobj(stream, handle)
                if sha256(output) != digest:
                    raise RuntimeError(f"Attribute file checksum mismatch: {name}")
        for name in ATTRIBUTE_FILES:
            os.replace(staging / name, target / name)
    return target


def ensure_yoloe(models_dir: Path, offline: bool = False) -> Path:
    """Bake text embeddings once, then omit the text encoder from normal runs."""
    import ultralytics

    target = models_dir / "yoloe-26s-trail-prompts.pt"
    metadata = target.with_suffix(".json")
    profile = {"base_sha256": ASSETS["yoloe_base"]["sha256"],
               "labels": LABELS, "ultralytics_version": ultralytics.__version__}
    if target.is_file() and metadata.is_file():
        try:
            saved = json.loads(metadata.read_text(encoding="utf-8-sig"))
            if all(saved.get(key) == value for key, value in profile.items()):
                digest = sha256(target)
                if saved.get("prepared_sha256") == digest:
                    return target
                # Compatibility with the original local study's saved checkpoint.
                if "prepared_sha256" not in saved and digest == "0519b3b142675964d01db8283447b69bc483cf49fe33184b1db4b2fbf452be33":
                    atomic_json(metadata, {**profile, "prepared_sha256": digest})
                    return target
        except (OSError, ValueError):
            pass
    base = _asset(models_dir, "yoloe_base", offline)
    _asset(models_dir, "text_encoder", offline)
    from ultralytics import YOLOE

    print("Preparing the 16 YOLOE trail categories once on CPU...", flush=True)
    previous = Path.cwd()
    temporary = target.with_name(target.stem + "." + uuid.uuid4().hex + ".tmp.pt")
    try:
        os.chdir(models_dir)  # MobileCLIP resolves its cache relative to cwd.
        model = YOLOE(str(base)).to("cpu")
        model.set_classes(LABELS)
        if list(model.names.values()) != LABELS:
            raise RuntimeError("Prepared YOLOE checkpoint has unexpected class labels")
        model.save(str(temporary))
        prepared_sha256 = sha256(temporary)
        os.replace(temporary, target)
        atomic_json(metadata, {**profile, "prepared_sha256": prepared_sha256})
    finally:
        os.chdir(previous)
        temporary.unlink(missing_ok=True)
    return target


def ensure_models(models_dir: Path, include_attributes: bool = True,
                  offline: bool = False) -> dict[str, Path]:
    """Return paths keyed by nano, pose, megadetector, yoloe, attributes."""
    models_dir = Path(models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = {key: _asset(models_dir, key, offline)
             for key in ("nano", "pose", "megadetector")}
    paths["yoloe"] = ensure_yoloe(models_dir, offline)
    if include_attributes:
        paths["attributes"] = ensure_attributes(models_dir, offline)
    return paths
