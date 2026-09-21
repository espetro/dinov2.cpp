#!/usr/bin/env python3
"""Numerical parity check: dinov2-cli embeddings vs the PyTorch HF reference.

Runs dinov2-cli on one or more images, recomputes the same embeddings with
the HuggingFace checkpoint the GGUF was converted from, and reports cosine
similarity for the cls / pooled / patch-token vectors plus a top-1
classification comparison. Exits 0 only if every thresholded check passes.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageClassification,
)

PATCH_SIZE: Final[int] = 14
IMAGENET_MEAN: Final = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD: Final = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEFAULT_CLI: Final[str] = "./build-debug/bin/dinov2-cli"
DEFAULT_GGUF: Final[str] = "models/model.gguf"
FALLBACK_GGUF: Final[str] = "models/dinov2-small/model.gguf"
DEFAULT_HF_MODEL: Final[str] = "facebook/dinov2-small-imagenet1k-1-layer"
DEFAULT_IMAGE: Final[str] = "assets/tench.jpg"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=str, default=DEFAULT_CLI, help="path to the dinov2-cli binary")
    parser.add_argument(
        "--gguf",
        type=str,
        default=DEFAULT_GGUF,
        help=f"GGUF weight path (falls back to {FALLBACK_GGUF} when the default is absent)",
    )
    parser.add_argument("--hf-model", type=str, default=DEFAULT_HF_MODEL, help="HuggingFace reference model")
    parser.add_argument(
        "--image",
        type=str,
        action="append",
        default=None,
        help=f"input image; repeat or comma-separate for several (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument("--cls-threshold", type=float, default=0.999, help="min cosine for the cls vector")
    parser.add_argument(
        "--patches-threshold", type=float, default=0.99, help="min cosine for patch tokens (flat and per-token min)"
    )
    parser.add_argument("--pooled-threshold", type=float, default=0.999, help="min cosine for the pooled vector")
    parser.add_argument(
        "--prob-tolerance", type=float, default=0.05, help="max abs difference for the top-1 softmax probability"
    )
    return parser.parse_args()


@dataclass
class Check:
    name: str
    ours: str = ""
    ref: str = ""
    metric: str = ""
    threshold: str = ""
    passed: Optional[bool] = None  # None marks an informational row


def parse_images(values: Optional[List[str]]) -> List[str]:
    if not values:
        return [DEFAULT_IMAGE]
    out: List[str] = []
    for value in values:
        out.extend(v.strip() for v in value.split(",") if v.strip())
    return out


def resolve_gguf(requested: str) -> Path:
    path = Path(requested)
    if path.is_file():
        return path
    fallback = Path(FALLBACK_GGUF)
    if requested == DEFAULT_GGUF and fallback.is_file():
        return fallback
    raise FileNotFoundError(f"GGUF not found at '{requested}' (fallback '{FALLBACK_GGUF}' also absent)")


def run_cli_json(cli: Path, gguf: Path, image: Path, extra: Sequence[str]) -> dict:
    cmd = [str(cli), "-m", str(gguf), "-i", str(image), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"dinov2-cli exited {proc.returncode}: {' '.join(cmd)}\nstderr:\n{proc.stderr}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"failed to parse dinov2-cli stdout as JSON: {exc}\nstdout head: {proc.stdout[:300]}\nstderr:\n{proc.stderr}"
        )


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0.0 else 0.0


def feature_preprocess(image_path: Path, patch_size: int) -> torch.Tensor:
    """Replicate the CLI feature pipeline: upscale the ORIGINAL image to the
    next patch_size multiple per axis ((dim // patch + 1) * patch) with
    bicubic on float [0,1] pixels, then ImageNet mean/std normalize.
    Per-channel PIL 'F' resize keeps the float pixels instead of the uint8
    rounding Image.resize would apply to an RGB image. Returns [1, 3, H, W]."""
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    new_w = (img.width // patch_size + 1) * patch_size
    new_h = (img.height // patch_size + 1) * patch_size
    channels = [
        np.asarray(
            Image.fromarray(arr[..., c]).resize((new_w, new_h), Image.BICUBIC),
            dtype=np.float32,
        )
        for c in range(3)
    ]
    resized = np.stack(channels, axis=-1)
    normed = (resized - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(normed.transpose(2, 0, 1).copy()).unsqueeze(0)


def check_image(
    image_path: Path,
    cli: Path,
    gguf: Path,
    feature_model: torch.nn.Module,
    processor,
    classifier: torch.nn.Module,
    num_register_tokens: int,
    args: argparse.Namespace,
) -> List[Check]:
    checks: List[Check] = []

    # --- feature mode ---
    emb = run_cli_json(cli, gguf, image_path, ["--print-embeddings", "--print-patch-tokens"])
    ours_cls = np.asarray(emb["cls"], dtype=np.float32)
    ours_pooled = np.asarray(emb["pooled"], dtype=np.float32)
    ours_patches = np.asarray(emb["patches"], dtype=np.float32).reshape(emb["n_patches"], emb["hidden"])

    pixel_values = feature_preprocess(image_path, PATCH_SIZE)
    with torch.no_grad():
        ref_out = feature_model(pixel_values=pixel_values)
    lhs = ref_out.last_hidden_state[0]
    ref_cls = lhs[0].numpy()
    ref_patches = lhs[1 + num_register_tokens :].numpy()
    ref_pooled = np.concatenate([ref_cls, ref_patches.mean(axis=0)])

    checks.append(
        Check("n_patches", str(emb["n_patches"]), str(ref_patches.shape[0]), "match", "equal",
              emb["n_patches"] == ref_patches.shape[0] == ours_patches.shape[0])
    )
    checks.append(
        Check("hidden", str(emb["hidden"]), str(ref_patches.shape[1]), "match", "equal",
              emb["hidden"] == ref_patches.shape[1])
    )

    cls_cos = cosine(ours_cls, ref_cls)
    checks.append(
        Check("cls", f"norm {np.linalg.norm(ours_cls):.4f}", f"norm {np.linalg.norm(ref_cls):.4f}",
              f"cos {cls_cos:.6f}", f">= {args.cls_threshold}", cls_cos >= args.cls_threshold)
    )
    checks.append(Check("cls", "", "", f"max|d| {np.max(np.abs(ours_cls - ref_cls)):.6f}", "info"))

    pooled_cos = cosine(ours_pooled, ref_pooled)
    checks.append(
        Check("pooled", f"norm {np.linalg.norm(ours_pooled):.4f}", f"norm {np.linalg.norm(ref_pooled):.4f}",
              f"cos {pooled_cos:.6f}", f">= {args.pooled_threshold}", pooled_cos >= args.pooled_threshold)
    )
    checks.append(Check("pooled", "", "", f"max|d| {np.max(np.abs(ours_pooled - ref_pooled)):.6f}", "info"))

    flat_cos = cosine(ours_patches.ravel(), ref_patches.ravel())
    checks.append(
        Check("patches flat", f"{ours_patches.shape[0]}x{ours_patches.shape[1]}",
              f"{ref_patches.shape[0]}x{ref_patches.shape[1]}", f"cos {flat_cos:.6f}",
              f">= {args.patches_threshold}", flat_cos >= args.patches_threshold)
    )
    if ours_patches.shape == ref_patches.shape:
        row_cos = np.array([cosine(ours_patches[i], ref_patches[i]) for i in range(ours_patches.shape[0])])
        checks.append(Check("patches token", "", "", f"cos mean {row_cos.mean():.6f}", "info"))
        checks.append(
            Check("patches token", "", "", f"cos min {row_cos.min():.6f}", f">= {args.patches_threshold}",
                  row_cos.min() >= args.patches_threshold)
        )
    else:
        checks.append(Check("patches token", "", "", "shape mismatch", ">= equal", False))

    # --- classify mode ---
    clf_json = run_cli_json(cli, gguf, image_path, ["-c", "--print-embeddings"])
    ours_top = clf_json["topk"][0]

    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = classifier(**inputs).logits[0]
    ref_probs = torch.softmax(logits, dim=-1)
    ref_idx = int(ref_probs.argmax())
    ref_label = classifier.config.id2label.get(ref_idx, str(ref_idx))
    ref_prob = float(ref_probs[ref_idx])

    checks.append(
        Check("classify top-1", f"{ours_top['idx']} \"{ours_top['label']}\"", f"{ref_idx} \"{ref_label}\"",
              "idx/label", "equal",
              ours_top["idx"] == ref_idx and ours_top["label"] == ref_label)
    )
    prob_diff = abs(float(ours_top["prob"]) - ref_prob)
    checks.append(
        Check("classify prob", f"{float(ours_top['prob']):.6f}", f"{ref_prob:.6f}",
              f"|diff| {prob_diff:.6f}", f"< {args.prob_tolerance}", prob_diff < args.prob_tolerance)
    )

    return checks


def print_table(image_path: Path, checks: List[Check]) -> None:
    print(f"== {image_path}")
    print(f"{'check':<22}{'ours':<28}{'ref':<28}{'metric':<20}{'threshold':<14}result")
    for c in checks:
        result = "-" if c.passed is None else ("PASS" if c.passed else "FAIL")
        print(f"{c.name:<22}{c.ours:<28}{c.ref:<28}{c.metric:<20}{c.threshold:<14}{result}")


def main() -> int:
    args = get_args()

    cli = Path(args.cli)
    if not cli.is_file():
        print(f"error: CLI binary not found at '{cli}'", file=sys.stderr)
        return 1

    try:
        gguf = resolve_gguf(args.gguf)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    image_paths = [Path(p) for p in parse_images(args.image)]
    for p in image_paths:
        if not p.is_file():
            print(f"error: image not found at '{p}'", file=sys.stderr)
            return 1

    print(f"cli={cli}  gguf={gguf}  hf={args.hf_model}")
    try:
        config = AutoConfig.from_pretrained(args.hf_model)
        num_register_tokens = getattr(config, "num_register_tokens", 0) or 0
        feature_model = AutoModel.from_pretrained(args.hf_model).eval()
        processor = AutoImageProcessor.from_pretrained(args.hf_model)
        classifier = AutoModelForImageClassification.from_pretrained(args.hf_model).eval()
    except Exception as exc:
        print(f"error: failed to load HF reference '{args.hf_model}': {exc}", file=sys.stderr)
        return 1

    total = 0
    failed = 0
    for image_path in image_paths:
        try:
            checks = check_image(
                image_path, cli, gguf, feature_model, processor, classifier, num_register_tokens, args
            )
        except Exception as exc:
            print(f"== {image_path}\nerror: {exc}", file=sys.stderr)
            failed += 1
            total += 1
            continue
        print_table(image_path, checks)
        for c in checks:
            if c.passed is not None:
                total += 1
                failed += 0 if c.passed else 1

    if failed == 0:
        print(f"verdict: PASS ({total}/{total} checks passed)")
        return 0
    print(f"verdict: FAIL ({failed}/{total} checks failed)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
