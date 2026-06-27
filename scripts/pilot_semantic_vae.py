#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SEMANTIC_PROMPT_TEMPLATE = (
    "A semantic segmentation video of a robot manipulation scene executing: {task}. "
    "Use the fixed RoboTwin RGB palette. Render only flat semantic colors with sharp "
    "object boundaries, no texture, no lighting, no shadows, and no natural RGB appearance."
)


def _import_runtime_deps(require_torch: bool):
    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit(
            "Missing runtime dependency. Run this script inside the FastWAM training "
            "environment with numpy/pillow installed."
        ) from exc
    torch = None
    if require_torch:
        try:
            import torch
        except ImportError as exc:
            raise SystemExit(
                "Missing torch. Use --skip-vae for target-only diagnostics, or run "
                "inside the FastWAM training environment for VAE reconstruction."
            ) from exc
    return np, torch, Image, ImageDraw


def _numeric_files(root: Path, suffix: str) -> list[Path]:
    files = []
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() == suffix and path.stem.isdigit():
            files.append(path)
    return [path for _, path in sorted((int(path.stem), path) for path in files)]


def _select_indices(num_available: int, num_raw_frames: int, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError(f"`stride` must be positive, got {stride}.")
    if num_available >= num_raw_frames:
        indices = list(range(0, num_raw_frames, stride))
    elif (num_available - 1) % 4 == 0:
        indices = list(range(num_available))
    else:
        raise ValueError(
            "Not enough frames for FastWAM-style sampling and available frames do not "
            f"already satisfy T % 4 == 1: available={num_available}, "
            f"num_raw_frames={num_raw_frames}, stride={stride}."
        )
    if (len(indices) - 1) % 4 != 0:
        raise ValueError(
            f"Selected video length must satisfy T % 4 == 1, got T={len(indices)}."
        )
    return indices


def _as_uint8_rgb(array, *, np):
    arr = np.asarray(array)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {arr.shape}.")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.shape[2] != 3:
        raise ValueError(f"Expected 3 RGB channels, got shape {arr.shape}.")
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max(initial=0.0) <= 1.0:
            arr = arr * 255.0
        return arr.clip(0, 255).astype(np.uint8)
    return arr.clip(0, 255).astype(np.uint8)


def _resize_uint8_hwc(frame, size_wh: tuple[int, int], *, image_cls, interpolation):
    return image_cls.fromarray(frame).resize(size_wh, resample=interpolation)


def _robotwin_layout(
    per_camera_frames: dict[str, list],
    camera_names: list[str],
    *,
    np,
    image_cls,
    interpolation,
):
    if len(camera_names) != 3:
        raise ValueError(f"RoboTwin layout expects exactly 3 cameras, got {camera_names}.")
    lengths = [len(per_camera_frames[name]) for name in camera_names]
    if len(set(lengths)) != 1:
        raise ValueError(f"Camera frame count mismatch: {dict(zip(camera_names, lengths))}.")

    out_frames = []
    for frame_idx in range(lengths[0]):
        top = _resize_uint8_hwc(
            per_camera_frames[camera_names[0]][frame_idx],
            (320, 256),
            image_cls=image_cls,
            interpolation=interpolation,
        )
        left = _resize_uint8_hwc(
            per_camera_frames[camera_names[1]][frame_idx],
            (160, 128),
            image_cls=image_cls,
            interpolation=interpolation,
        )
        right = _resize_uint8_hwc(
            per_camera_frames[camera_names[2]][frame_idx],
            (160, 128),
            image_cls=image_cls,
            interpolation=interpolation,
        )
        bottom = image_cls.new("RGB", (320, 128))
        bottom.paste(left, (0, 0))
        bottom.paste(right, (160, 0))
        canvas = image_cls.new("RGB", (320, 384))
        canvas.paste(top, (0, 0))
        canvas.paste(bottom, (0, 256))
        out_frames.append(np.asarray(canvas, dtype=np.uint8))
    return np.stack(out_frames, axis=0)


def _load_from_pkl_cache(
    cache_dir: Path,
    camera_names: list[str],
    semantic_key: str,
    num_raw_frames: int,
    stride: int,
    *,
    np,
    image_cls,
):
    pkl_files = _numeric_files(cache_dir, ".pkl")
    if not pkl_files:
        raise FileNotFoundError(f"No numeric .pkl files found in {cache_dir}.")
    indices = _select_indices(len(pkl_files), num_raw_frames=num_raw_frames, stride=stride)
    selected = [pkl_files[i] for i in indices]

    rgb_frames = {name: [] for name in camera_names}
    semantic_frames = {name: [] for name in camera_names}

    for path in selected:
        with path.open("rb") as f:
            payload = pickle.load(f)
        obs = payload.get("observation")
        if not isinstance(obs, dict):
            raise KeyError(f"Missing `observation` dict in {path}.")
        for camera_name in camera_names:
            camera_obs = obs.get(camera_name)
            if not isinstance(camera_obs, dict):
                raise KeyError(f"Missing camera `{camera_name}` in {path}.")
            if "rgb" not in camera_obs:
                raise KeyError(f"Missing `{camera_name}.rgb` in {path}.")
            if semantic_key not in camera_obs:
                raise KeyError(
                    f"Missing `{camera_name}.{semantic_key}` in {path}. "
                    "Collect a small RoboTwin episode with actor_segmentation or "
                    "mesh_segmentation enabled first."
                )
            rgb_frames[camera_name].append(_as_uint8_rgb(camera_obs["rgb"], np=np))
            semantic_frames[camera_name].append(
                _as_uint8_rgb(camera_obs[semantic_key], np=np)
            )

    rgb_video = _robotwin_layout(
        rgb_frames,
        camera_names,
        np=np,
        image_cls=image_cls,
        interpolation=image_cls.Resampling.BILINEAR,
    )
    semantic_video = _robotwin_layout(
        semantic_frames,
        camera_names,
        np=np,
        image_cls=image_cls,
        interpolation=image_cls.Resampling.NEAREST,
    )
    return {
        "rgb_video": rgb_video,
        "semantic_video": semantic_video,
        "selected_indices": indices,
        "source": str(cache_dir),
    }


def _decode_jpeg_bytes(value, *, image_cls, np):
    if isinstance(value, (bytes, bytearray)):
        buf = bytes(value).rstrip(b"\0")
    elif hasattr(value, "tobytes"):
        buf = value.tobytes().rstrip(b"\0")
    else:
        raise TypeError(f"Unsupported encoded image item: {type(value)}")
    with image_cls.open(io.BytesIO(buf)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _read_hdf5_image_sequence(dataset, indices: list[int], *, image_cls, np):
    frames = []
    for idx in indices:
        value = dataset[idx]
        if getattr(dataset, "dtype", None) is not None and dataset.dtype.kind == "S":
            frames.append(_decode_jpeg_bytes(value, image_cls=image_cls, np=np))
        else:
            frames.append(_as_uint8_rgb(value, np=np))
    return frames


def _load_from_hdf5(
    hdf5_path: Path,
    camera_names: list[str],
    semantic_key: str,
    num_raw_frames: int,
    stride: int,
    *,
    np,
    image_cls,
):
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("Missing h5py; install it or use --pkl-cache-dir.") from exc

    with h5py.File(hdf5_path, "r") as f:
        obs = f.get("observation")
        if obs is None:
            raise KeyError(f"Missing `observation` group in {hdf5_path}.")
        first_camera = obs.get(camera_names[0])
        if first_camera is None:
            raise KeyError(f"Missing camera `{camera_names[0]}` in {hdf5_path}.")
        if semantic_key not in first_camera:
            raise KeyError(
                f"Missing `{camera_names[0]}.{semantic_key}` in {hdf5_path}. "
                "Collect with actor_segmentation or mesh_segmentation enabled."
            )
        num_available = int(first_camera[semantic_key].shape[0])
        indices = _select_indices(num_available, num_raw_frames=num_raw_frames, stride=stride)

        rgb_frames = {}
        semantic_frames = {}
        for camera_name in camera_names:
            camera_group = obs.get(camera_name)
            if camera_group is None:
                raise KeyError(f"Missing camera `{camera_name}` in {hdf5_path}.")
            if "rgb" not in camera_group:
                raise KeyError(f"Missing `{camera_name}.rgb` in {hdf5_path}.")
            if semantic_key not in camera_group:
                raise KeyError(f"Missing `{camera_name}.{semantic_key}` in {hdf5_path}.")
            rgb_frames[camera_name] = _read_hdf5_image_sequence(
                camera_group["rgb"],
                indices,
                image_cls=image_cls,
                np=np,
            )
            semantic_frames[camera_name] = _read_hdf5_image_sequence(
                camera_group[semantic_key],
                indices,
                image_cls=image_cls,
                np=np,
            )

    rgb_video = _robotwin_layout(
        rgb_frames,
        camera_names,
        np=np,
        image_cls=image_cls,
        interpolation=image_cls.Resampling.BILINEAR,
    )
    semantic_video = _robotwin_layout(
        semantic_frames,
        camera_names,
        np=np,
        image_cls=image_cls,
        interpolation=image_cls.Resampling.NEAREST,
    )
    return {
        "rgb_video": rgb_video,
        "semantic_video": semantic_video,
        "selected_indices": indices,
        "source": str(hdf5_path),
    }


def _make_synthetic_video(num_frames: int, *, np):
    if (num_frames - 1) % 4 != 0:
        raise ValueError(f"Synthetic T must satisfy T % 4 == 1, got {num_frames}.")
    palette = np.array(
        [
            [0, 0, 0],
            [220, 20, 60],
            [0, 128, 255],
            [255, 215, 0],
            [34, 139, 34],
            [255, 255, 255],
        ],
        dtype=np.uint8,
    )
    frames = np.zeros((num_frames, 384, 320, 3), dtype=np.uint8)
    frames[:, :, :] = palette[0]
    for t in range(num_frames):
        x0 = 30 + t * 8
        y0 = 45 + t * 4
        frames[t, 20:260, 15:305] = palette[5]
        frames[t, y0:y0 + 82, x0:x0 + 92] = palette[1]
        frames[t, 205:300, 30 + t * 3:130 + t * 3] = palette[2]
        frames[t, 120:215, 190 - t * 5:270 - t * 5] = palette[3]
        frames[t, 305:360, 100:230] = palette[4]
    return {
        "rgb_video": frames.copy(),
        "semantic_video": frames,
        "selected_indices": list(range(num_frames)),
        "source": "synthetic",
    }


def _to_video_tensor(video_hwtc, *, torch):
    tensor = torch.from_numpy(video_hwtc).permute(3, 0, 1, 2).contiguous()
    return tensor.float().div(127.5).sub(1.0)


def _to_uint8_hwtc(video_cthw, *, torch):
    video = video_cthw.detach().float().clamp(-1, 1)
    video = video.add(1.0).mul(127.5).round().to(torch.uint8)
    return video.permute(1, 2, 3, 0).cpu().numpy()


def _resolve_dtype(dtype_name: str, *, torch):
    key = dtype_name.lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}.")


def _load_vae(args, *, torch):
    from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs

    if args.model_base_path:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(Path(args.model_base_path).expanduser())
    torch_dtype = _resolve_dtype(args.dtype, torch=torch)
    _, _, vae_config, _ = _resolve_configs(
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        redirect_common_files=args.redirect_common_files,
    )
    vae_config.download_if_necessary()
    vae = _load_registered_model(
        vae_config.path,
        "wan_video_vae",
        torch_dtype=torch_dtype,
        device=args.device,
    )
    vae.eval()
    return vae, str(vae_config.path), torch_dtype


def _run_vae_reconstruction(video_cthw, args, *, torch):
    vae, vae_path, torch_dtype = _load_vae(args, torch=torch)
    video_bcthw = video_cthw.unsqueeze(0).to(device=args.device, dtype=torch_dtype)
    with torch.no_grad():
        latents = vae.encode(video_bcthw, device=args.device, tiled=False)
        recon = vae.decode(latents, device=args.device, tiled=args.tiled_decode)
    recon_cthw = recon[0].detach().cpu()
    return recon_cthw, {
        "vae_path": vae_path,
        "latent_shape": list(latents.shape),
        "dtype": str(torch_dtype).replace("torch.", ""),
    }


def _unique_palette(video_hwtc, *, np):
    flat = video_hwtc.reshape(-1, 3)
    return np.unique(flat, axis=0)


def _palette_metrics(input_hwtc, recon_hwtc, max_palette_colors: int, *, np, torch):
    palette_np = _unique_palette(input_hwtc, np=np)
    if len(palette_np) > max_palette_colors:
        palette_np = palette_np[:max_palette_colors]
    palette = torch.from_numpy(palette_np.astype(np.float32))
    flat = torch.from_numpy(recon_hwtc.reshape(-1, 3).astype(np.float32))
    min_l1_chunks = []
    nearest_indices = []
    chunk_size = 262_144
    for start in range(0, flat.shape[0], chunk_size):
        cur = flat[start:start + chunk_size]
        dist = (cur[:, None, :] - palette[None, :, :]).abs().mean(dim=2)
        min_l1, nearest = dist.min(dim=1)
        min_l1_chunks.append(min_l1)
        nearest_indices.append(nearest)
    min_l1 = torch.cat(min_l1_chunks)
    nearest = torch.cat(nearest_indices)
    quantized = palette[nearest].round().clamp(0, 255).to(torch.uint8).numpy()
    quantized = quantized.reshape(recon_hwtc.shape)
    return {
        "palette": palette_np.tolist(),
        "palette_count": int(len(palette_np)),
        "nearest_palette_l1_mean_0_255": float(min_l1.mean().item()),
        "nearest_palette_l1_p95_0_255": float(torch.quantile(min_l1, 0.95).item()),
        "nearest_palette_within_8_frac": float((min_l1 <= 8).float().mean().item()),
        "nearest_palette_within_16_frac": float((min_l1 <= 16).float().mean().item()),
        "quantized_recon": quantized,
    }


def _save_video_frames(video_hwtc, out_dir: Path, prefix: str, *, image_cls):
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(video_hwtc):
        image_cls.fromarray(frame).save(out_dir / f"{prefix}_{idx:02d}.png")


def _make_labeled_tile(frame, label: str, *, image_cls, image_draw):
    img = image_cls.fromarray(frame).convert("RGB")
    label_h = 22
    canvas = image_cls.new("RGB", (img.width, img.height + label_h), color=(255, 255, 255))
    canvas.paste(img, (0, label_h))
    draw = image_draw.Draw(canvas)
    draw.text((6, 4), label, fill=(0, 0, 0))
    return canvas


def _save_contact_sheet(rows: list[tuple[str, object]], out_path: Path, *, image_cls, image_draw):
    if not rows:
        return
    num_frames = len(rows[0][1])
    if num_frames <= 3:
        frame_ids = list(range(num_frames))
    else:
        frame_ids = sorted(set([0, num_frames // 2, num_frames - 1]))

    tiles = []
    for label, video in rows:
        row_tiles = [
            _make_labeled_tile(video[i], f"{label} t={i}", image_cls=image_cls, image_draw=image_draw)
            for i in frame_ids
        ]
        tiles.append(row_tiles)

    tile_w, tile_h = tiles[0][0].size
    sheet = image_cls.new("RGB", (tile_w * len(frame_ids), tile_h * len(rows)), color=(240, 240, 240))
    for row_idx, row_tiles in enumerate(tiles):
        for col_idx, tile in enumerate(row_tiles):
            sheet.paste(tile, (col_idx * tile_w, row_idx * tile_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def _write_prompt(task: str, out_dir: Path):
    prompt = SEMANTIC_PROMPT_TEMPLATE.format(task=task)
    (out_dir / "semantic_prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    return prompt


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Pilot check for Vision-Banana-style semantic RGB targets in FastWAM. "
            "It builds a FastWAM RoboTwin-layout semantic video, runs Wan VAE "
            "reconstruction, and saves palette/reconstruction diagnostics."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pkl-cache-dir",
        type=Path,
        help=(
            "RoboTwin raw .cache/episodeN folder containing numeric .pkl files "
            "with rgb and actor_segmentation/mesh_segmentation."
        ),
    )
    input_group.add_argument(
        "--hdf5-path",
        type=Path,
        help=(
            "RoboTwin episode HDF5 produced with rgb and "
            "actor_segmentation/mesh_segmentation enabled."
        ),
    )
    input_group.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a synthetic semantic video. Useful for checking VAE palette drift.",
    )
    parser.add_argument("--semantic-key", default="actor_segmentation")
    parser.add_argument(
        "--camera-names",
        default="head_camera,left_camera,right_camera",
        help="Comma-separated RoboTwin camera names in FastWAM layout order.",
    )
    parser.add_argument("--num-raw-frames", type=int, default=33)
    parser.add_argument("--action-video-freq-ratio", type=int, default=4)
    parser.add_argument("--synthetic-num-frames", type=int, default=9)
    parser.add_argument("--task", default="the robot executes the manipulation instruction")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pilot_semantic_vae"))
    parser.add_argument("--skip-vae", action="store_true", help="Only save input target diagnostics.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--tiled-decode", action="store_true")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--tokenizer-model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--model-base-path", type=Path, default=None)
    parser.add_argument(
        "--redirect-common-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match FastWAM model.redirect_common_files behavior for locating the VAE.",
    )
    parser.add_argument("--max-palette-colors", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None):
    args = parse_args(argv)
    np, torch, Image, ImageDraw = _import_runtime_deps(require_torch=not args.skip_vae)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    camera_names = [name.strip() for name in args.camera_names.split(",") if name.strip()]
    if args.synthetic:
        loaded = _make_synthetic_video(args.synthetic_num_frames, np=np)
    elif args.hdf5_path is not None:
        loaded = _load_from_hdf5(
            args.hdf5_path.expanduser(),
            camera_names,
            args.semantic_key,
            args.num_raw_frames,
            args.action_video_freq_ratio,
            np=np,
            image_cls=Image,
        )
    else:
        loaded = _load_from_pkl_cache(
            args.pkl_cache_dir.expanduser(),
            camera_names,
            args.semantic_key,
            args.num_raw_frames,
            args.action_video_freq_ratio,
            np=np,
            image_cls=Image,
        )

    rgb_video = loaded["rgb_video"]
    semantic_video = loaded["semantic_video"]
    _save_video_frames(rgb_video, args.output_dir / "frames", "rgb_condition", image_cls=Image)
    _save_video_frames(semantic_video, args.output_dir / "frames", "semantic_target", image_cls=Image)

    prompt = _write_prompt(args.task, args.output_dir)
    metrics = {
        "source": loaded["source"],
        "selected_indices": loaded["selected_indices"],
        "semantic_shape_hwtc": list(semantic_video.shape),
        "prompt": prompt,
        "skip_vae": bool(args.skip_vae),
    }

    rows = [("rgb condition", rgb_video), ("semantic target", semantic_video)]
    if not args.skip_vae:
        semantic_tensor = _to_video_tensor(semantic_video, torch=torch)
        recon_tensor, vae_info = _run_vae_reconstruction(semantic_tensor, args, torch=torch)
        recon_video = _to_uint8_hwtc(recon_tensor, torch=torch)
        input_01 = torch.from_numpy(semantic_video).float().div(255.0)
        recon_01 = torch.from_numpy(recon_video).float().div(255.0)
        mse = (input_01 - recon_01).pow(2).mean().item()
        psnr = float("inf") if mse <= 0 else -10.0 * math.log10(mse)
        palette_info = _palette_metrics(
            semantic_video,
            recon_video,
            args.max_palette_colors,
            np=np,
            torch=torch,
        )
        quantized_recon = palette_info.pop("quantized_recon")
        _save_video_frames(recon_video, args.output_dir / "frames", "semantic_vae_recon", image_cls=Image)
        _save_video_frames(
            quantized_recon,
            args.output_dir / "frames",
            "semantic_vae_recon_nearest_palette",
            image_cls=Image,
        )
        rows.extend(
            [
                ("vae recon", recon_video),
                ("nearest palette", quantized_recon),
            ]
        )
        metrics.update(
            {
                "vae": vae_info,
                "reconstruction_mse_0_1": float(mse),
                "reconstruction_psnr_db": float(psnr),
                **palette_info,
            }
        )
    else:
        palette = _unique_palette(semantic_video, np=np)
        metrics.update(
            {
                "palette_count": int(len(palette)),
                "palette": palette[: args.max_palette_colors].tolist(),
            }
        )

    _save_contact_sheet(rows, args.output_dir / "contact_sheet.png", image_cls=Image, image_draw=ImageDraw)
    metrics_path = args.output_dir / "pilot_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[pilot] wrote {metrics_path}")
    print(f"[pilot] wrote {args.output_dir / 'contact_sheet.png'}")
    if args.skip_vae:
        print("[pilot] skipped VAE reconstruction")
    else:
        print(
            "[pilot] psnr={:.3f}dB palette_l1_mean={:.3f}".format(
                metrics["reconstruction_psnr_db"],
                metrics["nearest_palette_l1_mean_0_255"],
            )
        )


if __name__ == "__main__":
    main()
