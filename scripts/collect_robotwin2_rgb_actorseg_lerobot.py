#!/usr/bin/env python3
"""Replay RoboTwin2 clean ALOHA trajectories and write RGB/actor-seg LeRobot datasets."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import ImageColor


ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", "/data/robotwin_runtime/third_party/RoboTwin"))
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))

from envs import CONFIGS_PATH  # noqa: E402
from script.collect_data import class_decorator, get_embodiment_config  # noqa: E402


CAMERA_KEY_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}
LEROBOT_CAMERA_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
MOTORS = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]
VIDEO_INFO = {
    "video.height": 480,
    "video.width": 640,
    "video.codec": "mpeg4",
    "video.pix_fmt": "yuv420p",
    "video.is_depth_map": False,
    "video.fps": 50,
    "video.channels": 3,
    "has_audio": False,
}


def load_demo_clean_args(task: str, source_task_dir: Path) -> dict[str, Any]:
    with (ROBOTWIN_ROOT / "task_config" / "demo_clean.yml").open("r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task
    args["save_path"] = str(source_task_dir)
    args["episode_num"] = 50
    args["use_seed"] = True
    args["collect_data"] = False
    args["render_freq"] = 0
    args["save_data"] = False
    args["need_plan"] = False
    args["eval_video_log"] = False
    args["data_type"] = {
        "rgb": True,
        "third_view": False,
        "depth": False,
        "pointcloud": False,
        "observer": False,
        "endpose": False,
        "qpos": True,
        "mesh_segmentation": False,
        "actor_segmentation": True,
    }
    args["camera"]["collect_head_camera"] = True
    args["camera"]["collect_wrist_camera"] = True

    embodiment_type = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def embodiment_file(name: str) -> str:
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"missing embodiment files for {name}")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = str(embodiment_type[0]) + "+" + str(embodiment_type[1])
    else:
        raise RuntimeError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    args["task_config"] = "demo_clean"
    return args


def first_prompt(value: Any) -> str:
    preferred = ("instruction", "prompt", "text", "language_instruction", "task_description", "seen")
    if isinstance(value, dict):
        for key in preferred:
            if key in value:
                found = first_prompt(value[key])
                if found:
                    return found
        for item in value.values():
            found = first_prompt(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = first_prompt(item)
            if found:
                return found
    elif isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return ""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def object_to_name(obj: Any) -> str:
    for attr in ("get_name", "name"):
        try:
            val = getattr(obj, attr)
            return val() if callable(val) else str(val)
        except Exception:
            pass
    return type(obj).__name__


def object_to_id(obj: Any):
    for attr in ("get_per_scene_id", "per_scene_id", "get_id", "id", "get_global_id", "global_id"):
        try:
            val = getattr(obj, attr)
            return int(val() if callable(val) else val)
        except Exception:
            pass
    for attr in ("get_entity", "entity"):
        try:
            val = getattr(obj, attr)
            entity = val() if callable(val) else val
            entity_id = object_to_id(entity)
            if entity_id is not None:
                return entity_id
        except Exception:
            pass
    return None


def actor_id_map(env: Any) -> dict[int, list[str]]:
    mapping: dict[int, list[str]] = {}

    def add(obj: Any, prefix: str = "") -> None:
        obj_id = object_to_id(obj)
        if obj_id is None:
            return
        name = object_to_name(obj)
        if prefix:
            name = f"{prefix}/{name}"
        mapping.setdefault(obj_id, [])
        if name not in mapping[obj_id]:
            mapping[obj_id].append(name)

    for getter_name in ("get_all_actors", "get_actors"):
        getter = getattr(env.scene, getter_name, None)
        if getter is None:
            continue
        try:
            for actor in getter():
                add(actor)
        except Exception:
            pass

    for getter_name in ("get_all_articulations", "get_articulations"):
        getter = getattr(env.scene, getter_name, None)
        if getter is None:
            continue
        try:
            for articulation in getter():
                art_name = object_to_name(articulation)
                links_getter = getattr(articulation, "get_links", None)
                links = links_getter() if links_getter is not None else getattr(articulation, "links", [])
                for link in links:
                    add(link, art_name)
        except Exception:
            pass

    robot = getattr(env, "robot", None)
    if robot is not None:
        for attr in ("left_entity", "right_entity"):
            entity = getattr(robot, attr, None)
            links_getter = getattr(entity, "get_links", None)
            if links_getter is None:
                continue
            try:
                for link in links_getter():
                    add(link)
            except Exception:
                pass

    return mapping


def camera_dict_to_lerobot(rgb_or_seg: dict[str, dict[str, np.ndarray]], key: str) -> dict[str, np.ndarray]:
    out = {}
    for source_key, target_key in CAMERA_KEY_MAP.items():
        if source_key not in rgb_or_seg:
            raise KeyError(f"missing camera `{source_key}` in rendered frame")
        if key not in rgb_or_seg[source_key]:
            raise KeyError(f"missing `{key}` for camera `{source_key}`")
        frame = rgb_or_seg[source_key][key]
        if frame.shape[:2] != (480, 640):
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST if "segmentation" in key else cv2.INTER_AREA)
        out[target_key] = frame.astype(np.uint8)
    return out


def make_state(env: Any) -> np.ndarray:
    left_jointstate = env.robot.get_left_arm_jointState()
    right_jointstate = env.robot.get_right_arm_jointState()
    return np.asarray(left_jointstate + right_jointstate, dtype=np.float32)


def jpeg_bytes(frames: list[np.ndarray]) -> tuple[list[bytes], int]:
    encoded = []
    max_len = 0
    for frame in frames:
        ok, arr = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        data = arr.tobytes()
        encoded.append(data)
        max_len = max(max_len, len(data))
    return encoded, max_len


def write_processed_hdf5(path: Path, states: np.ndarray, actions: np.ndarray, frames: dict[str, list[np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("action", data=actions.astype(np.float32))
        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=states.astype(np.float32))
        image = obs.create_group("images")
        for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            encoded, max_len = jpeg_bytes(frames[cam])
            image.create_dataset(cam, data=encoded, dtype=f"S{max_len}")


def write_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise RuntimeError(f"cannot write empty video: {path}")
    h, w = frames[0].shape[:2]
    tmp_path = path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {tmp_path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    tmp_path.replace(path)


def feature_stats(array: np.ndarray, axis=0, keepdims=False) -> dict[str, list]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims).tolist(),
        "max": np.max(array, axis=axis, keepdims=keepdims).tolist(),
        "mean": np.mean(array, axis=axis, keepdims=keepdims).tolist(),
        "std": np.std(array, axis=axis, keepdims=keepdims).tolist(),
        "count": [int(len(array))],
    }


def episode_stats(states: np.ndarray, actions: np.ndarray, timestamps: np.ndarray, frame_indices: np.ndarray, episode_index: int, global_indices: np.ndarray, task_indices: np.ndarray) -> dict[str, dict]:
    return {
        "observation.state": feature_stats(states, axis=0),
        "action": feature_stats(actions, axis=0),
        "timestamp": feature_stats(timestamps, axis=0, keepdims=True),
        "frame_index": feature_stats(frame_indices, axis=0, keepdims=True),
        "episode_index": feature_stats(np.full((len(states),), episode_index, dtype=np.int64), axis=0, keepdims=True),
        "index": feature_stats(global_indices, axis=0, keepdims=True),
        "task_index": feature_stats(task_indices, axis=0, keepdims=True),
    }


def info_json(total_episodes: int, total_frames: int, total_tasks: int, total_videos: int, fps: int) -> dict[str, Any]:
    features = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": [MOTORS]},
        "action": {"dtype": "float32", "shape": [14], "names": [MOTORS]},
    }
    for key in LEROBOT_CAMERA_KEYS:
        features[key] = {
            "dtype": "video",
            "shape": [480, 640, 3],
            "names": ["height", "width", "rgb"],
            "info": {**VIDEO_INFO, "video.fps": fps},
        }
    features.update(
        {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }
    )
    return {
        "codebase_version": "v2.1",
        "robot_type": "aloha",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_videos,
        "total_chunks": max((total_episodes + 999) // 1000, 1),
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parquet_table(states: np.ndarray, actions: np.ndarray, timestamps: np.ndarray, episode_index: int, global_start: int, task_index: int) -> pa.Table:
    length = len(states)
    return pa.table(
        {
            "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32())),
            "timestamp": pa.array(timestamps.tolist(), type=pa.float32()),
            "frame_index": pa.array(np.arange(length, dtype=np.int64)),
            "episode_index": pa.array(np.full((length,), episode_index, dtype=np.int64)),
            "index": pa.array(np.arange(global_start, global_start + length, dtype=np.int64)),
            "task_index": pa.array(np.full((length,), task_index, dtype=np.int64)),
        }
    )


def append_task(task_to_index: dict[str, int], task_rows: list[dict[str, Any]], task: str) -> int:
    if task in task_to_index:
        return task_to_index[task]
    idx = len(task_to_index)
    task_to_index[task] = idx
    task_rows.append({"task_index": idx, "task": task})
    return idx


def color_legend_text(color_legend: list[dict[str, Any]]) -> str:
    lines = ["Actor segmentation color legend (RGB -> actor/link names observed in this episode):"]
    for item in color_legend:
        names = ", ".join(item["names"]) if item["names"] else "unknown"
        lines.append(f"- {item['hex']} RGB={item['rgb']} actor_id={item['raw_actor_id']}: {names}")
    return "\n".join(lines)


def render_episode(task: str, source_task_dir: Path, episode_idx: int) -> dict[str, Any]:
    seeds = [int(x) for x in (source_task_dir / "seed.txt").read_text().split()]
    instruction_json = load_json(source_task_dir / "instructions" / f"episode{episode_idx}.json")
    prompt = first_prompt(instruction_json) or f"{task} episode {episode_idx}"

    args = load_demo_clean_args(task, source_task_dir)
    env = class_decorator(task)
    states_all: list[np.ndarray] = []
    rgb_all = {cam: [] for cam in CAMERA_KEY_MAP.values()}
    seg_all = {cam: [] for cam in CAMERA_KEY_MAP.values()}
    raw_ids_seen: set[int] = set()
    color_by_raw_id: dict[int, tuple[int, int, int]] = {}
    color_palette = np.array(
        [ImageColor.getrgb(color) for color in sorted(set(ImageColor.colormap.values()))],
        dtype=np.uint8,
    )

    try:
        env.setup_demo(now_ep_num=episode_idx, seed=seeds[episode_idx], **args)
        id_to_name = actor_id_map(env)
        traj_data = env.load_tran_data(episode_idx)
        replay_args = copy.deepcopy(args)
        replay_args["need_plan"] = False
        replay_args["left_joint_path"] = traj_data["left_joint_path"]
        replay_args["right_joint_path"] = traj_data["right_joint_path"]
        env.set_path_lst(replay_args)

        def take_picture():
            env._update_render()
            env.cameras.update_picture()
            rgb = camera_dict_to_lerobot(env.cameras.get_rgb(), "rgb")
            seg = camera_dict_to_lerobot(env.cameras.get_segmentation(level="actor"), "actor_segmentation")
            state = make_state(env)
            states_all.append(state)
            for cam in rgb_all:
                rgb_all[cam].append(rgb[cam])
                seg_all[cam].append(seg[cam])
            for camera_obj in [env.cameras.left_camera, env.cameras.right_camera, *env.cameras.static_camera_list]:
                labels = camera_obj.get_picture("Segmentation")[..., 1].astype(np.int64)
                for raw_id in np.unique(labels).tolist():
                    raw_ids_seen.add(int(raw_id))
                    color_by_raw_id[int(raw_id)] = tuple(int(x) for x in color_palette[int(raw_id) % 256])

        env._take_picture = take_picture
        info = env.play_once()
        replay_check_success = False
        warnings = []
        try:
            replay_check_success = bool(env.check_success())
        except Exception as exc:
            warnings.append(f"check_success raised {type(exc).__name__}: {exc}")
        if not replay_check_success:
            warnings.append("replay did not pass check_success()")
        env.close_env(clear_cache=True)

        if len(states_all) < 2:
            raise RuntimeError(f"episode has too few captured frames: {len(states_all)}")

        color_legend = []
        for raw_id in sorted(raw_ids_seen):
            rgb = color_by_raw_id.get(raw_id, (0, 0, 0))
            color_legend.append(
                {
                    "raw_actor_id": raw_id,
                    "rgb": list(rgb),
                    "hex": "#%02x%02x%02x" % rgb,
                    "names": id_to_name.get(raw_id, []),
                }
            )

        states_arr = np.stack(states_all).astype(np.float32)
        return {
            "status": "ok",
            "task": task,
            "source_episode": episode_idx,
            "seed": seeds[episode_idx],
            "prompt": prompt,
            "seg_prompt": f"{prompt}\n\n{color_legend_text(color_legend)}",
            "states": states_arr[:-1],
            "actions": states_arr[1:],
            "rgb_frames": {cam: frames[:-1] for cam, frames in rgb_all.items()},
            "seg_frames": {cam: frames[:-1] for cam, frames in seg_all.items()},
            "raw_frame_count": len(states_all),
            "frame_count": len(states_all) - 1,
            "actor_color_legend": color_legend,
            "instruction_json": instruction_json,
            "replay_info": info,
            "replay_check_success": replay_check_success,
            "warnings": warnings,
        }
    except Exception:
        try:
            env.close_env(clear_cache=True)
        except Exception:
            pass
        raise


def write_episode_outputs(
    root: Path,
    raw_root: Path | None,
    episode_index: int,
    task_prompt: str,
    states: np.ndarray,
    actions: np.ndarray,
    frames: dict[str, list[np.ndarray]],
    fps: int,
    task_to_index: dict[str, int],
    task_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
    global_start: int,
    metadata: dict[str, Any],
) -> int:
    task_index = append_task(task_to_index, task_rows, task_prompt)
    timestamps = np.arange(len(states), dtype=np.float32) / float(fps)
    chunk = episode_index // 1000

    for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        video_key = f"observation.images.{cam}"
        video_path = root / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"
        write_mp4(video_path, frames[cam], fps=fps)

    parquet_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(parquet_table(states, actions, timestamps, episode_index, global_start, task_index), parquet_path)

    if raw_root is not None:
        hdf5_path = raw_root / f"episode_{episode_index:06d}" / f"episode_{episode_index:06d}.hdf5"
        write_processed_hdf5(hdf5_path, states, actions, frames)
        (hdf5_path.parent / "instructions.json").write_text(
            json.dumps({"instructions": [task_prompt]}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (hdf5_path.parent / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    episode_rows.append(
        {
            "episode_index": episode_index,
            "tasks": [task_prompt],
            "length": int(len(states)),
            "source_task": metadata["task"],
            "source_episode": metadata["source_episode"],
            "seed": metadata["seed"],
        }
    )
    indices = np.arange(global_start, global_start + len(states), dtype=np.int64)
    frame_indices = np.arange(len(states), dtype=np.int64)
    task_indices = np.full((len(states),), task_index, dtype=np.int64)
    stats_rows.append(
        {
            "episode_index": episode_index,
            "stats": episode_stats(states, actions, timestamps, frame_indices, episode_index, indices, task_indices),
        }
    )
    return int(len(states))


def selected_tasks(source_root: Path, task_args: list[str] | None, max_tasks: int | None) -> list[str]:
    tasks = task_args if task_args else sorted(p.name for p in source_root.iterdir() if p.is_dir())
    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="/data/datasets/robotwin2_aloha_clean_sources")
    parser.add_argument("--rgb-root", default="/data/datasets/robotwin2_rgb_50tasks_5eps_lerobot/robotwin2.0")
    parser.add_argument("--seg-root", default="/data/datasets/robotwin2_actorseg_50tasks_5eps_lerobot/robotwin2.0")
    parser.add_argument("--raw-rgb-root", default="/data/datasets/robotwin2_rgb_50tasks_5eps_hdf5")
    parser.add_argument("--raw-seg-root", default="/data/datasets/robotwin2_actorseg_50tasks_5eps_hdf5")
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    rgb_root = Path(args.rgb_root)
    seg_root = Path(args.seg_root)
    raw_rgb_root = None if args.skip_raw else Path(args.raw_rgb_root)
    raw_seg_root = None if args.skip_raw else Path(args.raw_seg_root)
    paths = [rgb_root, seg_root]
    if raw_rgb_root is not None:
        paths.append(raw_rgb_root)
    if raw_seg_root is not None:
        paths.append(raw_seg_root)
    for path in paths:
        if args.overwrite and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    tasks = selected_tasks(source_root, args.tasks, args.max_tasks)
    manifest_rows = []
    rgb_task_to_index: dict[str, int] = {}
    seg_task_to_index: dict[str, int] = {}
    rgb_task_rows: list[dict[str, Any]] = []
    seg_task_rows: list[dict[str, Any]] = []
    rgb_episode_rows: list[dict[str, Any]] = []
    seg_episode_rows: list[dict[str, Any]] = []
    rgb_stats_rows: list[dict[str, Any]] = []
    seg_stats_rows: list[dict[str, Any]] = []
    rgb_global_index = 0
    seg_global_index = 0
    episode_index = 0

    for task in tasks:
        task_dir = source_root / task
        if not task_dir.exists():
            print(json.dumps({"task": task, "status": "missing_source"}, ensure_ascii=False), flush=True)
            continue
        for source_episode in range(args.start_episode, args.start_episode + args.episodes_per_task):
            try:
                rendered = render_episode(task, task_dir, source_episode)
                metadata = {
                    key: rendered[key]
                    for key in (
                        "task",
                        "source_episode",
                        "seed",
                        "prompt",
                        "seg_prompt",
                        "raw_frame_count",
                        "frame_count",
                        "actor_color_legend",
                        "instruction_json",
                        "replay_info",
                        "replay_check_success",
                        "warnings",
                    )
                }
                rgb_len = write_episode_outputs(
                    root=rgb_root,
                    raw_root=raw_rgb_root,
                    episode_index=episode_index,
                    task_prompt=rendered["prompt"],
                    states=rendered["states"],
                    actions=rendered["actions"],
                    frames=rendered["rgb_frames"],
                    fps=args.fps,
                    task_to_index=rgb_task_to_index,
                    task_rows=rgb_task_rows,
                    episode_rows=rgb_episode_rows,
                    stats_rows=rgb_stats_rows,
                    global_start=rgb_global_index,
                    metadata=metadata,
                )
                seg_len = write_episode_outputs(
                    root=seg_root,
                    raw_root=raw_seg_root,
                    episode_index=episode_index,
                    task_prompt=rendered["seg_prompt"],
                    states=rendered["states"],
                    actions=rendered["actions"],
                    frames=rendered["seg_frames"],
                    fps=args.fps,
                    task_to_index=seg_task_to_index,
                    task_rows=seg_task_rows,
                    episode_rows=seg_episode_rows,
                    stats_rows=seg_stats_rows,
                    global_start=seg_global_index,
                    metadata=metadata,
                )
                rgb_global_index += rgb_len
                seg_global_index += seg_len
                row = {
                    "episode_index": episode_index,
                    "task": task,
                    "source_episode": source_episode,
                    "status": "ok",
                    "frames": rgb_len,
                    "seed": rendered["seed"],
                }
                manifest_rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                episode_index += 1
            except Exception as exc:
                row = {
                    "episode_index": episode_index,
                    "task": task,
                    "source_episode": source_episode,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                manifest_rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                raise

    for root, task_rows, episode_rows, stats_rows, total_frames in (
        (rgb_root, rgb_task_rows, rgb_episode_rows, rgb_stats_rows, rgb_global_index),
        (seg_root, seg_task_rows, seg_episode_rows, seg_stats_rows, seg_global_index),
    ):
        (root / "meta").mkdir(parents=True, exist_ok=True)
        (root / "meta" / "info.json").write_text(
            json.dumps(
                info_json(
                    total_episodes=len(episode_rows),
                    total_frames=total_frames,
                    total_tasks=len(task_rows),
                    total_videos=len(episode_rows) * 3,
                    fps=args.fps,
                ),
                ensure_ascii=False,
                indent=4,
            )
            + "\n",
            encoding="utf-8",
        )
        write_jsonl(root / "meta" / "tasks.jsonl", task_rows)
        write_jsonl(root / "meta" / "episodes.jsonl", episode_rows)
        write_jsonl(root / "meta" / "episodes_stats.jsonl", stats_rows)
        write_jsonl(root / "meta" / "collection_manifest.jsonl", manifest_rows)

    print(
        json.dumps(
            {
                "status": "complete",
                "episodes": len(rgb_episode_rows),
                "rgb_root": str(rgb_root),
                "seg_root": str(seg_root),
                "raw_rgb_root": str(raw_rgb_root),
                "raw_seg_root": str(raw_seg_root),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
