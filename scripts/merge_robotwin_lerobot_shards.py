#!/usr/bin/env python3
"""Merge sharded RoboTwin LeRobot v2.1 datasets into one clean dataset root."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def feature_stats(array: np.ndarray, axis=0, keepdims=False) -> dict[str, list]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims).tolist(),
        "max": np.max(array, axis=axis, keepdims=keepdims).tolist(),
        "mean": np.mean(array, axis=axis, keepdims=keepdims).tolist(),
        "std": np.std(array, axis=axis, keepdims=keepdims).tolist(),
        "count": [int(len(array))],
    }


def episode_stats(
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
    frame_indices: np.ndarray,
    episode_index: int,
    global_indices: np.ndarray,
    task_indices: np.ndarray,
) -> dict[str, dict]:
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


def parquet_table(
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
    episode_index: int,
    global_start: int,
    task_index: int,
) -> pa.Table:
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


def prompt_from_episode(row: dict[str, Any]) -> str:
    tasks = row.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    if isinstance(tasks, str):
        return tasks
    raise RuntimeError(f"episode row has no task prompt: {row}")


def append_task(task_to_index: dict[str, int], task_rows: list[dict[str, Any]], task: str) -> int:
    if task in task_to_index:
        return task_to_index[task]
    idx = len(task_to_index)
    task_to_index[task] = idx
    task_rows.append({"task_index": idx, "task": task})
    return idx


def read_episode_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(path)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float32)
    return states, actions, timestamps


def copy_episode_videos(source_root: Path, output_root: Path, source_episode: int, output_episode: int) -> None:
    source_chunk = source_episode // 1000
    output_chunk = output_episode // 1000
    for video_key in LEROBOT_CAMERA_KEYS:
        source = source_root / "videos" / f"chunk-{source_chunk:03d}" / video_key / f"episode_{source_episode:06d}.mp4"
        target = output_root / "videos" / f"chunk-{output_chunk:03d}" / video_key / f"episode_{output_episode:06d}.mp4"
        if not source.exists():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def merge(input_roots: list[Path], output_root: Path, fps: int, overwrite: bool) -> None:
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    task_to_index: dict[str, int] = {}
    task_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    global_index = 0
    output_episode = 0

    for source_root in input_roots:
        episodes = sorted(read_jsonl(source_root / "meta" / "episodes.jsonl"), key=lambda row: int(row["episode_index"]))
        manifest_by_episode = {
            int(row["episode_index"]): row
            for row in read_jsonl(source_root / "meta" / "collection_manifest.jsonl")
            if "episode_index" in row
        }
        for episode in episodes:
            source_episode = int(episode["episode_index"])
            prompt = prompt_from_episode(episode)
            task_index = append_task(task_to_index, task_rows, prompt)
            source_chunk = source_episode // 1000
            source_parquet = source_root / "data" / f"chunk-{source_chunk:03d}" / f"episode_{source_episode:06d}.parquet"
            states, actions, timestamps = read_episode_arrays(source_parquet)

            copy_episode_videos(source_root, output_root, source_episode, output_episode)
            output_chunk = output_episode // 1000
            output_parquet = output_root / "data" / f"chunk-{output_chunk:03d}" / f"episode_{output_episode:06d}.parquet"
            output_parquet.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(parquet_table(states, actions, timestamps, output_episode, global_index, task_index), output_parquet)

            length = int(len(states))
            merged_episode = dict(episode)
            merged_episode["episode_index"] = output_episode
            merged_episode["tasks"] = [prompt]
            merged_episode["length"] = length
            merged_episode["source_shard"] = str(source_root)
            merged_episode["source_shard_episode_index"] = source_episode
            episode_rows.append(merged_episode)

            frame_indices = np.arange(length, dtype=np.int64)
            indices = np.arange(global_index, global_index + length, dtype=np.int64)
            task_indices = np.full((length,), task_index, dtype=np.int64)
            stats_rows.append(
                {
                    "episode_index": output_episode,
                    "stats": episode_stats(states, actions, timestamps, frame_indices, output_episode, indices, task_indices),
                }
            )

            manifest = dict(manifest_by_episode.get(source_episode, {}))
            manifest["episode_index"] = output_episode
            manifest["source_shard"] = str(source_root)
            manifest["source_shard_episode_index"] = source_episode
            manifest_rows.append(manifest)
            global_index += length
            output_episode += 1

    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    (output_root / "meta" / "info.json").write_text(
        json.dumps(
            info_json(
                total_episodes=len(episode_rows),
                total_frames=global_index,
                total_tasks=len(task_rows),
                total_videos=len(episode_rows) * len(LEROBOT_CAMERA_KEYS),
                fps=fps,
            ),
            ensure_ascii=False,
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(output_root / "meta" / "tasks.jsonl", task_rows)
    write_jsonl(output_root / "meta" / "episodes.jsonl", episode_rows)
    write_jsonl(output_root / "meta" / "episodes_stats.jsonl", stats_rows)
    write_jsonl(output_root / "meta" / "collection_manifest.jsonl", manifest_rows)
    print(
        json.dumps(
            {
                "status": "complete",
                "episodes": len(episode_rows),
                "frames": global_index,
                "tasks": len(task_rows),
                "output_root": str(output_root),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-roots", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    merge([Path(path) for path in args.input_roots], Path(args.output_root), args.fps, args.overwrite)


if __name__ == "__main__":
    main()
