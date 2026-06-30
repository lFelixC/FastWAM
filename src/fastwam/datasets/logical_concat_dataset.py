from bisect import bisect_right
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


class LogicalConcatDataset(torch.utils.data.Dataset):
    """Concatenate instantiated or Hydra-configured datasets without rewriting data."""

    def __init__(self, datasets, shape_meta=None, processor=None, **metadata):
        if not datasets:
            raise ValueError("`datasets` must contain at least one dataset config.")

        self.datasets: list[torch.utils.data.Dataset] = []
        self._repeats: list[int] = []
        self.cumulative_sizes: list[int] = []
        total = 0

        for entry in datasets:
            dataset_cfg, repeat = self._parse_entry(entry)
            dataset = instantiate(dataset_cfg)
            if len(dataset) <= 0:
                raise ValueError(f"Child dataset has non-positive length: {len(dataset)}")
            self.datasets.append(dataset)
            self._repeats.append(repeat)
            total += len(dataset) * repeat
            self.cumulative_sizes.append(total)

        self.shape_meta = shape_meta
        self.processor = processor
        for key, value in metadata.items():
            setattr(self, key, value)
        self.lerobot_dataset = getattr(self.datasets[0], "lerobot_dataset", None)

    @staticmethod
    def _parse_entry(entry: Any):
        if isinstance(entry, (DictConfig, dict)) and "dataset" in entry:
            repeat = int(entry.get("repeat", 1))
            if repeat <= 0:
                raise ValueError(f"`repeat` must be positive, got {repeat}")
            return entry["dataset"], repeat
        return entry, 1

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        prev = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
        local_idx = (idx - prev) % len(self.datasets[dataset_idx])
        return self.datasets[dataset_idx][local_idx]
