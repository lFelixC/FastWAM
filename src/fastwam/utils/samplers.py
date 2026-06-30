from math import ceil
from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class ResumableStratifiedConcatSampler(Sampler[int]):
    """Sampler that gives each DataLoader micro-batch enough target child samples.

    This is intended for LogicalConcatDataset-like datasets that expose
    ``cumulative_sizes``. The sampler returns a flat index stream so the
    existing DataLoader(batch_size=...) path and Accelerate sharding can stay
    unchanged. It preserves the concat dataset's natural target fraction when
    that fraction already exceeds ``min_fraction``.
    """

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
        child_indices: list[int] | tuple[int, ...],
        min_fraction: float,
    ):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.child_indices = tuple(int(idx) for idx in child_indices)
        self.min_fraction = float(min_fraction)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

        if self.batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {self.batch_size}")
        if self.num_processes <= 0:
            raise ValueError(f"`num_processes` must be positive, got {self.num_processes}")
        if not (0.0 <= self.min_fraction <= 1.0):
            raise ValueError(f"`min_fraction` must be in [0, 1], got {self.min_fraction}")
        if not self.child_indices:
            raise ValueError("`child_indices` must contain at least one child dataset index.")

        self.target_indices, self.other_indices = self._split_indices()
        if self.min_fraction > 0.0 and not self.target_indices:
            raise ValueError("Stratified sampling requested target samples, but target index pool is empty.")

        self.natural_target_fraction = len(self.target_indices) / max(len(self.dataset), 1)
        self.target_fraction = max(self.min_fraction, self.natural_target_fraction)
        self.target_per_batch = int(ceil(self.batch_size * self.target_fraction))
        if self.min_fraction > 0.0:
            self.target_per_batch = max(self.target_per_batch, 1)
        self.target_per_batch = min(self.target_per_batch, self.batch_size)
        self.other_per_batch = self.batch_size - self.target_per_batch
        self.effective_target_fraction = self.target_per_batch / self.batch_size
        if self.target_per_batch > 0 and not self.target_indices:
            raise ValueError("Stratified sampling needs target samples, but target index pool is empty.")
        if self.other_per_batch > 0 and not self.other_indices:
            raise ValueError("Stratified sampling needs non-target samples, but non-target index pool is empty.")
        if self.target_per_batch > len(self.target_indices):
            raise ValueError(
                "`target_per_batch` is larger than the target index pool; "
                f"target_per_batch={self.target_per_batch}, target_pool={len(self.target_indices)}"
            )
        if self.other_per_batch > len(self.other_indices):
            raise ValueError(
                "`other_per_batch` is larger than the non-target index pool; "
                f"other_per_batch={self.other_per_batch}, other_pool={len(self.other_indices)}"
            )
        self.num_batches = int(ceil(len(self.dataset) / self.batch_size))
        self.total_samples = self.num_batches * self.batch_size

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def _split_indices(self) -> tuple[list[int], list[int]]:
        cumulative_sizes = getattr(self.dataset, "cumulative_sizes", None)
        if cumulative_sizes is None:
            raise TypeError(
                "`ResumableStratifiedConcatSampler` requires a concat dataset "
                "with a `cumulative_sizes` attribute."
            )

        target_children = set(self.child_indices)
        num_children = len(cumulative_sizes)
        invalid = sorted(idx for idx in target_children if idx < 0 or idx >= num_children)
        if invalid:
            raise ValueError(
                f"`child_indices` contains invalid child ids {invalid}; dataset has {num_children} children."
            )

        target_indices: list[int] = []
        other_indices: list[int] = []
        start = 0
        for child_idx, end in enumerate(cumulative_sizes):
            pool = target_indices if child_idx in target_children else other_indices
            pool.extend(range(start, int(end)))
            start = int(end)
        return target_indices, other_indices

    @staticmethod
    def _shuffled(indices: list[int], generator: torch.Generator) -> list[int]:
        if not indices:
            return []
        order = torch.randperm(len(indices), generator=generator).tolist()
        return [indices[i] for i in order]

    def _take(
        self,
        pool: list[int],
        shuffled: list[int],
        cursor: int,
        count: int,
        generator: torch.Generator,
    ) -> tuple[list[int], list[int], int]:
        out: list[int] = []
        while len(out) < count:
            if cursor >= len(shuffled):
                shuffled = self._shuffled(pool, generator)
                cursor = 0
            need = count - len(out)
            chunk = shuffled[cursor : cursor + need]
            out.extend(chunk)
            cursor += len(chunk)
        return out, shuffled, cursor

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)

        target_shuffled = self._shuffled(self.target_indices, g)
        other_shuffled = self._shuffled(self.other_indices, g)
        target_cursor = 0
        other_cursor = 0
        flat_indices: list[int] = []

        for _ in range(self.num_batches):
            batch: list[int] = []
            if self.target_per_batch:
                taken, target_shuffled, target_cursor = self._take(
                    self.target_indices,
                    target_shuffled,
                    target_cursor,
                    self.target_per_batch,
                    g,
                )
                batch.extend(taken)
            if self.other_per_batch:
                taken, other_shuffled, other_cursor = self._take(
                    self.other_indices,
                    other_shuffled,
                    other_cursor,
                    self.other_per_batch,
                    g,
                )
                batch.extend(taken)
            batch = [batch[i] for i in torch.randperm(len(batch), generator=g).tolist()]
            flat_indices.extend(batch)

        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            flat_indices = flat_indices[sample_offset:]
        return iter(flat_indices)

    def __len__(self) -> int:
        return self.total_samples
