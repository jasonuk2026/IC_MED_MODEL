"""
PyTorch Dataset for lazy row-by-row loading of EHRShot parquet files.

Each parquet file has these columns:
    patient_id, task, split, prediction_time, label,
    n_events_in_history, event_history, n_tokens

Usage:
    dataset = EHRShotDataset("path/to/task.parquet", split="train")
    # Optionally provide a transform to convert raw dicts to model inputs:
    dataset = EHRShotDataset("path/to/task.parquet", split="train", transform=my_fn)
    loader  = DataLoader(dataset, batch_size=8, num_workers=4)
"""

import bisect
import math

import pyarrow.parquet as pq
import torch.distributed as dist

from pathlib import Path
from typing import Callable, Optional, Iterator

from torch.utils.data import Dataset, Sampler



class EHRShotDataset(Dataset):
    """Lazy-loading Dataset over a single EHRShot parquet file.

    Each __getitem__ call reads one row group from disk and returns the
    requested row.  The ParquetFile handle is opened lazily (on first
    __getitem__ call) so fork-based DataLoader workers each get their own
    handle without sharing file descriptors.

    Args:
        parquet_path: Path to the .parquet file produced by extract_new_diagnosis.py.
        split:        If given ("train", "val", "test"), restrict to rows whose
                      `split` column matches.  Reads only the split column at
                      init time, not the full file.
        transform:    Optional callable applied to each raw row dict before
                      returning.  Fill this in to convert strings / timestamps
                      to tensors or whatever your model expects.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        split: Optional[str] = None,
        transform: Optional[Callable[[dict], object]] = None,
    ):
        self.parquet_path = str(parquet_path)
        self.transform = transform

        # ------------------------------------------------------------------ #
        # Read parquet metadata only — no row data loaded yet.
        # ------------------------------------------------------------------ #
        pf = pq.ParquetFile(self.parquet_path)
        meta = pf.metadata

        # Build a list of cumulative row-group start offsets for O(log n) lookup.
        self._rg_starts: list[int] = []
        cumulative = 0
        for rg in range(meta.num_row_groups):
            self._rg_starts.append(cumulative)
            cumulative += meta.row_group(rg).num_rows
        self._total_rows: int = cumulative

        # ------------------------------------------------------------------ #
        # Build index list: indices into [0, total_rows) that belong to
        # the requested split (or all rows if split is None).
        # ------------------------------------------------------------------ #
        if split is not None:
            splits = pq.read_table(self.parquet_path, columns=["split"])["split"].to_pylist()
            self._indices: list[int] = [i for i, s in enumerate(splits) if s == split]
        else:
            self._indices = list(range(self._total_rows))

        # ParquetFile handle — opened lazily so fork-based workers each get
        # their own handle without sharing file descriptors.
        self._pf: Optional[pq.ParquetFile] = None

    # ---------------------------------------------------------------------- #
    # PyTorch Dataset interface
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        abs_idx = self._indices[idx]
        row = self._read_row(abs_idx)
        if self.transform is not None:
            row = self.transform(row)
        return row

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _read_row(self, abs_idx: int) -> dict:
        """Read a single row from disk by loading its row group."""
        rg_idx, offset = self._locate(abs_idx)
        if self._pf is None:
            self._pf = pq.ParquetFile(self.parquet_path)
        rg_data = self._pf.read_row_group(rg_idx).to_pydict()
        return {col: rg_data[col][offset] for col in rg_data}

    def _locate(self, abs_idx: int) -> tuple[int, int]:
        """Map an absolute row index to (row_group_index, offset_within_group)."""
        rg_idx = bisect.bisect_right(self._rg_starts, abs_idx) - 1
        offset = abs_idx - self._rg_starts[rg_idx]
        return rg_idx, offset

class ContiguousDistributedSampler(Sampler[int]):
    """Splits the dataset into contiguous per-rank chunks.

    Unlike the standard DistributedSampler (which interleaves indices across
    ranks and destroys sort order), this sampler assigns each rank a single
    contiguous slice so that token-length ordering is preserved within each rank.

    Args:
        dataset:      The dataset to sample from.
        num_replicas: Number of processes (GPUs).  Defaults to dist world size.
        rank:         Rank of this process.  Defaults to dist rank.
        drop_last:    Drop tail samples that don't divide evenly across ranks.
                      Default False (pad the last rank instead).
    """

    def __init__(
        self,
        dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        drop_last: bool = False,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last

        n = len(dataset)
        if drop_last:
            n = (n // num_replicas) * num_replicas
        self._n = n

        self.num_samples = math.ceil(n / num_replicas) # samples per rank (may be uneven if drop_last=False)

    def __iter__(self) -> Iterator[int]:
        start = self.rank * self.num_samples
        end = min(start + self.num_samples, self._n)
        return iter(range(start, end))

    def __len__(self) -> int:
        return min(self.num_samples, max(0, self._n - self.rank * self.num_samples))
