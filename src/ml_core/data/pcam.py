from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    """
    PatchCamelyon (PCAM) Dataset reader for HDF5.

    Requirements:
    - H5 lazy loading: do NOT load the full H5 into memory; open file on first access.
    - Numerical clipping: clip pixel values to a safe range to handle corrupted samples.
    - Optional heuristic filtering: drop extreme black/white outliers if filter_data=True.
    """

    def __init__(
        self,
        x_path: str | Path,
        y_path: str | Path,
        transform: Optional[Callable] = None,
        clip_min: float = 0.0,
        clip_max: float = 255.0,
        scale_to_01: bool = True,
        filter_data: bool = False,
        filter_low_mean: float = 1.0,
        filter_high_mean: float = 254.0,
    ) -> None:
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform

        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.scale_to_01 = bool(scale_to_01)

        self.filter_data = bool(filter_data)
        self.filter_low_mean = float(filter_low_mean)
        self.filter_high_mean = float(filter_high_mean)

        if not self.x_path.exists():
            raise FileNotFoundError(f"X file not found: {self.x_path}")
        if not self.y_path.exists():
            raise FileNotFoundError(f"Y file not found: {self.y_path}")

        # Lazy handles (important with DataLoader(num_workers>0))
        self._fx: Optional[h5py.File] = None
        self._fy: Optional[h5py.File] = None
        self._x = None
        self._y = None

        # Determine length + indices once (without keeping file open)
        with h5py.File(self.x_path, "r") as f:
            xds = f["x"] if "x" in f else next(iter(f.values()))
            n = int(xds.shape[0])

            if self.filter_data:
                # Mean-based filtering: drop extreme black (mean 0) and white (mean 255) outliers
                means = np.mean(np.asarray(xds, dtype=np.float32), axis=(1, 2, 3))
                keep = (means > self.filter_low_mean) & (means < self.filter_high_mean)
                self.indices = np.nonzero(keep)[0].astype(np.int64)
            else:
                self.indices = np.arange(n, dtype=np.int64)

        self._length = int(self.indices.shape[0])

    def __len__(self) -> int:
        return self._length

    def _ensure_open(self) -> None:
        if self._fx is None:
            self._fx = h5py.File(self.x_path, "r")
            self._fy = h5py.File(self.y_path, "r")
            self._x = self._fx["x"] if "x" in self._fx else next(iter(self._fx.values()))
            self._y = self._fy["y"] if "y" in self._fy else next(iter(self._fy.values()))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()

        # Map dataset index -> original H5 index (filtered or full)
        h5_idx = int(self.indices[idx])

        x = self._x[h5_idx]  # usually (96, 96, 3) uint8
        y = self._y[h5_idx]  # usually scalar or shape (1,)

        # Numerical clipping + dtype safety
        x = np.asarray(x, dtype=np.float32)
        x = np.clip(x, self.clip_min, self.clip_max)

        if self.scale_to_01:
            x = x / 255.0

        # Optional transform (if provided)
        if self.transform is not None:
            x = self.transform(x)

        # If still numpy HWC, convert to torch CHW
        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
            x_t = torch.from_numpy(x).float()
        elif torch.is_tensor(x):
            x_t = x.float()
        else:
            raise TypeError(f"Transform returned unsupported type: {type(x)}")

        # Label to scalar long tensor
        y_np = np.asarray(y).reshape(-1)
        y_t = torch.tensor(int(y_np[0]), dtype=torch.long)

        return x_t, y_t

    def __del__(self) -> None:
        # Best-effort cleanup
        try:
            if self._fx is not None:
                self._fx.close()
            if self._fy is not None:
                self._fy.close()
        except Exception:
            pass
