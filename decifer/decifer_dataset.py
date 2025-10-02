#!/usr/bin/env python3

from typing import Dict, List, Optional, Union

import math

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

from decifer.utility import discrete_to_continuous_xrd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


class DeciferDataset(Dataset):

    @staticmethod
    def _slice_to_numpy(data_slice, dtype: np.dtype) -> np.ndarray:
        """Convert a possibly ragged HDF5 slice into a dense numpy array."""
        if isinstance(data_slice, np.ndarray) and data_slice.dtype == np.object_:
            # h5py returns object arrays when variable-length dtypes are used; pad to a shared length.
            converted = [np.asarray(entry, dtype=dtype) for entry in data_slice]
            if not converted:
                return np.empty((0,), dtype=dtype)
            max_len = max(arr.shape[0] for arr in converted)
            padded = np.zeros((len(converted), max_len), dtype=dtype)
            for i, arr in enumerate(converted):
                padded[i, :arr.shape[0]] = arr
            data_slice = padded
        else:
            data_slice = np.asarray(data_slice, dtype=dtype)
        return data_slice

    def __init__(
        self,
        h5_path,
        data_keys,
        *,
        precompute_conditioning: bool = False,
        conditioning_kwargs: Optional[Dict] = None,
        precompute_batch_size: int = 512,
        conditioning_device: Optional[Union[str, torch.device]] = None,
        progress_desc: Optional[str] = None,
        show_progress: bool = True,
    ):
        # Key mappings for backward compatibility
        KEY_MAPPINGS = {
            'cif_tokens': 'cif_tokenized',
            'xrd.q': 'xrd_disc.q',
            'xrd.iq': 'xrd_disc.iq',
        }
        self.h5_file = h5py.File(h5_path, 'r')
        self.data_keys = data_keys

        # Ensure that data_keys only contain datasets
        self.data = {}
        for key in self.data_keys:
            # Resolve mapped key or fallback to original
            mapped_key = KEY_MAPPINGS.get(key)
            if mapped_key and mapped_key in self.h5_file:
                item = self.h5_file[mapped_key]
            elif key in self.h5_file:
                item = self.h5_file[key]
            else:
                raise KeyError(f"Neither '{key}' nor its mapped key exists in the HDF5 file")

            # Validate type
            if isinstance(item, h5py.Dataset):
                self.data[key] = item
            else:
                raise TypeError(f"The key '{key}' does not correspond to an h5py.Dataset.")

        self.dataset_length = len(next(iter(self.data.values())))

        self.precompute_conditioning = precompute_conditioning
        self.conditioning_kwargs = conditioning_kwargs or {}
        if conditioning_device is not None:
            device = torch.device(conditioning_device)
            if device.type == 'cuda' and not torch.cuda.is_available():
                device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.precompute_device = device
        self.precompute_batch_size = max(1, precompute_batch_size)
        self._conditioning_cache: Optional[torch.Tensor] = None
        self._progress_desc = progress_desc
        self._show_progress = show_progress

        if self.precompute_conditioning:
            if self._show_progress and self._progress_desc:
                print(
                    f"[INFO] Precomputing conditioning cache for {self._progress_desc} "
                    f"({self.dataset_length} samples)...",
                    flush=True,
                )
            self._build_conditioning_cache()

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        data = {}
        for key in self.data_keys:
            sequence = self.data[key][idx]

            # Handle numeric data (np.ndarray)
            if isinstance(sequence, np.ndarray):
                dtype = torch.float32 if 'float' in str(sequence.dtype) else torch.long
                sequence = torch.tensor(sequence, dtype=dtype)
            elif isinstance(sequence, (bytes, str)):
                sequence = sequence.decode('utf-8') if isinstance(sequence, bytes) else sequence
            else:
                raise TypeError(f"Unsupported sequence type {type(sequence)}")

            data[key] = sequence

        if self._conditioning_cache is not None:
            data['xrd_cont'] = self._conditioning_cache[idx]

        return data

    @property
    def has_precomputed_conditioning(self) -> bool:
        return self._conditioning_cache is not None

    def _build_conditioning_cache(self):
        required_keys = {'xrd.q', 'xrd.iq'}
        missing = required_keys - set(self.data.keys())
        if missing:
            raise KeyError(
                "Cannot precompute conditioning signals without discrete XRD keys; "
                f"missing: {', '.join(sorted(missing))}."
            )

        if not self.conditioning_kwargs:
            raise ValueError(
                "conditioning_kwargs must be provided when precompute_conditioning=True to "
                "control the conversion to continuous XRD signals."
            )

        cache: List[torch.Tensor] = []

        total = self.dataset_length
        batch_size = min(self.precompute_batch_size, total)
        batch_iter = range(0, total, batch_size)
        if self._show_progress and tqdm is not None:
            total_batches = math.ceil(total / batch_size) if batch_size else 0
            desc = self._progress_desc + " XRD cache" if self._progress_desc else "Precomputing XRD cache"
            batch_iter = tqdm(batch_iter, total=total_batches, desc=desc, leave=False)

        conditioning_kwargs = dict(self.conditioning_kwargs)

        for start in batch_iter:
            end = min(start + batch_size, total)

            batch_q_np = self._slice_to_numpy(self.data['xrd.q'][start:end], np.float32)
            batch_iq_np = self._slice_to_numpy(self.data['xrd.iq'][start:end], np.float32)

            batch_q = torch.from_numpy(batch_q_np).to(self.precompute_device)
            batch_iq = torch.from_numpy(batch_iq_np).to(self.precompute_device)

            with torch.no_grad():
                result = discrete_to_continuous_xrd(batch_q, batch_iq, **conditioning_kwargs)

            cache.append(result['iq'].detach().cpu().clone())

        if self._show_progress and tqdm is not None:
            batch_iter.close()  # type: ignore[arg-type]

        if not cache:
            raise RuntimeError("Failed to precompute conditioning cache; dataset appears to be empty.")

        self._conditioning_cache = torch.cat(cache, dim=0)
