#!/usr/bin/env python3

import os
from typing import Dict, List

import h5py
import torch
from torch.utils.data import Dataset
import numpy as np


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
        progress_desc: str = "",
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
        self._progress_desc = progress_desc
        self._show_progress = show_progress

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

        return data
