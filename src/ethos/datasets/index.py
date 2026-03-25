from datetime import timedelta
from pathlib import Path

import polars as pl
import torch as th

from .base import InferenceDataset


class IndexDataset(InferenceDataset):
    """InferenceDataset driven by an explicit (subject_id, prediction_time) index.

    Instead of finding prediction points by scanning for specific tokens (as the
    task-specific datasets do), this dataset accepts an arbitrary index dataframe
    and looks up the last token in each patient's record that falls at or before
    the given prediction_time.

    Args:
        input_dir: Directory containing the ethos tokenized data (.safetensors
            shards, vocabulary, static_data.pickle, etc.).
        index_df: DataFrame with columns 'subject_id' (Int64) and
            'prediction_time' (Datetime or Int64 microseconds since epoch).
            Rows whose subject_id is absent from the dataset, or where the
            patient has no events before prediction_time, are silently dropped.
        n_positions: Total sequence length the model was trained with.
        time_limit: Maximum time horizon to generate into the future.
            Defaults to 2 years (same as InferenceDataset base default).
    """

    def __init__(
        self,
        input_dir: str | Path,
        index_df: pl.DataFrame,
        n_positions: int = 2048,
        time_limit: timedelta | None = None,
        **kwargs,
    ):
        super().__init__(input_dir, n_positions, **kwargs)

        if time_limit is not None:
            self.time_limit = time_limit

        # Ensure prediction_time is stored as int64 microseconds to match self.times.
        self._index = index_df.with_columns(
            pl.col("prediction_time").cast(pl.Datetime("us")).cast(pl.Int64)
        )

        # Build subject_id -> (global_start_token_idx, global_end_token_idx).
        # patient_offsets inside each shard are local; adding shard["offset"] gives
        # the global position in the flat token array.
        patient_ids = th.cat([s["patient_ids"] for s in self._data.shards])
        patient_offsets = th.cat(
            [s["patient_offsets"] + s["offset"] for s in self._data.shards]
        )
        end_offsets = th.cat([patient_offsets[1:], th.tensor([len(self.tokens)])])
        self._patient_range: dict[int, tuple[int, int]] = {
            pid.item(): (start.item(), end.item())
            for pid, start, end in zip(patient_ids, patient_offsets, end_offsets)
        }

        # Resolve each index row to a flat token index (None if unresolvable).
        self._token_indices, self._valid_row_indices = self._resolve_token_indices()

    def _resolve_token_indices(self) -> tuple[list[int | None], list[int]]:
        token_indices: list[int | None] = []
        valid_row_indices: list[int] = []

        for row_i, row in enumerate(self._index.iter_rows(named=True)):
            subject_id = row["subject_id"]
            pred_time_us = row["prediction_time"]  # int64 microseconds

            if subject_id not in self._patient_range:
                token_indices.append(None)
                continue

            start, end = self._patient_range[subject_id]
            times = self.times[start:end]  # tensor of int64 microseconds
            n_before = int((times <= pred_time_us).sum().item())

            if n_before == 0:
                token_indices.append(None)
                continue

            token_indices.append(start + n_before - 1)
            valid_row_indices.append(row_i)

        return token_indices, valid_row_indices

    def __len__(self) -> int:
        return len(self._valid_row_indices)

    def __getitem__(self, idx: int) -> tuple[th.Tensor | tuple[th.Tensor, th.Tensor], dict]:
        row_i = self._valid_row_indices[idx]
        token_idx = self._token_indices[row_i]
        row = self._index.row(row_i, named=True)

        # InferenceDataset.__getitem__ builds the context-prepended timeline,
        # applying the sliding-window truncation when the patient's history is
        # longer than timeline_size.
        x = super().__getitem__(token_idx)

        metadata = {
            "subject_id": row["subject_id"],
            "prediction_time": row["prediction_time"],       # int64 microseconds
            "window_last_observed": self.times[token_idx].item(),  # int64 microseconds
            "data_idx": token_idx,
        }
        return x, metadata
