from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ethos.evaluation.evaluate_trajectories import (
    compute_aurocs,
    compute_code_probabilities,
    load_trajectories,
)

PREDICTION_TIME = datetime(2024, 1, 1)
CODES = ["MEDS_DEATH", "ICU_ADMISSION"]


def _make_trajectory(
    rows: list[dict],
) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("prediction_time").cast(pl.Datetime("us")),
        pl.col("time").cast(pl.Datetime("us")),
    )


def _make_ground_truth(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(
        pl.col("prediction_time").cast(pl.Datetime("us")),
    )


class TestComputeCodeProbabilities:
    def test_all_rollouts_contain_code(self):
        """If every rollout has the code within duration, probability = 1.0."""
        row = {
            "subject_id": 1,
            "prediction_time": PREDICTION_TIME,
            "time": PREDICTION_TIME + timedelta(days=5),
            "code": "MEDS_DEATH",
        }
        trajectories = {i: _make_trajectory([row]) for i in range(4)}

        result = compute_code_probabilities(trajectories, ["MEDS_DEATH"], timedelta(days=30))
        prob = result.filter(pl.col("code") == "MEDS_DEATH").get_column("predicted_probability")[0]
        assert prob == 1.0

    def test_no_rollouts_contain_code(self):
        """If no rollout has the code, probability = 0.0."""
        row = {
            "subject_id": 1,
            "prediction_time": PREDICTION_TIME,
            "time": PREDICTION_TIME + timedelta(days=5),
            "code": "UNRELATED_CODE",
        }
        trajectories = {i: _make_trajectory([row]) for i in range(4)}

        result = compute_code_probabilities(trajectories, ["MEDS_DEATH"], timedelta(days=30))
        prob = result.filter(pl.col("code") == "MEDS_DEATH").get_column("predicted_probability")[0]
        assert prob == 0.0

    def test_half_rollouts_contain_code(self):
        """If 2/4 rollouts have the code, probability = 0.5."""
        base_row = {
            "subject_id": 1,
            "prediction_time": PREDICTION_TIME,
            "time": PREDICTION_TIME + timedelta(days=5),
        }
        hit = _make_trajectory([{**base_row, "code": "MEDS_DEATH"}])
        miss = _make_trajectory([{**base_row, "code": "UNRELATED_CODE"}])
        trajectories = {0: hit, 1: hit, 2: miss, 3: miss}

        result = compute_code_probabilities(trajectories, ["MEDS_DEATH"], timedelta(days=30))
        prob = result.filter(pl.col("code") == "MEDS_DEATH").get_column("predicted_probability")[0]
        assert prob == 0.5

    def test_code_outside_duration_excluded(self):
        """Code occurring after the duration window should not count."""
        row = {
            "subject_id": 1,
            "prediction_time": PREDICTION_TIME,
            "time": PREDICTION_TIME + timedelta(days=60),
            "code": "MEDS_DEATH",
        }
        trajectories = {i: _make_trajectory([row]) for i in range(4)}

        result = compute_code_probabilities(trajectories, ["MEDS_DEATH"], timedelta(days=30))
        prob = result.filter(pl.col("code") == "MEDS_DEATH").get_column("predicted_probability")[0]
        assert prob == 0.0

    def test_code_at_exact_duration_boundary(self):
        """Code occurring exactly at prediction_time + duration should count (<=)."""
        row = {
            "subject_id": 1,
            "prediction_time": PREDICTION_TIME,
            "time": PREDICTION_TIME + timedelta(days=30),
            "code": "MEDS_DEATH",
        }
        trajectories = {i: _make_trajectory([row]) for i in range(4)}

        result = compute_code_probabilities(trajectories, ["MEDS_DEATH"], timedelta(days=30))
        prob = result.filter(pl.col("code") == "MEDS_DEATH").get_column("predicted_probability")[0]
        assert prob == 1.0

    def test_multiple_codes(self):
        """Probabilities computed independently per code."""
        base = {"subject_id": 1, "prediction_time": PREDICTION_TIME}
        death_row = {**base, "time": PREDICTION_TIME + timedelta(days=5), "code": "MEDS_DEATH"}
        icu_row = {**base, "time": PREDICTION_TIME + timedelta(days=5), "code": "ICU_ADMISSION"}

        # Rollouts 0,1 have both codes; rollouts 2,3 have only ICU
        both = _make_trajectory([death_row, icu_row])
        icu_only = _make_trajectory([icu_row])
        trajectories = {0: both, 1: both, 2: icu_only, 3: icu_only}

        result = compute_code_probabilities(trajectories, CODES, timedelta(days=30))

        death_prob = result.filter(pl.col("code") == "MEDS_DEATH").get_column("predicted_probability")[0]
        icu_prob = result.filter(pl.col("code") == "ICU_ADMISSION").get_column("predicted_probability")[0]
        assert death_prob == 0.5
        assert icu_prob == 1.0

    def test_multiple_subjects(self):
        """Each subject gets its own probability."""
        base_time = PREDICTION_TIME + timedelta(days=5)

        def make_rollout(subj1_has_code: bool, subj2_has_code: bool):
            rows = [
                {"subject_id": 1, "prediction_time": PREDICTION_TIME, "time": base_time,
                 "code": "MEDS_DEATH" if subj1_has_code else "OTHER"},
                {"subject_id": 2, "prediction_time": PREDICTION_TIME, "time": base_time,
                 "code": "MEDS_DEATH" if subj2_has_code else "OTHER"},
            ]
            return _make_trajectory(rows)

        # subj1: 3/4 rollouts, subj2: 1/4 rollouts
        trajectories = {
            0: make_rollout(True, True),
            1: make_rollout(True, False),
            2: make_rollout(True, False),
            3: make_rollout(False, False),
        }

        result = compute_code_probabilities(trajectories, ["MEDS_DEATH"], timedelta(days=30))

        prob_s1 = result.filter(
            (pl.col("subject_id") == 1) & (pl.col("code") == "MEDS_DEATH")
        ).get_column("predicted_probability")[0]
        prob_s2 = result.filter(
            (pl.col("subject_id") == 2) & (pl.col("code") == "MEDS_DEATH")
        ).get_column("predicted_probability")[0]
        assert prob_s1 == 0.75
        assert prob_s2 == 0.25

    def test_duplicate_code_in_trajectory_counts_once(self):
        """A code appearing multiple times in one rollout still counts as 1 hit."""
        rows = [
            {"subject_id": 1, "prediction_time": PREDICTION_TIME,
             "time": PREDICTION_TIME + timedelta(days=5), "code": "MEDS_DEATH"},
            {"subject_id": 1, "prediction_time": PREDICTION_TIME,
             "time": PREDICTION_TIME + timedelta(days=10), "code": "MEDS_DEATH"},
        ]
        trajectories = {i: _make_trajectory(rows) for i in range(4)}

        result = compute_code_probabilities(trajectories, ["MEDS_DEATH"], timedelta(days=30))
        prob = result.filter(pl.col("code") == "MEDS_DEATH").get_column("predicted_probability")[0]
        assert prob == 1.0

    def test_raises_on_no_hits(self):
        """Should raise if no code hits found in any rollout."""
        row = {
            "subject_id": 1,
            "prediction_time": PREDICTION_TIME,
            "time": PREDICTION_TIME + timedelta(days=5),
            "code": "UNRELATED",
        }
        trajectories = {i: _make_trajectory([row]) for i in range(4)}

        with pytest.raises(ValueError, match="No code hits"):
            compute_code_probabilities(trajectories, ["MEDS_DEATH"], timedelta(days=30))


class TestComputeAurocs:
    def test_perfect_prediction(self):
        """AUROC should be 1.0 for perfect predictions."""
        probabilities = pl.DataFrame({
            "subject_id": [1, 2, 3, 4],
            "prediction_time": [PREDICTION_TIME] * 4,
            "code": ["MEDS_DEATH"] * 4,
            "predicted_probability": [0.9, 0.8, 0.1, 0.0],
        }).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))

        ground_truth = _make_ground_truth({
            "subject_id": [1, 2, 3, 4],
            "prediction_time": [PREDICTION_TIME] * 4,
            "code": ["MEDS_DEATH"] * 4,
            "boolean_value": [True, True, False, False],
        })

        result = compute_aurocs(probabilities, ground_truth)
        auroc = result.filter(pl.col("code") == "MEDS_DEATH").get_column("auroc")[0]
        assert auroc == 1.0

    def test_random_prediction(self):
        """AUROC for random predictions should be around 0.5."""
        import numpy as np
        rng = np.random.default_rng(42)
        n = 1000
        y_true = [True] * (n // 2) + [False] * (n // 2)
        y_pred = rng.uniform(0, 1, n).tolist()

        probabilities = pl.DataFrame({
            "subject_id": list(range(n)),
            "prediction_time": [PREDICTION_TIME] * n,
            "code": ["MEDS_DEATH"] * n,
            "predicted_probability": y_pred,
        }).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))

        ground_truth = _make_ground_truth({
            "subject_id": list(range(n)),
            "prediction_time": [PREDICTION_TIME] * n,
            "code": ["MEDS_DEATH"] * n,
            "boolean_value": y_true,
        })

        result = compute_aurocs(probabilities, ground_truth)
        auroc = result.filter(pl.col("code") == "MEDS_DEATH").get_column("auroc")[0]
        assert 0.4 < auroc < 0.6

    def test_single_class_returns_none(self):
        """AUROC should be None when only one class is present."""
        probabilities = pl.DataFrame({
            "subject_id": [1, 2],
            "prediction_time": [PREDICTION_TIME] * 2,
            "code": ["MEDS_DEATH"] * 2,
            "predicted_probability": [0.9, 0.8],
        }).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))

        ground_truth = _make_ground_truth({
            "subject_id": [1, 2],
            "prediction_time": [PREDICTION_TIME] * 2,
            "code": ["MEDS_DEATH"] * 2,
            "boolean_value": [True, True],
        })

        result = compute_aurocs(probabilities, ground_truth)
        auroc = result.filter(pl.col("code") == "MEDS_DEATH").get_column("auroc")[0]
        assert auroc is None

    def test_multiple_codes_independent_aurocs(self):
        """Each code gets its own AUROC."""
        probabilities = pl.DataFrame({
            "subject_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "prediction_time": [PREDICTION_TIME] * 8,
            "code": ["MEDS_DEATH"] * 4 + ["ICU_ADMISSION"] * 4,
            # perfect for death, inverted for ICU
            "predicted_probability": [0.9, 0.8, 0.1, 0.0, 0.1, 0.0, 0.9, 0.8],
        }).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))

        ground_truth = _make_ground_truth({
            "subject_id": [1, 2, 3, 4, 1, 2, 3, 4],
            "prediction_time": [PREDICTION_TIME] * 8,
            "code": ["MEDS_DEATH"] * 4 + ["ICU_ADMISSION"] * 4,
            "boolean_value": [True, True, False, False, True, True, False, False],
        })

        result = compute_aurocs(probabilities, ground_truth)
        death_auroc = result.filter(pl.col("code") == "MEDS_DEATH").get_column("auroc")[0]
        icu_auroc = result.filter(pl.col("code") == "ICU_ADMISSION").get_column("auroc")[0]
        assert death_auroc == 1.0
        assert icu_auroc == 0.0

    def test_partial_ground_truth_match(self):
        """Predictions without matching ground truth are excluded."""
        probabilities = pl.DataFrame({
            "subject_id": [1, 2, 3],
            "prediction_time": [PREDICTION_TIME] * 3,
            "code": ["MEDS_DEATH"] * 3,
            "predicted_probability": [0.9, 0.1, 0.5],
        }).with_columns(pl.col("prediction_time").cast(pl.Datetime("us")))

        # Only subjects 1 and 2 have ground truth
        ground_truth = _make_ground_truth({
            "subject_id": [1, 2],
            "prediction_time": [PREDICTION_TIME] * 2,
            "code": ["MEDS_DEATH"] * 2,
            "boolean_value": [True, False],
        })

        result = compute_aurocs(probabilities, ground_truth)
        row = result.filter(pl.col("code") == "MEDS_DEATH")
        assert row.get_column("n")[0] == 2
        assert row.get_column("auroc")[0] == 1.0


class TestLoadTrajectories:
    def test_loads_parquet_files(self, tmp_path):
        """Should load all .parquet files keyed by stem integer."""
        for i in range(3):
            df = _make_trajectory([{
                "subject_id": 1,
                "prediction_time": PREDICTION_TIME,
                "time": PREDICTION_TIME + timedelta(days=1),
                "code": f"CODE_{i}",
            }])
            df.write_parquet(tmp_path / f"{i}.parquet")

        result = load_trajectories(tmp_path)
        assert set(result.keys()) == {0, 1, 2}
        for i in range(3):
            assert result[i].get_column("code")[0] == f"CODE_{i}"

    def test_sorted_order(self, tmp_path):
        """Files should be loaded in numeric order."""
        for i in [9, 0, 15]:
            df = _make_trajectory([{
                "subject_id": 1,
                "prediction_time": PREDICTION_TIME,
                "time": PREDICTION_TIME + timedelta(days=1),
                "code": "X",
            }])
            df.write_parquet(tmp_path / f"{i}.parquet")

        result = load_trajectories(tmp_path)
        assert list(result.keys()) == [0, 9, 15]
