from __future__ import annotations

import pytest
import torch

from pylate.rank.rank import _compute_imputation_scores


class TestComputeImputationScores:
    """Unit tests for _compute_imputation_scores()."""

    def test_zero_imputation(self) -> None:
        scores = [[0.9, 0.7], [0.8, 0.6]]
        result = _compute_imputation_scores(scores, "zero", 10.0, 100.0, "cpu")
        assert torch.allclose(result, torch.zeros(2))

    def test_min_imputation_rectangular(self) -> None:
        scores = [[0.9, 0.7, 0.5], [0.8, 0.6, 0.4]]
        result = _compute_imputation_scores(scores, "min", 10.0, 100.0, "cpu")
        assert result[0] == pytest.approx(0.5, abs=1e-5)
        assert result[1] == pytest.approx(0.4, abs=1e-5)

    def test_mean_imputation_rectangular(self) -> None:
        scores = [[0.9, 0.3], [0.8, 0.4]]
        result = _compute_imputation_scores(scores, "mean", 10.0, 100.0, "cpu")
        assert result[0] == pytest.approx(0.6, abs=1e-5)
        assert result[1] == pytest.approx(0.6, abs=1e-5)

    def test_percentile_imputation_rectangular(self) -> None:
        scores = [[0.1, 0.5, 0.9], [0.2, 0.6, 1.0]]
        result = _compute_imputation_scores(scores, "percentile", 50.0, 100.0, "cpu")
        assert result[0] == pytest.approx(0.5, abs=1e-5)
        assert result[1] == pytest.approx(0.6, abs=1e-5)

    def test_power_law_imputation_executes(self) -> None:
        scores = [[0.9, 0.7, 0.5, 0.3], [0.8, 0.6, 0.4, 0.2]]
        result = _compute_imputation_scores(scores, "power_law", 10.0, 100.0, "cpu")
        assert result.shape == (2,)
        # Power-law imputed value should be non-negative and <= min of row.
        assert all(0.0 <= result[i].item() <= min(scores[i]) for i in range(2))

    def test_power_law_single_score_falls_back_to_min(self) -> None:
        scores = [[0.5]]
        result = _compute_imputation_scores(scores, "power_law", 10.0, 100.0, "cpu")
        assert result[0] == pytest.approx(0.5, abs=1e-5)

    def test_power_law_all_negative_falls_back_to_min(self) -> None:
        scores = [[-0.3, -0.5, -0.7]]
        result = _compute_imputation_scores(scores, "power_law", 10.0, 100.0, "cpu")
        # All scores <= 0, so valid_scores is empty → fallback to min.
        assert result[0] == pytest.approx(-0.7, abs=1e-5)

    def test_unknown_imputation_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown imputation strategy"):
            _compute_imputation_scores([[0.5]], "bogus", 10.0, 100.0, "cpu")

    # --- Ragged (non-rectangular) paths ---

    def test_min_imputation_ragged(self) -> None:
        scores = [[0.9, 0.7], [0.5]]
        result = _compute_imputation_scores(
            scores, "min", 10.0, 100.0, "cpu", is_rectangular=False
        )
        assert result[0] == pytest.approx(0.7, abs=1e-5)
        assert result[1] == pytest.approx(0.5, abs=1e-5)

    def test_mean_imputation_ragged(self) -> None:
        scores = [[0.9, 0.3], [0.4]]
        result = _compute_imputation_scores(
            scores, "mean", 10.0, 100.0, "cpu", is_rectangular=False
        )
        assert result[0] == pytest.approx(0.6, abs=1e-5)
        assert result[1] == pytest.approx(0.4, abs=1e-5)

    def test_percentile_imputation_ragged(self) -> None:
        scores = [[0.1, 0.5, 0.9], [0.2]]
        result = _compute_imputation_scores(
            scores, "percentile", 50.0, 100.0, "cpu", is_rectangular=False
        )
        assert result[0] == pytest.approx(0.5, abs=1e-5)
        assert result[1] == pytest.approx(0.2, abs=1e-5)

    def test_power_law_ragged(self) -> None:
        scores = [[0.9, 0.7, 0.5, 0.3], [0.8]]
        result = _compute_imputation_scores(
            scores, "power_law", 10.0, 100.0, "cpu", is_rectangular=False
        )
        assert result.shape == (2,)
        assert 0.0 <= result[0].item() <= min(scores[0])
        # Single element → fallback to min.
        assert result[1] == pytest.approx(0.8, abs=1e-5)

    # --- Empty / degenerate inputs ---

    def test_empty_rows_ragged(self) -> None:
        scores = [[], [0.5, 0.3]]
        result = _compute_imputation_scores(
            scores, "min", 10.0, 100.0, "cpu", is_rectangular=False
        )
        assert result[0] == pytest.approx(0.0, abs=1e-5)
        assert result[1] == pytest.approx(0.3, abs=1e-5)

    def test_all_empty_rectangular(self) -> None:
        scores = [[], []]
        result = _compute_imputation_scores(
            scores, "min", 10.0, 100.0, "cpu", is_rectangular=True
        )
        assert torch.allclose(result, torch.zeros(2))

    def test_is_rectangular_auto_detection(self) -> None:
        # Same-length rows should be auto-detected as rectangular.
        scores = [[0.9, 0.7], [0.8, 0.6]]
        rect = _compute_imputation_scores(scores, "min", 10.0, 100.0, "cpu")
        forced = _compute_imputation_scores(
            scores, "min", 10.0, 100.0, "cpu", is_rectangular=True
        )
        assert torch.allclose(rect, forced)

    def test_output_shape_matches_num_tokens(self) -> None:
        scores = [[0.5], [0.3], [0.1]]
        for imp in ["min", "zero", "mean", "percentile", "power_law"]:
            result = _compute_imputation_scores(scores, imp, 10.0, 100.0, "cpu")
            assert result.shape == (3,), f"Failed for imputation={imp}"
