from __future__ import annotations

import torch

from pylate.losses.padding import pad_embeddings_and_masks


class TestPadEmbeddingsAndMasks:
    def test_no_padding_needed(self):
        """All tensors same length — should be returned unchanged."""
        e1 = torch.randn(2, 5, 8)
        e2 = torch.randn(2, 5, 8)
        m1 = torch.ones(2, 5, dtype=torch.bool)
        m2 = torch.ones(2, 5, dtype=torch.bool)
        out_e, out_m = pad_embeddings_and_masks([e1, e2], [m1, m2])
        assert out_e[0].shape == (2, 5, 8)
        assert out_e[1].shape == (2, 5, 8)
        assert torch.equal(out_e[0], e1)
        assert torch.equal(out_m[0], m1)

    def test_pads_to_max(self):
        """Shorter tensors are zero-padded to the longest."""
        e_short = torch.ones(1, 3, 4)
        e_long = torch.ones(1, 7, 4)
        m_short = torch.ones(1, 3, dtype=torch.bool)
        m_long = torch.ones(1, 7, dtype=torch.bool)
        out_e, out_m = pad_embeddings_and_masks(
            [e_short, e_long], [m_short, m_long]
        )
        assert out_e[0].shape == (1, 7, 4)
        assert out_e[1].shape == (1, 7, 4)
        # Padded region of embedding should be zeros
        assert (out_e[0][:, 3:, :] == 0).all()
        # Original region should be ones
        assert (out_e[0][:, :3, :] == 1).all()
        # Padded region of mask should be False
        assert (out_m[0][:, 3:] == False).all()
        assert (out_m[0][:, :3] == True).all()

    def test_explicit_target_len(self):
        """Pad to an explicit target_len larger than any input."""
        e = torch.randn(2, 4, 6)
        m = torch.ones(2, 4, dtype=torch.bool)
        out_e, out_m = pad_embeddings_and_masks([e], [m], target_len=10)
        assert out_e[0].shape == (2, 10, 6)
        assert out_m[0].shape == (2, 10)
        assert (out_e[0][:, 4:, :] == 0).all()
        assert (out_m[0][:, 4:] == False).all()

    def test_target_len_equal_to_input(self):
        """target_len == input length — no padding."""
        e = torch.randn(1, 5, 3)
        m = torch.ones(1, 5, dtype=torch.bool)
        out_e, out_m = pad_embeddings_and_masks([e], [m], target_len=5)
        assert torch.equal(out_e[0], e)
        assert torch.equal(out_m[0], m)

    def test_multiple_varying_lengths(self):
        """Three tensors with different lengths all padded to max."""
        e1 = torch.randn(1, 2, 4)
        e2 = torch.randn(1, 5, 4)
        e3 = torch.randn(1, 3, 4)
        m1 = torch.ones(1, 2, dtype=torch.bool)
        m2 = torch.ones(1, 5, dtype=torch.bool)
        m3 = torch.ones(1, 3, dtype=torch.bool)
        out_e, out_m = pad_embeddings_and_masks(
            [e1, e2, e3], [m1, m2, m3]
        )
        for e in out_e:
            assert e.shape == (1, 5, 4)
        for m in out_m:
            assert m.shape == (1, 5)
