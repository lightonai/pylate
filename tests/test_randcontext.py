"""Tests for the GradCache RNG context used by CachedContrastive."""

from __future__ import annotations

import pytest
import torch

from pylate.losses.cached_contrastive import RandContext

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_randcontext_replays_the_forward_randomness(device: str) -> None:
    """The whole point of RandContext: the second forward must draw the same numbers."""
    tensor = torch.randn(4, 4, device=device)

    context = RandContext(tensor)
    with context:
        first = torch.randn(8, device=device).clone()
    with context:
        second = torch.randn(8, device=device).clone()

    assert torch.equal(first, second)


@pytest.mark.parametrize("device", DEVICES)
def test_randcontext_does_not_leak_rng_state(device: str) -> None:
    """Leaving the context must not disturb the surrounding RNG stream, or every
    training step after a cached backward would silently change its dropout."""
    tensor = torch.randn(4, 4, device=device)

    cpu_before = torch.get_rng_state().clone()
    mps_before = torch.mps.get_rng_state().clone() if device == "mps" else None

    with RandContext(tensor):
        torch.randn(64, device=device)

    assert torch.equal(cpu_before, torch.get_rng_state())
    if device == "mps":
        assert torch.equal(mps_before, torch.mps.get_rng_state())
