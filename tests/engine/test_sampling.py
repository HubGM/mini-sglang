from __future__ import annotations

import torch

from minisgl.engine.sample import BatchSamplingArgs, Sampler


def test_temperature_zero_sampling_is_deterministic() -> None:
    sampler = Sampler(torch.device("cpu"))
    logits = torch.tensor([[0.1, 4.0, 1.0], [3.0, 2.0, 1.0]])
    args = BatchSamplingArgs(temperatures=None)

    first = sampler.sample(logits.clone(), args)
    second = sampler.sample(logits.clone(), args)

    assert first.tolist() == [1, 0]
    assert torch.equal(first, second)
