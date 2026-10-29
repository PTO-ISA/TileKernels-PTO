import torch

VOCAB = 128
VL = 64
SAMPLE = 70


def reference(logprobs: torch.Tensor) -> torch.Tensor:
    pivot = logprobs[SAMPLE]
    total = torch.zeros((), dtype=torch.float32)
    for i in range(VOCAB):
        val = logprobs[i]
        if i // VL < SAMPLE // VL:
            keep = val <= pivot
        elif i // VL > SAMPLE // VL:
            keep = val < pivot
        else:
            keep = (val < pivot) or (val == pivot and (i % VL) <= (SAMPLE % VL))
        if bool(keep):
            total = total + torch.exp(val)
    return total.reshape(1)
