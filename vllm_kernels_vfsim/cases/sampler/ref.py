import torch


def reference(keys, values, query, src, accepted, indices):
    matched = torch.where(keys == query, values, torch.full_like(values, float("-inf")))
    selected = matched.max().reshape(1)
    finalized = src if int(accepted) != 0 else torch.zeros_like(src)
    pool = torch.full_like(values, float("-inf"))
    pool[indices.long()] = values
    gathered = pool[indices.long()]
    return selected, finalized, gathered
