import torch


def reference(logits: torch.Tensor):
    lse = torch.logsumexp(logits.float(), dim=0).reshape(1)
    argmax = torch.argmax(logits.float()).to(torch.int32).reshape(1)
    return lse, argmax
