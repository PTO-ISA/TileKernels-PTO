from test_parallel_device import NPU, rand, record
import pytest
import torch
from tile_kernels.moe.moe_topk_gate_forward_parallel import get_moe_topk_gate_forward_kernel_parallel
import tile_kernels.moe.moe_topk_gate_backward_parallel as backward
from tile_kernels.quant.batched_transpose_weight_sf_parallel import get_batched_transpose_weight_sf_kernel_parallel


def empty(*shape, dtype=torch.float32):
    return torch.empty(shape, dtype=dtype, device=NPU)


@pytest.mark.parametrize('n,e,k,masked,aux', [(3, 64, 4, False, False), (17, 128, 8, True, True), (33, 35, 9, False, True)])
def test_moe(n, e, k, masked, aux):
    x, b = rand(n, e, dtype=torch.float32), rand(e, dtype=torch.float32) * 0.1
    mask = torch.ones(n, dtype=torch.bool)
    if masked:
        mask[::3] = False
    xn, bn = x.to(NPU), b.to(NPU)
    mn = mask.to(NPU)
    unused = torch.zeros(n, dtype=torch.bool, device=NPU)
    idx = empty(n, k, dtype=torch.int64)
    unmapped = empty(n, k, dtype=torch.int64)
    weights, scores = empty(n, k), empty(n, e)
    mapping = torch.arange(e, dtype=torch.int32, device=NPU).reshape(e, 1)
    counts = torch.ones(e, dtype=torch.int32, device=NPU)
    f = get_moe_topk_gate_forward_kernel_parallel(k, e, 0, 1, masked, False, True, False, True, False, False, True, 4)
    record(f, f'moe_fwd_{n}_{e}_{k}')
    f(xn, bn, bn, unused, mn, unused, mapping, counts, idx, unmapped, weights, scores, unused, 0, 0.7, 0)
    score = torch.nn.functional.softplus(x).sqrt()
    expected_idx = torch.argsort(score + b, dim=-1, descending=True, stable=True)[:, :k]
    chosen = score.gather(1, expected_idx)
    expected_weight = chosen / (chosen.sum(-1, keepdim=True) + 1e-20) * 0.7
    expected_idx = torch.where(mask[:, None], expected_idx, -1)
    expected_weight = torch.where(mask[:, None], expected_weight, 0.0)
    torch.testing.assert_close(idx.cpu(), expected_idx, rtol=0, atol=0)
    torch.testing.assert_close(unmapped.cpu(), expected_idx, rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), expected_weight, rtol=3e-5, atol=1e-6)
    torch.testing.assert_close(scores.cpu(), score * mask[:, None], rtol=3e-5, atol=1e-6)
    # Use independent all-active routing for the backward check.
    order = torch.argsort(score, dim=-1, descending=True, stable=True)[:, :k]
    w = score.gather(1, order)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * 0.7
    g = rand(n, k, dtype=torch.float32)
    ag = rand(3 if n == 33 else 1, e, dtype=torch.float32) * 0.01
    xx = x.clone().requires_grad_()
    ss = torch.nn.functional.softplus(xx).sqrt()
    ww = ss.gather(1, order)
    ww = ww / (ww.sum(-1, keepdim=True) + 1e-20) * 0.7
    loss = (ww * g).sum()
    if aux:
        loss = loss + ((ss / (ss.sum(-1, keepdim=True))) * ag.repeat_interleave(n // ag.shape[0], 0) * mask[:, None]).sum()
    reference = torch.autograd.grad(loss, xx)[0]
    backward.get_num_vec_cores = lambda: 4
    f = backward.get_moe_topk_gate_backward_kernel_parallel(k, e, aux)
    record(f, f'moe_bwd_{n}_{e}_{k}')
    out = empty(n, e)
    f(score.to(NPU), order.to(NPU), w.to(NPU), g.to(NPU), mn, ag.to(NPU), out, 0.7)
    torch.testing.assert_close(out.cpu(), reference, rtol=3e-4, atol=3e-6)


@pytest.mark.parametrize('shapes', [[(1024, 1024, 1)], [(1024, 1024, 2), (1088, 1152, 1)]])
def test_batched_sf(shapes):
    sf, outs0, outs1 = [], [], []
    starts = []
    count = 0
    refs = []
    for n, k, g in shapes:
        exp = (torch.arange(g * (n // 32) * (k // 32)).reshape(g, n // 32, k // 32) % 7 + 124).to(torch.int32)
        x = torch.exp2((exp - 127).float())
        sf.append(x.to(NPU))
        outs0.append(empty(g, (k // 32 + 1) // 2, n, dtype=torch.uint16))
        outs1.append(empty(g, (n // 32 + 1) // 2, k, dtype=torch.uint16))
        refs.append(
            (
                (exp[:, :, 0::2] | (exp[:, :, 1::2] << 8)).transpose(1, 2).repeat_interleave(32, 2).to(torch.uint16),
                (exp[:, 0::2, :] | (exp[:, 1::2, :] << 8)).repeat_interleave(32, 2).to(torch.uint16),
            )
        )
        starts.append(count)
        count += ((n + 1023) // 1024) * ((k + 1023) // 1024) * g
    infos = torch.tensor(shapes, dtype=torch.int32, device=NPU)
    pointers = torch.tensor([[a.data_ptr(), b.data_ptr(), c.data_ptr()] for a, b, c in zip(sf, outs0, outs1)], dtype=torch.int64, device=NPU)
    f = get_batched_transpose_weight_sf_kernel_parallel(32, 4)
    record(f, f'batched_sf_{len(shapes)}')
    f(torch.tensor(starts, dtype=torch.int32, device=NPU), infos, pointers, count)
    for a, b, (ra, rb) in zip(outs0, outs1, refs):
        torch.testing.assert_close(a.cpu().int(), ra.int(), rtol=0, atol=0)
        torch.testing.assert_close(b.cpu().int(), rb.int(), rtol=0, atol=0)
