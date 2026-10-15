"""Three-way precision analysis of the pre_split_mixes A/B artifacts:
ASC vs VMI vs fp64 ground truth.  CPU only (TORCH_DEVICE_BACKEND_AUTOLOAD=0)."""
import glob, os
import torch

ART_DIR = "/data/sunkexin/tkvmi/work/ab_artifacts"

def rel_ulp(a, b):
    a32, b32 = a.float(), b.float()
    spacing = torch.where(b32 == 0, torch.ones_like(b32), b32.abs() * torch.finfo(torch.float32).eps)
    return (a32 - b32).abs() / spacing

for asc_path in sorted(glob.glob(os.path.join(ART_DIR, "pre_split_asc_*.pt"))):
    vmi_path = asc_path.replace("pre_split_asc_", "pre_split_vmi_")
    if not os.path.exists(vmi_path):
        continue
    asc = torch.load(asc_path, weights_only=True)
    vmi = torch.load(vmi_path, weights_only=True)
    inp = asc["inputs"]
    x = inp["input_mixes"].double()
    n0, n1, _ = x.shape
    n = n0 * n1
    scale = inp["mhc_scale"].double()
    base = inp["mhc_base"].double()

    x2 = x.reshape(n, 24)
    scaled = x2 * torch.cat([scale[0].expand(4), scale[1].expand(4), scale[2].expand(16)]) + base
    pre64 = torch.sigmoid(scaled[:, :4]) + 1e-2
    post64 = torch.sigmoid(scaled[:, 4:8]) * 2.0

    g = {k: inp[k].double().reshape(n, -1) for k in ("pre_grad", "post_grad", "comb_grad")}
    post_in = inp["post_for_bwd"].double().reshape(n, 4)
    sig_pre = torch.sigmoid(scaled[:, :4])
    d_pre = g["pre_grad"] * sig_pre * (1 - sig_pre)
    sig_post = post_in / 2.0
    d_post = g["post_grad"] * post_in * (1 - sig_post)
    base_grad64 = torch.cat([d_pre.sum(0), d_post.sum(0), g["comb_grad"].sum(0)])
    scale_grad64 = torch.stack([(d_pre * x2[:, :4]).sum(), (d_post * x2[:, 4:8]).sum(), (g["comb_grad"] * x2[:, 8:]).sum()])

    print(f"--- {os.path.basename(asc_path)} ---")
    for name, a, v, ref64 in (
        ("fwd pre", asc["fwd_out"]["pre"].reshape(n, 4), vmi["fwd_out"]["pre"].reshape(n, 4), pre64),
        ("fwd post", asc["fwd_out"]["post"].reshape(n, 4), vmi["fwd_out"]["post"].reshape(n, 4), post64),
        ("bwd base_grad", asc["bwd"]["base_grad_partial"].sum(0), vmi["bwd"]["base_grad_partial"].sum(0), base_grad64),
        ("bwd scale_grad", asc["bwd"]["scale_grad_partial"].sum(0), vmi["bwd"]["scale_grad_partial"].sum(0), scale_grad64),
    ):
        ua = rel_ulp(a, ref64).max().item()
        uv = rel_ulp(v, ref64).max().item()
        av = rel_ulp(a, v).max().item()
        print(f"  {name:15s} max_ulp(ASC,fp64)={ua:8.2f}  max_ulp(VMI,fp64)={uv:8.2f}  max_ulp(ASC,VMI)={av:8.2f}", flush=True)
