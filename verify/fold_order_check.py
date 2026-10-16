import torch

art = torch.load("/data/sunkexin/tkvmi/work/ab_artifacts/sinkhorn_asc_stages.pt", weights_only=True)
e = art["e"].flatten().double()
rsum_asc = art["rsum"].flatten()

rows = e.reshape(16, 4)
a, b, c, d = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
f32 = lambda t: t.float()
cands = {
    "seq ((a+b)+c)+d":  f32(((f32(a + b) + c) + d)),
    "pair (a+b)+(c+d)": f32(f32(a + b) + f32(c + d)),
    "a+((b+c)+d)":      f32(a + f32(f32(b + c) + d)),
    "a+(b+(c+d))":      f32(a + f32(b + f32(c + d))),
    "(a+c)+(b+d)":      f32(f32(a + c) + f32(b + d)),
    "((a+c)+b)+d":      f32(((f32(a + c) + b) + d)),
    "((a+d)+b)+c":      f32(((f32(a + d) + b) + c)),
    "(d+c)+(b+a)":      f32(f32(d + c) + f32(b + a)),
    "((d+c)+b)+a":      f32(((f32(d + c) + b) + a)),
    "d+((c+b)+a)":      f32(d + f32(f32(c + b) + a)),
    "((a+b)+(c+d))+0":  f32(f32(f32(a + b) + f32(c + d)) + 0.0),
}
rsum_rows = rsum_asc.reshape(16, 4)[:, 0]
for name, cand in cands.items():
    eq = torch.equal(cand, rsum_rows)
    n = int((cand != rsum_rows).sum())
    tag = "EXACT MATCH" if eq else (str(n) + "/16 rows differ")
    print(name + ": " + tag)
