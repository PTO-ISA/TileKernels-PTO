# ptoas / tilelang-pto-dev 缺陷上报（sinkhorn VMI 移植过程中发现）

环境：238 服务器 NPU0 Ascend950PR（a5），tilelang pto-dev@5f105e99（含 PR#6，已删
vload/vstore dist_mode）+ ptoas master@d4f1c8a5d（0.68），conda 环境
`/data/sunkexin/tkvmi/conda-env`。三个问题均来自 mhc sinkhorn ASC→VMI 移植
（`tile_kernels/mhc/sinkhorn_vmi.py`），复现脚本在本目录同级 `../probe*.py`，
均可独立运行（不依赖本仓算子）。

复现运行模板：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0 TMPDIR=/data/sunkexin/tkvmi/tmp PATH=/data/sunkexin/tkvmi/conda-env/bin:$PATH
cd /data/sunkexin/tkvmi/work && python probe2_ops.py    # 问题 1
python probe5_vdiv.py                                  # 问题 2
# 问题 3 需先在 tl-run 环境跑 probe6_asc_stages.py 落盘 ASC 基准，再跑 probe7_vdiv_exact.py
```

## 问题 1：`vcmax`（及可能同族的 `vcmin`）在 partial element mask + group 下数值错误

**最小复现**（`verify/probe2_ops.py`，fp32x64 输入 = ramp 0..63）：

```python
idx_f = V.vcvt(V.vci(T.int32(0), size=64), 'float32')          # 0..63
mask_lo = V.vcmps(V.vand(idx, V.vbrc(T.int32(7), size=64)), T.int32(4), mask_full, 'lt')  # [1,1,1,1,0,0,0,0]×8
res8 = V.vcmax(idx_f, mask_lo, group=8)                         # 期望 lane g = max(lanes[8g,8g+4))
res64 = V.vbrc(res8, size=64, group=8)                          # 期望组内复制
```

**期望**：`res64[8g+k] = max(第 g 组前 4 lane)`；负斜坡（-63..0）时为 -60/-56/...

**实际**（两个错误叠加）：

1. inactive lane 以 0.0 参与比较：负斜坡输入下 `res64[0..7] = 0.0`（期望 -60）；
2. 结果取的是 group 0 的归约值铺满所有 lane：正斜坡下 `res64[8..15] = 3`
   （期望 11，即第 1 组的 max），全部 8 组结果相同。

对照：同写法的 `V.vcadd(v, mask_lo, group=8, reassoc=True)` 行为正确（probe1 中与
numpy 逐位一致），说明问题特定于 vcmax 的 mask/group lowering。官方测试
`test_vmi_bf16_group8_carrier.py` / `test_jit_compile.py:3844` 只用**满 mask**且只查
IR 文本，不覆盖 partial mask 数值。

**触发条件**：fp32x64 + 元素粒度 partial mask（vcmps 产生）+ group=8。

**规避**（本仓已用，非修复）：行内 4 元素 max 用 `vselr`+`vmax` 二叉树表达，逐位
等价（max 顺序无关）。

## 问题 2：`vcadd` 产生、后续被 `vdiv` 消费的寄存器值被错误编译（消费者相关）

**最小复现**（`verify/probe5_vdiv.py`，自包含）：

两个几乎相同的 kernel，行和链完全一致
（`vsel(mask_lo, vselr(vbrc(vcadd(e, mask_lo, group=8, reassoc=True), size=64, group=8), bidx), ...`）：

- kernel A（只 `vstore` 行和，无 div 消费者）：**数值正确**（与 numpy 逐位一致，见 probe5 注释与 probe2 的 stage 检查）；
- kernel B（probe5）：同一行和额外被 `V.vdiv(mat_1, rsum_b_0)` 消费 →
  **rsum 本身错误**（lane 8 起、即 group ≥1 全错，group 0 正确），显式 mask、
  UB store/load 回读后除，结果同样错（错在值产生处）。

**结论**：`vcadd(group)+partial mask` 的结果作为 `vdiv` 操作数时被错误调度/物化，
属于消费者相关的错误编译。同 kernel 中 `vselr/vadd/vmax` 产生的值喂 `vdiv`/
`vexpdif` 无此问题。

**规避**（本仓已用）：行和改用 `vselr`+`vadd` pairwise 树
（`(a+b)+(c+d)`，与 ASC vcgadd 折叠顺序逐位一致）。

## 问题 3：默认（hw）`vdiv` 与 ASC 链路 `S.vdiv` 舍入不一致；`precision='exact'` 正确但被前端展开成指令序列

**最小复现**（`verify/probe7_vdiv_exact.py`，对照 `probe6_asc_stages.py` 在 tl-run 环境的 ASC 落盘）：

```python
sm_hw    = V.vdiv(mat_1, rsum_b_0)                                # 5/64 lane 差 1 ULP
sm_exact = V.vdiv(mat_1, rsum_b_0, mask_full, precision='exact')  # 与 ASC 逐位一致
```

ASC 侧（tl-run 环境 tilelang 0.1.12 的 `S.vdiv`，CCE 除法指令）为正确舍入除法；
VMI 默认 hw vdiv 约 8% 商差 1 ULP。sinkhorn 每 token 需 2×repeat=20 次归一化除法，
1-ULP 差异在链上放大（终态 fwd 差 3–9 bit）。

`precision='exact'` 在 tilelang `vmi.py` 前端展开为约 15 条 vmi 指令的修正序列
（hw 除 + 指数缩放 + ±1ULP 候选选择），数值正确但性能代价显著：本算子
n=8192 时 VMI(exact) 比 ASC 慢 13–16%，VMI(hw) 与 ASC 持平（见验收报告 §3）。

**建议**：把 `vdiv precision='exact'`（或默认 vdiv）lower 到 ASC 链路同款的
单条 0-ULP 除法指令（CANN `vdiv_0ulp`），既逐位一致又无展开开销——这同时是
本算子性能 gap 的主修复项。

## 版本

- tilelang: pto-dev@5f105e99（`/data/sunkexin/tkvmi/src/tilelang`，editable）
- ptoas: master@d4f1c8a5d（`/data/sunkexin/tkvmi/src/ptoas`，editable）
- 运行时：CANN 9.1（torch_npu），NPU Ascend950PR，`--pto-arch=a5 --pto-backend=vpto`
- kernel.pto 中间产物：tilelang jit 缓存目录（TMPDIR 下）可按需归档
