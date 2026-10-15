# mhc pre_apply_mix ASC → VMI 移植验收报告（pre_apply_mix_vmi 分支）

- 移植对象：`tile_kernels/mhc/pre_apply_mix_asc.py` → `tile_kernels/mhc/pre_apply_mix_vmi.py`
  （`mhc_pre_apply_mix_fwd_vmi` / `mhc_pre_apply_mix_bwd_vmi`；外层 schedule 与 host 契约逐字保留，
  程序化 AST diff 验证：`verify/diff_pre_outer.py` OK）
- 环境（238 服务器，NPU 0，Ascend950PR / a5）：
  - VMI 侧：tilelang pto-dev **5f105e99**（含 PR #6 "Adapt TileLang VMI APIs to PTOAS
    compositions"）+ ptoas **0.68**（master d4f1c8a5d），环境 `/data/sunkexin/tkvmi/conda-env`
  - ASC 侧：tl-run 环境 tilelang **0.1.12**（pto-dev 5f105e99 的 ASC C++ 模板用了
    CANN 9.1.0 不存在的 `asc_pack_to_high` 等内建，ASC 链路只能用旧 tilelang 编译；
    同 post_vmi 报告"环境备注"）
- 数据流：ASC 侧（tl-run env）固定种子生成输入并跑 ASC kernel 落盘；VMI 侧（tkvmi env）
  加载同一输入跑 VMI kernel，逐位对比。脚本：`verify/ab_pre_asc_side.py` / `verify/ab_pre_vmi_side.py`。
- bwd 的 `x_grad` 是读改写累加语义（`x_grad += mix·o_grad`），A/B 时两侧均以随机值预填充
  累加器以覆盖该路径（仓内官方测试只用零填充）。

## 1. SimdVF 逐条对照（ASC → VMI）

| ASC（pre_apply_mix_asc.py） | VMI（pre_apply_mix_vmi.py） | 说明 |
|---|---|---|
| `S.alloc_local((mhc,), T.float32)` | `V.alloc_local((mhc,), V.vreg(64, T.float32))` | 逻辑 64-lane vreg 数组 |
| `S.alloc_var(T.float32)` | `V.alloc_var(T.float32, size=64)` | 同 |
| `S.vdup(0.0, T.float32)` | `V.vbrc(T.float32(0.0), size=64)` | 标量广播 |
| `S.vld(mix_ub[m], dist='BRC_B32')` | `V.vbrc(V.vload(mix_ub[m], size=1), size=64)` | ptoas 已移除 vload 广播 dist_mode；官方迁移写法 = 1-lane vload + vbrc |
| `S.vcvt(S.vld(x_ub[m, col], dist='UNPK_B16'), T.float32, part=0)` | `V.vcvt(V.vload(x_ub[m, col], size=64), 'float32')` | 装载逻辑 64-lane bf16 向量再逐元素拓宽；物理 unpack 布局交给 ptoas |
| `S.vsts(o_ub[col], S.vcvt(v, T.bfloat16), dist='PK_B32')` | `V.vstore(V.vcvt(v, 'bfloat16'), o_ub[col])` | 对称地，物理打包交给 ptoas |
| `S.vmul(a, b)` | `V.vmul(a, b)` | 满宽省 mask |
| `S.vmula(dst, a, b)`（原地累加） | `dst = V.vmula(dst, a, b, mask_full)`（值形式） | vmula 必须显式 mask（discussion #1616 清单） |
| `S.vld(mix_grad_acc[m, 0])`（NORM 满宽） | `V.vload(mix_grad_acc[m, 0], size=64)` | 满宽装载无 dist |
| `S.vsts(acc_ub[m, 0], v)`（NORM 满宽） | `V.vstore(v, acc_ub[m, 0])` | 值前地址后 |
| `S.vsts(mix_grad_ub[m], S.vcadd(S.vld(mix_grad_acc[m, 0])), dist='ONEPT_B32')` | `V.vstore(V.vcadd(V.vload(mix_grad_acc[m, 0], size=64), mask_full, reassoc=True), mix_grad_ub[m])` | **有意差异**，见下 |

有意差异点（均为 ptoas 现行 API 约束，非语义变化，与 post_vmi 完全同款）：

1. **标量广播装载**：ptoas commit `2729bb556`（2026-10-13）移除了 vload 的
   `dist_mode="brc"`；其迁移指南规定"one-element vload + vbrc"。数学上等价
   （同一 fp32 标量填充 64 lane）。
2. **ONEPT 归约存储**：ASC `vcadd` 返回满宽向量取 lane0 单点写；VMI `vcadd` 返回
   1-lane vreg，直接做无 mask 的 1-lane（标量）store。"广播回 64 lane + lane0
   mask store"写法会被 ptoas 拒绝（`VMI-UNSUPPORTED: masked_store requires a
   destination address with a proven store alignment`）。`reassoc=True` 保持 ASC
   vcadd 的两两树归约顺序。
3. **accumulation 顺序**：fwd 的 `vmul(xs[0],mix[0])` + mhc-1 条 vmula、bwd 的
   每元素单次读改写累加与 `mix_grad_reg` 跨 chunk 累加链，求和顺序与 ASC 循环
   嵌套完全一致——精度逐位一致（§2）即由此保证。

## 2. 精度（验收②）

同种子同输入，6 个 config 全部输出张量 **torch.equal 逐位一致**（0 mismatch、
0 bit distance）：

| config (n0×n1, h) | fwd o | bwd x_grad | bwd mix_grad |
|---|---|---|---|
| 1×128, 1280 | eq | eq | eq |
| 1×128, 2560 | eq | eq | eq |
| 1×128, 4096 | eq | eq | eq |
| 1×128, 5120 | eq | eq | eq |
| 1×128, 7168 | eq | eq | eq |
| 2×512, 7168 | eq | eq | eq |

无 1-ULP 残余。torch reference（`tile_kernels.torch.mhc.mhc_pre_apply_mix_ref`）sanity
亦全过（`verify/smoke_pre.py` 8/8 PASS，覆盖 h_blk=hidden 与 h_blk<hidden、
num_stages=3/4 全部 regime；判据同 `tests/mhc/test_pre_apply_mix.py`）。

测试参考实现备注（后续算子可复用的教训）：bwd 的 `x_grad` 参考**不能**直接用
autograd 的 `x_ref.grad`——`o_ref` 是 bf16，autograd 会把 `mix·o_grad` 先舍入成
bf16 再加到初值上；kernel（ASC 与 VMI 一致）是乘积保持 fp32、只对最终和做一次
bf16 舍入。正确参考：`bf16(f32(x_init) + mix_f32 · f32(o_grad))`，此写法下
h≤4096 的 config 与 kernel 逐位一致。

## 3. 性能（验收③，flush L2，NPU Event 计时，min of 20）

ratio = ASC_us / VMI_us，达标线 0.90。覆盖 `test_pre_apply_mix.py` 全部 5 个
bench config（另加 (1,512,5120) 一行）× fwd/bwd：

| op | n1 | h | ASC_us | VMI_us | ratio | 结论 |
|---|---|---|---|---|---|---|
| bwd | 512 | 4096 | 127.84 | 107.08 | 1.19 | 达标 |
| bwd | 512 | 5120 | 153.73 | 125.66 | 1.22 | 达标 |
| bwd | 512 | 7168 | 180.60 | 159.29 | 1.13 | 达标 |
| bwd | 8192 | 4096 | 820.88 | 824.33 | 1.00 | 达标 |
| bwd | 8192 | 5120 | 1023.62 | 996.78 | 1.03 | 达标 |
| bwd | 8192 | 7168 | 1364.38 | 1340.39 | 1.02 | 达标 |
| fwd | 8192 | 4096 | 366.88 | 387.45 | 0.95 | 达标 |
| fwd | 8192 | 5120 | 437.80 | 445.04 | 0.98 | 达标 |
| fwd | 8192 | 7168 | 563.91 | 588.71 | 0.96 | 达标 |
| fwd | 512 | 4096 | 40.28 | 51.73 | 0.78 | **正确但低于门槛** |
| fwd | 512 | 5120 | 47.24 | 57.87 | 0.82 | **正确但低于门槛** |
| fwd | 512 | 7168 | 63.23 | 76.48 | 0.83 | **正确但低于门槛** |

**fwd 小 batch（n1=512）gap 分析（与 post_vmi 报告同因同量级）**：每个
(n, h_blk) tile 的 SimdVF 序言要做 mhc=4 次标量广播装载；ptoas 移除 brc vload
后，VMI 每次广播是 `vload(size=1)+vbrc` 两条指令（ASC 是一条 `vlds BRC_B32`），
即每 tile 多 4 条向量指令。小 batch 时 tile 数多（512×h/h_blk = 512–1024）、
单 tile 有效计算少，该固定开销占比放大。差值 11–13us / tile 数 ≈ 13–22 ns/tile，
与每 tile 多 4 条向量指令的量级吻合。潜在优化点在 ptoas/编译器侧（如把
`vbrc(vload(1))` 模式重新融合为单条 BRC 装载），kernel 侧写法已是最简逻辑形式。
bwd 不受此影响（其瓶颈在 x_grad 读改写与 mix_grad 跨 chunk 累加的主循环，
序言开销占比小），反而因 VMI 布局摆放更优而全面快于 ASC。

有效带宽复核：全部行 IO 带宽 0.4–1.14 TB/s，无 >4TB/s 行，无 bench 异常剔除项。

## 环境备注（沿袭 post_vmi，后续 PR/issue 素材）

1. tilelang pto-dev ≥ 5f105e99 的 **ASC（非 PTO）链路**需要 CANN **9.2.0+**
   （9.1.0 缺 `asc_pack_to_high` 等 AscendC 内建）。本次 A/B 的 ASC 基准沿用
   tl-run 环境 tilelang 0.1.12 + CANN 9.1（同 post_vmi 的既有结论）。
2. ptoas masked store 对齐约束：对逐元素标量槽做 lane0-mask store 会被拒；
   合法替代是 1-lane 无 mask store（本 kernel bwd 的 mix_grad 归约存储即用此）。
3. VMI 侧运行需 `PATH` 含 `/data/sunkexin/tkvmi/conda-env/bin`（ptoas CLI 由
   tilelang libgen 以 `ptoas` 裸命令调起）。

## 复现命令（238）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0 TMPDIR=/data/sunkexin/tkvmi/tmp
export PATH=/data/sunkexin/tkvmi/conda-env/bin:$PATH
cd /data/sunkexin/tkvmi/work
# 冒烟 + torch 参考
/data/sunkexin/tkvmi/conda-env/bin/python smoke_pre.py
# A/B：ASC 侧（tl-run env）→ VMI 侧（tkvmi env）
/data/sunkexin/tl-run/conda-env/bin/python ab_pre_asc_side.py
/data/sunkexin/tkvmi/conda-env/bin/python ab_pre_vmi_side.py
# 性能
/data/sunkexin/tl-run/conda-env/bin/python bench_pre_side.py asc
/data/sunkexin/tkvmi/conda-env/bin/python bench_pre_side.py vmi
```

原始结果：`bench_pre_asc.json` / `bench_pre_vmi.json` / `smoke_pre.log` /
`ab_pre_asc.log` / `ab_pre_vmi.log`（本目录）。
