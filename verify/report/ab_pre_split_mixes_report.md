# mhc pre_split_mixes ASC → VMI 移植验收报告（pre_split_mixes_vmi 分支）

- 移植对象：`tile_kernels/mhc/pre_split_mixes_asc.py` → `tile_kernels/mhc/pre_split_mixes_vmi.py`
  （`mhc_pre_split_mixes_fwd_simd_mhc4_vmi` / `mhc_pre_split_mixes_bwd_mhc4_vmi`；外层 schedule 与
  host 契约逐字保留，程序化 AST diff 验证：`verify/diff_pre_split_outer.py` OK）
- ASC 模块中的另两个 SimtVF-only 通用 fallback kernel（`mhc_pre_split_mixes_{fwd,bwd}_asc`）
  不含 SimdVF 体、dispatch 层不调用，不属于移植对象，未搬运。
- 环境（238 服务器，NPU 0，Ascend950PR / a5）：
  - VMI 侧：tilelang pto-dev（editable，`/data/sunkexin/tkvmi/src/tilelang`）+ ptoas 0.68
    （editable，master d4f1c8a5d），环境 `/data/sunkexin/tkvmi/conda-env`，CANN
    **`/data/sunkexin/tkvmi/cann-9.2.0`**（原 `/data/s30066936/ascend/cann-9.2.0` 已迁移且副本
    签名校验失败起不了设备，勿再用）
  - ASC 侧：tl-run 环境 tilelang 0.1.12 + 系统 CANN 9.1（同 post_vmi / pre_apply_mix_vmi 报告
    "环境备注"：pto-dev 的 ASC 链路要 CANN 9.2+ 内建，ASC 基准只能用旧 tilelang 编译）
- 数学（torch 参考 `mhc_pre_split_mixes_ref`，mhc_mult=4，mhc_mult3=24）：
  `scaled = x·scale[段] + base[列]`（列 0–3 用 scale[0]，4–7 用 scale[1]，8–23 用 scale[2]）；
  `pre = sigmoid(scaled[:, :4]) + pre_eps`；`post = sigmoid(scaled[:, 4:8])·post_mult`；
  `comb = scaled[:, 8:24]`。bwd 按三段求导并产出逐核 partial（host 侧 `sum(0)` 归约）。

## 1. SimdVF 逐条对照（ASC → VMI）

| ASC（pre_split_mixes_asc.py） | VMI（pre_split_mixes_vmi.py） | 说明 |
|---|---|---|
| `S.pset(32)`（b32 元素全 lane = 64 lane 全开） | `V.create_mask(64, size=64)` | pset 首参是元素位宽非 lane 数；满宽二元/一元算子直接省 mask |
| `S.vld(scale_ub[i, 0], dist='BRC_B32')` | `V.vbrc(V.vload(scale_ub[i, 0], size=1), size=64)` | ptoas 已移除 vload 广播 dist_mode，官方迁移写法 |
| `S.vdup(c, T.float32, mask)` | `V.vbrc(T.float32(c), size=64)` | 标量广播 |
| `S.vadds/vmuls/vshls/vshrs(v, c, m)` | `V.vadd/vmul/vshl/vshr(v, V.vbrc(T.int32(c), size=64))` | 标量算子 = vbrc + 普通向量算子（discussion #1616 规则 3） |
| `S.vand(idx, S.vdup(3, ...), m)` | `V.vand(idx, V.vbrc(T.int32(3), size=64))` | 满宽省 mask |
| `S.vci(T.int32(0), T.int32)` | `V.vci(T.int32(0), size=64)` | lane 序号向量 |
| `S.vcmps(col3, c, m, 'eq')` | `V.vcmps(col3, T.int32(c), mask_full, 'eq')` | seed=mask 前置语义不变 |
| `S.vsel(new, old, mask)` | `V.vsel(mask, new, old)` | mask 前置（参数顺序变化，#1616 对照表） |
| `T.reinterpret(i, 'uint32x64')` | `V.vinterpret_cast(i, 'uint32')` | 同宽 reinterpret，lane 数不变 |
| `S.vgather2(base, u32idx, m)` | `V.vgather(base, u32idx, m)` | fp32 源接受 uint32 offsets；mask 必填 |
| `S.vscatter(v, base, u32idx, m)` | `V.vscatter(v, base, u32idx, m)` | 同序，mask 必填 |
| `S.vexpdif(zeros, val, m)`（= exp(0−val)） | `V.vexpdif(zeros, val, mask_full)` | sigmoid 保持 `1/(1+exp(−val))`，两侧均默认硬件除法 |
| `S.vsts(dst, v, m)`（NORM 满宽） | `V.vstore(v, dst)` | 值前地址后，满宽省 mask |
| `S.vsts(dst, vcadd(v, m), lane0, 'ONEPT_B32')` | `V.vstore(V.vcadd(v, m, reassoc=True), dst)` | **有意差异**，见下 |
| `S.mem_bar('VST_VLD')` | `T.simd.mem_bar('VST_VLD')` | 不变 |
| `S.alloc_local((3,), T.float32)` | `V.alloc_local((3,), V.vreg(64, T.float32))` | 逻辑 64-lane vreg 数组 |

有意差异点（均为 ptoas 现行 API 约束，非语义变化）：

1. **常量趟数循环展开**：ASC 的 `for base_id in range(4/16)` 与 `for vec_id in T.serial(4/16)`
   改写为 `T.unroll(n, explicit=True)`。原因：PTODSL 前端把 range/T.serial 循环变量定为
   `index` 类型，喂给 `V.vbrc` 会生成非法的 `vreg<64xindex>`（ptoas 断言
   "expected an 8/16/32-bit logical element type"）；`T.unroll(..., explicit=True)` 在 trace 期
   折叠为常量，保持 `pto.si32` dtype。趟数均为编译期常量、迭代顺序不变，语义等价；这也是
   tilelang 编译器自身报错信息规定 的写法（`CheckVmiLocalRegisterIndex`），post_vmi /
   norm_fn_vmi 的全部常量趟数循环同款。运行时标量（`valid_tokens`）进 `vcmps` 不受影响。
2. **标量广播装载**：ptoas 移除了 vload 的 `dist_mode="brc"`，改为 one-element vload + vbrc
   （同 post_vmi / pre_apply_mix_vmi 报告，ptoas commit 2729bb556）。
3. **ONEPT 归约存储**：ASC `vcadd` 满宽结果取 lane0 单点写；VMI `vcadd` 返回 1-lane vreg，
   直接无 mask 单 lane store（"广播回 64 lane + lane0 mask store"会被 ptoas 拒绝，
   `VMI-UNSUPPORTED: masked_store requires a proven store alignment`）。`reassoc=True` 保持
   ASC vcadd 的两两树归约顺序。
4. **除法精度**：两侧均为默认硬件除法（ASC `S.vdiv` 无 precision 参数 ↔ VMI `V.vdiv`
   precision=None→"hw"），未引入 `precision='exact'`。

## 2. 精度（验收②）

方法：同种子同输入（CPU 固定种子生成，ASC 侧落盘 .pt），ASC（tl-run env）与 VMI（tkvmi
env）分别运行，逐张量 `torch.equal` / 位距统计（`verify/ab_pre_split_asc_side.py` /
`verify/ab_pre_split_vmi_side.py`）；另做 fp64 真值三方对照（`ulp_analysis.py`）。
网格覆盖 `tests/mhc/test_pre_split_mixes.py` 全部精度 config：n0∈{1,2} × n1∈{1001,1024,4096}
（1001=15×64+41，覆盖 bwd 逐 token lane mask 的 ragged tail 路径）。bwd 的
post_layer_mix 输入由 torch 参考生成，保证两侧输入逐位相同。

结果（5 config 形态一致，逐 config 明细见 `ab_pre_split_vmi.log` / `ulp_three_way.log`）：

| 张量 | ASC↔VMI | 结论 |
|---|---|---|
| fwd comb_res_mix | 全 config **逐位一致**（0 mismatch） | 纯乘加链，无舍入差 |
| fwd pre / post | 最大位距 1–2（≈≤1 ULP），~10% 元素 | sigmoid 硬除法链舍入差，见下 |
| bwd input_mixes_grad post/comb 列 | **逐位一致** | |
| bwd input_mixes_grad pre 列 | 最大位距 ≤9（含近零元素） | pre 列梯度携带 sigmoid 的 ULP 差 |
| bwd scale/base_grad_partial 求和后 | 典型 ≤2.3 rel-ULP；灾难抵消元素处放大 | 见 fp64 三方对照 |

fp64 三方对照（`max_ulp(侧, fp64)` 与 `max_ulp(ASC, VMI)`，取各 config 最大）：

- fwd pre/post：ASC 与 VMI 到 fp64 距离相当（1.2–4.6 ULP，同一元素同侧舍入），ASC↔VMI ≤1.3 ULP。
- bwd 求和梯度：ASC↔VMI 典型 ≤2.3 rel-ULP；个别 config 的个别元素达 ~100–385 rel-ULP，但该处
  两侧各自离 fp64 也达 6–290 rel-ULP（n0=1 n1=4096 的 base_grad：ASC 290 vs VMI 94，VMI 反而
  更近真值）——fp32 归约在近零抵消处的固有现象，绝对差 ≤4e-5，非单侧错误。

结论：**逐位一致目标在纯乘加路径达成；sigmoid 除法链与 fp32 归约的残余差按 #1616 验收注 2
文档化为舍入差异**（与 norm_fn "除法链 ≤3 ULP" 同类）。torch 参考 sanity 亦全过
（`smoke_pre_split.py` 5/5 PASS，判据同官方测试 rtol=1e-5/atol=2e-5）。

## 3. 性能（验收③，flush L2 512MB，NPU Event 计时，min of 20）

本算子 IO 极小（192 B/token），两种计时：单发（含 launch 开销）与 50 连发摊销（kernel 本体）。
网格 = `test_pre_split_mixes.py` `_PRE_SPLIT_BENCH_CASES` 全部 config（mhc_mult=4）× fwd/bwd。

| op | n1 | ASC 单发/摊销 us | VMI 单发/摊销 us | ratio 单发/摊销 | 结论 |
|---|---|---|---|---|---|
| fwd | 512 | 25.03 / 14.08 | 38.58 / 24.28 | 0.65 / 0.58 | **正确但低于门槛** |
| fwd | 8192 | 35.33 / 14.29 | 36.03 / 23.91 | 0.98 / 0.60 | **正确但低于门槛** |
| bwd | 512 | 34.39 / 24.75 | 65.47 / 32.23 | 0.53 / 0.77 | **正确但低于门槛** |
| bwd | 8192 | 23.45 / 24.58 | 46.34 / 32.56 | 0.51 / 0.76 | **正确但低于门槛** |

有效带宽复核：全部行 IO 带宽 2–110 GB/s（本算子本身远非带宽受限），无 >4TB/s 行，无 bench
异常剔除项。

**gap 分析**：两侧摊销时间均与 n1 无关（512 与 8192 同为 ~14us / ~24us 的地板值，40 vec core
下并行度早已打满），差距是**每 persistent block 的固定向量序言开销**，不是数据通路：

1. 广播装载翻倍：本 kernel 每 block 序言做 3(scale)+4+4+16(base)=27 次标量广播；ptoas 移除
   brc vload 后每次是 `vload(size=1)+vbrc` 两条（ASC 一条 `vlds BRC_B32`），每 block 多 27 条
   向量指令。与 post_vmi / pre_apply_mix 报告的 fwd 小 batch gap 同因，但本算子每 block 有效
   计算更少、序言占比更大。
2. bwd 额外有 accumulate UB 的读写（scale/base_acc 3+3 个 64-lane vreg 装载/回写）与
   T.unroll(16) 展开后的 codegen 体积，摊销后 ratio 0.76–0.77 好于 fwd 但仍低于门槛。

潜在优化点在 ptoas/编译器侧：(a) 把 `vbrc(vload(1))` 模式重新融合为单条 BRC 装载（三份报告
重复出现的同一诉求）；(b) 对常量趟数 unroll 的寄存器分配/codegen 优化。kernel 侧写法已是最简
逻辑形式（shift/add 索引式与 ASC 逐条对应，未做 ×24 单乘等改写以保持对照可读性）。

## 编译问题记录（供 issue/PR 素材，本次未绕过、已解决）

1. **PTODSL 循环变量 index 类型**：`V.vbrc(T.int32(vec_id * 16), size=64)` 中 vec_id 为
   `T.serial(4)` 循环变量时，发射为 `pto.vmi.vbrc((vec_id * 16), size=64)`，前端推断为
   `vreg<64xindex>`，ptoas 断言失败（"expected an 8-bit, 16-bit, or 32-bit logical element
   type"，MLIR StorageUniquer assert）。复现：任何把 range/T.serial 循环变量表达式喂进
   VMI 向量算子的 SimdVF。合规解法 = `T.unroll(..., explicit=True)`（tilelang 自身
   `CheckVmiLocalRegisterIndex` 的指引）。**前端在 vbrc/vcmps 入口对 index 标量自动
   coerce 到 si32 是合理的编译器侧改进项。**
2. CANN 9.2.0 位置变更：skill 记录的 `/data/s30066936/ascend/cann-9.2.0` 已迁移到
   `/data/s30066936/ascend/1015/` 且副本因签名校验失败无法起设备；可用副本在
   `/data/sunkexin/tkvmi/cann-9.2.0`。

## 复现命令（238）

```bash
export ASCEND_RT_VISIBLE_DEVICES=0 TMPDIR=/data/sunkexin/tkvmi/tmp
# 冒烟 + torch 参考（VMI 侧）
source /data/sunkexin/tkvmi/cann-9.2.0/set_env.sh
export PATH=/data/sunkexin/tkvmi/conda-env/bin:$PATH
cd /data/sunkexin/tkvmi/work
python smoke_pre_split.py
# A/B：ASC 侧（tl-run env + 系统 CANN 9.1）落盘 → VMI 侧逐位比对
source /usr/local/Ascend/ascend-toolkit/set_env.sh
/data/sunkexin/tl-run/conda-env/bin/python ab_pre_split_asc_side.py
source /data/sunkexin/tkvmi/cann-9.2.0/set_env.sh; export PATH=/data/sunkexin/tkvmi/conda-env/bin:$PATH
python ab_pre_split_vmi_side.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python ulp_analysis.py
# 性能（两侧各跑一次）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
/data/sunkexin/tl-run/conda-env/bin/python bench_pre_split_side.py asc
source /data/sunkexin/tkvmi/cann-9.2.0/set_env.sh; export PATH=/data/sunkexin/tkvmi/conda-env/bin:$PATH
python bench_pre_split_side.py vmi
```

原始结果：本目录 `smoke_pre_split.log` / `ab_pre_split_vmi.log` / `ulp_three_way.log` /
`bench_pre_split_{asc,vmi}.json` / `bench_pre_{asc,vmi}_inner.log` / `ulp_analysis.py`。
