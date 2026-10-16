# mhc pre_big_fuse ASC → VMI 移植验收报告

- 分支：`pre_big_fuse_vmi`（基于 `vmi-impl`@66258df），文件 `tile_kernels/mhc/pre_big_fuse_vmi.py`
  （`pre_big_fuse_asc.py` 原样保留，便于 A/B；外层调度经 `verify/diff_prebig_outer.py` AST 校验逐字一致）
- 环境：238 服务器 NPU 0（Ascend950PR），conda env `/data/sunkexin/tkvmi/conda-env`，
  tilelang pto-dev@5f105e99（editable）+ ptoas master@d4f1c8a5d（editable），
  CANN `/data/l00975381/Ascend/cann-9.2.0`（tkvmi 下的 cann-9.2.0 拷贝缺 `asc_pack_to_high`，
  ASC 侧编译须用该安装）
- 日期：2026-10-15

## 一、SimdVF 逐条对照（ASC → VMI）

通用映射（与 norm_fn / post / pre_apply_mix 三次移植同款）：

| ASC | VMI | 备注 |
|---|---|---|
| `S.pset(32)`（fp32 满宽） | `V.create_mask(64, size=64)` | pset(32) 在 CCE 即 64-lane fp32 全满 |
| `S.vdup(c, T.float32, mask)` | `V.vbrc(T.float32(c), size=64)` | |
| `S.vld(buf, dist='BRC_B32')` | `V.vbrc(V.vload(buf, size=1), size=64)` | dist_mode 已被上游删除 |
| `S.vld(buf)`（满宽） | `V.vload(buf, size=64)` | |
| `S.vsts(dst, v, mask, 'NORM_B32')` | `V.vstore(v, dst)` | 值前地址后，满宽免 mask |
| `S.vadds / vmuls(v, c)` | `V.vadd / vmul(v, V.vbrc(T.float32(c), size=64))` | |
| `S.vsel(a, b, mask)` | `V.vsel(mask, a, b)` | **mask 前置**（本次踩过：顺序照抄 ASC 会报 vreg 类型不一致） |
| `S.vcmps(v, c, mask, 'ge')` | `V.vcmps(v, T.int32(c), mask_full, 'ge')` | |
| `S.vand(a, b, mask)` | `V.vand(a, b)` | |
| `S.vshrs / vshls(v, c, mask)` | `V.vshrs / vshls(v, T.int32(c), mask_full)` | ptodsl 会自动升到 vshr(vector)；仅 deprecation warning |
| `S.vgather2(buf, u32_idx, mask)` | `V.vgather(buf, V.vinterpret_cast(idx, 'uint32'), mask_full)` | |
| `S.vcvt(S.vld(..., 'UNPK_B16'), f32, part=0)` | `V.vcvt(V.vload(..., size=64), 'float32')` | 解包布局交 ptoas |
| `S.vsts(.., S.vcvt(v, bf16), 'PK_B32')` | `V.vstore(V.vcvt(v, 'bfloat16'), ..)` | |
| `S.vmula(acc, a, b)`（就地） | `acc = V.vmula(acc, a, b, mask_full)` | 值语义写回，mask 必填 |
| `S.vdiv(a, b, mask)` | `V.vdiv(a, b, mask_full, precision='exact')` | **见下方"精度关键点"** |
| `S.mem_bar('VST_VLD')` | `T.simd.mem_bar('VST_VLD')` | |

Phase 专有映射：

| 相位 | ASC | VMI |
|---|---|---|
| A（reduce+rmsnorm） | `vselr`/`vcgadd` 无；lane mask 涂抹 rms | 同构照搬 norm_fn_vmi 已验收写法：`vand(vcmps ge, vcmps lt)` + `vsel(token_mask, rms_val, rms_vec)` |
| B（split mixes） | `vgather2` + 基向量 `vsel` 链 | `vgather` + `vsel(base_mask, vbrc(vload size=1), acc)`（mask 前置） |
| C（sinkhorn）行归约 | `_row_max/_row_reduce = vcgmax/vcgadd ×2(mask_lo/hi) + vselr(bcast_idx2) ×2 + vsel` | `vsel(mask_lo, vbrc(vcmax/vcadd(mat, mask_lo/hi, group=8, …), size=64, group=8), …同 hi…)`；`vbrc(group=8)` 分块广播 ≡ ASC `vselr(bcast_idx2)`，省掉 bcast_idx2/token2/local_div8 全部索引算术 |
| C 转置 | `vselr(mat, t_idx)` | `V.vselr(mat, t_idx)`（保留，逻辑 lane 置换） |
| D（pre-apply） | `vmula` 就地累加 | `acc = V.vmula(acc, …, mask_full)`，同 pre_apply_mix_vmi |

## 二、精度（discussion #1616 验收 2）

`ab_prebig_asc_side.py`（固定 CPU 种子生成 gemm partials + 输入，落盘）→
`ab_prebig_vmi_side.py`（同输入重放，逐位比对）：

**9/9 config 全部逐位一致（torch.equal）**：n1∈{16,128,512,1021} ×
h∈{1280,2560,4096,4160,5120,7168} 组合，三个输出 `post_mix`(fp32)、
`comb_mix`(fp32，经 10 轮 sinkhorn 迭代)、`layer_input`(bf16) mismatch 均 0。
冒烟另含 tail 用例（n1=1021，非 16 倍数，T.copy 裁剪路径）5/5 通过。

### 精度关键点：vdiv 必须显式 `precision='exact'`

- tilelang `enable_fast_math` 默认 False → ASC `S.vdiv` 默认走 CANN **0-ULP 精确除法**；
  而 `V.vdiv` 默认是裸 SFU 硬件除法（tilelang 测试
  `test_tilelang_ascend_vmi_vdiv_precision.py` 明确了这一差别）。
- 不带 `precision='exact'` 的实测后果：post_mix 单次 sigmoid 即漂 1–2 bit，
  comb_mix 经 10 轮 sinkhorn 放大到 4–8 bit，layer_input 少量元素跨 bf16 舍入边界。
- 加上后（19 处 vdiv）全量逐位一致；vsqrt/vexpdif 两侧本就同一 SFU 指令，无需处理。
- 注意 `precision='exact'` 还要求显式 mask（内部残差搜索链需要）。

## 三、性能（discussion #1616 验收 3）

> **2026-10-16 补：官方 pytest 基准复验（最终口径）见第五节——5/5 config ratio≥1.00 全部达标，
> 本节 npu-event 口径的"n1=512 低于门槛"结论被其推翻（当时卡 0 共驻干扰 + 含 launch 开销）。**

`bench_prebig_side.py`：每侧 5 warmup + 20 次计时，每次先 memset 512MB flush L2；
min 为 20 次最小值，amort50 为 50 连发摊销（kernel 本体，无逐发 launch 噪声）。
有效带宽全部 ≤ ~1TB/s，无 >4TB/s 的 bench 异常行。

| n1 | h | ASC_us(min) | VMI_us(min) | ratio(min) | ASC_us(amort50) | VMI_us(amort50) | ratio(amort) | 结论 |
|---|---|---|---|---|---|---|---|---|
| 512 | 4096 | 43.4–63.4 | 80.6–151.2 | 0.46–0.79 | 25.6–28.7 | 43.5（最佳观测） | ~0.59 | 正确但低于门槛 |
| 512 | 7168 | 72.5–116.9 | 99.7–150.8 | 0.55–0.73 | 26.7–46.0 | 43.3（最佳观测） | ~0.62 | 正确但低于门槛 |
| 8192 | 4096 | 378.1–418.4 | 401.7–419.8 | 0.94–0.99 | 268.8–270.8 | 269.2–270.6 | **1.00** | **达标** |
| 8192 | 5120 | 437.5–461.0 | 470.6–596.7 | 0.77–0.97 | 336.5–337.8 | 335.9–338.8 | **1.00** | **达标** |
| 8192 | 7168 | 576.7–612.9 | 614.6–715.7 | 0.81–0.99 | 468.8–470.8 | 470.0–471.8 | **1.00** | **达标** |

- 覆盖 `tests/mhc/test_pre_big_fuse.py` `_PRE_BIG_FUSE_BENCH_CASES` 全部 5 个 ASC-legal config。
- 大 batch（8192）摊销计时下两侧逐微秒级一致（ratio≈1.00），多次重跑稳定。
- 小 batch（512）两侧均进入 launch/队列噪声区（ASC amort 恒 ~26.7us 触底），且测量期间
  NPU 0 有共驻 python 进程（npu-smi 可见 PID 3803420/3542570），同配置摊销值在
  43–121us 间摆动；上表取各侧最佳观测。可确认的真实差距与 post / pre_apply_mix 两次
  移植记录的 **ptoas 侧 `vbrc(vload(size=1))` 标量广播材料化开销** 同源：本融合核每
  token block 的标量广播数更多（Phase B 24 个 base + 3 个 scale，Phase D 每 tile 每
  token 4 个 mix），小 batch 时无足够并行度掩盖。属框架侧已知问题，kernel 不绕
  （硬性规定 4），沿用前两次移植的上报口径。

## 四、有意差异点（PR 说明）

1. **`range()` → `T.unroll(..., explicit=True)`**（Phase B pre/post 与 comb 两处循环）：
   TIR 循环下 PTODSL 打印器会丢失循环携带 int 标量的 int32 类型（`vbrc(T.int32(vec_id*4))`
   被打成裸 `(vec_id * 4)`，ptodsl 推断为 index → ptoas 断言 `vreg<64xindex>`）。
   显式展开后 vec_id 成为编译期常量，语义不变（trip count 分别为 1 和 4）。
   最小复现：`verify/repro_ptodsl_loop_scalar_vbrc.py`（已实测触发同一断言），建议上报
   tilelang/ptoas。
2. **sinkhorn 行归约的 `vselr(bcast_idx2)` → `vbrc(..., group=8)`**：逻辑等价（分块广播
   ≡ 该索引置换），少建 3 个索引向量；`vcgadd/vcgmax` → `vcadd/vcmax(mask_lo/hi, group=8)`
   保持两遍互补 mask 结构（VMI 归约 group 只允许 1/2/4/8，与 ASC 硬件 8-lane 对齐约束
   相同，单趟 group=16 不可达）。`vcadd` 浮点必须显式 reassoc，取 True（配对树，
   实测逐位一致）。
3. **`vdiv` 全部带 `precision='exact'` + 显式 mask**（见第二节）。
4. 其余（含 `S.vselr` 转置、vgather 索引算术、UB 布局、T.Pipelined pre-apply 结构）
   与 ASC 逐条同构。

## 五、复现

```bash
# 238, NPU 0
source /data/l00975381/Ascend/cann-9.2.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0 TMPDIR=/data/sunkexin/tkvmi/tmp PATH=/data/sunkexin/tkvmi/conda-env/bin:$PATH
cd /data/sunkexin/tkvmi/TileKernels-PTO
python verify/smoke_prebig.py                     # 冒烟 + torch 参考 sanity
python verify/ab_prebig_asc_side.py               # 生成 ASC 侧 artifact
python verify/ab_prebig_vmi_side.py               # 逐位比对（ALL BITWISE EQUAL）
python verify/bench_prebig_side.py asc|vmi        # 性能（json 落 /data/sunkexin/tkvmi/work/）
python verify/diff_prebig_outer.py                # 外层调度 AST 一致性
```

产物：`verify/report/{smoke_prebig.log, ab_prebig_vmi.log, bench_prebig_asc.json, bench_prebig_vmi.json}`

## 五、官方 pytest 基准复验（2026-10-16，最终性能口径）

按 vmi-verify skill 第七节：ASC 侧原生 dispatch 直接跑 pytest，VMI 侧用通用劫持模板
（`verify/run_pre_big_fuse_bench.py`，劫持 `tile_kernels.mhc.pre_big_fuse_asc` 的
`mhc_pre_big_fuse_fwd`）跑同一条命令；计时 = 仓库 benchmark 插件的 `do_bench(msprof)`
**设备侧 kernel 时间**（torch_npu.profiler + tilelang L2 flush kernel，不含 launch 开销，
与第三节 npu-event 口径不可混表）。**两侧均加 `TILELANG_DISABLE_CACHE=1`**，进程内全新
编译后计时，杜绝缓存产物串扰（见下方"首跑异常"）。JSONL 落盘
`work/pre_big_fuse_bench_{asc,vmi}.jsonl`（该文件为追加式，最终数据前已清空重写）。

| n1 | h | ASC_us | VMI_us | ratio | ASC 带宽 | VMI 带宽 | 结论 |
|---|---|---|---|---|---|---|---|
| 512 | 4096 | 18.7 | 18.6 | **1.005** | 1127 GB/s | 1130 GB/s | **达标** |
| 512 | 7168 | 31.8 | 31.3 | **1.016** | 1158 GB/s | 1177 GB/s | **达标** |
| 8192 | 4096 | 268.8 | 268.1 | **1.003** | 1254 GB/s | 1257 GB/s | **达标** |
| 8192 | 5120 | 335.6 | 335.1 | **1.001** | 1254 GB/s | 1256 GB/s | **达标** |
| 8192 | 7168 | 466.6 | 466.2 | **1.001** | 1262 GB/s | 1263 GB/s | **达标** |

- GEOMEAN：ASC 120.1us / VMI 119.5us（1210 vs 1215 GB/s），VMI 全 config 略优。
- 带宽全部 ≤1.26TB/s，无 >4TB/s 的 bench 异常行。
- **结论：5/5 config（= 全部 ASC-legal bench config）ratio≥0.90，达标；与 ASC 逐位一致 +
  官方基准达标，本算子验收三项全部通过。**

### 首跑异常与排查记录（方法论）

当天第一次 VMI 侧劫持跑曾得到 58.4/74.2/415.2/468.8us 的虚高数字（当天该卡首批 VMI
执行、紧随 22.6s 进程内编译后计时）。后续 5+ 次复跑、独立缓存目录冷跑、以及
`TILELANG_DISABLE_CACHE=1` 进程内现场编译跑全部稳定复现 ≈ASC 的数字（例如 512-4096
= 18.6–18.7us），且劫持经 spy 包裹验证确系 VMI jit 函数被调用 137 次。故判定首跑为瞬态
异常，最终数据取自上表同条件成对复验。**教训：VMI 侧首跑（冷编译后立即计时）数字不可
直接采信，须复跑确认。**

### 环境

- 卡：NPU 2（Ascend950PR，0000:61:00.0），跑前跑后 npu-smi 均 Health OK（前序共驻进程
  已退出后开跑；跑后 186.5W / 59°C / util 0）
- CANN：`/data/llf/Ascend/cann-9.2.0-weekly.20260902.01/cann-9.2.0`（2026-08-30 周构建，
  bisheng 含 `asc_pack_to_high`，msprof 计时正常）
- 工具链：tilelang pto-dev@5f105e99 + ptoas 0.68@d4f1c8a5d（tkvmi conda env，editable），
  pytest 9.1.1
- 代码：ASC 侧 `pre_big_fuse_asc.py`@vmi-impl(66258df) 原样；VMI 侧 `pre_big_fuse_vmi`
  分支（96dff8b），238 树 md5 与本地提交一致（83da2415989748b3a2f26ac42962b1df）
- 复现：`source <CANN>/set_env.sh; export ASCEND_RT_VISIBLE_DEVICES=2 TMPDIR=...
  PATH=<env>/bin:$PATH TILELANG_DISABLE_CACHE=1; cd TileKernels-PTO;
  pytest tests/mhc/test_pre_big_fuse.py -m benchmark --run-benchmark
  --benchmark-output=work/pre_big_fuse_bench_asc.jsonl`（VMI 侧换
  `python work/run_pre_big_fuse_bench.py vmi`）
