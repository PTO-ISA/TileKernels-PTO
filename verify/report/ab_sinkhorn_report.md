# mhc sinkhorn ASC → VMI 移植验收报告（sinkhorn_vmi 分支）

- 移植对象：`tile_kernels/mhc/sinkhorn_asc.py` → `tile_kernels/mhc/sinkhorn_vmi.py`
  （`mhc_sinkhorn_fwd_vmi` / `mhc_sinkhorn_bwd_vmi`；外层 schedule 与 host 契约逐字保留，
  程序化 AST diff 验证：`verify/diff_sinkhorn_outer.py` OK）
- 环境（238 服务器，NPU 0，Ascend950PR / a5）：
  - VMI 侧：tilelang pto-dev **5f105e99**（含 PR #6）+ ptoas **0.68**（master d4f1c8a5d），
    环境 `/data/sunkexin/tkvmi/conda-env`
  - ASC 侧：tl-run 环境 tilelang **0.1.12** + CANN 9.1（pto-dev 的 ASC 链路需 CANN 9.2，
    沿袭 post_vmi / pre_apply_mix_vmi 的既有结论与基线）
- 数据流：ASC 侧（tl-run env）固定种子生成输入并跑 ASC kernel 落盘；VMI 侧（tkvmi env）
  加载同一输入跑 VMI kernel，逐位对比。脚本：`verify/ab_sinkhorn_asc_side.py` /
  `verify/ab_sinkhorn_vmi_side.py`。

## 0. 算子概况与 lane 布局

sinkhorn 归一化（`sinkhorn_normalize_ref`）：`x = softmax(x)+eps; x /= colsum+eps;
repeat-1 次 { x /= rowsum+eps; x /= colsum+eps }`，fwd 单 pass、bwd 含前向重计算 +
反传两阶段。每 64-lane fp32 VR 装 4 个 token×16 元素（`lane = 16t+4r+c`），行 = 连续
4 lane，4×4 转置 = 按索引 vselr。fwd 每 token-block 4 token；bwd 8 token（2 VR），
中间态存 xs_ub/sums_ub，两阶段间 `mem_bar('VST_VLD')`。

## 1. SimdVF 逐条对照（ASC → VMI）

| ASC（sinkhorn_asc.py） | VMI（sinkhorn_vmi.py） | 说明 |
|---|---|---|
| `S.pset(32)` | `V.create_mask(64, size=64)` | 64-lane fp32 的满宽 mask |
| `S.vci(0, T.int32)` | `V.vci(T.int32(0), size=64)` | lane 递增斜坡 |
| `S.vdup(c, T.int32, m)` | `V.vbrc(T.int32(c), size=64)` | 整型标量广播 |
| `S.vand/vadd(a, b, m)` | `V.vand/vadd(a, b)` | 满宽逐元素省 mask |
| `S.vshls/vshrs(v, c, m)` | `V.vshls/vshrs(v, c, mask_full)` | mask 必须显式 |
| `S.vcmps(v, c, m, op)` | `V.vcmps(v, c, mask_full, op)` | mask 作 seed |
| `S.vsel(a, b, m)` | `V.vsel(m, a, b)` | mask 前置 |
| `S.vselr(v, i)`（转置/选播） | `V.vselr(v, i)` | ASC 前端的 f32→u32 位重解释由 VMI lowering 承担 |
| `S.vexpdif(x, mx, m)` | `V.vexpdif(x, mx, mask_full)` | mask 必须显式 |
| `S.vadds(v, eps, m)` | `V.vadd(v, eps_b)`（`eps_b = V.vbrc(T.float32(eps), size=64)`） | 广播 + 普通向量加 |
| `S.vmul/vsub(a, b, m)` | `V.vmul/vsub(a, b)` | 满宽省 mask |
| `S.vld(a)` | `V.vload(a, size=64)` | |
| `S.vsts(a, v, m)` | `V.vstore(v, a)` | 值前地址后 |
| `S.mem_bar('VST_VLD')` | `T.simd.mem_bar('VST_VLD')` | 仍走 simd 侧 |
| `S.alloc_var(T.float32)` | `V.alloc_var(T.float32, size=64)` | |
| `@tilelang.jit` | `@tilelang.jit(target='pto')` | |

**行/列归约与除法（本算子核心，含 3 处有意差异）**：

| ASC | VMI | 说明 |
|---|---|---|
| `_row_max`：`vcgmax×2 + vselr×2 + vsel` | `vselr`+`vmax` 二叉树（2 轮） | **有意差异①**（见下）；max 顺序无关 → 逐位一致 |
| `_row_reduce`：`vcgadd×2 + vselr×2 + vsel` | `vselr`+`vadd` pairwise 树 `(a+b)+(c+d)` | **有意差异②**；与 ASC vcgadd 折叠顺序逐位一致（`verify/fold_order_check.py` 对落盘 ASC rsum 枚举验证，pairwise 0/16 行差异） |
| `S.vdiv(a, b, m)` | `V.vdiv(a, b, mask_full, precision='exact')` | **有意差异③**；ASC vdiv 是正确舍入除法，VMI 默认 hw 除法 ~8% 商差 1 ULP 且被 2×repeat=20 次归一化放大；`exact` 逐位一致（`verify/probe7_vdiv_exact.py`） |
| 索引序言：`mod8/mask_lo/mask_hi/bcast_idx2`（为 vcg 的奇偶行拆分服务） | 删除；新增 `pair_swap_idx = idx^1`、`cross_pair_idx`（行内配对索引） | 归约改树后奇偶拆分与选播索引不再需要；`t_idx`（4×4 转置）逐字保留 |

有意差异点均为 ptoas 现行缺陷或舍入差异所迫，非语义变化，无 ASC 物理布局回退、
无 host 侧规避、无 shape 分支。缺陷详情与最小复现：**`verify/report/sinkhorn_ptoas_issues.md`**
（问题 1：`vcmax` partial mask + group 数值错；问题 2：`vcadd` 结果被 `vdiv` 消费时
错误编译；问题 3：hw `vdiv` 舍入与 ASC 不一致、`exact` 需单条 0-ULP 指令 lowering）。

## 2. 精度（验收②）

同种子同输入，6 个 config（随机 logits + 大同号 logits ±100、eps 1e-6/1e-2），
fwd/bwd 全部输出张量 **torch.equal 逐位一致**（0 mismatch、0 bit distance）：

| config | fwd out | bwd grad_input |
|---|---|---|
| random n=8 | eq | eq |
| random n=128 | eq | eq |
| random n=1024 | eq | eq |
| random n=4096 | eq | eq |
| logits+100 n=8, eps=1e-6 | eq | eq |
| logits−100 n=8, eps=1e-2 | eq | eq |

无 1-ULP 残余。torch reference sanity：`verify/smoke_sinkhorn.py` **10/10 PASS**
（判据同 `tests/mhc/test_sinkhorn.py`，含有限性检查）。config 覆盖说明：ASC-legal
grid 为 mhc=4（hidden=4）、repeat=10、eps∈{1e-6,1e-2}、num_tokens 为 8 的倍数
（原实现对非整块 token 数本就无 tail 处理，固定形状块拷贝越界依赖分配器余量，
两侧外层结构相同故行为同类，不在 A/B 范围）；A/B 覆盖 {8,128,1024,4096}，
bench 覆盖 {512,2048,8192}。

## 3. 性能（验收③，flush L2，NPU Event 计时，min of 20）

ratio = ASC_us / VMI_us，达标线 0.90。本算子 IO 极小（0.07–1.6 MB），全程
0–25 GB/s，**计算受限**，无 >4TB/s bench 异常行。测量时服务器有其他租户负载，
单轮 min 漂移明显（两侧同向），下表取 **3 轮 ASC/VMI 交替 + 2 轮单侧共 5 轮的
各侧最优 min**（逐轮原始数据见 `bench_sinkhorn_asc.json` / `bench_sinkhorn_vmi.json`
与过程记录）：

| op | n | ASC_us | VMI_us | ratio | 结论 |
|---|---|---|---|---|---|
| fwd | 512 | 27.89 | 47.40 | 0.59 | 正确但低于门槛 |
| fwd | 2048 | 25.56 | 43.06 | 0.59 | 正确但低于门槛 |
| fwd | 8192 | 46.58 | 69.17 | 0.67 | 正确但低于门槛 |
| bwd | 512 | 24.98 | 45.21 | 0.55 | 正确但低于门槛 |
| bwd | 2048 | 24.37 | 44.22 | 0.55 | 正确但低于门槛 |
| bwd | 8192 | 62.33 | 87.26 | 0.71 | 正确但低于门槛 |

**Gap 分析（已定位到指令级）**：

1. **主因：`vdiv precision='exact'` 的前端展开**。逐位一致要求 exact 除法
   （§1 差异③），而 tilelang `vmi.py` 将其展开为约 15 条 vmi 指令的修正序列；
   sinkhorn 每 token 需 20 次除法（fwd）更多（bwd），该开销乘 20 放大。对照实验
   （同一 VMI kernel 仅把 vdiv 换回 hw，较安静时段测量）：n=8192 时
   VMI-hw fwd 62.85us / bwd 73.02us vs ASC 60.27/73.06us → **ratio ≈ 0.96 / 1.00**。
   即大 batch 的 gap 几乎全部来自 exact 展开。**修复路径在编译器侧**：把
   `precision='exact'`（或默认 vdiv）lower 到 ASC 链路同款单条 0-ULP 除法指令
   （CANN `vdiv_0ulp`），可同时获得逐位一致与单指令性能（issue 材料 §问题 3）。
2. **小 batch（n=512）另有固定开销差**：hw-div 变体下 VMI fwd 512 仍 41.0us vs
   ASC 27.9us，与 post_vmi / pre_apply_mix_vmi 的小 batch 模式同因（每 tile 序言
   指令数与 launch 路径差异），该算子 tile 极小（4/8 token）而轮数多，固定开销
   占比更高。
3. kernel 侧写法已是最简逻辑形式（归约树 4 op vs ASC 的 5 op），无进一步 kernel
   侧优化空间；待编译器侧修复后 ratio 预期回到 ≥0.95。

## 环境备注（沿袭前序分支，PR/issue 素材）

1. tilelang pto-dev ≥ 5f105e99 的 ASC（非 PTO）链路需 CANN 9.2.0+；ASC 基准沿用
   tl-run 环境 tilelang 0.1.12 + CANN 9.1。
2. **ptoas 三项缺陷**（本次新发现，附最小复现）：见
   `sinkhorn_ptoas_issues.md` —— vcmax partial-mask+group 数值错、vcadd→vdiv
   消费者误编译、hw vdiv 舍入差异 + exact 除法需单条指令 lowering。
3. VMI 侧运行需 `PATH` 含 `/data/sunkexin/tkvmi/conda-env/bin`（ptoas CLI 裸命令调起）。

## 复现命令（238）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0 TMPDIR=/data/sunkexin/tkvmi/tmp
export PATH=/data/sunkexin/tkvmi/conda-env/bin:$PATH
cd /data/sunkexin/tkvmi/work
# 冒烟 + torch 参考（VMI 侧）
/data/sunkexin/tkvmi/conda-env/bin/python smoke_sinkhorn.py
# 精度 A/B：ASC 侧（tl-run env）→ VMI 侧（tkvmi env）
/data/sunkexin/tl-run/conda-env/bin/python ab_sinkhorn_asc_side.py
/data/sunkexin/tkvmi/conda-env/bin/python ab_sinkhorn_vmi_side.py
# 性能（建议 ASC/VMI 交替多轮取各侧最优 min，见 §3 说明）
/data/sunkexin/tl-run/conda-env/bin/python bench_sinkhorn_side.py asc
/data/sunkexin/tkvmi/conda-env/bin/python bench_sinkhorn_side.py vmi
# 外层结构 AST diff（本机）
python3 verify/diff_sinkhorn_outer.py
```

原始结果：`smoke_sinkhorn.log` / `ab_sinkhorn_asc.log` / `ab_sinkhorn_vmi.log` /
`bench_sinkhorn_asc.json` / `bench_sinkhorn_vmi.json`（本目录）。
