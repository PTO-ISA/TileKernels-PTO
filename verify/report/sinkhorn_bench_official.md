# mhc sinkhorn 官方 pytest 基准（vmi-verify skill §7 方法，2026-10-16）

**这是性能验收的权威口径**（repo 官方测试 + msprof 设备侧 kernel 时间，含 tilelang
L2 flush，不含 launch 开销），取代 `ab_sinkhorn_report.md` §3 的 npu-event 辅助口径。

## 方法

- 两侧同一会话环境（同 conda、同 CANN、同卡），只换 dispatch：
  - **asc 侧**：原生 dispatch 跑 `pytest tests/mhc/test_sinkhorn.py -m benchmark --run-benchmark`；
  - **vmi 侧**：`verify/run_sinkhorn_bench.py`（pytest collection 前对
    `tile_kernels.mhc.sinkhorn_asc` 的 `mhc_sinkhorn_fwd_asc / mhc_sinkhorn_bwd_asc`
    setattr 劫持为 `_vmi` 版本），同一条 pytest 命令；
  - `--benchmark-output` 落 JSONL：`sinkhorn_bench_{asc,vmi}.jsonl`（本目录）。
- bench grid = `test_sinkhorn.py` 声明的全部 benchmark config：
  `_SINKHORN_BENCH_CASES = [(1, 512, 4), (1, 8192, 4)]` × fwd/bwd，repeat=10、eps=1e-6。

## 环境

- 238 服务器 **NPU 2**（Ascend950PR；跑前跑后 npu-smi 均 OK/空闲，卡 0 Critical 弃用）
- CANN：`/data/llf/Ascend/cann-9.2.0-weekly.20260902.01/cann-9.2.0`（2026-08-31 构建）
- conda：`/data/sunkexin/tkvmi/conda-env`（tilelang pto-dev@5f105e99 editable +
  ptoas 0.68@d4f1c8a5d）——**ASC 侧亦在此环境原生编译运行**（9.2-weekly 下 pto-dev 的
  ASC 链路可用，不再依赖 tl-run 旧环境）
- TileKernels 树：`/data/sunkexin/tkvmi/TileKernels-PTO`（sinkhorn_vmi 分支内容 =
  commit 1ad1ef6）

## 逐 config 结果（msprof 设备侧时间）

| op | config (n0×n1, mhc=4, repeat=10) | ASC_us | VMI_us | ratio | VMI 带宽 | 结论 |
|---|---|---|---|---|---|---|
| fwd | 1×512 | 4.60 | 4.78 | **0.96** | 0.02 TB/s | **达标**（≥0.90） |
| fwd | 1×8192 | 41.52 | 44.22 | **0.94** | 0.02 TB/s | **达标** |
| bwd | 1×512 | 6.95 | 7.35 | **0.95** | 0.01 TB/s | **达标** |
| bwd | 1×8192 | 54.59 | 78.97 | **0.69** | 0.01 TB/s | 正确但低于门槛 |

geomean ratio ≈ 0.88。全部行 IO 带宽 0.01–0.02 TB/s（计算受限），无 >4TB/s bench 异常行。

## 结论与归因

1. **4 个 config 中 3 个达标（0.94–0.96）**。此前的 npu-event 口径（含 launch 开销、
   测量期服务器有租户负载，ratio 0.55–0.71）把固定开销混入了 kernel 时间；官方
   msprof 设备侧口径下 fwd 两档与 bwd 小 batch 均在门槛之上。
2. **唯一未达标：bwd n=8192，ratio 0.69**，原因与此前定位一致且在此口径下更纯粹：
   bwd 是除法最密的路径（前向重算 + 反传两阶段，每 token-block 约 2×repeat 次
   归一化除法 × 2 个 VR），逐位一致要求的 `vdiv(precision='exact')` 被 tilelang
   前端展开为 ~15 条指令（对照实验：同一内核换 hw 除法后 n=8192 ratio 回到
   0.96–1.00，见 `ab_sinkhorn_report.md` §3）。**修复路径在编译器侧**：把 exact
   除法 lower 为 ASC 同款单条 0-ULP 除法指令（`sinkhorn_ptoas_issues.md` 问题 3），
   落地后该行预期回到 ≥0.95。
3. kernel 侧写法已是最简逻辑形式，无进一步优化空间；无 ASC 回退、无 shape 分支。

## 复现命令（238，会话环境四行）

```bash
source /data/llf/Ascend/cann-9.2.0-weekly.20260902.01/cann-9.2.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2 TMPDIR=/data/sunkexin/tkvmi/tmp PATH=/data/sunkexin/tkvmi/conda-env/bin:$PATH
cd /data/sunkexin/tkvmi/TileKernels-PTO
python /data/sunkexin/tkvmi/work/run_sinkhorn_bench.py asc   # → work/sinkhorn_bench_asc.jsonl
python /data/sunkexin/tkvmi/work/run_sinkhorn_bench.py vmi   # → work/sinkhorn_bench_vmi.jsonl
```
