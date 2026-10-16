# pre_split_mixes 官方 pytest 基准 A/B（vmi-verify skill §7/§8 口径）

本报告补充并**修正** `ab_pre_split_mixes_report.md` §3 的性能结论：官方 do_bench(msprof)
设备侧计时 + 两侧同工具链编译下，VMI 四个 config 全部达标（ratio 0.91–1.03）。
此前"低于门槛"的结论源于 (a) npu-event 计时含 launch 开销（本算子 launch 主导）与
(b) ASC 基准用旧 tilelang 0.1.12（tl-run env）编译的跨工具链对照。两种口径数字不可混表
（skill §7 注意事项 1）。

## 方法

- 命令：`pytest tests/mhc/test_pre_split_mixes.py -m benchmark --run-benchmark
  --benchmark-output=work/pre_split_mixes_bench_{asc,vmi}.jsonl`
- ASC 侧：原生 dispatch（`pre_split_mixes_kernel.py` → `pre_split_mixes_asc.py`）。
- VMI 侧：`run_pre_split_mixes_bench.py vmi`（skill §7 劫持模板，pytest collection 前
  setattr 替换 `mhc_pre_split_mixes_{fwd_simd_mhc4,bwd_mhc4}_asc` → `_vmi`，官方测试全链路走 VMI）。
- 计时：`benchmark_timer = do_bench(msprof)`，torch_npu.profiler 设备侧 kernel 时间 +
  tilelang L2 flush kernel，不含 launch 开销。
- 两侧各跑 3 轮（首轮含 kernel 编译；msprof 偶发污染行见下），取 min 口径（污染只会加时）。

## 逐 config 表（min of 3 runs）

| operation | param_id | ASC_us | VMI_us | ratio=ASC/VMI | ASC BW | VMI BW | 结论 |
|---|---|---|---|---|---|---|---|
| pre_split_mixes_forward | 1-512-4 | 1.99 | 2.02 | 0.99 | 49.6 GB/s | 47.0 GB/s | **达标** |
| pre_split_mixes_forward | 1-8192-4 | 3.09 | 3.40 | 0.91 | 508.6 GB/s | 462.8 GB/s | **达标**（贴线，VMI fwd 大 shape 恒定 ~10% 差，同前报告序言开销主题） |
| pre_split_mixes_backward | 1-512-4 | 7.23 | 7.41 | 0.98 | 20.4 GB/s | 19.9 GB/s | **达标** |
| pre_split_mixes_backward | 1-8192-4 | 8.81 | 8.58 | 1.03 | 267.9 GB/s | 273.3 GB/s | **达标**（VMI 更快） |

GEOMEAN ratio ≈ **0.98**。全部行 IO 带宽 ≤509 GB/s，无 >4TB/s 的 bench 异常行。

三轮原始明细见 `pre_split_mixes_bench_{asc,vmi}.jsonl`（每侧 12 条 = 4 config × 3 轮）。
观测到的 msprof 污染行（2–10× 偏高、不同轮打在不同 config、不可复现，min 口径剔除）：
ASC 轮1 bwd@512=45.62us；ASC 轮3 fwd@512=14.72 / fwd@8192=17.12 / bwd@8192=26.34us；
VMI 轮1 bwd@8192=18.37us。跑时卡 1 他人满载（100% util），共享主机抖动为疑似来源。

## 环境

- 238 服务器（ssh -p 10118），**NPU 卡 2** Ascend950PR，跑前跑后 `npu-smi` Health=OK（卡 0
  Critical、卡 1 Alarm 满载，均未使用）
- CANN：`/data/llf/Ascend/cann-9.2.0-weekly.20260902.01/cann-9.2.0`（weekly 20260902.01，
  08-31 构建，bisheng ≥08 月版，msprof 实测正常）
- 环境 `/data/sunkexin/tkvmi/conda-env`：tilelang 0.1.15（editable，pto-dev@5f105e99）+
  ptoas 0.68（editable，master@d4f1c8a5d）——**两侧同 env 同 CANN 同工具链编译**，只换 kernel 源
- 代码版本：ASC = `pre_split_mixes_asc.py`（vmi-impl@66258df）；VMI = `pre_split_mixes_vmi.py`
  （pre_split_mixes_vmi 分支@61619f4，238 树内已同步 T.unroll 修正版）
- 服务器归档：`/data/sunkexin/tkvmi/work/report/`；本地归档：
  `TileKernelVMI/verify/report/pre_split_mixes_official_bench/`
