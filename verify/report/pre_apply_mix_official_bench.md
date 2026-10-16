# pre_apply_mix 官方 pytest 基准验收（vmi-verify skill §7/§8，2026-10-16）

- 对象：`tests/mhc/test_pre_apply_mix.py`（`-m benchmark --run-benchmark`，5 fwd + 5 bwd = 10 项）
- kernel：`tile_kernels/mhc/pre_apply_mix_asc.py`（原生 dispatch）vs
  `tile_kernels/mhc/pre_apply_mix_vmi.py`（skill §7 劫持模板
  `work/run_pre_apply_mix_bench.py`，劫持基名 `mhc_pre_apply_mix_fwd` / `mhc_pre_apply_mix_bwd`）
- 原始 JSONL：`pre_apply_mix_bench_{asc,vmi}.jsonl`（--benchmark-output 落盘）

## 环境

| 项 | 值 |
|---|---|
| 卡 | NPU 2（Ascend950PR）；跑前 OK/58°C、跑后 OK/60°C（卡 0 Critical 为既有故障，未使用） |
| CANN | `/data/llf/Ascend/cann-9.2.0-weekly.20260902.01/cann-9.2.0`（2026-08-31 构建，bisheng ≥08 月版） |
| python env | 两侧同 env `/data/sunkexin/tkvmi/conda-env`：tilelang 0.1.15（pto-dev@5f105e99）+ ptoas 0.68（master@d4f1c8a5d） |
| ASC 侧代码 | `pre_apply_mix_asc.py` = vmi-impl@66258df 原生 dispatch，无任何改动 |
| VMI 侧代码 | `pre_apply_mix_vmi.py` = pre_apply_mix_vmi 分支@b53c9ee（服务器树为同步副本） |
| 计时口径 | 官方 harness `benchmark_timer = do_bench(msprof)`：**设备侧 kernel 时间**，不含 launch 开销（skill §7 注 1；与 2026-10-15 报告 §3 的 npu-event 口径不可混表比较） |

两侧命令（同一 conda、同一 CANN、同卡，只换 side）：

```bash
source /data/llf/Ascend/cann-9.2.0-weekly.20260902.01/cann-9.2.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2 TMPDIR=/data/sunkexin/tkvmi/tmp
export PATH=/data/sunkexin/tkvmi/conda-env/bin:$PATH
python /data/sunkexin/tkvmi/work/run_pre_apply_mix_bench.py asc   # → pre_apply_mix_bench_asc.jsonl
python /data/sunkexin/tkvmi/work/run_pre_apply_mix_bench.py vmi   # → pre_apply_mix_bench_vmi.jsonl
```

## 逐 config 表（ratio = ASC_us / VMI_us，达标 ≥0.90）

| kernel | n1 | h | ASC_us | VMI_us | ratio | ASC GB/s | VMI GB/s | 结论 |
|---|---|---|---|---|---|---|---|---|
| fwd | 512 | 7168 | 29.89 | 29.81 | 1.003 | 1228.0 | 1231.5 | 达标 |
| fwd | 512 | 4096 | 17.51 | 17.44 | 1.004 | 1198.2 | 1202.8 | 达标 |
| fwd | 8192 | 7168 | 459.10 | 461.17 | 0.996 | 1279.3 | 1273.6 | 达标 |
| fwd | 8192 | 5120 | 327.61 | 329.19 | 0.995 | 1280.7 | 1274.5 | 达标 |
| fwd | 8192 | 4096 | 263.00 | 264.17 | 0.996 | 1276.3 | 1270.7 | 达标 |
| bwd | 512 | 7168 | 76.75 | 77.81 | 0.986 | 1243.5 | 1226.6 | 达标 |
| bwd | 512 | 4096 | 43.71 | 44.48 | 0.983 | 1247.8 | 1226.2 | 达标 |
| bwd | 8192 | 7168 | 1199.10 | 1276.21 | **0.940** | 1273.4 | 1196.5 | 达标（最接近门槛） |
| bwd | 8192 | 5120 | 860.85 | 863.91 | 0.996 | 1267.1 | 1262.6 | 达标 |
| bwd | 8192 | 4096 | 686.63 | 689.60 | 0.996 | 1271.0 | 1265.5 | 达标 |

- GEOMEAN：ASC 185.9 us / 1256.3 GB/s，VMI 187.9 us / 1242.7 GB/s（ratio 0.99）
- 覆盖：test_pre_apply_mix.py 声明的全部 5 个 bench config × fwd/bwd，10/10，两侧均 `10 passed`
- 带宽复核：全部行 1196–1281 GB/s，无 >4 TB/s 行，无 bench 异常剔除项
- 结论：**官方基准口径下全部 config 达标（0.940–1.004）**；尚无"低于门槛/无实现"行

## 与 2026-10-15 npu-event 口径结果的口径差异说明

昨日报告（`ab_pre_apply_mix_report.md` §3）的 fwd 小 batch（n1=512）ratio 0.78–0.83 是
npu-event 口径（含 launch 开销，ASC 侧在 tl-run 旧 env、卡 0）下测得；今日官方 msprof
口径（纯设备侧 kernel 时间，两侧同 env 同卡）下同批 config ratio 为 1.003–1.004，
说明该 gap 不在设备侧 kernel 执行本身，而在 host/launch 侧固定开销（`vbrc(vload(size=1))`
展开为两条指令带来的指令流膨胀等），两组数字口径不同不可混比。bwd (8192,7168) 的
0.940 为本口径下最接近门槛的行。
