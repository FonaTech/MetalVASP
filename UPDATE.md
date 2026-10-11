# Figure and benchmark-data update / 图表与基准数据更新

This is a documentation and derived-data bundle for an existing MetalVASP source checkout. It does not contain an installer, backend source, VASP executable or licensed input files. It changes no numerical kernel or build option.

本包用于已有 MetalVASP 源码目录的文档与派生数据更新，不包含安装器、后端源码、VASP 可执行程序或授权输入文件，不改变数值内核与编译选项。

## Snapshot / 本次快照

Dataset A contains seven reviewed fixtures. Dataset B contains 15/20 completed background systems. All 22 structures have matched CPU/GPU inputs, one warmup and three formal runs per route. The bundle includes 176 selected-route processes, 132 formal timing samples and 88 CPU/GPU comparisons including warmups. Five incomplete background systems are excluded. The remaining local serial campaign continues independently.

A 数据集含七个已审核样本，B 数据集含后台已完成的 15/20 体系。全部 22 个结构使用匹配的 CPU/GPU 输入，每路线一次预热、三次正式计算，包含 176 个所选路线进程、132 个正式计时样本和 88 个含预热 CPU/GPU 对照。其余五个未完成体系不纳入；本地串行测试队列继续运行。

New completed case: Ag-54. Measured CPU/GPU whole-program medians are 337.369775/88.763793 s, with 3.800759× speedup and 73.689% elapsed-time reduction. Paired formal ratios range from 3.710692× to 4.268313×. Every raw formal sample is retained.

新增已完成体系为 Ag-54。实测整程序 CPU/GPU 中位耗时为 337.369775/88.763793 秒，加速 3.800759 倍，总耗时减少 73.689%。正式配对加速比范围为 3.710692–4.268313 倍，全部原始正式样本保留。

All comparisons use VASP 6.4.1 on Apple M5 Max with 128 GiB unified memory, one MPI rank and one BLAS thread. The CPU reference uses the same executable with Metal disabled. Small-system slowdowns, missing observations and negative stage gains remain visible. These measurements do not establish a fully threaded CPU comparison, physical DFT accuracy or a universal speed guarantee.

所有对照在 128 GiB 统一内存的 Apple M5 Max 上使用 VASP 6.4.1、单 MPI rank、单 BLAS 线程；CPU 为同一可执行程序关闭 Metal。小体系减速、缺失观察和阶段负收益均保留。这些测量不代表全线程 CPU 对照、物理 DFT 精度或普遍加速保证。

## Use / 使用方法

Review and copy `docs/assets/`, `benchmarks/data/`, `benchmarks/plot_release.py`, `benchmarks/figure-captions.json`, `docs/FIGURES.md` and both `docs/BENCHMARKS` documents into your local repository. Merge the measured-results sections from the supplied English/Chinese README into your own README. The other documents and validation record are reference copies from the source release; preserve any newer local edits. No GitHub action is performed by this bundle.

审核后将 `docs/assets/`、`benchmarks/data/`、`benchmarks/plot_release.py`、`benchmarks/figure-captions.json`、`docs/FIGURES.md` 及两份 `docs/BENCHMARKS` 文档复制到本地仓库。将附带中英文 README 的实测结果段落合并到你当前的 README。其他文档与校验记录为源码发布包的参考副本，应保留你较新的本地修改。本包不会操作 GitHub。

To reproduce all seven figure families without VASP or a GPU:

无需 VASP 或 GPU，即可重绘全部七组图片：

```sh
python3 -m pip install -r requirements-figures.txt
python3 benchmarks/plot_release.py
```

The export includes 400-dpi PNG, embedded-font PDF, editable-text SVG and sourced CSV/JSON. See [figure index](docs/FIGURES.md), [methods and results](docs/BENCHMARKS.md), [中文说明](docs/BENCHMARKS.zh-CN.md) and [summary CSV](benchmarks/data/summary.csv). Original project text, code and derived graphics remain MIT licensed; separately licensed VASP and third-party rights are unchanged. The source wrapper's queued native rebuild is still pending in the supplied validation record.

提供 400-dpi PNG、嵌入字体 PDF、可编辑文字 SVG 与可追溯 CSV/JSON，见[图片索引](docs/FIGURES.md)、[英文方法与结果](docs/BENCHMARKS.md)、[中文说明](docs/BENCHMARKS.zh-CN.md)及[摘要 CSV](benchmarks/data/summary.csv)。原创文字、代码与派生图仍采用 MIT 授权；VASP 和第三方的独立版权归属保持不变。附带校验记录中，源码入口的排队原生重编译仍未完成。
