# Complete-campaign figure update / 完整实验图表更新

This documentation/data overlay includes all20 completed chemistry/size systems and seven supplemental fixtures, 27 structures in total. Each CPU/GPU route has one warmup and three formal runs. It contains 216 CPU/GPU processes, 162 formal timing samples and 108 input/numerical/full-SCF comparisons including warmups. All existing gates passed. The plots compare the released GPU implementation against matched CPU, with one MPI rank and one BLAS thread on Apple M5 Max/VASP6.4.1. Small-system slowdowns, exact zeros, missing memory samples, fallback and negative stage gains are retained.

本图表/数据更新包包括全部20个已完成化学组成/规模体系及七组补充测试，共 27 个结构。每 CPU/GPU 路线一次预热、三次正式计算，包含 216 个 CPU/GPU 进程、162 个正式计时样本及 108 个含预热输入/数值/完整 SCF 对照，原校验门全部通过。图片比较发布版 GPU 与匹配 CPU，采用 Apple M5 Max/VASP6.4.1、单 MPI rank、单 BLAS 线程。小体系减速、精确零值、缺失内存采样、回退及阶段负收益均保留。

Measured median speedups range from 0.907953× to 5.129296×; 26/27 structures are faster. In the complete20-system campaign, 19/20 are faster. These are controlled software runtime observations, not fully threaded CPU comparisons, physical DFT accuracy or a universal speed guarantee.

实测中位加速比范围为 0.907953–5.129296 倍，26/27 个结构更快；完整20体系实验中 19/20 更快。这是受控的软件耗时观察，不代表全线程 CPU 对照、物理 DFT 精度或普遍加速保证。

Review and copy `docs/assets/`, `benchmarks/data/`, `benchmarks/plot_release.py`, `benchmarks/figure-captions.json`, `docs/FIGURES.md` and both benchmark-methods documents into your existing repository. Merge the measured-results sections from the supplied English/Chinese README. Other documentation and the validation record are reference copies; preserve newer local edits. This overlay contains no installer/backend source, VASP binaries or licensed input files. It changes no kernel or build setting and performs no GitHub operation.

审核后将 `docs/assets/`、`benchmarks/data/`、`benchmarks/plot_release.py`、`benchmarks/figure-captions.json`、`docs/FIGURES.md` 和中英文基准方法文档复制到已有仓库，将附带 README 的实测段落合并到你当前的中英文 README。其他文档与校验记录是参考副本，应保留你较新的本地修改。本包不含安装器/后端源码、VASP 二进制或授权输入文件，不改变内核与构建选项，也不操作 GitHub。

```sh
python3 -m pip install -r requirements-figures.txt
python3 benchmarks/plot_release.py
```

Seven figure families are supplied as400-dpi PNG, embedded-font PDF and editable SVG. See [figure index](docs/FIGURES.md), [English methods](docs/BENCHMARKS.md), [中文方法](docs/BENCHMARKS.zh-CN.md) and [summary CSV](benchmarks/data/summary.csv). Original text/code/derived graphics retain MIT licensing, with independent VASP/third-party rights unchanged.

七组图提供400-dpi PNG、嵌入字体 PDF 与可编辑 SVG，见[图片索引](docs/FIGURES.md)、[英文方法](docs/BENCHMARKS.md)、[中文方法](docs/BENCHMARKS.zh-CN.md)及[摘要 CSV](benchmarks/data/summary.csv)。原创文字/代码/派生图仍采用 MIT；VASP 与第三方的独立权利保持不变。

Installer validation is separate from benchmark completion. Backend compilation succeeded, but one lifecycle test exited with SIGTRAP (33/34 passed), stopping before VASP variant builds and installer smoke. This known failed native gate is recorded in `release-validation.json`; this figure update does not claim a fix.

安装器验证独立于基准完成。后端编译成功，一项生命周期测试以 SIGTRAP 异常退出（33/34通过），因此尚未进入 VASP 变体构建与安装器烟测。此已知原生验证失败记录在 `release-validation.json`，本次图表更新不声称已修复。
