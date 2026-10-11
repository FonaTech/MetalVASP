# MetalVASP

[简体中文](README.zh-CN.md) · [User guide](docs/USER_GUIDE.md) · [Technical implementation](docs/TECHNICAL.md) · [Measured results](docs/BENCHMARKS.md) · [Rights and third-party notices](NOTICE.md)

**Metal acceleration for locally licensed VASP on Apple Silicon.** This source-only research release provides double-single compute kernels, bounded workspaces, reusable wave-basis data, and a scoped nonlocal force/stress projector. A local installer identifies the machine and toolchain, checks the VASP source version, and prepares an isolated build.

Version **0.15.0-rc3** provides a local GPU execution path with CPU fallback. **Compared with the matched CPU route, Si128 completed 4.713× faster and used 78.8% less elapsed time.** The public snapshot covers 21 structures, 2–128 atoms, including 14 completed cases from the active 20-system campaign. It contains no VASP source, pseudopotentials or compiled VASP executable.

## Quick start

Requires an arm64 Mac, macOS 13 or newer, Apple developer tools, Python 3.10+, CMake 3.24+, GNU Fortran, matching Open MPI, LP64 OpenBLAS/ScaLAPACK, and FFTW. A dependency's own minimum macOS may be higher than 13. Read the [compatibility matrix](docs/COMPATIBILITY.md).

```sh
# Read-only: identifies hardware, SDK, tools and missing dependencies.
./setup.sh doctor --json diagnostics.json

# Interactive: asks for your LOCAL licensed source/archive path and a NEW work directory.
./setup.sh

# Or inspect and plan without compiling:
./setup.sh plan --source /path/to/vasp.6.4.1.tgz --json plan.json

# Build all variants sequentially; jobs are chosen from performance cores and memory.
./setup.sh build --source /path/to/vasp.6.4.1.tgz --work "$HOME/metalvasp-build" --profile auto

# Execute in your own licensed calculation directory:
VASP_METAL_STATS="$PWD/metal-stats.json" "$HOME/metalvasp-build/run-metal.sh"
```

The source path stays on your computer. Nothing is uploaded or downloaded by the installer. It does not install dependencies automatically. Source/archive paths may contain spaces; Make work/dependency paths must avoid spaces and expansion characters. Existing work directories are never overwritten.

**Validated native adapter: VASP 6.4.1.** Every affected file must match its source fingerprint before changes are applied. Other versions receive an inspection report. An explicit experimental mode allows other 6.x version metadata only when *all* fingerprints still match. Changed layouts require a new adapter and fresh numerical tests; see [version adaptation](docs/COMPATIBILITY.md).

## What is accelerated

| Area | Implemented behavior | Boundary |
| --- | --- | --- |
| Linear algebra | Complex GEMM, reductions, overlap/subspace operations; GPU eigensolver policies and bounded blocks | LP64 interposition; CPU fallback remains |
| FFT/local Hamiltonian | Batched transforms and local operator kernels, reusable allocations | ABI transfers still occur; distributed FFT remains open |
| Wave basis | Generation-checked basis reuse with incremental imports | Complete SCF wave/PAW/Hamiltonian residency remains open |
| Nonlocal force/stress | Factorized radial × phase × wave projection, shared force derivatives, 64-bit logical offsets and memory-budget windows | Opt-in; std complex, one rank/k-point/spin/spinor, nonspiral |
| Modern Metal | Actual device-family/SDK/OS guards choose shared-tile kernels and supported MSL | No ANE/tensor/Metal 4 encoder implementation |

`auto` is recommended. `portable` chooses baseline MSL and a generic ARM CPU target. `advanced` requests eligible tiled kernels but retains capability checks and fallback. Device model names are recorded; they do not override runtime capability checks. These are capability-based choices, not a claim of universally optimal settings. Double-single kernels always use safe/precise math and disabled implicit contraction.

## Why use MetalVASP?

- **Shorter calculation time on Apple Silicon.** Si64 measured 3.282×, Au48 4.302× and Si128 4.713× whole-program speedup against CPU. The reference uses one MPI rank and one BLAS thread; this is not a comparison against a fully threaded CPU configuration.
- **CPU-consistent electronic outputs.** All 84 CPU/GPU input, numerical and full-SCF comparisons passed. Across the snapshot, maximum energy difference was 1.000001×10⁻⁸ eV and maximum force-component difference was 7.00×10⁻⁸ eV/Å.
- **Local setup with explicit device and source checks.** The installer reports missing tools, identifies the actual Metal capabilities, and builds from your own licensed source in an isolated directory.
- **Reuse with bounded operation workspaces.** Wave-basis generations, incremental imports and memory-budget windows avoid unconditional re-import and oversized projector buffers. Layouts that are not eligible retain their CPU path.

## GPU versus CPU: measured results

All figures compare **the released GPU path with CPU** using identical inputs and the same executable with Metal enabled/disabled. They show 21 reviewed structures on Apple M5 Max with VASP 6.4.1, one MPI rank and one BLAS thread. Each route used one warmup and three fresh formal runs: **168 selected-route processes, 126 formal timing samples and 84 CPU/GPU numerical/full-SCF comparisons**. Dataset A contains seven completed fixtures; dataset B contains 14/20 completed background cases. Running or unfinished cases are excluded.

![GPU versus CPU whole-program speedup](docs/assets/01-speed.png)

GPU route medians were faster in **20/21** reviewed structures. Observed median ratios span **0.908–4.713×**. The 2-atom C fixture was slower at **0.908×**, preserving small-workload startup/call overhead. Si128 CPU/GPU medians were **940.097 / 199.475 s**. Dots show all three paired runs; ranges are observed min–max, not confidence intervals. One device and three repeats do not establish universal speedup.

![CPU/GPU numerical agreement](docs/assets/02-precision.png)

Maximum printed-output differences including warmups: energy **1.000001×10⁻⁸ eV**, force component **7.00×10⁻⁸ eV/Å**, stress component **4.52×10⁻⁶ kbar**, full-SCF energy sequence **1.970001×10⁻⁷ eV**. These describe software agreement, not physical DFT accuracy. Exact zero differences are retained.

![CPU/GPU absolute total and SCF times](docs/assets/03-times.png)

The [detailed benchmark document](docs/BENCHMARKS.md) contains every structure, absolute CPU/GPU times, SCF timings, speedups, settings, source-stage results, memory scopes, import/cache telemetry and fallback behavior. [All seven figure families](docs/FIGURES.md) include 400-dpi PNG, PDF and editable SVG, with CSV/JSON and Matplotlib source. CPU-relative slowdowns and missing observations are kept.

## Documentation and reproducibility

| Document | English | 中文 |
| --- | --- | --- |
| Installation, diagnostics, running and troubleshooting | [User guide](docs/USER_GUIDE.md) | [用户指南](docs/USER_GUIDE.zh-CN.md) |
| Version/device/deployment support | [Compatibility](docs/COMPATIBILITY.md) | [兼容性](docs/COMPATIBILITY.zh-CN.md) |
| Kernels, precision, ownership, memory and source map | [Technical implementation](docs/TECHNICAL.md) | [技术实现](docs/TECHNICAL.zh-CN.md) |
| Methods, measured evidence and limitations | [Benchmarks](docs/BENCHMARKS.md) | [基准说明](docs/BENCHMARKS.zh-CN.md) |
| Copyright boundaries and disclaimer | [Legal](docs/LEGAL.md) | [法律声明](docs/LEGAL.zh-CN.md) |
| Publishing this source-only release | [Release guide](docs/RELEASING.md) | [发布说明](docs/RELEASING.zh-CN.md) |

```sh
python3 -m unittest discover -s Tests -p test_installer.py -v   # Installer safety; no VASP or GPU required
python3 -m pip install -r requirements-figures.txt
python3 benchmarks/plot_release.py        # Rebuild all measured PNG/PDF/SVG figures
./setup.sh backend --work "$HOME/metalvasp-backend" --profile portable
```

Use the exact `Tests/` capitalization on every platform. Installer CI runs independently on Linux and macOS; the package audit rejects incorrect discovery paths and empty test selections.

The backend command runs strict GPU tests on your Mac. Source reconstruction matches the reviewed native build byte-for-byte. Installer safety and source-only checks passed; its complete native rebuild remains queued behind the active serial benchmark. This distinction is recorded in `release-validation.json`.

## Repository layout

```text
setup.sh                    Local interactive/CLI entry point
Scripts/metalvasp.py         Device/environment/source diagnostics and build transaction
Sources/, Fortran/          Original project backend and interfaces
adapters/                   Fingerprint-gated local COPY/INSERT source rules
Tests/                      Numerical backend and installer tests
benchmarks/data/             Nonconfidential measured derived data with SHA/locators
benchmarks/plot_release.py   Matplotlib figure generator
docs/                       Bilingual documentation and publication-quality assets
```

![Distinct GPU-route API, CPU, GPU, wait and conversion timers](docs/assets/06-api-timers.png)

The timer figure separates observed durations without stacking overlapping measurements or calling GPU command time hardware occupancy.

## License and attribution

Original project code, documentation and derived figures are licensed under [MIT](LICENSE). VASP remains separately licensed proprietary software. Apple tools/frameworks, GNU runtimes, MPI, BLAS, FFTW and other dependencies retain their own rights and terms. MIT grants no VASP access or redistribution rights. See [NOTICE](NOTICE.md) and [legal disclaimer](docs/LEGAL.md), including the FFTW licensing boundary for locally linked artifacts.

This project is independent of VASP Software GmbH and Apple and is not endorsed by either. Cite the exact release/commit and artifact checksum using [CITATION.cff](CITATION.cff). No software DOI has been assigned. Documentation organization was informed by [FonaTech/E3-miu-GNN](https://github.com/FonaTech/E3-miu-GNN); no model code or data was copied from that repository.
