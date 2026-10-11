# Compatibility and adaptation contract

| Axis | This release | Evidence boundary |
| --- | --- | --- |
| VASP 6.4.1 native adapter | Sixteen exact source/template fingerprints; std/gam/ncl build rules | Reviewed native builds and 21 repeated static cases |
| Other VASP 6.x | Version inspection; explicit experimental metadata-only route if all fingerprints match | No general native-build or precision claim |
| VASP 5.x / changed 6.x files / VASP 6.6 CMake layout | Inspect or project-only backend; native adaptation rejects | Needs independently implemented adapter |
| Apple Silicon M-series | Runtime device-family/limits and SDK/OS guards; generic and eligible advanced kernels | Current release numerical/performance evidence from M5; historical old-device tests do not validate every new binary |
| macOS source target ≥13 | Availability/SDK guards and portable MSL path | Installed dependency closure may require newer OS; tested native dependencies require macOS26 |
| MSL 4.0 and 2.3 | Backend offline compilation/loading and strict/portable tests reviewed | Does not prove a newly linked VASP binary can run on macOS13 |
| std non-SOC static single rank | Actual native runs and force/stress/SCF comparisons | Eligible new nonlocal route constrained to one k-point/spin/spinor, nonspiral |
| Gamma and SOC/NCL | Six fresh compatibility calculations reviewed; unaffected or CPU fallback operations retained | Gamma had 13 scalar GPU reductions; SOC/NCL had 68 complex GPU GEMMs, FFT GPU0 and new nonlocal0 for those inputs |
| MPI / multiple ranks / distributed FFT/ScaLAPACK | Existing CPU/collective boundaries preserved | New distributed GPU communication and ILP64 remain open |
| HSE, long AIMD, very large systems | Workload-wide certification remains open | No universal precision/speed/scale claim |
| ANE/tensor/Metal 4 encoder | Not production implementations | Optional experimental probe files are not acceleration evidence |

The [official VASP installation guide](https://vasp.at/wiki/Installing_VASP.6.X.X) describes current supported build systems. It is not a compatibility promise for these custom hooks. A name or minor-version similarity cannot replace a source-layout check.

To add a version adapter, work locally on legally obtained clean source. Describe changed interfaces, array layouts, rank/spinor/Gamma guards and collective semantics. Produce local COPY/INSERT instructions that preserve upstream text through copy ranges rather than distributing routines. Require exact input/output hashes, independently test precision and failure atomicity, and match actual shader/native objects. Run completion, full-SCF energy/force/stress comparisons and repeated whole-program measurements before updating this matrix. The new version remains unvalidated until all those gates pass.

`auto` and `advanced` do not force Metal4 onto older SDKs. The runtime chooses supported language versions and kernels from the actual MTLDevice. Host DS compilation disables fast-math and implicit FP contraction; the Metal compiler uses safe math and precise FP32 functions. Portable profile reduces the language/CPU target without pretending to remove the minimum OS of linked libraries.
