# GPU versus CPU: methods and measured results

## Population and execution controls

This source release contains one GPU implementation. CPU is the identical executable with Metal disabled; GPU enables Metal, ownership and the scoped nonlocal hook. The Apple M5 Max host has 128 GiB unified memory. VASP is 6.4.1, std complex, with one MPI rank, one BLAS thread and Gamma-centered 1×1×1 sampling. This reference is a controlled single-thread-BLAS CPU route, not an optimized fully threaded CPU comparison.

Dataset A has seven complete 20–128-atom fixtures. Dataset B snapshots 14 completed cases from the active 20-case campaign. Only cases with all routes normally completed, input/numerical/full-SCF gates passed and clean timing were eligible. All eligible cases at the snapshot are included. The 6 unfinished cases are excluded. The combined public population is 21 structures, 168 CPU/GPU processes (including one warmup per route), 126 formal timing samples and 84 GPU-to-CPU comparisons including warmups. These counts describe only the public CPU/GPU subset.

The two datasets are different input populations, not different software versions. Do not combine them into a size-only scaling law: chemistry, spin, cutoffs and bands co-vary. One warmup and three fresh formal runs were used per CPU/GPU route. Timing validation checks owned process identities, process-stop/clock events and competing native work. Exact owned normal-exit identity events were resolved by retained read-only evidence; unknown events were not silently accepted. Raw states and private evidence remain archived. Nothing was rerun or tolerance-relaxed to improve a figure.

## Results and formulas

| Structure | Atoms | CPU median (s) | GPU median (s) | CPU/GPU | Time saved (%) | SCF CPU/GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C-2 | 2 | 0.879352 | 0.968500 | 0.907953 | -10.138 | 0.703668 |
| GaAs-4 | 4 | 1.666861 | 1.625407 | 1.025504 | 2.487 | 1.101519 |
| BN-6 | 6 | 1.622885 | 1.580279 | 1.026962 | 2.625 | 1.130400 |
| LiF-8 | 8 | 2.129493 | 1.885772 | 1.129242 | 11.445 | 1.319451 |
| NaCl-10 | 10 | 4.077223 | 2.978691 | 1.368797 | 26.943 | 1.694616 |
| MgO-12 | 12 | 5.619840 | 4.127943 | 1.361414 | 26.547 | 1.679989 |
| SiC-16 | 16 | 9.643923 | 6.274369 | 1.537035 | 34.940 | 1.885538 |
| TiO2-18 | 18 | 17.480233 | 9.132300 | 1.914111 | 47.756 | 2.146677 |
| Al20 | 20 | 6.309551 | 3.952992 | 1.596146 | 37.349 | 1.952888 |
| BN20 | 20 | 7.573804 | 4.620557 | 1.639154 | 38.993 | 2.031643 |
| GaAs20 | 20 | 12.034225 | 6.593992 | 1.825029 | 45.206 | 2.222195 |
| MgO20 | 20 | 12.695285 | 7.110474 | 1.785434 | 43.991 | 2.155646 |
| Ni-20 | 20 | 72.941117 | 32.309082 | 2.257604 | 55.705 | 2.496257 |
| Si20 | 20 | 10.122421 | 5.624866 | 1.799584 | 44.432 | 2.222716 |
| Fe-24 | 24 | 111.295788 | 43.620043 | 2.551483 | 60.807 | 2.840798 |
| Co-32 | 32 | 215.035842 | 79.777157 | 2.695456 | 62.901 | 3.031464 |
| ZnO-36 | 36 | 149.486364 | 45.250193 | 3.303552 | 69.730 | 3.428688 |
| Al2O3-40 | 40 | 51.769037 | 17.378933 | 2.978839 | 66.430 | 3.209740 |
| Au-48 | 48 | 199.764991 | 46.436966 | 4.301853 | 76.754 | 4.385000 |
| Si64 | 64 | 127.961023 | 38.993789 | 3.281575 | 69.527 | 3.358927 |
| Si128 | 128 | 940.096923 | 199.475406 | 4.712846 | 78.781 | 4.738176 |

Whole-program speedup is `median(T_CPU) / median(T_GPU)`. Time saved is `100 × (1 − median(T_GPU)/median(T_CPU))`; a negative value is extra GPU-route time. Paired dots are `T_CPU,j / T_GPU,j`, which need not have the same median as the ratio of route medians. SCF speedup uses the sum of LOOP wall times inside each run, then compares medians. Min–max whiskers show the three observations, not statistical confidence intervals. C-2 is slower and is retained; larger-case gains do not prove every input will accelerate.

## Numerical agreement

All 84 comparisons passed their original input, numerical, eligible-GPU and complete electronic-sequence gates. Maxima including warmups are total energy 1.00000079328311e-08 eV; force component 6.99999999995843e-08 eV/Å; force-vector root-mean-square error 3.77491721729444e-08 eV/Å; stress component 4.52000000006336e-06 kbar; complete self-consistent-field (SCF) energy sequence 1.97000076695986e-07 eV. SCF iteration counts match per pair. Three formal points per structure are plotted; warmup records remain in data.

For atom `i`, `ΔF_i = F_i^GPU − F_i^CPU`. Vector RMSE is `sqrt(mean_i(sum_component(ΔF_i²)))`; component RMSE averages all 3N components. Maximum vector norm differs from maximum component error. Energy/atom difference divides total energy difference by atom count. Exact printed zeros are retained. These values quantify software agreement for the given fixtures, not physical density-functional-theory accuracy, unseen inputs or a universal error bound.

## Conditions and implementation scope

`benchmarks/data/input-settings.json` lists every INCAR setting, k-point mesh and input hash, including POTCAR hashes without distributing PAW data. Typical controls are Accurate precision, Normal SCF, no real-space projection, no symmetry, static fresh starts and EDIFF=1e-6; cutoffs, band counts, smearing and spin are case-specific. These are benchmark assumptions, not general convergence recommendations. CPU/GPU inputs and algorithms match within each pair.

The nonlocal force/stress GPU projector is currently eligible for std complex, single rank/k-point/spin/spinor and nonspiral layouts. Ni20, Fe24 and Co32 use ISPIN=2: nonlocal projection falls back to CPU, while eligible FFT, linear-algebra and other kernels still execute on the GPU. A zero projection counter in these cases is not a failed calculation or evidence of zero overall GPU work. Actual Gamma/SOC fallback tests are recorded separately in `release-validation.json`.

## Resource and timing scopes

RSS is native-process resident set size sampled every two seconds. Some sub-second/short runs lack RSS observations; they remain NA and are omitted from that panel. Metal peaks are observed at tracked project allocations, not driver-wide VRAM or total unified memory. Planned nonlocal workspace is a bound for one operation. Host transactional staging is unmeasured (NA). These scopes overlap and must not be added. Other applications and unsampled driver/process peaks are excluded.

Wave-basis imports are cumulative logical ABI bytes; cache reuse counters refer to this path. They are neither physical bus transactions nor all communication or peak storage. Current telemetry is exported without a historical-version baseline. Source-stage wall times include waits, and parent LOOP/STRESS stages overlap children. API wall/process-CPU/GPU-command/wait/conversion values also overlap; no figure stacks them or equates command duration with hardware occupancy.

## Reproducible public evidence

`measured.json` supplies every case and route summary, precision, resources, timers, sources and accounting scopes. Companion CSV files expose all selected-route runs and formal samples. Neutral evidence IDs link each derived measurement to retained evidence SHA256 and a JSON/VTIME locator; the private path mapping is retained by maintainers. Original licensed source/PAW data, binaries and private logs are excluded. Derived data alone do not replace a VASP license or reproduce private inputs.

Run `python3 benchmarks/plot_release.py` with `requirements-figures.txt`. [Figure index](FIGURES.md) describes all seven arguments and both-language captions. PNG uses 400 dpi, PDF embeds fonts and SVG retains text. File hashes, data hashes and 180 mm design specs are in `docs/assets/figure-manifest.json` and `figure-specs.json`.

The public backend/source adapter is unchanged and source-reconstruction checked. The new public wrapper's full native rebuild remains queued behind the active serial benchmark. Current source/installer checks do not relabel that pending build as complete. See `release-validation.json`; full SCF residency, remaining wide-index domains, ILP64, distributed work and other projector layouts remain open.
