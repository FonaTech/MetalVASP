# User guide

## 1. Prepare your own licensed installation source

Obtain and verify VASP through its [official portal and checksum process](https://vasp.at/info/faq/). Provide a clean source directory, `.tar.gz`/`.tgz`, another tar compression supported by Python, or ZIP. An executable-only installation is insufficient. Keep your original archive unchanged. `setup.sh` asks for a local path; there is no browser upload field or remote service.

Run from the extracted MetalVASP directory. Python 3.10+ is needed even for diagnostics. The native build requires an arm64 Mac; Linux/Intel hosts can run installer safety tests and inspect project data, but are not supported native acceleration targets.

## 2. Diagnose dependencies and identify the device

```sh
./setup.sh doctor --json diagnostics.json
```

The JSON records OS/architecture, model and chip, memory/performance cores when available, displayed GPU/Metal information, SDK path/version, tools, likely dependency roots, offline compiler availability and suggested parallel jobs. Serial numbers and UUIDs are excluded. Missing fields stay null or explicitly unprobed. `doctor` executes version queries only; it does not compile kernels or certify actual GPU correctness. Local paths can appear in this report; redact them before sharing.

If dependencies are missing and you use Homebrew, a typical command is `brew install cmake gcc open-mpi openblas fftw scalapack`. The installer displays this suggestion but does not execute it. Use matching GNU Fortran/Open MPI and LP64 libraries. For custom installations, export `OPENBLAS_ROOT`, `FFTW_ROOT`, `SCALAPACK_ROOT`, and optionally `CLANG_RUNTIME_FILE`. All roots need the expected include/lib files. Newer dependencies may require newer macOS.

## 3. Inspect the version and plan

```sh
./setup.sh inspect --source '/path with spaces/vasp.6.4.1.tgz'
./setup.sh plan --source /path/to/source --profile auto --json plan.json
```

Version detection uses source metadata, including major/minor/patch declarations; an archive filename alone is insufficient. Sixteen source/build files are checked against the adapter's base fingerprints and reconstructed output hashes. A mismatch stops native adaptation. `inspect` and `plan` do not write into the input source or compile. Temporary archive extraction is removed after inspection; a directory input is read directly.

`--allow-unvalidated-version` only allows other VASP 6.x metadata when every affected file has the same known fingerprint. It does not relax layout or output hash gates and does not confer tested support. A changed 6.5/6.6 or VASP 5.x layout needs a separately implemented adapter. Use `backend` for project-only builds without VASP.

## 4. Compile in a new private directory

```sh
./setup.sh build --source /path/to/vasp.6.4.1.tgz --work "$HOME/metalvasp-build" --profile auto
# Prepare hooks/dependency manifests without building VASP:
./setup.sh build --source /path/to/vasp.6.4.1.tgz --work "$HOME/metalvasp-configured" --configure-only
# Conservative source profile:
./setup.sh build --source /path/to/vasp.6.4.1.tgz --work "$HOME/metalvasp-portable" --profile portable --jobs 4
```

`--configure-only` still checks dependencies and may compile tiny CPU integer-ABI probes; it is different from read-only `plan`. Build jobs default to the minimum of detected performance cores, memory/2 GiB and 16, with a floor of one; `--jobs` overrides this heuristic. It is a compilation heuristic, not a scientific-performance tuning result. CPU target is `-march=native` for local auto/advanced and `-march=armv8-a` for portable.

The input is copied/extracted into `licensed-input/`; links, special files, duplicate members and traversal paths are rejected. All fingerprints and output reconstructions are checked before writes. Existing copied build/bin directories are moved aside; the user input remains untouched. Project backend tests run through CMake/CTest with strict GPU requirements, then std/gam/ncl build in sequence using `DEPS=1`. `--targets std` limits variants. Failures retain the private copy and `build-status.json` for diagnosis; choosing an existing work directory is an error.

`--deployment-target 13.0` sets source targets, not the minimum OS of existing dependency binaries. After native compilation, the installer audits non-system Mach-O dependencies. A failed/unresolved closure gate is recorded explicitly and blocks any compatibility claim; the user may still have a usable local executable on their newer OS. Build against libraries actually supporting the intended target to distribute any independently licensed compatible artifact.

## 5. Runtime or offline shader compilation

Default builds compile Metal source at runtime, trying only languages admitted by SDK/OS/device guards. A missing offline Metal Toolchain is acceptable for that path. `--offline` requests a matching source-provenanced `.metallib`; the script uses conservative MSL 2.3 for portable and 3.0 for other profiles. Runtime guards can still choose newer source paths. Advanced offline MSL4 generation is an explicit developer workflow with `Scripts/build-metallib.py`; do not infer device support from a chip name.

Offline manifests contain shader part hashes, compile macros, language, deployment target and compiler commands. Loader checks are mandatory. A metallib from another build/device target is not automatically compatible. Do not distribute locally linked VASP binaries or dependency libraries as part of this MIT project.

## 6. Run and validate locally

```sh
cd /path/to/your/calculation
VASP_METAL_STATS="$PWD/metal-stats.json" "$HOME/metalvasp-build/run-metal.sh"
# Matching CPU reference, with a separate fresh calculation directory:
VASP_METAL=0 "$HOME/metalvasp-build/BUILD_COPY/bin/vasp_mps"
```

Replace `BUILD_COPY` with the `licensed_work_copy` location in `build-status.json`. The launcher enables `VASP_METAL=1`, wave owner, Davidson owner and `VASP_METAL_NONLOCAL=1`. Native eligibility checks still retain CPU fallback for unsupported layouts. Use the explicit gam/ncl binary paths recorded under `binaries` when running those variants; the generated launcher selects std when present.

Make clean CPU/GPU calculation directories with identical inputs, licensed POTCAR, thread counts and convergence settings. Compare energies, force vectors, stress, complete SCF sequence/iterations and completion before comparing time. `Scripts/compare-vasp.py --help` describes the comparator. Warm up once, repeat whole programs at least three times, prevent sleep, and record background contention. Never mix CPU/GPU artifacts from different inputs or multiply speedups across separate campaigns.

Review actual GPU/fallback counters and API/command/wait/conversion timers. Command time is not occupancy. Activity Monitor's aggregate display does not locate individual kernels. Memory planning, Metal allocation observations and RSS have distinct scopes. The [technical guide](TECHNICAL.md) explains these distinctions.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Cannot identify version | Use a complete clean source archive; do not rename an executable to look like source |
| Fingerprint/layout mismatch | Preserve input; inspect adapter report and port locally with fresh gates, not fuzzy patching |
| Missing gfortran/mpif90 | Install/configure matching GNU Fortran and MPI; inspect wrapper compiler version |
| ILP64/default integer-8 rejected | Use LP64; ILP64 is not implemented |
| Work/SDK/dependency path rejected | Choose a plain path; source archive paths may still contain spaces |
| GPU unavailable or strict test fails | Inspect actual device/probe/test logs; CPU fallback is not a strict-GPU pass |
| Offline compiler absent | Omit `--offline`; runtime-source compilation remains available |
| Older macOS rejected by dependency audit | Rebuild dependencies for that OS; source guards alone cannot fix binary minimum versions |
| Low aggregate GPU use | Check supported route counters and CPU stages; unsupported MPI/spin/Gamma operators can fall back |
| Allocation/budget failure | Keep failure evidence; use documented workspace settings and smaller windows; do not lower accuracy |

Diagnostics are not license, numerical, safety or performance certification. See [legal scope](LEGAL.md) before publishing logs or artifacts.
