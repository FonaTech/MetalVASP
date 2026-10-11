# 用户指南

## 1. 准备自己的合法授权安装源码

通过 [VASP 官方下载及校验流程](https://vasp.at/info/faq/) 获取并验证安装包。可以提供干净的源码目录、`.tar.gz`/`.tgz`、Python 支持的其他 tar 压缩格式或 ZIP。只有可执行程序的安装目录不足以构建。保留原始归档不变。`setup.sh` 请求本机路径，不提供浏览器上传或远程服务。

在解压后的 MetalVASP 目录运行。诊断也需要 Python 3.10+。原生构建要求 arm64 Mac；Linux/Intel 主机可以运行安装器安全测试和检查项目数据，但不是支持的原生加速目标。

## 2. 诊断环境并确定设备

```sh
./setup.sh doctor --json diagnostics.json
```

JSON 记录系统/架构、型号与芯片、可获取的内存/性能核心、显示的 GPU/Metal 信息、SDK 路径/版本、工具、发现的依赖路径、离线编译器状态及建议并发数。序列号和 UUID 不记录。缺失信息保留 null 或明确未探测。`doctor` 只查询版本，不编译内核，也不认证 GPU 正确性。报告可能包含本地路径，公开前请脱敏。

如果使用 Homebrew 且缺少依赖，典型命令为 `brew install cmake gcc open-mpi openblas fftw scalapack`。安装器只显示建议，不自动执行。GNU Fortran/Open MPI 必须匹配，库必须采用 LP64。自定义安装可设置 `OPENBLAS_ROOT`、`FFTW_ROOT`、`SCALAPACK_ROOT` 和可选的 `CLANG_RUNTIME_FILE`。依赖路径需要包含预期的 include/lib 文件。新依赖可能要求更高版本 macOS。

## 3. 检查版本并规划

```sh
./setup.sh inspect --source '/path with spaces/vasp.6.4.1.tgz'
./setup.sh plan --source /path/to/source --profile auto --json plan.json
```

版本识别读取源码元数据，包括 major/minor/patch 声明，不仅依赖归档文件名。适配器检查 16 个源码/构建文件的基础指纹和重建输出 SHA。任何不匹配都停止原生适配。`inspect` 和 `plan` 不修改输入源码，不编译。检查归档时使用的临时副本随后删除；目录输入直接只读检查。

`--allow-unvalidated-version` 只允许其他 VASP 6.x 版本元数据，同时要求全部受影响文件仍具相同已知指纹。此选项不放宽布局或输出校验，不表示已测试支持。发生变化的 6.5/6.6 或 VASP 5.x 需要独立实现适配器。不使用 VASP 时可通过 `backend` 构建项目后端。

## 4. 在新的私有目录中编译

```sh
./setup.sh build --source /path/to/vasp.6.4.1.tgz --work "$HOME/metalvasp-build" --profile auto
# 配置 hook 与依赖清单，不构建 VASP：
./setup.sh build --source /path/to/vasp.6.4.1.tgz --work "$HOME/metalvasp-configured" --configure-only
# 保守源码配置：
./setup.sh build --source /path/to/vasp.6.4.1.tgz --work "$HOME/metalvasp-portable" --profile portable --jobs 4
```

`--configure-only` 仍检查依赖，可能编译很小的 CPU 整数 ABI 探针，与只读 `plan` 不同。默认编译并发取性能核心数、内存/2 GiB、16 的最小值，至少为一；`--jobs` 可以覆盖。这是编译策略，不是科学计算性能调优结论。auto/advanced 使用本机 `-march=native`，portable 使用 `-march=armv8-a`。

输入复制/解压到 `licensed-input/`；拒绝链接、特殊文件、重复条目和路径逃逸。写入前先检查全部指纹及重建输出。副本中的旧 build/bin 移到其他位置；用户输入保持不变。CMake/CTest 运行严格 GPU 后端测试，然后用 `DEPS=1` 依次构建 std/gam/ncl。`--targets std` 可限制变体。失败保留私有副本和 `build-status.json`；已有工作目录会导致错误。

`--deployment-target 13.0` 设置源码目标，不能降低已有依赖二进制的最低系统要求。原生编译后安装器审核非系统 Mach-O 依赖。闭包失败或无法解析时会明确记录，不能据此声称目标系统兼容；较新的本机系统仍可能运行该程序。如需分发任何另行许可的兼容产物，需要为目标系统真实重建依赖。

## 5. 运行时或离线 shader 编译

默认在运行时编译 Metal 源码，只尝试 SDK/OS/设备 guard 允许的语言。此路线不要求离线 Metal Toolchain。`--offline` 请求带匹配源码溯源的 `.metallib`：portable 使用保守 MSL 2.3，其他 profile 使用 3.0。运行时 guard 仍可选择更新的源码路径。离线 MSL4 属于显式开发者流程，可使用 `Scripts/build-metallib.py`，不能仅从芯片名称推定支持。

离线清单记录各 shader 的 SHA、编译宏、语言、部署目标和编译器命令。loader 检查不能省略。其他构建或设备目标的 metallib 不自动兼容。不要把本地链接的 VASP 或依赖库作为 MIT 项目的一部分分发。

## 6. 运行并在本机验证

```sh
cd /path/to/your/calculation
VASP_METAL_STATS="$PWD/metal-stats.json" "$HOME/metalvasp-build/run-metal.sh"
# 在另一个干净的计算目录中运行匹配 CPU 参考：
VASP_METAL=0 "$HOME/metalvasp-build/BUILD_COPY/bin/vasp_mps"
```

将 `BUILD_COPY` 替换为 `build-status.json` 中的 `licensed_work_copy`。启动器启用 `VASP_METAL=1`、wave owner、Davidson owner 和 `VASP_METAL_NONLOCAL=1`。原生布局检查继续保留 CPU 回退。运行 gam/ncl 时使用清单 `binaries` 中相应路径；有 std 时生成的启动器默认选择 std。

为 CPU/GPU 建立独立干净目录，输入、合法授权的 POTCAR、线程数和收敛设置完全相同。先比较能量、力向量、应力、完整 SCF 序列/次数及正常完成，再比较时间。`Scripts/compare-vasp.py --help` 提供比较器参数。一次预热后至少三次正式整程序计算，防止休眠并记录其他负载。不要混用不同输入产物，不要跨实验相乘加速比。

查看实际 GPU/回退计数以及 API/command/wait/conversion 计时。command 时间不等于 occupancy。活动监视器的总体显示不能定位单个内核。内存计划、Metal 分配观察和 RSS 口径不同，详见[技术实现](TECHNICAL.zh-CN.md)。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 无法识别版本 | 使用完整干净源码，不要把二进制重命名成源码 |
| 指纹/布局不匹配 | 保留输入，检查报告并独立适配及重新验证，不做模糊替换 |
| 缺 gfortran/mpif90 | 配置匹配的 GNU Fortran/MPI，核对 wrapper 编译器版本 |
| ILP64/default integer-8 被拒绝 | 使用 LP64；ILP64 尚未实现 |
| 工作/SDK/依赖路径被拒绝 | 选择普通路径；源码归档路径仍可含空格 |
| GPU 不可见或严格测试失败 | 检查真实 probe/日志；CPU 回退不算严格 GPU 通过 |
| 离线编译器缺失 | 不用 `--offline`；保留运行时源码编译 |
| 旧 macOS 依赖审核失败 | 为旧系统重建依赖；源码 guard 不能修复二进制最低版本 |
| 总体 GPU 使用低 | 检查实际路线计数及 CPU 阶段；MPI/spin/Gamma 未支持算子可回退 |
| 分配/预算失败 | 保留证据，按文档调工作区/窗口，不降低精度 |

诊断不构成许可、数值、安全或性能认证。公开日志或产物前请读[法律声明](LEGAL.zh-CN.md)。
