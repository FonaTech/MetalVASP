# 兼容性与适配约定

| 维度 | 当前版本 | 证据边界 |
| --- | --- | --- |
| VASP 6.4.1 原生适配 | 16 个精确源码/模板指纹，std/gam/ncl 构建规则 | 已审核原生构建及 27 个重复静态体系 |
| 其他 VASP 6.x | 识别版本；所有指纹匹配时，可显式尝试仅元数据不同的实验路线 | 不声称普遍原生构建或数值兼容 |
| VASP 5.x / 改变的 6.x / VASP 6.6 CMake 布局 | 检查或仅项目后端；拒绝原生适配 | 需要独立实现适配器 |
| Apple Silicon M 系列 | 按真实 device family、限制和 SDK/OS 检查，保留通用及符合条件的先进内核 | 本版数值/性能证据来自 M5；历史旧设备测试不能验证所有新二进制 |
| macOS 源码目标 ≥13 | availability/SDK guard 和 portable MSL 路线 | 已安装依赖可能要求更新系统；本次已测原生依赖要求 macOS26 |
| MSL 4.0 和 2.3 | 已审核后端离线编译/加载及严格/portable 测试 | 不证明新链接 VASP 二进制可在 macOS13 运行 |
| std 非 SOC 静态单 rank | 实际原生运行及力/应力/SCF 对照 | 新非局域路线要求单 k 点/自旋/spinor、非螺旋 |
| Gamma 和 SOC/NCL | 已审核六个独立兼容性运行；保留其他操作或 CPU 回退 | 指定 Gamma 输入为 13 次标量 GPU 归约；SOC/NCL 为 68 次复数 GPU GEMM，FFT GPU0、新 nonlocal0 |
| MPI / 多 rank / 分布式 FFT/ScaLAPACK | 保留既有 CPU/collective 边界 | 新分布式 GPU 通信及 ILP64 尚开放 |
| HSE、长 AIMD、很大体系 | 广泛工作负载验证仍开放 | 不声称普遍精度/速度/规模 |
| ANE/tensor/Metal 4 encoder | 无生产实现 | 可选实验性 probe 文件不构成加速证据 |

[VASP 官方安装文档](https://vasp.at/wiki/Installing_VASP.6.X.X) 说明目前支持的构建系统，不对本项目 hook 作兼容承诺。相似的名称或小版本不能代替源码布局检查。

新增版本适配需要在合法取得的干净源码上本地进行。说明接口、数组布局、rank/spinor/Gamma guard 和 collective 语义变化。生成本地 COPY/INSERT 指令，以复制范围保留上游文本，不分发子程序。检查输入/输出精确 SHA，独立验证精度和失败原子性，并匹配实际 shader/native 对象。更新矩阵前，完成正常退出、完整 SCF 能量/力/应力对照及整程序重复计时。全部门通过前，新版本仍属于未验证。

`auto`/`advanced` 不强行在旧 SDK 上启用 Metal4。运行时根据真实 MTLDevice 选择可用语言和内核。host DS 编译关闭 fast-math 和隐式 FP 收缩，Metal 编译采用 safe 数学和 precise FP32 函数。portable 降低语言/CPU 目标，不宣称可以移除依赖库自身的最低系统版本。
