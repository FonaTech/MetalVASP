# Copyright and third-party boundaries / 版权与第三方边界

Copyright © 2026 FonaTech and MetalVASP contributors applies to the original project code, local adaptation instructions, documentation, and nonconfidential derived benchmark records/figures. These components use MIT. This notice does not certify ownership of externally supplied material or replace any dependency license.

本项目原创代码、本地适配指令、文档及非保密基准派生记录/图片的版权归 FonaTech 与贡献者，采用 MIT。本声明不替外部材料确认权属，也不取代依赖许可。

| Component / 组件 | Rights holder / 权利方 | Terms / 条款 | Included here / 本包包含 |
| --- | --- | --- | --- |
| VASP source, executables and PAW/pseudopotential database | VASP Software GmbH and relevant licensors | Proprietary licensed access; user's agreement controls | **None / 无** |
| Apple Metal, Foundation, SDK, Xcode/Metal tools | Apple Inc. and respective contributors | Apple developer/SDK agreements | Calls and guards only; no SDK/framework binaries |
| GNU GCC/GFortran and runtimes | Free Software Foundation and contributors | GPL; runtime components may have GCC Runtime Library Exception; check exact artifact | None; invoked locally |
| LLVM/Clang/libc++ | LLVM contributors | Apache-2.0 WITH LLVM-exception; exact installed tool distribution controls | None; invoked/linked locally |
| Open MPI | Open MPI contributors and organizations listed upstream | BSD-3-Clause-Open-MPI and component notices | None; linked locally |
| OpenBLAS | OpenBLAS/Xianyi Zhang and contributors; inherited BLAS/LAPACK contributors | BSD-3-Clause and component notices | None; linked locally |
| ScaLAPACK | University of Tennessee, UC Berkeley, University of Colorado Denver and contributors | Upstream BSD-style license | None; linked locally |
| FFTW | Matteo Frigo, MIT and other listed contributors | GPL-2.0-or-later or separately purchased non-free terms | None; linked locally |
| CMake | Kitware and contributors | BSD-3-Clause | None; build tool |
| Python | Python Software Foundation and contributors | PSF and component licenses | None; interpreter |
| NumPy | NumPy developers and contributors | BSD-3-Clause and bundled-component notices | None; optional plotting dependency |
| Matplotlib | Matplotlib development team and contributors | Matplotlib/PSF-based license and bundled-component notices | None; optional plotting dependency |

No dependency source or binary is vendored. Linking a local executable does not make all of its components MIT. In particular, FFTW's default GPL terms and VASP's proprietary terms need their own analysis or applicable alternative FFTW license; no compatible distribution permission is inferred. This source-only package does not resolve or grant rights for distributing a combined executable.

没有依赖源码或二进制被打包。本地链接的程序不因此整体变成 MIT。尤其 FFTW 默认 GPL 条款与 VASP 专有许可需要独立分析，或使用适用的 FFTW 其他许可；本项目不据此推定组合产物可以再分发。此源码包不解决、也不授予组合程序的分发权。

Primary sources / 官方来源 (checked 2026-10-11):

- [VASP FAQ and confidential-material guidance](https://vasp.at/info/faq/)
- [VASP installation](https://vasp.at/wiki/Installing_VASP.6.X.X)
- [Apple developer agreements](https://developer.apple.com/support/terms/)
- [GCC Runtime Library Exception](https://www.gnu.org/licenses/gcc-exception-3.1.html)
- [LLVM license](https://llvm.org/docs/DeveloperPolicy.html#copyright-license-and-patents)
- [Open MPI license](https://www.open-mpi.org/community/license.php)
- [OpenBLAS license](https://github.com/OpenMathLib/OpenBLAS/blob/develop/LICENSE)
- [ScaLAPACK license](https://netlib.org/scalapack/LICENSE)
- [FFTW licensing](https://www.fftw.org/doc/License-and-Copyright.html)
- [CMake license](https://cmake.org/licensing/)
- [Python license](https://docs.python.org/3/license.html)
- [NumPy license](https://numpy.org/doc/stable/license.html)
- [Matplotlib license](https://matplotlib.org/stable/project/license.html)

Apple, Metal and VASP names identify compatible products and remain the property of their respective holders. No sponsorship, partnership or endorsement is claimed. / Apple、Metal、VASP 名称仅用于标识适用产品，归各自权利方；不表示赞助、合作或背书。
