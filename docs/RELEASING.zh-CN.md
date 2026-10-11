# GitHub 源码发布

把本发布目录内容上传到新的项目仓库，不上传私有研究工作区。确定仓库名称/URL，核对原创权属和第三方边界，保留 `0.15.0-rc2` 预发布标签。此次准备没有创建或推送 GitHub 仓库。

执行 `python3 Scripts/audit-public-package.py .` 与 `python3 -m unittest discover -s tests -v`，用独立 `SHA256SUMS` 核对归档。阅读 `release-validation.json`，区分已审核后端/原生证据与新安装器源码级检查。公开入口完整原生构建和正在运行的 20 体系实验尚未完成，因此应标为 **pre-release** 并保留这些说明。

GitHub 首页使用英文 README，链接中文 README。保留七组图及 PDF/SVG/CSV/JSON 配套数据。引用实际 tag/commit 和归档 SHA，不虚构 DOI。GitHub CI 不需要 VASP fixture、秘密信息或许可下载，只检查安装器和公开包。

不附私有修改 VASP 树、POTCAR、链接 VASP 程序、依赖库或私有日志；MIT 不授权这些分发。其他 VASP 版本在宣传支持前须完成兼容性约定。只有单独解决每项许可/依赖/运行条件后才能考虑二进制发布，本包不含二进制。

随着更多体系完成，先审核其 CPU/GPU 输入、完整 SCF、计时和资源门，再扩展公开快照。仅导出发布版 GPU 对 CPU；运行中的体系不是已完成测量。保留来源哈希、缺失值和慢结果。
