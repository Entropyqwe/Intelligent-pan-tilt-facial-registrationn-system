# 手动上传GitHub与文件校验

## 应上传什么

E盘外层目录分为工作副本、离线模型、干净上传副本与本机检查记录。最方便上传的是 `browser_upload_ready/repository/` 内部内容。它由 `tools/prepare_browser_upload.py` 生成，排除了权重、姓名特征库、日志、缓存和编译中间文件。

不要将整个外层目录上传。`offline_models_DO_NOT_UPLOAD`用于本机离线恢复，不进入仓库；`source_baseline_LOCAL_ONLY.json`等是本机来源核查记录，也不上传。

GitHub网页有单文件与单次文件数量限制；为避开不同限制，本包按**每个文件小于25MiB、每批最多90个文件**整理。SFace约36.9MiB，不放网页上传目录。官网规则可查 GitHub Docs “Adding a file to a repository”；如果网站提示有变化，以当时提示为准。

## 网页上传

1. 自己创建GitHub仓库，选择公开或私有；公开前确认自有代码许可及第三方声明。
2. 进入仓库根，选择 Add file → Upload files。
3. 若一次拖所有文件超出数量，依次打开 `browser_upload_ready/batches/batch_01`、`batch_02`、`batch_03`。
4. 每次拖入该批次**内部的文件及子目录**，保留 `app/`、`firmware/`等相对层级；不要把 `batch_01`作为仓库顶层目录。
5. 检查页面列出的路径，确认 `.gitignore`、`.github/workflows/check.yml` 等隐藏文件也上传。
6. 每一批提交后再传下一批，最后确认README位于仓库根。

`FaceLock_Gimbal_Face_GitHub.zip`是便于传输的压缩备份。把ZIP上传到GitHub不会自动展开为源代码，正式仓库请上传其展开后的文件。

## 修改代码后重新生成干净包

```bash
python tools/prepare_browser_upload.py --output ../browser_upload_ready_new
```

工具拒绝覆盖已有输出目录，避免误删旧包；使用新目录名。它会先审计、生成SHA清单，再复制干净目录与每批最多90文件的批次目录。仅修改代码而不重新生成时，旧干净目录不会自动更新。

## 如何校验

源码校验用于判断复制/下载/传输后文件是否与打包时一致，**不能证明电路接线或实车功能正确**。

```bash
python tools/verify_files.py
```

输出“不一致：0”表示清单内所有源码/配置/HEX字节一致。清单不包括自身、模型和本机运行数据。模型另外执行：

```bash
python tools/models.py --verify
```

Windows只检查HEX时：

```powershell
Get-FileHash -Algorithm SHA256 .\firmware\Objects\FaceLock_GIMBAL.hex
```

应得到 `6caf86c0711a6bbe533088cfacfa31bbe1e9537db4fcbe8a68f344b723c649a8`。

SHA不符先确认是否自己修改过该文件；不要为了“通过校验”随意覆盖自己的有效修改。重新生成发布包后会得到新的源码清单，旧固件保持原SHA。
