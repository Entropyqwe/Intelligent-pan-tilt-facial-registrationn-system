# 来源与第三方许可

本副本来源于项目已有云台工程；本次复制保留其中的第三方代码声明。没有为所有历史代码擅自添加统一MIT或其他开源许可证。公开发布前，请由项目所有者核对自有代码及历史来源的授权。

| 内容 | 来源 | 当前处理 |
|---|---|---|
| YuNet检测ONNX | OpenCV Zoo `models/face_detection_yunet` | 权重不进上传包；保留模型MIT许可证 |
| SFace识别ONNX | OpenCV Zoo `models/face_recognition_sface` | 权重不进上传包；保留Apache-2.0许可证 |
| NanoDet CPU对照ONNX | OpenCV Zoo `models/object_detection_nanodet` | 权重不进上传包；保留Apache-2.0许可证 |
| YOLOv5 NPU `.nb` | 既有板端 `/opt/yolov5/model/yolov5.nb` | 不随本仓库再分发；须核对SDK、权重来源和相应许可 |
| VIPLite运行库/内核驱动 | 实际板卡官方SDK | 不再分发；仅提供本项目桥接源码与本机编译步骤 |
| STM32标准外设库/启动文件 | 原工程 `firmware/Library`、`Start`、`System` | 保留源文件原有版权与许可声明，未重新授权 |
| espeak-ng | 本机系统软件包 | 通过独立进程调用；未附带其二进制/语音数据；遵循其自身GPL许可 |
| OpenCV、numpy、Pillow、pyserial | requirements中的上游依赖 | 通过包管理安装，依各自许可证 |

OpenCV Zoo固定提交：`48b10fa505aeb51fbd127fc60a5c660040e823b7`。各权重哈希在 `tools/models.py` 和环境文档中。许可证文件在 `app/models/`。

上游参考地址：

```text
https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7
https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html
https://github.com/espeak-ng/espeak-ng
https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository
```

本次网络访问不稳定，未重新下载核验上游网页或运行板端语音；本机模型通过原有SHA256及实际OpenCV加载核对。这里的地址是供复核的来源入口，不意味着已经完成全部第三方发布授权审查。

若后续要选择本项目自有部分的开源许可证，先明确代码所有权和可授权范围，再添加LICENSE，不应覆盖第三方文件声明。
