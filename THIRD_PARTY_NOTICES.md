# 来源与许可

本项目基于 [Chengqian-Zhang/deepmd-kit, 260720_classification](https://github.com/Chengqian-Zhang/deepmd-kit/tree/260720_classification)，固定 commit 为 `9b779d9162c3dc5b1612df31a58f4a547f6fde9b`。

`vendor/deepmd-kit-classification/` 是实际历史训练上传的 869 个运行源码文件，保留原始版权声明和 LGPL-3.0-or-later 许可；该子集依赖既有编译库，不是完整的可独立构建发行版。`upstream/patches/` 和 `upstream/overlay/` 保存本项目相对固定上游的 15 个文件改动。完整构建请使用固定上游加补丁。

本仓库的新增工具代码同样采用根目录的 LGPL-3.0-or-later 许可。实验模型、原始配置和参考预测是本项目实验产物，来源和校验值见 `metadata/`；软件许可不改变第三方背景数据及原始预训练权重各自的权利条款。仓库不包含训练数据、背景数据或原始 DPA4 预训练权重。

构建完整源码时请同时保留上游的依赖许可文件。运行镜像及其编译库来自原实验环境，未被本项目重新分发。
