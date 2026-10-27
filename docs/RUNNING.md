# 环境与运行入口

## 两种环境

本地：Python≥3.11、NumPy，用于源码/文件校验、指标重算、准备运行脚本；ASE 仅用于 POSCAR 格式转换；dpdispatcher 仅用于提交/下载 Bohrium 作业。

模型运行：原实验镜像 `registry.dp.tech/dptech/dp/native/prod-435364/dpa4-rl-matdesign:0.0.7`，V100，12 CPU / 1 GPU。历史环境记录为 Python 3.13、DeepMD-kit 3.2.0、PyTorch 2.11.0+cu126、CUDA 12.6。原作业没有保存完整 pip freeze，不能把这个版本摘要称为完整依赖锁；新入口会保存 `environment.json` 和 `pip-freeze.txt`。

镜像需要使用者具备访问权限。如果无法访问，应从固定完整上游源码加补丁构建，并单独核对环境与输出。只安装官方稳定版 DeepMD，或只克隆上游未打补丁，都不能直接代替本实验环境。

## 为什么有 vendor 源码和 source activation

`vendor/deepmd-kit-classification/` 是实际训练上传的运行源码快照，所有 869 个原文件都固定 SHA256。它借用镜像既有的编译库。`use_classification_source.activate()` 在导入 DeepMD 实现模块前，切换当前进程的 Python 模块搜索路径，保留已安装的二进制资源，不覆盖系统环境。

`workflow/` 从三分类训练脚本整理为二/三分类共用的流程，保留共享参数、两分支梯度与更新次数的诊断。`scripts/` 统一执行 classification schema 校验、原生 dp test 和与历史相同的 checkpoint 排名规则。**使用 `scripts/reproduce.py prepare` 生成 `run_job.sh`**，不需要旧的数据压缩归档。

## 四种模式

| `--mode` | 加载哪个模型 | 使用哪些数据 | 行为 |
|---|---|---|---|
| `smoke` | 仓库固定的最佳模型 | 完整 valid；准备时校验 train/valid 格式 | GPU 分类测试 + valid 与历史输出对比，不训练 |
| `evaluate` | 仓库固定的最佳模型 | train/valid/test | 对三集合做原生 dp test 与历史输出对比 |
| `train` | 外部原始 DPA4 预训练模型 | 背景 train/valid + 性质 train/valid，选定后 test | 100k 步正式训练、40 个候选验证、固定模型后测试 |
| `predict` | 仓库固定的最佳模型 | 外部无标签结构 | 输出逐结构及配方预测，不读取实验标签 |

`--dataset-root`、`--background-root`、`--pretrained` 是 **Bohrium 作业内**的路径；prepare 不检查本机这些路径是否存在，也不把对应数据加入 `payload.tar.gz`。请通过自己的数据盘、数据集挂载或既有远程目录让作业访问它们。本项目不会自动购买、创建或上传这些数据。

外部性质数据应是固定原始 one-hot 格式。若使用其他新数据，需明确修改索引、划分和标签，不应关闭校验来伪装成原实验。

## 训练超参数

| 项目 | 配置 |
|---|---|
| 总步数 / 随机种子 | 100,000 / training seed 1000 |
| 分类头 | [64,64]，SiLU，float32，seed 42 |
| 任务抽样 | opencsp 0.5，synthesizability 0.5 |
| 分类 batch size | 1 |
| 学习率 | WSD，1e-4→1e-6，warmup_ratio=0.003，decay_phase_ratio=0.65，cosine decay |
| 优化器 | HybridMuon，weight_decay=0.001 |
| 梯度裁剪 | gradient_max_norm=5.0 |
| checkpoint | 每 5,000 步普通+EMA，EMA decay=0.999，各保留21份 |
| 训练曲线 | lcurve.out，每 1,000 步；曲线是抽样 batch 指标 |
| 选模 | 完整 valid 的结构 macro-F1，其次 balanced accuracy、较低概率 MAE、较早步数 |

实际共享与梯度记录写入 `multitask_diagnostics.json`。自然二分类和三分类原运行的分类更新次数是 50,054，背景更新 49,946；100k 不是分类头单独做满100k次更新。

## 资源与日志

`scripts/bohrium.py` 从调用者指定的 `--machine` 文件读取账号和项目 ID，不在仓库中提供作者 machine.json，不把密码/令牌保存进状态文件。它沿用文件中的账号与项目，不硬编码作者项目。作者在自己的项目执行时，应继续使用项目 AGENTS 指定的 machine.json。

真实日志文件固定为 `bohrium_reproduction.log`，同时设为 Task 的 `outlog` 与 `backward_files`；错误文件为 `bohrium_reproduction.err`。训练另返回 `lcurve.out`。同一 `--batch` 目录的多个任务启用 grouped 并复用 `job_group_id`；提交尝试有防重复记录。

运行目录下 `submission_state.json`、`submission_status.json` 仅用于本次任务恢复，整个 `runs/` 默认不进 Git。不要手工把 machine.json 加入 Git。

手动在 Bohrium 网页提交时，上传 `payload.tar.gz` 和 `run_job.sh`，选上述镜像、V100和自己的项目，挂载外部数据。命令：

```bash
bash run_job.sh > bohrium_reproduction.log 2>&1
```

将 `bohrium_reproduction.log` 同时列入日志文件与输出文件。输出文件还包括 `bohrium_reproduction.err`、`reproduction_result.tar.gz`；train 模式加 `lcurve.out`。

## 返回文件

下载并解压后，`repro_outputs/` 包含实际运行日志、环境、测试日志、输入、选定模型及评估目录。正式训练还返回普通/EMA checkpoint、完整验证集排名、梯度诊断与曲线。

原生分类 `.classification.out` 每个 system 一段，包含：

```text
二分类：frame true_class predicted_class logit_0 logit_1 prob_0 prob_1
三分类：frame true_class predicted_class logit_0 logit_1 logit_2 prob_0 prob_1 prob_2
```

混淆矩阵行是真实类别，列是预测类别。误判率指真实 impure 中被判为可合成的比例，即 FP/(TN+FP)。

无标签的 `predict` 模式输出 CSV 与 `logits.npy`，没有原生带真值 dp test，也不会伪造标签或计算准确率。

## 检查范围

本仓库的 CPU 自动检查可验证源码与配置身份、固定抽样顺序、完整历史预测的指标重算、类型映射、按配方汇总规则，以及训练 payload 不带数据和已训练分类模型。它不替代 GPU smoke。

整理时已通过本地检查和上游补丁还原核对；尚未用这些新的可移植入口提交额外 GPU 作业。校验记录见 `metadata/repository_verification.json`。

复现时先跑 `smoke`，确认 GPU 测试通过、valid 逐结构类别一致，再做完整评估或训练。历史已完成的 Bohrium 结果保留为参考。新训练的数值不保证逐位一致，尤其当 GPU、库版本、背景数据或几何版本发生变化时。
