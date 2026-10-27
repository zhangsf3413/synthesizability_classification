# 钙钛矿可合成性 classification 模型

基于 DeepMD-kit 的 DPA4 描述符，联合训练能量/力/应力分支与可合成性分类分支。这里固定真实实验的源码、配置、最佳模型、数据划分清单和 `dp test` 结果，支持其他人在准备好外部数据后复现训练与预测。

**训练数据、背景数据、原始预训练权重和账号凭据不包含在仓库中。** 本地入口仅做文件校验、格式转换和已有输出的指标重算；训练与模型推理入口用于 Bohrium V100。

## 先理解模型在做什么

```text
结构：元素 + 坐标 + 晶胞
             │
       共享、可训练的 DPA4 描述符
             │
       ┌─────┴──────────────────┐
背景数据分支                 可合成性分支
能量/力/应力                 classification [64,64]
原模板全部混合背景           每原子 K 列 logits
                             ↓ 对有效原子求平均
                         每结构 K 列 logits
                             ↓ 交叉熵 + 整数类别标签
                         预测时取最大列的类别
```

两个分支以 `0.5 : 0.5` 的概率交替更新，共享描述符也会被更新。10 万步是两个任务合计的优化步数。三分类训练时保留三个独立类别，最后可以再把 nearly pure 和 pure 合并为“可合成”。

交叉熵直接接收原始 logits，在内部进行 log-softmax；**损失前不再额外加 softmax**。输出概率时再用 softmax，分类结果取 argmax。

分类原理见 [实现说明](docs/IMPLEMENTATION.md)；与上游的命令差异和代码精简见 [DeepMD 对照审查](docs/DEEPMD_AUDIT.md)。

## 收录的实验

三组均为按配方分组的 8/1/1 划分，100,000 步、V100、每 5,000 步保存普通与 EMA checkpoint。最佳 checkpoint 由完整验证集的结构级 macro-F1 选择。

| 实验目录 | 任务 | 抽样 | 已选模型 | 原 Bohrium 训练任务 |
|---|---|---|---|---|
| `experiments/binary_natural` | 二分类 CE | 自然 | 95,000 步 EMA | 23560328 |
| `experiments/binary_balanced` | 二分类 CE | 类别均衡 | 50,000 步 EMA | 23560329 |
| `experiments/three_natural` | 三分类 CE | 自然 | 95,000 步 EMA | 23562056 |

| 测试集，结构级 | 原任务准确率 | 原任务 macro-F1 | 合并后二分类准确率 | 不纯相误判率 |
|---|---:|---:|---:|---:|
| 二分类自然 | 100.00% | 1.0000 | 100.00% | 0/21 = 0% |
| 二分类均衡 | 91.53% | 0.8432 | 91.53% | 0/21 = 0% |
| 三分类自然 | 75.14% | 0.7056 | 99.44% | 0/21 = 0% |

不纯相测试样本只有 **3 条配方、21 个结构**。上述 0% 是这次固定测试集的结果，不能理解为任意新配方都不会误判。三分类最弱的类别是 nearly pure，测试 F1 为 0.3175。

完整 train/valid/test、每类 F1 与配方级补充结果见 [结果说明](docs/RESULTS.md) 和 `experiments/*/reference/`。配方级采用先平均原始 logits，再取 argmax。

## 目录

```text
experiments/       三组原始 input.json、最佳模型、lcurve.out、参考 dp test 输出
vendor/            实际历史训练的 869 个运行源码文件，不含编译库
upstream/          上游 commit 锁定、完整补丁、15 个改动文件及源码校验值
metadata/          固定划分清单、索引、标签映射、模型与环境来源
scripts/           数据校验、Bohrium 任务准备/提交、训练、评估、预测入口
workflow/          标准 dp 命令的源码启动器与镜像编译库适配
tests/             不执行模型的科学结果与打包检查
docs/              实现、数据、运行方法和实验结果说明
.github/workflows/ GitHub Actions 自动复核已有结果，不训练模型
```

## 1. 不需要 GPU：先验证仓库

建议 Python 3.11 或以上。在仓库根目录运行：

```bash
python -m pip install -r requirements-local.txt
python scripts/reproduce.py verify
python -m unittest discover -s tests -v
python scripts/reproduce.py report --output runs/reports
```

最后一条从九份历史 `.classification.out` 重算结构级与配方级指标，生成 `runs/reports/summary.csv`，并逐项核对历史指标。它不加载 checkpoint，不训练或推理模型。

## 2. 在外部准备数据

复现训练需要三类外部文件，均不由本仓库打包：

1. 8/1/1 的可合成性 DeepMD/Numpy 数据，包含 `train/`、`valid/`、`test/`，每条配方一个 system。原三列 one-hot 保存在 `synthesizability.npy`。
2. 原模板**全部混合背景**的 `train/`、`valid/`。配置中的 `opencsp` 是分支名，实际背景还含 interface、miao、iter 等来源。
3. 原始 DPA4 预训练 checkpoint `model.ckpt-200000.pt`，SHA256 为 `316133698c00f3b74112616d8a6dee1e597b502746de68553b635b7861df5a74`。

可先对外部数据做只读校验：

```bash
python scripts/check_data.py --dataset-root /path/to/synthesizability_dpdata_811
```

数据格式、计数与二/三分类标签转换见 [数据说明](docs/DATA.md)。**运行模型时，这些路径必须在 Bohrium 容器内可见**，例如已挂载的数据盘路径。仅有作者本机路径不能使远程容器自动访问文件。

## 3. Bohrium 先做 smoke

下面命令中的 `/mounted/...` 是示例，换成容器实际可访问的数据路径。`--dataset-root` 只写入配置，不把训练数据放进上传文件。

```bash
python scripts/reproduce.py prepare \
  --experiment three_natural --mode smoke \
  --dataset-root /mounted/synthesizability_dpdata_811 \
  --output runs/three_smoke

python scripts/bohrium.py submit runs/three_smoke \
  --machine /private/path/machine.json --batch runs/batch_smoke
```

smoke 使用仓库中的固定最佳模型，在 GPU 上执行一次完整 valid 的标准 `dp --pt test`，并比较历史逐结构 logits 与类别。它不训练，也不读取 test。

查看、下载同一批次：

```bash
python scripts/bohrium.py status runs/three_smoke \
  --machine /private/path/machine.json --batch runs/batch_smoke
python scripts/bohrium.py fetch runs/three_smoke \
  --machine /private/path/machine.json --batch runs/batch_smoke
```

也可以在已配置好数据盘的 Bohrium V100 作业中上传生成目录内的两个文件，然后执行 `bash run_job.sh > bohrium_reproduction.log 2>&1`。镜像、日志文件配置和返回文件见 [运行说明](docs/RUNNING.md)。

## 4. 重新训练 100,000 步

```bash
python scripts/reproduce.py prepare \
  --experiment three_natural --mode train \
  --dataset-root /mounted/synthesizability_dpdata_811 \
  --background-root /mounted/background \
  --pretrained /mounted/model.ckpt-200000.pt \
  --output runs/three_train

python scripts/bohrium.py submit runs/three_train \
  --machine /private/path/machine.json --batch runs/batch_training
```

改为 `binary_natural` 或 `binary_balanced` 即可复现相应实验。多任务一起提交时，将目录一起传给 `submit`，使用同一个 `--batch`；驱动启用 `grouped=True` 并复用组 ID。

正式训练从原 DPA4 权重开始；上传内容不含已经训练好的分类模型。训练完对每个普通和 EMA checkpoint 做完整验证集评估，选定后才建立 test 标签目录并评估 test。历史测试结果只作为参照，不用于新一轮 checkpoint 选择。

作业内实际调用的是标准 DeepMD 命令：

```bash
dp --pt train input.json --finetune pretrained/model.ckpt-200000.pt --skip-neighbor-stat
dp --pt test -m best_checkpoint.pt -s data/phase/valid -n 0 --head synthesizability -d evaluation/valid/dptest
```

第二条以三分类 valid 为例；二分类使用 `data/binary/valid`。训练、损失、优化器和抽样均由 DeepMD 执行，复现脚本只准备标签、组织命令、选模和汇总结果。

## 5. 只重测已有模型

```bash
python scripts/reproduce.py prepare \
  --experiment three_natural --mode evaluate \
  --dataset-root /mounted/synthesizability_dpdata_811 \
  --output runs/three_evaluate
```

提交方式相同，返回完整 train/valid/test 的原生 `.classification.out`、指标 JSON，以及结构和配方预测 CSV。

## 6. 预测没有标签的新结构

如果已有 DeepMD/Numpy 数据，可以直接作为外部 `--dataset-root`。如果是 POSCAR，先准备 CSV：

```csv
recipe_id,path
recipe_001,/path/to/recipe_001/POSCAR_01
recipe_001,/path/to/recipe_001/POSCAR_02
recipe_002,/path/to/recipe_002/POSCAR_01
```

```bash
python scripts/poscars_to_dpdata.py --index /path/to/poscars.csv --output /path/to/new_dpdata
python scripts/reproduce.py prepare \
  --experiment three_natural --mode predict \
  --dataset-root /mounted/new_dpdata --output runs/new_prediction
```

把转换后的新结构数据自行放到 Bohrium 可访问位置，再提交。转换只处理格式；新配方的建模、MD 和几何优化需沿用原结构处理流程。预测程序按元素符号映射类型，支持不同原子数的 system。没有实验真值时只输出 logits、概率、分类和配方汇总，不生成准确率或 F1。

## 上游固定与复现边界

上游：[Chengqian-Zhang/deepmd-kit / 260720_classification](https://github.com/Chengqian-Zhang/deepmd-kit/tree/260720_classification)。固定 commit：`9b779d9162c3dc5b1612df31a58f4a547f6fde9b`。

直接 clone 该分支不足以还原本实验：还需要仓库中的 DPA4/SeZM 分类适配、统计缩放处理、分类推理和 `dp test` 补丁。完整源码重建方法见 [实现说明](docs/IMPLEMENTATION.md#完整源码重建)。

历史运行环境为 DeepMD 3.2.0、PyTorch 2.11.0+cu126、CUDA 12.6、V100；镜像固定在 `metadata/environment.json`。重新训练可能受 GPU 非确定性影响，不能承诺逐位相同的模型或相同最佳步数；已保存模型的输出可以直接核对。

历史结构数据的保存帧存在几何收敛审计问题，详见 [数据质量边界](docs/DATA.md#几何质量边界)。这里保留历史数据身份与结果，不将它们描述为全部严格收敛。

## 上传 GitHub

仓库已配置 `.gitignore` 和 GitHub Actions。训练数组、背景数据、运行目录和 `machine*.json` 不纳入版本管理。三个最佳模型每个约 2 MB，可以直接用 Git 保存。

在 GitHub 创建空仓库后，在当前仓库根目录执行：

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

许可和上游归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [LICENSE](LICENSE)。
