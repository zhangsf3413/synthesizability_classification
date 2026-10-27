# classification 的实际实现与审查

## 固定审查对象

这份说明对应后来三个 8/1/1 的原生 classification + CE multitask 实验。以实际上传给 Bohrium 的运行源码为准，二分类与三分类代码归档中的 869 个文件逐个 SHA256 一致；配置和 checkpoint 分别保留原始校验值。

上游固定 commit：`9b779d9162c3dc5b1612df31a58f4a547f6fde9b`。2026-10-27 检查时，远端 `260720_classification` 分支 HEAD 也是该 commit。

## 从结构到分类结果

1. 读取每个结构的元素、坐标和晶胞。DeepMD 按元素符号把数据中的局部类型表映射到模型的 118 元素类型表。
2. DPA4/SeZM 描述符生成各原子的环境特征。它与背景分支共享同一组参数，并参加训练。
3. `ClassificationFittingNet` 使用 `[64,64]` 的隐藏层、SiLU、float32，为每个原子产生 K 个未归一化分数。
4. `intensive=True` 让框架对有效原子的分数求平均，得到每个结构的 K 个 logits。padding/mask 不作为真实原子参与归约。这些原子输出不是逐原子的实验分类标签。
5. 结构标签是一个类别整数。`ClassificationLoss` 调用 `F.cross_entropy(logits, target)`，梯度从分类头回到共享描述符。
6. 预测取 logits 最大列。要查看概率时，使用稳定 softmax；它不改变 argmax。

二分类：`K=2`，0 为不可合成（impure），1 为可合成（nearly pure + pure）。三分类：`K=3`，0 impure，1 nearly pure，2 pure。

```python
# 概念示意；实际实现由 DeepMD 处理原子 mask 和归约。
atom_logits = fitting(descriptor(structure))
logits = masked_atom_mean(atom_logits)
loss = torch.nn.functional.cross_entropy(logits, class_id)
probabilities = torch.softmax(logits, dim=-1)  # 用于报告/推理
predicted_class = logits.argmax(dim=-1)
```

## 上游已经有什么，本实验补了什么

| 环节 | 固定上游已有实现 | 本实验额外适配 |
|---|---|---|
| 基础分类头 | `ClassificationFittingNet` 与 CE | 明确 `get_intensive()=True`，补全 DPA4 原子平均路径 |
| DPA4 工厂与配置 | 原路径主要支持 energy/property | 接受 `classification`，注册 `SeZMClassificationModel`，补充配置 schema |
| 输出统计 | DPA4/property 的输出统计与 bias 调整 | 分类分支跳过输出均值/方差及类别标签 bias 拟合，原样返回 logits |
| 损失 | 原生 CE 基础逻辑 | 检查 logits/整数标签/范围/有限性；补充概率误差指标 |
| 推理 | 普通分类模型路径 | 添加 `DeepClassification`，按 task type 识别分类头，支持概率输出 |
| 测试 | 原测试入口不足以覆盖本路径 | 原生分类 `.out`、全局混淆矩阵与指标 JSON |
| 系统发现 | 目录发现顺序可能受归档展开影响 | 排序自动发现的 system，固定 `sys_probs` 的位置对应关系 |

所有差异包含 **10 个修改文件和 5 个新增文件**。完整补丁在 `upstream/patches/dpa4-classification.patch`；`upstream/lock.json` 列出全部文件；`upstream/overlay/` 便于逐文件阅读。

关键代码：

- `vendor/deepmd-kit-classification/deepmd/pt/model/task/classification.py`
- `vendor/deepmd-kit-classification/deepmd/pt/model/model/sezm_classification_model.py`
- `vendor/deepmd-kit-classification/deepmd/pt/model/atomic_model/sezm_atomic_model.py`
- `vendor/deepmd-kit-classification/deepmd/pt/loss/classification.py`
- `vendor/deepmd-kit-classification/deepmd/infer/deep_classification.py`
- `vendor/deepmd-kit-classification/deepmd/entrypoints/test.py`
- `vendor/deepmd-kit-classification/deepmd/utils/classification.py`

## 为什么关闭输出缩放

0、1、2 表示类别身份，不是有物理量纲、存在连续距离关系的回归值。不能把类别编号当作连续标量统计均值/方差后，再用于缩放 K 维分类输出。

本实现对 classification 跳过 `compute_or_load_out_stat`、`change_out_bias`，`apply_out_stat` 直接返回 logits。能量/回归分支继续使用原统计逻辑。关闭这一回归统计不等于关闭描述符的必要数据统计，也不等于模型没有可训练的网络 bias。

## 交叉熵与 MAE/RMSE 的关系

实际反向传播的分类损失只有 CE，未启用 label smoothing 或 class-weighted CE。这里的类别均衡通过 `training_data.sys_probs` 实现。

配置中保留的 MAE、MSE、RMSE 是 `softmax(logits)` 与 one-hot 真值之间的误差，位于 `torch.no_grad()` 中，仅用于报告及既定同分规则，不参加反向传播，也不是整数类别编号之间的距离。保留这些字段是为了原样复现 classification 的输入和指标。

CE 可能在少量“非常自信地错判”的样本上很大。三分类测试 CE≈11.84，同时准确率≈75.14%，并不矛盾；分类正确率和概率校准是两个方面。

## multitask 更新及抽样

每个总步随机选择一个任务，背景与可合成性各 0.5。选中的任务执行一次损失反向传播；不是每一步把两个 batch 的损失一起相加。诊断脚本验证共享描述符参数对象一致、两个分支都产生有限非零梯度、描述符参数确实变化。

自然抽样配置 `auto_prob="prob_sys_size"`。若 system i 有 n_i 个结构，则选中该 system 的概率为 n_i/Σn_j；在 batch_size=1 时近似给所有结构相同的边际抽样机会。它是有放回的随机抽样，不保证每隔一个“epoch”逐个看完所有结构。

二分类均衡抽样先给两个类别各 0.5，再在同类别的训练配方 system 间均分：`p_i = 0.5 / N_recipe(class_i)`。system 内再抽结构。因此这是类别均衡、类内配方均匀，不是类内所有结构完全等概率。`sys_probs` 顺序必须对应按目录名排序的训练 system。

## checkpoint 选择和预测口径

普通及 EMA 两套 checkpoint，共检查 40 个候选。完整 valid 的结构级 macro-F1 最高者胜出；同分按 balanced accuracy、较低概率 MAE、较早步数比较。若这些完全相同，历史脚本保留遍历中先遇到的候选，遍历顺序为同一步普通在前、EMA 在后。

配方级：先对同配方所有结构的原始 logits 取算术平均，再 argmax；不是结构多数投票，也不是平均 softmax 概率。配方指标不用于主要 checkpoint 选择。

三分类的补充二分类：先取三列 argmax，再将 1、2 合并为可合成。例如概率 `[0.4,0.3,0.3]` 仍判为不纯相，不能用后两项之和 0.6 自动改判。

## 完整源码重建

`vendor/` 是实际运行子集，不含 `pyproject.toml`、完整 C++ 构建文件或二进制库；不要对这个子目录执行 `pip install -e`。原实验借用固定镜像的已编译库，通过 `use_classification_source.py` 在当前进程切换 Python 源码，不修改系统安装。

需要完整源码时，在仓库根目录执行：

```bash
git clone --branch 260720_classification https://github.com/Chengqian-Zhang/deepmd-kit.git /path/to/full-deepmd
git -C /path/to/full-deepmd checkout 9b779d9162c3dc5b1612df31a58f4a547f6fde9b
git -C /path/to/full-deepmd apply /absolute/path/to/this-repo/upstream/patches/dpa4-classification.patch
python scripts/verify_checkout.py --checkout /path/to/full-deepmd
```

之后按该固定版本的上游构建说明编译。独立重建的环境还需实际 GPU 验证，不能视为与历史镜像逐位等价。

## 审查结论

从配置、归档源码、运行诊断和输出的对应关系看，这三个实验确实使用原生 `classification` + CE 的结构分类路径，并联合更新共享 DPA4 描述符。二/三分类的整数标签、输出列数、损失和评估映射一致。

仍需保留的实验边界：几何保存帧质量问题；不纯相独立测试配方少；三分类 nearly pure 辨别偏弱及过高置信度；只复现此 8/1/1 固定划分，不代表完成跨数据集或合成条件外推验证。
