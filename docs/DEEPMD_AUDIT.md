# DeepMD 命令与冗余审查

审查对象：本仓库与上游 [260720_classification](https://github.com/Chengqian-Zhang/deepmd-kit/tree/260720_classification)，固定 commit `9b779d9162c3dc5b1612df31a58f4a547f6fde9b`。2026-10-27 检查远端分支 HEAD 与该 commit 一致。

结论：训练、checkpoint 加载和测试可以沿用标准 DeepMD 入口。本实验的额外工作是 DPA4 分类适配、数据标签转换、完整验证集选模和 Bohrium 调度。这几部分不应被误认为 DeepMD 的另一套训练算法。

## 逐项对照

| 项目 | 审查发现 | 当前做法 |
|---|---|---|
| 训练 | 原脚本已调用上游训练，但额外改写 `ModelWrapper` 方法来记录梯度 | 删除这些 hook，直接执行 `dp --pt train` |
| multitask | 共享描述符、任务概率、数据抽样由原生配置实现 | 保留原始 input，不增加抽样或共享参数的 Python 封装 |
| 测试 | 原包装调用内部 `deepmd.entrypoints.test.test()` | 改为 `dp --pt test`，由原生解析器处理参数 |
| checkpoint | 固定 PyTorch 推理入口支持本实验 `.pt` 文件 | 直接测试普通/EMA checkpoint，不增加必需的 freeze 步骤 |
| 无标签预测 | 没有通用的原生 `dp predict` 命令可直接替代本任务 | 使用补丁提供的 `DeepClassification` 推理 API，按元素符号映射类型 |
| Bohrium 提交 | 上游没有指定 Bohrium 或 dpdispatcher 的提交约定 | 保留独立调度脚本；资源调度与 DeepMD 模型计算分开 |

作业中调用的命令如下，三分类 valid 为例：

```bash
dp --pt train input.json --finetune pretrained/model.ckpt-200000.pt --skip-neighbor-stat
dp --pt test -m best_checkpoint.pt -s data/phase/valid -n 0 --head synthesizability -d evaluation/valid/dptest
```

`-n 0` 明确要求所有帧；`--head` 必须选可合成性分支。数据路径和输出前缀按集合变化，其余默认行为交给原生解析器。`--finetune` 用于从原始预训练模型开始，`--restart` 用于恢复已有训练，两者不能混用。

固定镜像仍需要源码启动器：payload 中的 `bin/dp` 只激活归档 Python 源码，再调用未修改的 `deepmd.main.main()`。它没有自建命令解析器或训练循环。如果完整构建安装了同一份补丁源码，可直接使用安装后的 `dp`。

## 删除了哪些冗余

- 删除训练中的梯度 hook、参数快照、任务更新计数及对应的训练包装脚本。它们用于历史审计，训练本身不依赖这些诊断。
- 删除每次作业自动执行的额外模型单元测试，以及逐字段重复检查原始配置的脚本；schema 检查交给 DeepMD。smoke 直接测试完整 valid。
- 将训练、重测和选模的作业编排合并为一个 `run_remote.py`，消除重复的初始化与模式入口。
- 推理不再显式填写 `no_jit=True`、`nlist_backend="native"`、`atomic=False`、`auto_batch_size=False`。DPA4 推理已自动关闭不适用的 JIT 并使用内置邻居表，其余使用 API 默认值。
- 删除依赖内部 `state_dict` 命名的逐张量比对；保留模型 SHA256、输出形状和历史逐结构结果核对。
- 删除重复 GPU 字段、重复清空凭据的语句和重复日志重定向。保留阻止 dpdispatcher 内部序列化凭据的必要处理。
- 分类作业生成整数标签后，只复制坐标、晶胞和原子类型，不再把原始 one-hot 数组重复复制到分类数据目录。

公开运行代码（`scripts/` + `workflow/`，不含不可变 DeepMD 源码和测试）由 15 个文件、1,766 行，减为 13 个文件、1,339 行，减少约 24%。历史 input、模型、源码快照和参考结果的字节保持不变。

## 仍需保留的适配

上游已提供 `ClassificationFittingNet` 和交叉熵；本实验补丁继续用于 DPA4/SeZM 模型工厂与 schema、有效原子平均、跳过分类输出的回归统计、分类推理和分类 `dp test` 输出。直接换成未修改的上游不能覆盖这条完整路径。15 个补丁文件及理由见 [实现说明](IMPLEMENTATION.md)。

标签转换、数据指纹、配方分组划分、完整 valid 的结构级选模和最终 test 隔离用于保证实验口径，应当保留。Bohrium 的真实日志登记、任务组复用和外部凭据读取用于正确提交，也需要保留。附加资源列表不再被清空，以免破坏外部数据访问。

原始 input 中部分字段虽然等于该版本默认值，仍保留原样，因为它们是已完成实验的溯源依据。仓库中的通用 DeepMD 回归模块及 classification 的概率 MAE/MSE/RMSE 指标也予以保留；本仓库暴露的分类训练损失只有 CE。

## 验证范围

本地只进行文件哈希、静态检查、NumPy 指标重算和不加载模型的测试。train/test 参数经固定上游真实解析器验证，评估适配流程用历史 `.out` 和指标 JSON 回放核对；新入口未提交额外 GPU smoke，也未重新训练。三组历史成功任务及原生结果仍作为参考，重新运行前按 [运行说明](RUNNING.md) 先提交 smoke。
