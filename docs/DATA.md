# 数据、标签与固定划分

## 仓库保存什么

仓库只保存数据格式说明、配方/结构索引、固定划分、标签映射和校验值。训练坐标、晶胞、背景数据与原始预训练权重由使用者自行提供，不附带数据压缩包，也不通过 `prepare` 打包上传。

`metadata/three/dataset_index.csv` 和 `metadata/binary/dataset_index.csv` 对应相同的 1,845 个结构；差别是三分类与二分类标签列。坐标/晶胞/原子类型的结构 SHA256 以及所属配方、frame 都保留在索引中。`development_index.csv` 只含 train/valid，`holdout_index.csv` 只含 test。

## 划分口径

沿用历史 8/1/1、seed=20261016 的划分，按配方分组防止同配方不同结构跨集合。它从此前 117/40/40 的配方划分调整而来：保留原 valid、test 中分层抽取的各 20 条，其余回到 train。精确名单以 CSV 为准，不是重新随机分一次 80/10/10。

| 集合 | 不纯相配方（结构） | 几乎纯相配方（结构） | 纯相配方（结构） | 合计配方（结构） |
|---|---:|---:|---:|---:|
| train | 22（200） | 36（347） | 99（929） | 157（1476） |
| valid | 3（30） | 4（37） | 13（125） | 20（192） |
| test | 3（21） | 4（38） | 13（118） | 20（177） |
| 全部 | 28（251） | 44（422） | 125（1172） | 197（1845） |

二分类合并 nearly pure 与 pure：train 为 22/135 条配方、200/1276 个结构；valid 为 3/17 条、30/162 个结构；test 为 3/17 条、21/156 个结构，前者均为不可合成。

每配方最初计划 10 个结构，筛选后实际数目可能不足 10。模型训练/验证/主要评估的样本是单个结构。按配方划分不意味着把 10 个结构在训练前平均为一个样本。

## 外部数据目录

```text
synthesizability_dpdata_811/
  train/
    row_003/
      type.raw
      type_map.raw
      set.000/
        coord.npy
        box.npy
        real_atom_types.npy
        synthesizability.npy
  valid/row_009/...
  test/row_006/...
```

| 文件 | 每个 system 的形状 | 含义 |
|---|---|---|
| `coord.npy` | `(N, 3*A)` | N 个结构、每结构 A 个原子的笛卡尔坐标，Å |
| `box.npy` | `(N, 9)` | 每行展开的 3×3 晶胞矩阵，Å |
| `real_atom_types.npy` | `(N, A)` | 混合类型模式中每个结构各原子的局部元素编号 |
| `type_map.raw` | 文本元素列表 | 局部编号到元素符号的映射 |
| `type.raw` | A 个整数 | DeepMD system 的原子类型入口；混合类型以 real_atom_types 为准 |
| `synthesizability.npy` | `(N, 3)` | 原始 one-hot，顺序为 impure/nearly pure/pure |

历史局部元素表：Ba、Ca、Co、Cu、Fe、K、La、Mg、Mn、Ni、O、Sm、Sr、Ti、Y。模型类型表是从 H 到 Og 的 118 元素。不能把这 15 元素的局部编号直接当作 118 元素的全局编号；原生数据读取器和新预测脚本按符号进行映射。

## CE 读取的标签

原 one-hot 保留用于追溯。Bohrium `stage_data.py` 从外部数据读取它，在作业内部创建整数 CE 标签：

```python
phase_class = onehot.argmax(axis=1).astype(np.int32).reshape(-1, 1)
synthesizable = (phase_class != 0).astype(np.int32)
```

| 实验 | 标签文件 | dtype / shape | 类别 |
|---|---|---|---|
| 三分类 | `phase_class.npy` | int32 / `(N,1)` | 0 impure；1 nearly pure；2 pure |
| 二分类 | `synthesizable.npy` | int32 / `(N,1)` | 0 不可合成；1 可合成 |

分类训练直接读取上述整数标签。原始三列 one-hot 只用于从实验数据转换标签及核查，不作为 CE 的直接 target。

训练时只在 `data/binary/` 或 `data/phase/` 下建立 train/valid。选定 checkpoint 后再在 `heldout/` 下建立 test，test 不参与训练、描述符统计拟合和 checkpoint 排名。外部根目录可以包含 test，但正式训练不对这些数组进行读取。

## 背景数据

`opencsp` 分支沿用原输入中的全部混合背景来源。原清单统计：

| 集合 | systems | frames |
|---|---:|---:|
| train | 1,290 | 3,027,987 |
| valid | 336 | 363,663 |

这是能量、力和应力训练数据，不是可合成性实验标签，也不能替换为“仅 opencsp 数据”后仍声称与历史实验完全相同。数据需沿用各来源原本的格式、单位与类型表。

## 结构质量和标签边界

实验标签来自配方层面的实验相纯度结果，同一配方的结构共享标签。它不是每个计算微观结构经独立实验验证得到的相稳定性标签；模型学习的是在当前数据和实验条件下与配方可合成性相关的结构特征。

### 几何质量边界

数据源曾发现：优化轨迹每 5 步保存一次，导出的 `optimized_POSCAR` 使用最后保存帧，可能早于优化器真正最终状态；结果 JSON 的收敛判定也不足以核实全部晶胞自由度。原 1,847 个标为收敛的输出中，1,451 个保存帧的最大原子力高于 0.01 eV/Å。剔除两个明显异常后，历史训练数据为 1,845 个结构。

因此历史训练集应称为按当时流程筛选的暂定几何数据集，不能说全部严格满足最终优化阈值。本仓库以保留历史实验身份为目的，不自动替换几何。若重新导出真正最终结构或继续优化，应创建新的数据版本、重算结构 SHA256 并从原始预训练模型重训，不与这里的结果混用。

具体历史核查记录见 `metadata/historical_dataset_audit.json`。其中的作者本机绝对路径仅作来源记录，不是运行依赖。
