# ViT (Vision Transformer) 示例项目

这是一个基于 PyTorch 的简单 Vision Transformer (ViT) 示例实现，包含训练与评估脚本，配套的小型数据加载器。项目主要用于教学与实验用途，可运行在 CPU 或 GPU（CUDA）上。

## 目录结构

# ViT (Vision Transformer) - 教学与实验示例

这是一个基于 PyTorch 的简化 Vision Transformer (ViT) 示例实现，包含训练、评估与一个 One-vs-Rest (OvR) 的训练包装器。该仓库适合用于教学、实验或小规模数据集（如 CIFAR10）的原型开发。

## 项目结构（重要文件）

- `train.py` - 训练脚本：训练 ViT 并在每个 epoch 保存 checkpoint（checkpoint 中包含 `model_cfg` 与 `num_classes`，便于后续重建模型）。
- `test.py` - 评估脚本：加载由 `train.py` 保存的 checkpoint，在测试集上计算总体与每类准确率。
- `Agent/run.py` - 运行入口：可直接运行原始训练或者通过 `--ovr` 使用 OvR 模式。
- `Agent/ovr_trainer.py` - One-vs-Rest 包装：为每个类训练一个二分类器（会把整个训练集加载到内存，适合 CIFAR10 这种小数据集）。
- `vit_pytorch/` - 模型与数据加载：
  - `vit.py` - ViT 模型实现（使用 `einops` 进行张量重排）。
  - `dataloader.py` - `get_dataloaders`：基于 `torchvision.datasets.ImageFolder` 创建 `DataLoader`，返回 `(train_loader, test_loader, num_classes)`。
- `dataset/` - 示例数据目录（仓库内不包含数据）。

## 依赖

- Python 3.8+
- torch
- torchvision
- einops
- tensorboard（可选，用于可视化训练日志）

推荐在虚拟环境中安装依赖（PowerShell 示例）：

```powershell
# 创建并激活虚拟环境
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip

# 请根据你的 CUDA 版本选择合适的 torch 安装命令，例如：
# CPU 版本示例：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# CUDA 11.8 示例：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 其它依赖
pip install einops tensorboard
```

## 数据格式

代码使用 `torchvision.datasets.ImageFolder` 加载数据，期望数据组织为每个类一个子文件夹的结构。例如（以 CIFAR10 为例）：

```
dataset/
  CIFAR10_imbalance/    # 训练集（示例：类别不均衡）
    0/
    1/
    ...
    9/
  CIFAR10_balance/      # 测试集（示例：类别均衡）
    0/
    1/
    ...
    9/
```

每个子文件夹中放该类的图片。运行脚本时直接把训练/测试目录路径通过 `--train-dir` / `--test-dir` 传入即可。

注意：`Agent/ovr_trainer.py` 会一次性把训练集全部加载到内存并生成二分类标签，适合小数据集（如 CIFAR10）。若数据集很大，请改写为流式标签包装器以节省内存。

## 常用运行示例

训练（默认行为为在可用 CUDA 上运行，否则回退到 CPU）：

```powershell
python .\train.py --train-dir dataset/CIFAR10_imbalance --epochs 10 --batch-size 128 --image-size 32 --patch-size 4 --checkpoint checkpoint.pth --log-dir runs/exp1
```

评估（`test.py` 要求提供由 `train.py` 保存且包含 `model_cfg` 的 checkpoint）：

```powershell
python .\test.py --test-dir dataset/CIFAR10_balance --checkpoint checkpoint.pth --batch-size 64
```

使用 `Agent/run.py`（包含 OvR 的便捷入口）：

```powershell
# 使用 OvR：为每个类训练一个二分类器，生成 ovr_cls_{i}.pth
python .\Agent\run.py --ovr --train-dir dataset/CIFAR10_imbalance --epochs 5 --batch-size 64

# 或者运行原始多类训练（等价于直接运行 train.py）
python .\Agent\run.py --train-dir dataset/CIFAR10_imbalance --epochs 5 --batch-size 64
```

TensorBoard 查看训练日志（如果使用了 `--log-dir`）：

```powershell
tensorboard --logdir runs --bind_all
# 在浏览器打开 http://localhost:6006
```

## Checkpoint 与模型重建

`train.py` 在保存 checkpoint 时会把训练使用的模型配置（`model_cfg`）和 `num_classes` 一并存入 `.pth` 文件。这样 `test.py` 可以直接从 checkpoint 中读取配置并精确重建模型结构。

如果你的 checkpoint 不包含 `model_cfg`，`test.py` 会报错并提示使用由 `train.py` 保存的 checkpoint 重试。

## 模型与数据加载要点

- `vit_pytorch/vit.py`：实现了简化 ViT（patch embedding、position embedding、Transformer 层、分类头），使用 `einops` 做张量重排。构造函数会检查 `image_size` 是否能被 `patch_size` 整除。
- `vit_pytorch/dataloader.py`：`get_dataloaders(train_dir, test_dir, batch_size, image_size, num_workers)` 返回 `(train_loader, test_loader, num_classes)`。

常见边界情况：
- 图像尺寸必须能被 patch 大小整除。
- OvR 模式会把整个训练集一次性加载到内存（注意内存消耗）。

## 类别不平衡与处理（重要）

在很多实际场景中，训练数据可能存在类别不平衡（某些类样本远多于其他类）。本仓库的 `train.py` 已针对不平衡问题提供了内置的处理方法，同时也给出一些常见可选方案：

- 内置方法：Class-Balanced Focal Loss
  - `train.py` 中实现了 `ClassBalancedFocalLoss`：先根据每类样本数计算 effective number 权重（基于 beta 参数），然后结合 focal loss（gamma）来对困难样本给予更高关注并对少数类给予更高权重。默认超参：`beta=0.9999`, `gamma=2.0`。
  - 优点：同时处理类别不平衡（通过 class-balanced 权重）与难易样本（通过 focal term）问题，适用于中小规模不平衡情形。
  - 在 `train.py` 中该损失会被移动到训练 device（`.to(device)`），并直接用于训练循环。

- 可选/补充方法：
  1. 重采样（过采样/欠采样）
     - 过采样少数类（如使用 `torch.utils.data.WeightedRandomSampler` 或数据增强复制）可以增加少数类的出现频率。
     - 欠采样多数类会减少多数类样本数量，但可能造成信息丢失。
  2. 类别加权交叉熵
     - 直接在交叉熵中传入 class weights（如 torch.nn.CrossEntropyLoss(weight=...)）。权重通常取反比于类频率或基于 effective number 计算得到。
  3. 数据增强
     - 对少数类做更激进的数据增强（旋转、裁剪、颜色抖动等）来增加其多样性。
  4. One-vs-Rest (OvR)
     - 仓库提供的 `Agent/ovr_trainer.py` 会为每个类训练一个二分类器，适合某些场景下提升少数类检出率，但会增加训练总量并且 OvR 的评估需要额外合并策略。

- 推荐实践与超参提示：
  - 先从内置的 ClassBalancedFocalLoss 开始（`train.py` 默认使用），观察训练曲线与 per-class accuracy；如果少数类仍然表现很差，再尝试：
    1. 使用 `WeightedRandomSampler` 做过采样；
    2. 在损失中使用 class weights（CrossEntropyLoss 或调整 `ClassBalancedFocalLoss` 的 beta）；
    3. 对少数类进行更强的数据增强；
    4. 在资源允许的情况下尝试 OvR 作为对照实验。
  - 如果使用重采样，请注意训练集的 batch 内类分布波动可能影响学习率与批归一化统计，建议配合较小学习率或使用 Group/LayerNorm 等不依赖 batch 统计的归一化层。

示例：如果想尝试使用加权交叉熵而不是内置 loss，可以在 `train.py` 中替换损失定义为：

```python
# 伪代码示例：基于 class_counts 计算权重并替换 criterion
# counts = [n0, n1, ..., nC-1]
import torch
counts = torch.tensor(class_counts, dtype=torch.float32)
weights = 1.0 / (counts + 1e-12)
weights = weights / weights.sum() * len(weights)
criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
```

总之：先用仓库默认的 ClassBalancedFocalLoss 做基线，再根据少数类表现逐步采用重采样、加权损失、数据增强或 OvR 等策略。

## 其它说明 / 建议

- 若你希望在训练过程中同时保留验证集，请自行实现验证 loader 或在训练循环中加入验证步骤（当前 `train.py` 默认不接收 `--val-dir`）。
- 若计划在大数据集上使用 OvR，请改写 `Agent/ovr_trainer.py` 中的数据包装逻辑，避免一次性将全部图像张量加载到内存。

如果你需要我把 README 翻译成英文版、添加快速开始动画（GIF）、或为常用配置添加示例脚本（比如 GPU/CPU 的安装命令），告诉我需要哪些内容，我可以继续补充。

---

更新摘要：已根据仓库当前代码（`train.py` / `test.py` / `Agent/ovr_trainer.py` / `vit_pytorch/*`）整理并更新 README，包含运行示例与注意事项。

- 若要在 CPU 上运行，传入 `--device cpu`（`test.py` 支持 `--device` 参数）。`train.py` 通过检测 CUDA 可用性自动回退到 CPU，但没有显式的 `--device` 参数。
