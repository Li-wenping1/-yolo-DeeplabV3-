# 两阶段目标检测与语义分割评估

本项目用于对混凝土表面病害进行检测与像素级分割评估。推理脚本
[`twostep_new9_27.py`](./twostep_new9_27.py) 将本地 YOLO 模型和 DeepLabV3+ 模型串联起来，
支持两阶段推理和单阶段对照实验，并自动输出预测结果、可视化图片及评估指标。

## 方法流程

### 两阶段模式（默认）

```text
输入图像
   |
   v
YOLO 检测病害目标
   |
   v
检测框向外扩展 20%，裁剪 ROI
   |
   v
DeepLabV3+ 对 ROI 进行语义分割
   |
   v
将 ROI 分割结果映射回原图并计算指标
```

### 单阶段模式

关闭 YOLO ROI 后，整张原图直接输入 DeepLabV3+，可用于与两阶段方法进行消融对比。

## 主要功能

- YOLO 置信度过滤和 NMS。
- YOLO 检测框扩边后进行 ROI 裁剪。
- DeepLabV3+（MobileNet backbone，`downsample_factor=8`）语义分割。
- 自动从 `.pth` 权重中推断类别数量，并兼容常见 checkpoint 格式。
- 自动选择 CUDA 或 CPU。
- 计算 mIoU、mPA、Precision、Recall、F1 和 Accuracy。
- 保存纯预测掩码、彩色掩码、预测/真实标注对比图和 YOLO 阶段可视化图。
- 输出全局混淆矩阵及逐图评估结果 CSV。

## 类别与标注编码

当前脚本按四分类任务处理：

| 像素值 | 类别 | 可视化颜色 |
|---:|---|---|
| 0 | background | 黑色 |
| 1 | spall（剥落） | 黄色 |
| 2 | rebar（钢筋） | 红色 |
| 3 | crack（裂缝） | 白色 |

真实标注必须是灰度 PNG，且与原图文件同名。例如：

```text
images/
├── 0001.jpg
└── 0002.png

labels/
├── 0001.png
└── 0002.png
```

脚本支持 `.jpg`、`.jpeg` 和 `.png` 输入图像，标注统一查找为同名 `.png` 文件。

## 项目结构

```text
.
├── twostep_new9_27.py              # 两阶段/单阶段评估主脚本
├── twostep_new.py                  # 其他实验脚本
├── twostep_evaluate.py             # 其他评估脚本
├── yolov12/                        # 本地 YOLO 代码库
│   └── runs/.../weights/best.pt    # YOLO 权重示例位置
├── Deeplabv3/
│   └── deeplabv3-plus-pytorch-main/ # DeepLab 网络定义
└── logs/
    └── *.pth                       # DeepLab 权重
```

YOLO 代码库和 DeepLab 代码库需要保留在项目目录中。脚本会优先加载项目内的本地模块，
以确保自定义的 YOLO 网络层能够被正确导入。

## 环境安装

建议使用 Python 3.9 或更高版本，并根据 CUDA 版本安装对应的 PyTorch。

```bash
pip install torch torchvision
pip install opencv-python numpy tqdm pandas scikit-learn
```

如果使用 GPU，请从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装匹配 CUDA
版本的 PyTorch。YOLO 本地代码库还需要其自身声明的依赖。

## 使用方法

### 1. 准备模型

准备以下两个权重文件：

- YOLO 检测权重：`.pt`
- DeepLabV3+ 分割权重：`.pth`

权重文件通常不建议直接提交到 GitHub，可使用 Git LFS、Release 附件或外部模型仓库存储。

### 2. 修改路径

打开 [`twostep_new9_27.py`](./twostep_new9_27.py)，修改配置区中的路径：

```python
YOLO_MODEL_PATH = r"D:\path\to\yolo\best.pt"
DEEPLAB_MODEL_PATH = r"D:\path\to\deeplab\model.pth"
TEST_IMAGES_DIR = r"D:\path\to\images"
TEST_LABELS_DIR = r"D:\path\to\labels"
OUTPUT_DIR = r"D:\path\to\output"
```

### 3. 选择推理模式

两阶段模式（默认）：

```python
USE_YOLO_ROI = True
```

单阶段对照模式：

```python
USE_YOLO_ROI = False
```

常用参数包括：

```python
CONF_THRESH = 0.15          # YOLO 置信度阈值
NMS_IOU = 0.6               # NMS IoU 阈值
DEEPLAB_INPUT_SIZE = (640, 640)
SAVE_VISUALIZATION = True
SAVE_RAW_PRED_MASK = True
SAVE_YOLO_STAGE1 = True
```

### 4. 执行评估

在项目根目录运行：

```bash
python twostep_new9_27.py
```

脚本会自动创建输出目录并逐张处理测试图像。没有对应标注、无法读取或尺寸不一致的样本
会被跳过并打印提示。

## 输出文件

假设 `OUTPUT_DIR` 为 `output`，运行后会生成：

```text
output/
├── two_step_metrics.csv          # 每张图像的评估指标
├── confusion_matrix_global.csv   # 测试集全局混淆矩阵
├── pred_masks/
│   ├── *_pred_mask.png           # 类别 ID 灰度掩码
│   └── *_color_pred_mask.png     # 彩色预测掩码
├── visualizations/
│   └── *_compare.png             # 左侧预测叠加，右侧 GT 叠加
└── yolo_stage1/
    └── *_yolo.jpg                # YOLO 原始框与扩展 ROI 框
```

其中 YOLO 可视化图中：

- 蓝色框：YOLO 原始检测框，带类别和置信度；
- 红色框：向外扩展后的 ROI 框；
- ROI 扩展比例目前在脚本中设置为 `0.2`，可用于消融实验。

## 指标说明

脚本同时保存逐图指标和测试集整体指标。终端打印的整体指标基于整个测试集累加的全局
混淆矩阵计算，并与 DeepLabV3+ 项目的评估口径保持一致。默认包含 background 类，
因此 mIoU、mPA 等指标与只统计前景类别的结果可能不同。

## 注意事项

1. YOLO 权重必须与项目内的本地 YOLO 代码兼容。
2. DeepLab 权重的网络配置必须与训练时一致，目前固定为 MobileNet 和下采样倍率 8。
3. 标注图必须使用类别 ID 像素值，而不是 RGB 调色板或类别名称。
4. 输入图像和标注图的宽高必须一致。
5. 如果 YOLO 没有检测到目标，两阶段模式会将该图像预测为全背景。
6. 请不要将数据集、模型权重和本地绝对路径提交到公开仓库。

## 许可证与致谢

本项目整合了本地 YOLO 实现和 DeepLabV3+ 代码。发布到 GitHub 前，请分别确认所使用
代码库和预训练权重的许可证要求，并在仓库中保留相应的版权及许可证声明。
