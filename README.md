# 图像处理作业 YOLOv11 图像分割项目

本项目基于 YOLOv11 实现图像和视频中的实例分割功能。

## 项目结构

```
.
├── predict_image.py     # 图像分割推理脚本
├── predict_video.py     # 视频分割推理脚本
├── train.py             # 模型训练脚本
├── yolo11-seg.yaml      # 模型结构配置文件
├── coco.yaml            # COCO数据集配置文件
├── yolo11x-seg.pt       # 预训练分割模型
├── 1.jpg                # 示例图片
├── 1.mp4                # 示例视频
└── README.md            # 项目说明文档
```

## 功能特性

- 基于 YOLOv11 实现实例分割
- 支持图像和视频的推理处理
- 支持训练自定义模型
- 可视化分割结果

## 环境依赖

- Python 3.x
- PyTorch
- OpenCV
- Ultralytics YOLO
- NumPy

安装依赖：
```bash
pip install torch opencv-python ultralytics numpy
```

## 使用方法

### 图像分割

```bash
python predict_image.py
```

该脚本会对 [1.jpg](file:///c%3A/Users/guojia/PycharmProjects/tuxiang/1.jpg) 进行实例分割，并显示两个窗口：
1. 带有分割掩码和标签的原始图像
2. 仅包含分割出的人体对象的黑色背景图像

### 视频分割

```bash
python predict_video.py
```

该脚本会对 [1.mp4](file:///c%3A/Users/guojia/PycharmProjects/tuxiang/1.mp4) 进行逐帧实例分割，并实时显示分割结果。

按 `q` 键退出视频播放。

### 模型训练

```bash
python train.py
```

该脚本使用COCO数据集训练YOLOv11实例分割模型。

## 代码说明

### predict_image.py

主要功能：
- 加载预训练的 YOLOv11 分割模型
- 对单张图像进行推理
- 只处理标签为 "person" 的对象
- 将所有分割出的人体对象显示在同一张图像上

### predict_video.py

主要功能：
- 加载预训练的 YOLOv11 分割模型
- 对视频逐帧进行推理
- 只处理标签为 "person" 的对象
- 实时显示分割结果

### train.py

主要功能：
- 使用COCO数据集训练YOLOv11实例分割模型
- 包含训练参数配置

## 配置文件

- [yolo11-seg.yaml](tuxiang/yolo11-seg.yaml) - 模型结构定义文件
- [coco.yaml](tuxiang/coco.yaml) - COCO数据集配置文件

## 注意事项

1. 确保输入图像/视频文件存在
2. 如果有 GPU 环境，代码会自动使用 GPU 加速推理
3. 可根据需要修改代码中的模型路径和输入文件路径
4. 训练模型需要大量的计算资源和时间