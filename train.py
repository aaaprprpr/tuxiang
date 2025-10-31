import torch
from ultralytics import YOLO
import torch.multiprocessing as mp

mp.freeze_support()


if __name__ == '__main__':
    model = YOLO('./yolo11-seg.yaml')
    model.train(data='./coco.yaml', 
                # time=5.0, # (float, optional) number of hours to train for, overrides epochs if supplied
                epochs=300,                 # 实例分割任务推荐使用300个epochs
                batch=16,                   # 根据GPU内存调整，较小的batch适用于分割任务
                imgsz=640,                  # 标准的YOLO训练尺寸
                device=torch.device('cuda'),                
                cache=False,                # (bool) 是否缓存数据到内存或磁盘中（True/ram、disk 或 False）                
                workers=8,                  # (int) 数据加载线程数
                amp=True,                   # (bool) 是否使用自动混合精度训练（AMP）
                lr0=0.01,                   # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
                lrf=0.01,                   # (float) final learning rate (lr0 * lrf)
                optimizer='SGD',            # 优化器选择，SGD通常在分割任务中表现良好
                close_mosaic=10,            # 关闭Mosaic增强的epoch数，默认为10
                rect=False,                 # 是否使用矩形训练，实例分割通常设置为False
                box=7.5,                    # 目标检测损失权重
                cls=0.5,                    # 分类损失权重
                dfl=1.5,                    # Distribution Focal Loss权重
                pose=1.0,                   # 关键点损失权重
                kobj=1.0,                   # 关键点对象性损失权重
                hsv_h=0.015,                # HSV-Hue增广强度
                hsv_s=0.7,                  # HSV-Saturation增广强度
                hsv_v=0.4,                  # HSV-Value增广强度
                degrees=0.0,                # 图像旋转角度
                translate=0.1,              # 图像平移强度
                scale=0.5,                  # 图像缩放强度
                shear=0.0,                  # 图像剪切强度
                perspective=0.0,            # 图像透视变换强度
                mosaic=1.0,                 # Mosaic增强概率
                mixup=0.0,                  # MixUp增强概率，对于分割任务通常设置为0
                copy_paste=0.0,             # Copy-Paste增强概率
                auto_augment='randaugment', # 自动数据增强策略
                patience=50,                # 早停机制
                # Val/Test settings ----------------------------------------------------------------------------------------------------
                val=True,                   # (bool) 是否在训练过程中执行验证
                iou=0.7,                    # (float) NMS 中使用的 IoU 阈值
                max_det=300,                # (int) 每张图像最多检测多少个目标
                half=False,                 # (bool) 是否使用 FP16 半精度推理
    )