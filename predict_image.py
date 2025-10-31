import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os

image_path = "1.jpg" 


def predict_single_image():
    model_path = "yolo11x-seg.pt"  
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在")
        return

    model = YOLO(model_path)
    if torch.cuda.is_available():
        device = 'cuda'
        print("使用 GPU 进行推理")
    else:
        device = 'cpu'
        print("使用 CPU 进行推理")
        
    model.to(device)
    print(f"模型当前运行在设备上: {device}")

    if not os.path.exists(image_path):
        print(f"图片文件 {image_path} 不存在")
        return

    # 进行推理
    results = model(image_path, device=device)
    result = results[0]
    plotted_image = result.plot(
        conf=True,  # 显示置信度
        line_width=1,  # 线条粗细
        font_size=1,  # 字体大小
        labels=True,  # 显示标签
        boxes=True,  # 显示边界框
        masks=True  # 显示分割掩码
    )

    cv2.imshow("YOLOv11 Instance Segmentation", plotted_image)
    
    if result.masks is not None and result.boxes is not None:
        orig_image = result.orig_img
        segmented_objects = np.zeros_like(orig_image)
        for i, (mask, box) in enumerate(zip(result.masks.data, result.boxes)):
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # 只处理标签为"person"的对象
            if class_name == "person":
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (orig_image.shape[1], orig_image.shape[0]))
                for c in range(orig_image.shape[2]):
                    segmented_objects[:, :, c] = np.where(
                        mask_resized > 0, 
                        orig_image[:, :, c], 
                        segmented_objects[:, :, c]
                    )
            
        cv2.imshow("All Segmented Persons", segmented_objects)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if result.boxes is not None:
        print(f"检测到 {len(result.boxes)} 个对象")
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            print(f"对象 {i+1}: {class_name}, 置信度: {confidence:.2f}")

if __name__ == "__main__":
    predict_single_image()