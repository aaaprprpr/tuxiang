import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os

video_path = "1.mp4" 

def predict_video():
    model_path = "yolo11n-seg.pt" 
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
    
    if not os.path.exists(video_path):
        print(f"视频文件 {video_path} 不存在")
        return

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入器，用于保存分割结果
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_mask = cv2.VideoWriter('mask_output.mp4', fourcc, fps, (960, 540))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 进行推理
        results = model(frame,device=device,imgsz=(384, 640))
        result = results[0]
        plotted_frame = result.plot(
        conf=True,  # 显示置信度
        line_width=1,  # 线条粗细
        font_size=1,  # 字体大小
        labels=True,  # 显示标签
        boxes=True,  # 显示边界框
        masks=True  # 显示分割掩码
        )

        frame_count += 1
        if frame_count % 30 == 0:  # 每30帧打印一次进度
            print(f"已处理 {frame_count} 帧")

        
        cv2.imshow("YOLOv11 Instance Segmentation", plotted_frame)

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
        segmented_objects =cv2.resize(segmented_objects, (960, 540))
        # out_mask.write(segmented_objects)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    out_mask.release()
    print(f"总共处理了 {frame_count} 帧")

if __name__ == "__main__":
    predict_video()
