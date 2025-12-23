"""
:Date        : 2025-07-13 15:41:08
:LastEditTime: 2025-12-23 23:10:18
:Description : 
"""
from ultralytics import YOLO
import torch

if __name__ == "__main__":
    # Clear Cache
    torch.cuda.empty_cache()
    # Load YOLOv8 model
    model = YOLO('yolov8m.pt')

    # Configure the transfer learning
    results = model.train(
        data='dataset.yml',
        epochs=200,
        imgsz=1440,
        batch=2,
        device=0,
        optimizer='SGD',
        momentum=0.937,
        weight_decay=0.0005,
        lr0=0.01,
        lrf=1e-5,
        cos_lr=True,
        rect=True,
        box=10.0,
        cls=1.0,
        dfl=0.2,
        label_smoothing=0.1,
        dropout=0.2,
        patience=15,
        save_period=10,
        project='cilia',
        name='exp1',
        augment=True,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.0,
        mixup=0,
        copy_paste=0,
        erasing=0.2
    )
