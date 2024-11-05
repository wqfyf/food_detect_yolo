import os
import torch
from ultralytics import YOLO
import time

# 定义模型对应的预训练权重
pretrained_weights = 'yolov8m.pt'  # 预训练权重

# 定义通用训练参数
data_yaml = 'UECFOOD_all.yaml'  # 数据集配置文件
epochs = 10  # 训练轮数
batch_size = 32  # 批量大小
lr0 = 0.001
# lrf = 0.1
save_dir = '../../runs/train_fine_tune_v8_7_UECFOOD_all_layering_augment_freeze_default_opt_257-m'  # 结果保存路径

# 定义训练结果存储
results = []


# 定义训练函数
def train_yolov8():
    print("\nStarting fine-tuning for YOLOv8m with layer-wise optimization...\n")

    # 记录训练开始时间
    start_time = time.time()

    # 加载模型
    model = YOLO(pretrained_weights)  # 加载预训练权重

    model.model.nc = 257  # 假设有 256 个类别

    # 训练模型
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        name='yolo_v8',  # 保存文件夹名
        project=save_dir,  # 结果存储位置
        device='0',
        augment=True,
        lr0=lr0,
        # lrf=lrf
    )

    # 记录训练结束时间
    end_time = time.time()

    # 计算训练时间
    training_time = end_time - start_time

    # 验证模型性能
    metrics = model.val()

    # 获取 mAP@0.5 和 mAP@0.5:0.95
    map50 = metrics.box.map50  # mAP@0.5
    map5095 = metrics.box.map  # mAP@0.5:0.95

    print(f"Finished fine-tuning YOLOv8. mAP@0.5: {map50}, mAP@0.5:0.95: {map5095}")

    # 保存结果
    results.append({
        'mAP@0.5': map50,
        'mAP@0.5:0.95': map5095,
        'training_time': training_time
    })

    # 保存结果为 CSV 文件
    with open(os.path.join(save_dir, 'results_layerwise.csv'), 'w') as f:
        f.write("mAP@0.5,mAP@0.5:0.95,Training Time (s)\n")
        for result in results:
            f.write(f"{result['mAP@0.5']},{result['mAP@0.5:0.95']},{result['training_time']}\n")


# 主函数
if __name__ == "__main__":
    train_yolov8()
