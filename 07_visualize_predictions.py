import cv2
import os
import random
import matplotlib.pyplot as plt
from ultralytics import YOLO


def visualize_model_predictions(model_path, val_images_path, num_samples=4):
    """
    model_path: 训练好的权重路径 (如 runs/detect/train/weights/best.pt)
    val_images_path: 验证集图片的文件夹路径
    num_samples: 随机抽样几张图来看
    """
    # 1. 加载模型
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return

    model = YOLO(model_path)

    # 2. 获取所有图片列表
    image_files = [f for f in os.listdir(val_images_path) if f.endswith(('.jpg', '.png'))]
    if not image_files:
        print(f"错误：在 {val_images_path} 找不到图片")
        return

    # 随机抽取图片
    selected_files = random.sample(image_files, min(len(image_files), num_samples))

    # 3. 设置绘图
    plt.figure(figsize=(15, 5 * ((len(selected_files) + 1) // 2)))

    for i, file_name in enumerate(selected_files):
        img_path = os.path.join(val_images_path, file_name)

        # 运行推理
        results = model(img_path)

        # 绘制结果 (Ultralytics 自带的绘图功能)
        # plot() 会返回一个 BGR 格式的 numpy 数组
        res_plotted = results[0].plot()

        # 转换 BGR 到 RGB 以便 Matplotlib 显示
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

        plt.subplot(2, 2, i + 1)
        plt.imshow(res_rgb)
        plt.title(f"Prediction: {file_name}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 请修改为你的实际路径
    # 模型路径
    my_model = "runs/detect/cube_detect/yellow_block_v19/weights/best.pt"
    # 验证集图片文件夹
    my_val_imgs = "my_data/JPEGImages"  # 或者 "my_data/images/val"

    visualize_model_predictions(my_model, my_val_imgs, num_samples=4)