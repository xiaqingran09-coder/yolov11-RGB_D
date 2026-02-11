import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_training_metrics(results_csv_path):
    # 检查文件是否存在
    if not os.path.exists(results_csv_path):
        print(f"错误：找不到文件 {results_csv_path}")
        print("请确保你已经完成了训练，且路径指向了正确的 runs 文件夹。")
        return

    # 读取 CSV 数据 (去除列名的空格)
    data = pd.read_csv(results_csv_path)
    data.columns = [c.strip() for c in data.columns]

    # 设置画布
    plt.figure(figsize=(15, 5))

    # 1. 绘制 Box Loss (定位误差)
    plt.subplot(1, 3, 1)
    plt.plot(data['epoch'], data['train/box_loss'], label='Train Box Loss', color='blue')
    plt.plot(data['epoch'], data['val/box_loss'], label='Val Box Loss', color='orange')
    plt.title('Box Loss (Lower is Better)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. 绘制 mAP50 (检出率/准确率综合指标)
    plt.subplot(1, 3, 2)
    plt.plot(data['epoch'], data['metrics/mAP50(B)'], label='mAP@50', color='green')
    plt.title('mAP50 (Higher is Better)')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)

    # 3. 绘制 mAP50-95 (高精度指标)
    plt.subplot(1, 3, 3)
    plt.plot(data['epoch'], data['metrics/mAP50-95(B)'], label='mAP@50-95', color='purple')
    plt.title('mAP50-95 (Higher is Better)')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # 保存图片
    save_path = 'training_evaluation.png'
    plt.savefig(save_path)
    print(f"评估图表已保存为: {save_path}")
    plt.show()


if __name__ == "__main__":
    # 注意：请修改为你实际训练结果的路径
    # 通常在 runs/detect/yellow_block_v1/results.csv
    csv_path = "runs/detect/cube_detect/yellow_block_v19/results.csv"
    plot_training_metrics(csv_path)