import cv2
import numpy as np
from ultralytics import YOLO

# 配置区域
MODEL_PATH = "runs/detect/cube_detect/yellow_block_v19/weights/best.pt"
TEST_IMAGE_PATH = "my_data/JPEGImages/image_134.jpg"
CAMERA_INTRINSICS = (424.7, 419.3, 320.0, 200.0)
BLOCK_SIZE_MM = 20.0


def get_cube_center_optimized(rgb_roi, depth_roi, bbox, intrinsics, block_size):
    # 1. HSV 颜色过滤 (提取黄色区域)
    hsv = cv2.cvtColor(rgb_roi, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 2. 深度有效性过滤
    valid_depth_mask = (depth_roi > 0)
    final_mask = mask & valid_depth_mask

    # 如果有效点太少，视为噪声
    if np.count_nonzero(final_mask) < 10:
        return None

    # 3. 提取有效点的 3D 数据
    ys, xs = np.where(final_mask)
    zs = depth_roi[ys, xs]

    # 简单的离群值过滤 (只取中位数附近的深度)
    z_median = np.median(zs)
    # 容差设为方块边长的一半
    z_valid_idx = np.abs(zs - z_median) < (block_size * 0.5)

    if np.count_nonzero(z_valid_idx) == 0:
        return None

    # 再次筛选
    ys = ys[z_valid_idx]
    xs = xs[z_valid_idx]
    zs = zs[z_valid_idx]

    # 4. 像素坐标 -> 相机坐标
    # 还原到整张图的坐标系
    u_global = xs + bbox[0]
    v_global = ys + bbox[1]

    fx, fy, cx, cy = intrinsics
    X_points = (u_global - cx) * zs / fx
    Y_points = (v_global - cy) * zs / fy
    Z_points = zs

    # 5. 计算体心 (表面质心 + Z轴补偿)
    body_center = np.array([
        np.mean(X_points),
        np.mean(Y_points),
        np.mean(Z_points) + (block_size / 2.0)  # 往内部推一半边长
    ])

    return body_center


def run_prediction():
    # 1. 加载 YOLO 模型
    print(f"正在加载模型: {MODEL_PATH} ...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"错误: 找不到模型文件，请检查路径。详细信息: {e}")
        return

    # 2. 读取图片
    frame = cv2.imread(TEST_IMAGE_PATH)
    if frame is None:
        print(f"错误: 无法读取图片 {TEST_IMAGE_PATH}")
        return

    print("警告: 正在使用模拟深度数据 (全图 600mm)，仅供测试流程！")
    depth_image = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint16) * 600
    # =========================================

    # 3. YOLO 推理
    results = model(frame, conf=0.6)

    print("\n" + "=" * 30)
    print("开始输出检测结果")
    print("=" * 30)

    # ... (前面的代码不变) ...

    for result in results:
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            area = w * h

            if area > 8000:
                print(f"跳过大面积误检 (Area: {area})")
                continue

            if x1 < 5 or y1 < 5 or x2 > 635 or y2 > 395:
                print(f"跳过边缘误检 (Edge)")
                continue
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # 只计算黄色方块
            if label == 'yellow':
                # 裁剪 ROI
                roi_rgb = frame[y1:y2, x1:x2]
                roi_depth = depth_image[y1:y2, x1:x2]

                # === 核心调用：计算体心 ===
                center_3d = get_cube_center_optimized(
                    roi_rgb, roi_depth, (x1, y1), CAMERA_INTRINSICS, BLOCK_SIZE_MM
                )

                if center_3d is not None:
                    # 格式化坐标字符串
                    coord_text = f"X:{center_3d[0]:.1f} Y:{center_3d[1]:.1f} Z:{center_3d[2]:.1f}"

                    # 1. 终端打印 (Read Out)
                    print(f"[物体 {i}] {label} | 体心坐标: {coord_text}")

                    # 2. 图片绘制
                    # 画框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 画中心点
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    # 写坐标文字
                    cv2.putText(frame, coord_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    print("=" * 30 + "\n")

    # 显示结果图
    save_path = "result_prediction.jpg"
    cv2.imwrite(save_path, frame)
    print(f"检测完成！结果图已保存为: {save_path}")
    print("请在左侧文件列表中找到该图片并双击查看。")


if __name__ == '__main__':
    run_prediction()