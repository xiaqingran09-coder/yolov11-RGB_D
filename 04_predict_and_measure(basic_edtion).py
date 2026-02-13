import cv2
import numpy as np
import math
import os
from ultralytics import YOLO

MODEL_PATH = "runs/detect/cube_detect/yellow_block_v19/weights/best.pt"
TEST_IMAGE_PATH = "my_data/JPEGImages/image_1.jpg"
CAMERA_INTRINSICS = (424.7, 419.3, 320.0, 200.0)
BLOCK_SIZE_MM = 20.0

CAMERA_HEIGHT_MM = 400.0  # 相机镜头离地高度
CAMERA_OFFSET_FORWARD_MM = 50.0  # 相机相对于机械臂原点的前后偏移
CAMERA_PITCH_DEGREE = 35.0  # 相机往下低头的角度
# 0度=平视前方，90度=垂直俯视地面

def calculate_world_coord_on_floor(u, v, intrinsics):
    """
    基于“物体在地面上”的假设，从像素坐标 (u,v) 反推世界坐标 (X,Y)
    """
    fx, fy, cx, cy = intrinsics

    angle_pixel_y = math.atan((v - cy) / fy)
    pitch_rad = math.radians(CAMERA_PITCH_DEGREE)
    total_angle = pitch_rad + angle_pixel_y
    if total_angle <= 0.01: total_angle = 0.01

    obj_height = BLOCK_SIZE_MM / 2.0
    distance_horizontal = (CAMERA_HEIGHT_MM - obj_height) / math.tan(total_angle)
    X_world = distance_horizontal + CAMERA_OFFSET_FORWARD_MM
    Y_world = distance_horizontal * (cx - u) / fx
    Z_world = obj_height

    return np.array([X_world, Y_world, Z_world])


def run_prediction():
    print(f"加载模型: {MODEL_PATH} ...")
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型不存在")
        return
    model = YOLO(MODEL_PATH)

    if os.path.exists(TEST_IMAGE_PATH):
        frame = cv2.imread(TEST_IMAGE_PATH)
    else:
        print("图片不存在")
        return

    # YOLO 推理
    results = model(frame, conf=0.6)

    print("\n" + "=" * 50)
    print(f"坐标系: [X: 前进] [Y: 左侧] [Z: 垂直向上]")
    print(f"计算模式: 地面投影 (假设相机下倾 {CAMERA_PITCH_DEGREE} 度)")
    print("=" * 50)

    for result in results:
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 简单的过滤
            if (x2 - x1) * (y2 - y1) > 15000: continue
            if x1 < 5 or y1 < 5 or x2 > 635 or y2 > 395: continue

            cls_id = int(box.cls[0])
            if model.names[cls_id] == 'yellow':
                u_center = (x1 + x2) / 2.0
                v_center = (y1 + y2) / 2.0

                center_world = calculate_world_coord_on_floor(
                    u_center, v_center, CAMERA_INTRINSICS
                )

                coord_str = f"X:{center_world[0]:.1f} Y:{center_world[1]:.1f} Z:{center_world[2]:.1f}"
                print(f"[目标 {i}] | 坐标: {coord_str}")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int(u_center), int(v_center)), 5, (0, 0, 255), -1)
                cv2.putText(frame, coord_str, (x1, y1 - 25), 0, 0.5, (0, 255, 255), 2)

        save_filename = "result_ground_projection.jpg"
        cv2.imwrite(save_filename, frame)
        dir_path = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(dir_path, "result_final.jpg")
        cv2.imwrite(save_path, frame)
        print(f"图片保存在: {save_path}")

        print("\n" + "=" * 50)
        print(f"检测完成！")
        print(f"因为在服务器环境无法弹窗，结果图已保存为: {save_filename}")
        print(f"请在 PyCharm 左侧文件列表中找到它，并双击查看。")
        print("=" * 50)


if __name__ == '__main__':
    run_prediction()
