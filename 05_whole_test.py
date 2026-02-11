import os


# ... (前面的配置和 get_cube_center_optimized 函数保持不变) ...

def run_batch_prediction():
    # 1. 加载模型
    print(f"正在加载模型: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)

    # 2. 定义图片所在的文件夹 (Windows 本地路径)
    images_dir = r"my_data/JPEGImages"

    # 3. 读取 test.txt (获取测试集的文件名列表)
    # 假设 test.txt 在当前目录下，或者你可以写绝对路径
    test_txt_path = "my_data/ImageSets/test.txt"  # 根据你实际存放位置修改

    if not os.path.exists(test_txt_path):
        print(f"错误: 找不到 {test_txt_path}，请检查路径")
        return

    with open(test_txt_path, 'r') as f:
        lines = f.readlines()

    print(f"准备测试 {len(lines)} 张图片...")

    # 4. 循环处理每一张图
    for line in lines:
        line = line.strip()
        if not line: continue

        # 提取文件名 (例如从 /home/.../image_191.jpg 提取出 image_191.jpg)
        filename = os.path.basename(line)

        # 拼接本地真实路径
        image_path = os.path.join(images_dir, filename)

        if not os.path.exists(image_path):
            print(f"跳过: 找不到图片 {image_path}")
            continue

        # 读取图片
        frame = cv2.imread(image_path)
        if frame is None: continue

        # --- 模拟深度图 (实际使用时请替换为真实深度) ---
        depth_image = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint16) * 600
        # -------------------------------------------

        # 推理
        results = model(frame, conf=0.5, verbose=False)  # verbose=False 不让它刷屏

        # 绘制和计算
        has_detection = False
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if label == 'yellow':
                    has_detection = True
                    roi_rgb = frame[y1:y2, x1:x2]
                    roi_depth = depth_image[y1:y2, x1:x2]

                    center_3d = get_cube_center_optimized(
                        roi_rgb, roi_depth, (x1, y1), CAMERA_INTRINSICS, BLOCK_SIZE_MM
                    )

                    if center_3d is not None:
                        coord_text = f"X:{center_3d[0]:.1f} Y:{center_3d[1]:.1f} Z:{center_3d[2]:.1f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, coord_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        print(f"[{filename}] 检测到黄色方块: {coord_text}")

        # 显示图片
        cv2.imshow("Batch Prediction (Press 'q' to quit, any key for next)", frame)

        # 按任意键看下一张，按 'q' 退出
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    print("测试结束")


if __name__ == '__main__':
    run_batch_prediction()