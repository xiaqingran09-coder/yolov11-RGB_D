import os
import shutil
from pathlib import Path


def organize_dataset():
    root_dir = "my_data"

    source_images = os.path.join(root_dir, "JPEGImages")
    source_labels = os.path.join(root_dir, "labels")

    split_files = {
        "train": os.path.join(root_dir, "train.txt"),
        "val": os.path.join(root_dir, "val.txt"),
        "test": os.path.join(root_dir, "test.txt")
    }

    print("开始整理数据集结构...")

    for split_name, txt_file_path in split_files.items():
        if not os.path.exists(txt_file_path):
            print(f"警告: 找不到 {txt_file_path}，跳过。")
            continue

        with open(txt_file_path, 'r') as f:
            lines = f.readlines()

        print(f"正在处理 {split_name} 集，共 {len(lines)} 张图片...")

        target_img_dir = os.path.join(root_dir, "images", split_name)
        target_lbl_dir = os.path.join(root_dir, "labels", split_name)
        os.makedirs(target_img_dir, exist_ok=True)
        os.makedirs(target_lbl_dir, exist_ok=True)

        count = 0
        for line in lines:

            clean_line = line.strip()
            if not clean_line: continue

            filename = os.path.basename(clean_line)
            name_no_ext = os.path.splitext(filename)[0]

            src_img = os.path.join(source_images, filename)
            dst_img = os.path.join(target_img_dir, filename)

            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            else:
                print(f"  [丢失] 图片未找到: {src_img}")
                continue

            label_name = name_no_ext + ".txt"
            src_lbl = os.path.join(source_labels, label_name)
            dst_lbl = os.path.join(target_lbl_dir, label_name)

            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, dst_lbl)
            else:
                pass

            count += 1

        print(f"  -> 完成 {count} 组文件的整理。")

    print("\n 数据集整理完毕！现在的结构符合 YOLO 标准。")


if __name__ == '__main__':
    organize_dataset()