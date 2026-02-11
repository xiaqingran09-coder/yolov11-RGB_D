import xml.etree.ElementTree as ET
import os
import glob

# 配置你的类别名称（必须与你的 classes.names 一致）
CLASSES = ['yellow', 'log', 'cherry']


def convert_annotation(xml_file, output_path):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    out_file = open(output_path, 'w')

    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in CLASSES:
            continue
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')

        # 坐标转换逻辑: (xmin, xmax, ymin, ymax) -> (x_center, y_center, w, h)
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
             float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))

        # 归一化
        bb = ((b[0] + b[1]) / 2.0 / w, (b[2] + b[3]) / 2.0 / h,
              (b[1] - b[0]) / w, (b[3] - b[2]) / h)

        out_file.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

    out_file.close()


# 设置你的路径
xml_dir = r'C:\Users\Len\Desktop\my_data\Annotations'  # XML文件夹路径
txt_dir = r'C:\Users\Len\Desktop\my_data\labels'  # 输出txt文件夹路径

os.makedirs(txt_dir, exist_ok=True)
xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))

print(f"开始转换 {len(xml_files)} 个文件...")
for xml_file in xml_files:
    file_name = os.path.basename(xml_file).replace('.xml', '.txt')
    convert_annotation(xml_file, os.path.join(txt_dir, file_name))
print("转换完成！")