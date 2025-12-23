"""
:Date        : 2025-07-13 15:59:03
:LastEditTime: 2025-12-23 23:10:24
:Description : 
"""
import os
import sys
from pathlib import Path
from shapely.geometry import shape
from PIL import Image
import geojson


def get_all_file_under(pathdir: str, skip_suffix: str = None):
    """
    获取指定目录下的所有文件并存入列表
    """
    # 指定目录路径
    directory = Path(pathdir)
    # 获取目录下的所有文件名
    if skip_suffix:
        filenames = [f.name.removesuffix(skip_suffix) for f in directory.iterdir() if f.is_file()]
    else:
        filenames = [f.name for f in directory.iterdir() if f.is_file()]
    return filenames


def name2idx(name: str):
    """
    名称转id
    """
    index_list = [
        {"id": 1, "name": "positive"}, # 有纤毛
        {"id": 0, "name": "negative"}, # 有疑似纤毛的非纤毛结构
        {"id": 0, "name": "ignore"},   # 无纤毛
    ]
    for index_item in index_list:
        if index_item["name"] in name.lower():
            return index_item["id"]
    return -1

def voc_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
    """
    VOC格式转为YOLO格式
    """
    # 计算中心点坐标
    cx = (xmin + xmax) / (2 * img_width)
    cy = (ymin + ymax) / (2 * img_height)

    # 计算宽度和高度
    w = (xmax - xmin) / img_width
    h = (ymax - ymin) / img_height

    return f"{cx} {cy} {w} {h}"

def translate(geojson_dir: str, pic_dir: str, output_dir: str):
    """
    主方法，转换格式并统计比例
    """
    id_zero_count = 0
    id_one_count = 0
    label_files = get_all_file_under(geojson_dir, ".geojson")
    for label_file in label_files:
        lines = []
        img_path = os.path.join(pic_dir, f"{label_file}.tif")
        img = Image.open(img_path)
        width, height = img.size
        img.close()

        geojson_path = os.path.join(geojson_dir, f"{label_file}.geojson")
        output_path = os.path.join(output_dir, f"{label_file}.txt")
        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson_data = geojson.load(f)

        for feat in geojson_data["features"]:
            # 从GeoJSON中解析几何图形和类别
            try:
                classid = name2idx(feat["properties"]["classification"]["name"])
                if classid == -1:
                    continue
                if classid == 0:
                    id_zero_count += 1
                elif classid == 1:
                    id_one_count += 1
            except KeyError as e:
                print(repr(e))
                print(geojson_path)
            geom = shape(feat["geometry"])
            x_min, y_min, x_max, y_max = geom.bounds  # 获取边界框
            yolo_lbl = voc_to_yolo(x_min, y_min, x_max, y_max, width, height)
            lines.append(f"{classid} {yolo_lbl}\n")
        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    print("no cilia:", id_zero_count, "cilia:", id_one_count, "ratio:", id_zero_count/id_one_count)

if __name__ == "__main__":
    geojson_directory = sys.argv[1]
    pic_directory = sys.argv[2]
    output_directory = sys.argv[3]
    translate(geojson_directory, pic_directory, output_directory)
