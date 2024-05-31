import json

# 加载原始的 COCO 格式 JSON 文件
with open('data/COCO/VOC2007/annotations/test2007.json', 'r') as file:
    coco_data = json.load(file)

# 图像 ID 列表，替换成你选择的图像 ID
selected_image_ids = [168, 297, 371, 616]

# 过滤出选定的图像和标注
selected_images = [img for img in coco_data['images'] if img['id'] in selected_image_ids]
selected_annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] in selected_image_ids]

# 创建新的 JSON 结构
new_coco_data = {
    "images": selected_images,
    "annotations": selected_annotations,
    "categories": coco_data['categories']  # 保留类别信息
}

# 写入新的 JSON 文件
with open('data/COCO/VOC2007/annotations/test4show.json', 'w') as file:
    json.dump(new_coco_data, file, indent=4)