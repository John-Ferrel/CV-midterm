import json
import os

import cv2

# 指定 JSON 文件和图像的目录
json_dir = './data/VOCdevkit'
train_path = os.path.join(json_dir, 'train2007.json')
val_path = os.path.join(json_dir, 'val2007.json')
test_path = os.path.join(json_dir, 'test2007.json') # 指定测试集的 JSON 文件
img_path = './data/VOCdevkit/VOC2007/JPEGImages/'

# 指定输出目录
train_dir = './data/COCO/train2024'
val_dir = './data/COCO/val2024'
test_dir = './data/COCO/test2024'  # 指定测试集的输出目录

  
   
# 读取 JSON 文件
train_img = json.load(open(train_path))['images']
val_img = json.load(open(val_path))['images']
test_img = json.load(open(test_path))['images']  # 读取测试集数据

# 创建输出目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)  # 创建测试集的输出目录

# 复制训练集图像
for img in train_img:
    img_name = img['file_name']
    load_img = cv2.imread(os.path.join(img_path, img_name))
    cv2.imwrite(os.path.join(train_dir, img_name), load_img)

# 复制验证集图像
for img in val_img:
    img_name = img['file_name']
    load_img = cv2.imread(os.path.join(img_path, img_name))
    cv2.imwrite(os.path.join(val_dir, img_name), load_img)

# 复制测试集图像
for img in test_img:
    img_name = img['file_name']
    load_img = cv2.imread(os.path.join(img_path, img_name))
    cv2.imwrite(os.path.join(test_dir, img_name), load_img)

# 输出每个集合的图像数量
print(f"Number of training images: {len(train_img)}")
print(f"Number of validation images: {len(val_img)}")
print(f"Number of test images: {len(test_img)}")