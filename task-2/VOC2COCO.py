from __future__ import absolute_import, division, print_function

import json
import logging
import os.path as osp
from collections import OrderedDict

import mmcv
import xmltodict

logger = logging.getLogger(__name__)

class PASCALVOC2COCO(object):
    def __init__(self):
        self.cat2id = {
            'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4,
            'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8,
            'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12,
            'horse': 13, 'motorbike': 14, 'person': 15, 'pottedplant': 16,
            'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20
        }

    def get_img_item(self, file_name, image_id, size):
        image = OrderedDict()
        image['file_name'] = file_name
        image['height'] = int(size['height'])
        image['width'] = int(size['width'])
        image['id'] = image_id
        return image

    def get_ann_item(self, obj, image_id, ann_id):
        x1 = int(obj['bndbox']['xmin']) - 1
        y1 = int(obj['bndbox']['ymin']) - 1
        w = int(obj['bndbox']['xmax']) - x1
        h = int(obj['bndbox']['ymax']) - y1
        annotation = OrderedDict()
        annotation['segmentation'] = [[x1, y1, x1, (y1 + h), (x1 + w), (y1 + h), (x1 + w), y1]]
        annotation['area'] = w * h
        annotation['iscrowd'] = 0
        annotation['image_id'] = image_id
        annotation['bbox'] = [x1, y1, w, h]
        annotation['category_id'] = self.cat2id[obj['name']]
        annotation['id'] = ann_id
        annotation['ignore'] = int(obj['difficult'])
        return annotation

    def get_cat_item(self, name, id):
        category = OrderedDict()
        category['supercategory'] = 'none'
        category['id'] = id
        category['name'] = name
        return category
    
    def read_file_to_list(self,filepath):
        with open(filepath, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
        return lines

    def convert(self, devkit_path, year, splits, save_dir):
        for split in splits:
            split_file = osp.join(devkit_path, f'VOC{year}/ImageSets/Main/{split}.txt')
            ann_dir = osp.join(devkit_path, f'VOC{year}/Annotations')

            name_list = self.read_file_to_list(split_file)  # 使用自定义的文件读取函数

            images, annotations = [], []
            ann_id = 1
            for name in name_list:
                image_id = int(''.join(name.split('_'))) if '_' in name else int(name)
                xml_file = osp.join(ann_dir, name + '.xml')
                with open(xml_file, 'r') as f:
                    ann_dict = xmltodict.parse(f.read(), force_list=('object',))
                image = self.get_img_item(name + '.jpg', image_id, ann_dict['annotation']['size'])
                images.append(image)

                if 'object' in ann_dict['annotation']:
                    for obj in ann_dict['annotation']['object']:
                        annotation = self.get_ann_item(obj, image_id, ann_id)
                        annotations.append(annotation)
                        ann_id += 1
                else:
                    logger.warning(f'{name} does not have any object')

            categories = [self.get_cat_item(name, id) for name, id in self.cat2id.items()]
            ann = OrderedDict()
            ann['images'] = images
            ann['type'] = 'instances'
            ann['annotations'] = annotations
            ann['categories'] = categories
            save_file = osp.join(save_dir, f'{split}{year}.json')
            logger.info(f'Saving annotations to {save_file}')
            with open(save_file, 'w') as f:
                json.dump(ann, f, indent=4)

if __name__ == '__main__':
    converter = PASCALVOC2COCO()
    devkit_path = './data/VOCdevkit'
    year = '2007'  # 2007年数据集
    splits = ['train', 'val', 'test']  # 包括训练集、验证集和测试集
    save_dir = './data/VOCdevkit'
    converter.convert(devkit_path, year, splits, save_dir)