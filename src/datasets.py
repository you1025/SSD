import os
import numpy as np
import cv2
from xml.etree import ElementTree
import torch
from torch.utils.data import Dataset

class VOCDataset(Dataset):
    def __init__(self, image_path_list, annotation_path_list, phase, transformer, annotation_extractor):
        self.image_path_list = image_path_list
        self.annotation_path_list = annotation_path_list
        self.phase = phase
        self.transform = transformer
        self.annotation_extract = annotation_extractor

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        image, ground_truth, _, _ = self.pull_item(index)
        return (image, ground_truth)

    def pull_item(self, index):
        # 画像情報を取得
        image_file_path = self.image_path_list[index]
        image = cv2.imread(image_file_path)
        height, width, channels = image.shape

        # アノテーション情報を取得
        annotation_file_path = self.annotation_path_list[index]
        annotation_list = self.annotation_extract(annotation_file_path, width, height)

        # 画像の変換を実施
        transformed_image, boxes, labels = self.transform(image, self.phase, annotation_list[:, :4], annotation_list[:, 4])

        # 画像フォーマットの変換
        # - BGR -> RGB
        # - 前チャネルへの変換: (height, weight, channel) -> (channel, height, weight)
        image_tensor = torch.from_numpy(transformed_image[:, :, (2, 1, 0)]).permute(2, 0, 1)

        ground_truth = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return (image_tensor, ground_truth, height, width)

def make_datapath_list(root_path):
    # ファイルパスを定義
    train_id_file = os.path.join(root_path, "ImageSets/Main/train.txt")
    valid_id_file = os.path.join(root_path, "ImageSets/Main/val.txt")

    # 訓練用の情報を取得
    train_image_path_list = []
    train_annotation_path_list = []
    with open(train_id_file) as f:
        for line in f:
            file_id = line.strip()
            train_image_path = os.path.join(root_path, f"JPEGImages/{file_id}.jpg")
            train_annotation_path = os.path.join(root_path, f"Annotations/{file_id}.xml")
            train_image_path_list.append(train_image_path)
            train_annotation_path_list.append(train_annotation_path)

    # 検証用の情報を取得
    valid_image_path_list = []
    valid_annotation_path_list = []
    with open(valid_id_file) as f:
        for line in f:
            file_id = line.strip()
            valid_image_path = os.path.join(root_path, f"JPEGImages/{file_id}.jpg")
            valid_annotation_path = os.path.join(root_path, f"Annotations/{file_id}.xml")
            valid_image_path_list.append(valid_image_path)
            valid_annotation_path_list.append(valid_annotation_path)

    return (train_image_path_list, train_annotation_path_list, valid_image_path_list, valid_annotation_path_list)

class AnnotationExtractor():
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, xml_path, width, height):
        ret = []

        xml = ElementTree.parse(xml_path).getroot()
        for obj in xml.iter("object"):
            is_difficult = (obj.find("difficult").text == "1")
            if is_difficult:
                continue

            bndbox = obj.find("bndbox")
            xmin = (int(bndbox.find("xmin").text) - 1) / width
            xmax = (int(bndbox.find("xmax").text) - 1) / width
            ymin = (int(bndbox.find("ymin").text) - 1) / height
            ymax = (int(bndbox.find("ymax").text) - 1) / height

            name = obj.find("name").text.strip().lower()
            class_id = self.classes.index(name)

            bounding_box = [xmin, ymin, xmax, ymax, class_id]
            ret.append(bounding_box)

        return np.array(ret)
