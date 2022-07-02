import os, sys
import warnings

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from data_transforms import DataTransformer
from datasets import make_datapath_list, VOCDataset, AnnotationExtractor
from data_loaders import od_collate_fn
from models import SSD
from criterions import MultiBoxLoss
from train import train

if __name__ == "__main__":

    # 不要な警告を無視
    warnings.simplefilter("ignore")

    # デバイスを取得
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # VOC のクラス定義
    # TODO: config に移動
    voc_classes = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor"
    ]

    # データのファイルパスを取得
    # TODO: パスを config に移動
    root_path = os.path.join(os.path.dirname(__file__), "data/VOCdevkit/VOC2012/")
    train_image_path_list, train_annotation_path_list, valid_image_path_list, valid_annotation_path_list = make_datapath_list(root_path)

    # Data Augumentation 用の transformer を作成
    input_size = 300
    color_mean = (104, 117, 123)
    data_transformer = DataTransformer(input_size, color_mean)

    # アノテーション抽出用の extractor を作成
    annotation_extractor = AnnotationExtractor(voc_classes)

    # DataSet を作成
    train_dataset = VOCDataset(
        train_image_path_list,
        train_annotation_path_list,
        phase="train",
        transformer=data_transformer,
        annotation_extractor=annotation_extractor
    )
    valid_dataset = VOCDataset(
        valid_image_path_list,
        valid_annotation_path_list,
        phase="valid",
        transformer=data_transformer,
        annotation_extractor=annotation_extractor
    )

    # DataLoader を作成
    # TODO: batch_size と num_workers を config に移動
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,  collate_fn=od_collate_fn, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, collate_fn=od_collate_fn, num_workers=2)

    # モデルの作成
    # TODO: ssd_config を config に移動
    # TODO: VGG のパスを config に移動
    ssd_config = {
        "num_classes": 21,
        "image_size": 300,
        "bbox_aspect_num": [4, 6, 6, 6, 4, 4],
        "feature_map_regions": [38, 19, 10, 5, 3, 1],
        "region_sizes": [8, 16, 32, 64, 100, 300],
        "min_sizes": [30, 60, 111, 162, 213, 264],
        "max_sizes": [60, 111, 162, 213, 264, 315],
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    }
    net = SSD(phase="train", config=ssd_config)
    # 既存モデルで初期値を設定
    vgg_weight_path = os.path.join(os.path.dirname(__file__), "weights/vgg/vgg16_reducedfc.pth")
    vgg_weights = torch.load(vgg_weight_path)
    net.init_parameters(vgg_weights)

    criterion = MultiBoxLoss(jaccard_thresh=0.5, negpos_ratio=3, device=device)
    optimizer = SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    # 学習の実行
    # TODO: epochs を config に移動
    # TODO: model_weight_output_dir を config に移動
    train(net, train_loader, valid_loader, criterion, optimizer, device, epochs=1, model_weight_output_dir="weights/models")
