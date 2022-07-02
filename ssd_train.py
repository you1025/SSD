import os, sys
import yaml
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

    # 設定ファイルの読み込み
    config_path = os.path.join(os.path.dirname(__file__), "train_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # デバイスを取得
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # VOC のクラス定義
    voc_classes = config["classes"]

    # データのファイルパスを取得
    source_dir = config["source_dir"]
    root_path = os.path.join(os.path.dirname(__file__), source_dir)
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
    batch_size  = config["data_loader"]["batch_size"]
    num_workers = config["data_loader"]["num_workers"]
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,  collate_fn=od_collate_fn, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, collate_fn=od_collate_fn, num_workers=num_workers)

    # モデルの作成
    ssd_config = config["ssd_config"]
    net = SSD(phase="train", config=ssd_config)

    # 既存モデルで初期値を設定
    vgg_weight_path = config["model_weight"]["vgg"]
    weight_path     = os.path.join(os.path.dirname(__file__), vgg_weight_path)
    weights         = torch.load(weight_path)
    net.init_parameters(weights)

    # 損失の設定
    jaccard_thresh = config["criterion"]["jaccard_thresh"]
    negpos_ratio   = config["criterion"]["negpos_ratio"]
    criterion = MultiBoxLoss(jaccard_thresh, negpos_ratio, device=device)

    # 最適化の設定
    learning_rate = eval(config["optimizer"]["SGD"]["learning_rate"])
    momentum      = config["optimizer"]["SGD"]["momentum"]
    weight_decay  = eval(config["optimizer"]["SGD"]["weight_decay"])
    optimizer = SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # 学習の実行
    epochs                  = config["train"]["epochs"]
    model_weight_output_dir = config["train"]["model_weight_output_dir"]
    iter_per_log            = config["train"]["iter_per_log"]
    train(net, train_loader, valid_loader, criterion, optimizer, device, epochs, model_weight_output_dir, iter_per_log)
