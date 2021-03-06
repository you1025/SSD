import os, sys
import yaml
import warnings

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from data_transforms import DataTransformer
from datasets import make_datapath_list, VOCDataset, AnnotationExtractor
from data_loaders import od_collate_fn
from models import SSD, set_learned_weight
from criterions import MultiBoxLoss
from train import train

ROOT = os.path.dirname(__file__)

if __name__ == "__main__":

    # 不要な警告を無視
    warnings.simplefilter("ignore")

    # 設定ファイルの読み込み
    config_path = os.path.join(ROOT, "configs/train_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 乱数シードの設定
    manual_seed      = config["seed"]["torch_manual_seed"]
    cuda_manual_seed = config["seed"]["cuda_manual_seed"]
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(cuda_manual_seed)
    ## 決定論的アルゴリズムの可否
    #backends_cudnn_deterministic = config["deterministic"]["backends_cudnn_deterministic"]
    #use_deterministic_algorithms = config["deterministic"]["use_deterministic_algorithms"]
    #torch.backends.cudnn.deterministic = backends_cudnn_deterministic
    #torch.use_deterministic_algorithms(use_deterministic_algorithms)

    # デバイスを取得
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # VOC のクラス定義
    classes = config["source_data"]["classes"]

    # データのファイルパスを取得
    dir_path = config["source_data"]["dir_path"]
    root_path = os.path.join(ROOT, dir_path)
    train_image_path_list, train_annotation_path_list, valid_image_path_list, valid_annotation_path_list = make_datapath_list(root_path)

    # Data Augumentation 用の transformer を作成
    input_size = 300
    color_mean = (104, 117, 123)
    data_transformer = DataTransformer(input_size, color_mean)

    # アノテーション抽出用の extractor を作成
    annotation_extractor = AnnotationExtractor(classes)

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
    ssd_config = config["model"]["ssd_config"]
    net = SSD(phase="train", config=ssd_config)
    # 既存モデルで初期値を設定
    model_weight_path = config["model"]["initial_weight"]["vgg"]
    net.init_parameters(model_weight_path)
    # 追加学習の場合は学習途中のモデルに復帰
    if config["train"]["additional_learning"]:
        learned_model_path = config["train"]["learned_model_path"]
        set_learned_weight(net, learned_model_path, device)

    # 損失の設定
    jaccard_thresh = config["criterion"]["jaccard_thresh"]
    negpos_ratio   = config["criterion"]["negpos_ratio"]
    criterion = MultiBoxLoss(jaccard_thresh, negpos_ratio, device=device)

    # 最適化の設定
    learning_rate = eval(config["optimizer"]["SGD"]["learning_rate"])
    momentum      = config["optimizer"]["SGD"]["momentum"]
    weight_decay  = eval(config["optimizer"]["SGD"]["weight_decay"])
    optimizer = SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # 追加学習の場合は学習途中の optimizer に復帰
    if config["train"]["additional_learning"]:
        learned_optimizer_path = config["train"]["learned_optimizer_path"]
        set_learned_weight(optimizer, learned_optimizer_path, device)

    # 最適化スケジュールの設定
    gamma      = config["optimizer"]["scheduler"]["gamma"]
    milestones = config["optimizer"]["scheduler"]["milestones"]
    scheduler = MultiStepLR(optimizer, milestones, gamma)
    # 追加学習の場合は学習途中の scheduler に復帰
    if config["train"]["additional_learning"]:
        learned_scheduler_path = config["train"]["learned_scheduler_path"]
        set_learned_weight(scheduler, learned_scheduler_path, device)

    # 学習の実行
    train_config = config["train"]
    train(net, train_loader, valid_loader, criterion, optimizer, scheduler, device, train_config)
