import os, sys
import numpy as np
from glob import glob
import yaml
import cv2
import matplotlib.pyplot as plt

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from models import SSD
from data_transforms import DataTransformer

def get_image_infos(config):
    # デバイスを取得
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # モデルの作成
    ssd_config = config["model"]["ssd_config"]
    net = SSD(phase="inference", config=ssd_config)

    # 学習済みモデルを適用
    weight_path = os.path.join(os.path.dirname(__file__), config["model"]["weight_path"])
    ssd_weights = torch.load(weight_path, map_location=device)
    net.load_state_dict(ssd_weights)

    # 対象画像パスから変換済画像と画像情報を取得
    source_dir = os.path.join(os.path.dirname(__file__), config["source_dir"])
    transformed_images, image_infos = get_source_images_from_dir(source_dir)

    # BGR->RGB & 前チャネル へ変換
    transformed_image_tensors = torch.tensor(np.array(transformed_images)[:, :, :, (2, 1, 0)]).permute(0, 3, 1, 2)

    # 推論処理の実行
    net.eval()
    detections = net(transformed_image_tensors).cpu().detach().numpy()

    # image_info に検出された物体情報を追加
    update_image_info_by_detections(image_infos, detections)

    return image_infos

def update_image_info_by_detections(image_infos, detections, confidence_level=0.6):
    # detections の中から背景を対象外として confidence が confidence_level 以上のものだけを抽出
    valid_indexes = (detections[:, 1:, :, 0] >= confidence_level).nonzero()

    # 対象となる検出情報を image_info に埋め込む
    for batch_id, class_id, order_id in zip(*valid_indexes):
        # 対象の画像情報を取得
        image_info = image_infos[batch_id]

        # オリジナル画像の高さと幅を取得
        height, width = image_info["height"], image_info["width"]

        # detections は id=0 として "背景" クラスが挿入されている事に注意(class_id に 1 を足して補正)
        confidence = detections[batch_id, class_id+1, order_id][0]
        bbox       = detections[batch_id, class_id+1, order_id][1:] * [width, height, width, height]

        # image_info に object_info(物体情報) が存在しない場合は作成する
        # 同一画像(batch_id)に対し複数の物体が検出されるケースでは初回のみ対象
        if "object_info" not in image_info:
            image_info["object_info"] = {
                "confidences": [],
                "bboxes": [],
                "class_ids": []
            }

        # 検出物体情報を登録
        object_info = image_info["object_info"]
        object_info["confidences"].append(confidence)
        object_info["bboxes"].append(bbox)
        object_info["class_ids"].append(class_id)

def get_sopurce_image(image_path):
    image_path_list = [image_path]
    return transform_and_extract_image_info(image_path_list)

def get_source_images_from_dir(dir_path):
    image_path_list = glob(f"{dir_path}/*")
    return transform_and_extract_image_info(image_path_list)

def transform_and_extract_image_info(image_path_list):
    # 画像の変換処理
    color_mean = (104, 117, 123)
    input_size = 300
    transformer = DataTransformer(input_size, color_mean)

    # 変換済画像および画像属性の抽出
    transformed_images = []
    image_infos = []
    for image_path in image_path_list:
        transformed_image, image_info = image_transform(image_path, transformer)
        transformed_images.append(transformed_image)
        image_infos.append(image_info)

    return (transformed_images, image_infos)

def image_transform(image_path, transformer):
    # 画像属性の取得
    original_image = cv2.imread(image_path)
    height, width, _ = original_image.shape

    # BGR -> RGB
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # 画像の変換
    transformed_image, _, _ = transformer(original_image, "valid", "", "")

    return (
        transformed_image,
        {
            "image_path": image_path,
            "rgb_image": rgb_image,
            "height": height,
            "width": width            
        }
    )

def plot_images(image_infos, classes, figsize=(100, 100)):
    num_classes = len(classes)
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    image_num = len(image_infos)

    _, axes = plt.subplots(nrows=image_num, figsize=figsize)

    for image_id in range(image_num):
        image_info = image_infos[image_id]

        ax = axes[image_id]

        # オリジナル画像を表示
        image = image_info["rgb_image"]
        ax.imshow(image)

        # 検出された物体が存在しない場合は次の画像へ
        if "object_info" not in image_info:
            continue

        # 検出された物体をオリジナル画像に重ねて表示
        object_info = image_info["object_info"]
        bboxes      = object_info["bboxes"]
        class_ids   = object_info["class_ids"]
        confidences = object_info["confidences"]
        for bbox, class_id, confidence in zip(bboxes, class_ids, confidences):
            color = colors[class_id]

            # 検出物体の位置
            x, y = (bbox[0], bbox[1])
            width  = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # 枠の表示
            ax.add_patch(plt.Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=2))

            # テキストの表示
            label_name = classes[class_id]
            display_text = f"{label_name}: {confidence:.3f}"
            ax.text(x, y-5, display_text, bbox={"facecolor": color, "alpha": 0.5})

# TODO: matplotlib 以外で書き直したい
# TODO: 画面に表示させたくない
def save_images(image_infos, classes, output_dir):
    num_classes = len(classes)
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    image_num = len(image_infos)

    for image_id in range(image_num):
        # 対象の画像情報を取得
        image_info = image_infos[image_id]

        fig, ax = plt.subplots(dpi=100, figsize=(image_info["width"]/100, image_info["height"]/100))

        # ファイル名を取得し名前と拡張子に分解
        file_name = os.path.basename(image_info["image_path"])
        name, extention = file_name.split(".")

        # オリジナル画像を表示
        image = image_info["rgb_image"]
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.imshow(image)

        # 検出された物体が存在しない場合は次の画像へ
        if "object_info" not in image_info:
            continue

        # 検出された物体をオリジナル画像に重ねて表示
        object_info = image_info["object_info"]
        bboxes      = object_info["bboxes"]
        class_ids   = object_info["class_ids"]
        confidences = object_info["confidences"]
        for bbox, class_id, confidence in zip(bboxes, class_ids, confidences):
            color = colors[class_id]

            # 検出物体の位置
            x, y = (bbox[0], bbox[1])
            width  = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # 枠の表示
            ax.add_patch(plt.Rectangle((x, y), width, height, fill=False, edgecolor=color, linewidth=2))

            # テキストの表示
            label_name = classes[class_id]
            display_text = f"{label_name}: {confidence:.3f}"
            ax.text(x, y-5, display_text, bbox={"facecolor": color, "alpha": 0.5})

        # 画像の出力
        output_file_path = f"{output_dir}/{name}_detected.png"
        plt.savefig(output_file_path)

if __name__ == "__main__":
    # 設定ファイルの読み込み
    config_path = os.path.join(os.path.dirname(__file__), "configs/inference_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 画像情報を取得
    image_infos = get_image_infos(config)

    # 物体検出済みの画像を保存
    classes    = config["classes"]
    output_dir = config["target_dir"]
    save_images(image_infos, classes, output_dir)
