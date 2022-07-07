import os, sys
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from inference import get_image_infos, save_images

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
