import os, json
from logging import getLogger, config

def get_logger():
    # 設定の読み込み
    config_path = os.path.join(os.path.dirname(__file__), "../configs/logger_config.json")
    with open(config_path, "r") as f:
        logging_config = json.load(f)
    config.dictConfig(logging_config)

    return getLogger("SSD")
