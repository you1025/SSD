from time import time
import torch
from torch.nn.utils import clip_grad_value_
from criterions import calc_total_loss

# Logger の取得
from logger import get_logger
logger = get_logger("train")

def train(net, train_loader, valid_loader, criterion, optimizer, scheduler, device, config):
    epochs                   = config["epochs"]
    iters_per_log            = config["iters_per_log"]
    epochs_per_log           = config["epochs_per_log"]
    epochs_per_save          = config["epochs_per_save"]
    model_weight_output_dirs = config["model_weight_output_dirs"]

    net.to(device)
    logger.info(f"device: {device}")

    torch.backends.cudnn.benchmark = True

    iteration = 1
    for epoch in range(epochs):
        logger.info(f"epoch: {epoch+1}/{epochs}")

        epoch_train_loss = 0

        # エポック開始時刻
        epoch_start_time = time()
        # イテレーション開始時刻
        iter_start_time = time()

        net.train()
        for images, targets in train_loader:
            # GPU にデータを転送
            images  = images.to(device)
            targets = [ annotation.to(device) for annotation in targets ]

            # 勾配をリセット
            optimizer.zero_grad()

            # 順伝播
            outputs = net(images)

            # ロスを算出
            loss_l, loss_c = criterion(outputs, targets)
            loss = loss_l + loss_c

            # 逆伝播
            loss.backward()
            clip_grad_value_(net.parameters(), clip_value=2.0)
            optimizer.step()

            # 一定期間ごとにログを出力
            if (iteration % iters_per_log == 0):
                iter_duration = time() - iter_start_time
                logger.info(f"iteration: {iteration:5d} - loss: {loss.item():6.3f}, {iters_per_log} iter: {iter_duration:4.1f} sec.")
                iter_start_time = time()

            epoch_train_loss += loss.item()
            iteration += 1

        # スケジュールに従って optimizer の学習率を変更
        scheduler.step()
        logger.debug(f"optimizer: {optimizer}")

        # 毎エポック終了ごとに訓練誤差を出力
        logger.info(f"epoch: {epoch+1} - train_loss: {epoch_train_loss:7.3f}")

        # 1 エポックに費やした時間を出力
        epoch_duration = time() - epoch_start_time
        logger.info(f"epoch: {epoch+1} - {epoch_duration:4.1f} sec in epoch.")

        # 指定エポック毎にログを出力
        if((epoch+1) % epochs_per_log == 0):
            # 検証ロスを算出
            epoch_valid_loss = calc_total_loss(net, criterion, valid_loader, device)
            logger.info(f"epoch: {epoch+1} - valid_loss: {epoch_valid_loss:7.3f}")

        # 指定エポック毎にパラメータを保存
        if((epoch+1) % epochs_per_save == 0):
            # モデルのパラメータを保存
            for output_dir in model_weight_output_dirs:
                torch.save(net.state_dict(), f"{output_dir}/ssd300_{epoch+1}.pth")
                torch.save(optimizer.state_dict(), f"{output_dir}/optimizer_{epoch+1}.pth")
                torch.save(scheduler.state_dict(), f"{output_dir}/scheduler_{epoch+1}.pth")
