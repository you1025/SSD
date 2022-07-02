import os, sys
from time import time
import torch
from torch.nn.utils import clip_grad_value_
from criterions import calc_total_loss

# Logger の取得
sys.path.append(os.path.join(os.path.dirname(__file__), "../logger/"))
from logger import get_logger
logger = get_logger()

def train(net, train_loader, valid_loader, criterion, optimizer, device, epochs, model_weight_output_dir="./", last_iteration=0):
    net.to(device)
    print(f"device: {device}")

    torch.backends.cudnn.benchmark = True

    iteration = last_iteration + 1
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
            if (iteration % 10 == 0):
                iter_duration = time() - iter_start_time
                logger.info(f"iteration: {iteration:5d} - loss: {loss.item():6.3f}, 10 iter: {iter_duration:4.1f} sec.")
                iter_start_time = time()

            epoch_train_loss += loss.item()
            iteration += 1

        # 毎エポック終了ごとに訓練誤差を出力
        logger.info(f"epoch: {epoch+1} - train_loss: {epoch_train_loss:7.3f}")

        # 1 エポックに費やした時間を出力
        epoch_duration = time() - epoch_start_time
        logger.info(f"epoch: {epoch+1} - {epoch_duration:4.1f} sec in epoch.")

        if((epoch+1) % 10 == 0):
            # 検証ロスを算出
            epoch_valid_loss = calc_total_loss(net, criterion, valid_loader, device)
            logger.info(f"epoch: {epoch+1} - valid_loss: {epoch_valid_loss:7.3f}")

            # モデルのパラメータを保存
            torch.save(net.state_dict(), f"{model_weight_output_dir}/ssd300_{epoch+1}.pth")
