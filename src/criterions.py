import torch
from torch.nn import Module
from torch.nn.functional import smooth_l1_loss, cross_entropy
from match import match

class MultiBoxLoss(Module):
    def __init__(self, jaccard_thresh=0.5, negpos_ratio=3, device="cpu"):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh
        self.negpos_ratio = negpos_ratio
        self.device = device

    def forward(self, predictions, targets):
        loc, conf, dbox_list = predictions

        num_batch   = loc.size(0)
        num_dbox    = dbox_list.size(0)
        num_classes = conf.size(2)

        # match 関数に渡す変数を確保
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

        # ミニバッチ内 idx+1 番目のデータに対応するアノテーション情報を用いて conf_t_label, loc_t に値を設定する
        for idx in range(num_batch):
            # アノテーションから bbox(truths) とラベル(labels)を抽出
            truths = targets[idx][:, :-1].to(self.device)
            labels = targets[idx][:, -1].to(self.device)

            dbox = dbox_list.to(self.device)

            # loc_t と conf_t_label の値を更新
            variance = [0.1, 0.2]
            match(self.jaccard_thresh, truths, dbox, variance, labels, loc_t, conf_t_label, idx)

        # 物体が検出された BBox を示すマスク
        pos_mask = (conf_t_label > 0)

        # 検出された物体全体の Smooth L1 Loss を算出
        loc_p = loc[pos_mask]
        loc_t = loc_t[pos_mask]
        loss_l = smooth_l1_loss(loc_p, loc_t, reduction="sum")


        ## Start: Hard Negative Mining ##
        # 適切な負例を選別するために仮のロスを算出
        loss_c = cross_entropy(
            conf.view(-1, num_classes),
            conf_t_label.view(-1),
            reduction="none"
        )
        loss_c = loss_c.view(num_batch, -1)
        # 正例のロスは 0 に設定して無効化(適切な負例の検索には不要)
        loss_c[pos_mask] = 0
        # loss_c の各値がバッチ内で何番目に大きいのかを算出(=idx_rank)
        _, loss_idx = loss_c.sort(dim=1, descending=True)
        _, idx_rank = loss_idx.sort(dim=1)
        # 負例として用いるサンプル数を指定(正例の negpos_ratio 倍)
        num_pos = pos_mask.sum(dim=1, keepdim=True)
        num_neg = torch.clamp(num_pos * self.negpos_ratio, max=num_dbox)
        # loss の大きい順に負例サンプル数だけ指定するマスクを作成
        neg_mask = (idx_rank < num_neg.expand_as(idx_rank))
        ## End: Hard Negative Mining ##


        # Hard Negative Mining によって指定された負例を含むロスを算出
        conf_hnm = conf[pos_mask + neg_mask]
        conf_t_label_hmn = conf_t_label[pos_mask + neg_mask]
        loss_c_hnm = cross_entropy(conf_hnm, conf_t_label_hmn, reduction="sum")

        # クラス分類のロスは負例の数を考慮していないけど良いのか？？？
        N = num_pos.sum()
        loss_l     /= N
        loss_c_hnm /= N

        return (loss_l, loss_c_hnm)

def calc_total_loss(net, criterion, data_loader, device):
    net.eval()

    total_loss = 0
    with torch.no_grad():
        for images, targets in data_loader:
            # GPU にデータを転送
            images  = images.to(device)
            targets = [ annotation.to(device) for annotation in targets ]

            # 順伝播
            outputs = net(images)

            # ロスを算出
            loss_l, loss_c = criterion(outputs, targets)
            loss = loss_l + loss_c

            total_loss += loss.item()

    return total_loss
