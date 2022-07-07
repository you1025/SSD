import os
import numpy as np
from itertools import product
import torch
from torch.nn import Module, ModuleList, Parameter, Conv2d, MaxPool2d, ReLU
from torch.nn.functional import relu
from torch.nn.init import kaiming_normal_, constant_

ROOT_DIR = os.path.join(os.path.dirname(__file__), "..")

class SSD(Module):
    def __init__(self, phase, config):
        super(SSD, self).__init__()

        self.phase = phase
        self.num_classes = config["num_classes"]

        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc = make_loc(self.num_classes)
        self.conf = make_conf(self.num_classes)

        self.dbox_list = DBox(config).make_dbox_list()

        if phase == "inference":
            self.detect = Detect()

    def forward(self, x):
        for k in range(23):
            x = self.vgg[k](x)

        source1 = self.L2Norm(x)
        loc1  = self.loc[0](source1).permute(0, 2, 3, 1).contiguous()
        conf1 = self.conf[0](source1).permute(0, 2, 3, 1).contiguous()

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        source2 = x
        loc2  = self.loc[1](source2).permute(0, 2, 3, 1).contiguous()
        conf2 = self.conf[1](source2).permute(0, 2, 3, 1).contiguous()

        x = relu(self.extras[0](x), inplace=True)
        x = relu(self.extras[1](x), inplace=True)
        source3 = x
        loc3  = self.loc[2](source3).permute(0, 2, 3, 1).contiguous()
        conf3 = self.conf[2](source3).permute(0, 2, 3, 1).contiguous()

        x = relu(self.extras[2](x), inplace=True)
        x = relu(self.extras[3](x), inplace=True)
        source4 = x
        loc4  = self.loc[3](source4).permute(0, 2, 3, 1).contiguous()
        conf4 = self.conf[3](source4).permute(0, 2, 3, 1).contiguous()

        x = relu(self.extras[4](x), inplace=True)
        x = relu(self.extras[5](x), inplace=True)
        source5 = x
        loc5  = self.loc[4](source5).permute(0, 2, 3, 1).contiguous()
        conf5 = self.conf[4](source5).permute(0, 2, 3, 1).contiguous()

        x = relu(self.extras[6](x), inplace=True)
        x = relu(self.extras[7](x), inplace=True)
        source6 = x
        loc6  = self.loc[5](source6).permute(0, 2, 3, 1).contiguous()
        conf6 = self.conf[5](source6).permute(0, 2, 3, 1).contiguous()

        loc = torch.cat([
            loc1.view(loc1.size(0), -1),
            loc2.view(loc2.size(0), -1),
            loc3.view(loc3.size(0), -1),
            loc4.view(loc4.size(0), -1),
            loc5.view(loc5.size(0), -1),
            loc6.view(loc6.size(0), -1)
        ], dim=1)
        loc = loc.view(loc.size(0), -1, 4)

        conf = torch.cat([
            conf1.view(conf1.size(0), -1),
            conf2.view(conf2.size(0), -1),
            conf3.view(conf3.size(0), -1),
            conf4.view(conf4.size(0), -1),
            conf5.view(conf5.size(0), -1),
            conf6.view(conf6.size(0), -1)
        ], dim=1)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        if self.phase == "inference":
            return self.detect.forward(loc, conf, self.dbox_list)
        else:
            return (loc, conf, self.dbox_list)

    def init_parameters(self, model_weight_path):
        # 訓練済みのパラメータを設定
        weight_path = os.path.join(ROOT_DIR, model_weight_path)
        weights     = torch.load(weight_path)
        self.vgg.load_state_dict(weights)

        # VGG 以外のパラメータを初期化(He)
        self.extras.apply(self.__weights_init)
        self.loc.apply(self.__weights_init)
        self.conf.apply(self.__weights_init)

    def __weights_init(self, layer):
        if isinstance(layer, Conv2d):
            kaiming_normal_(layer.weight.data)
            if layer.bias is not None:
                constant_(layer.bias, 0.0)

class DBox():
    def __init__(self, config):
        self.config = config

    def make_dbox_list(self):
        image_size = self.config["image_size"]

        boxes = []
        for source_id in range(6):
            # 元画像(300)に対する region_size(ex. 8) が 0〜1.0 にスケーリングされる際の比率
            scale_ratio = image_size / self.config["region_sizes"][source_id]

            # 画像の分割数
            n_feature_map_regions = self.config["feature_map_regions"][source_id]
            for (i, j) in product(range(n_feature_map_regions), repeat=2):
                # 0〜1 スケールにおける中心点の算出
                cx = (j + 0.5) / scale_ratio
                cy = (i + 0.5) / scale_ratio

                # 小さめ正方形
                min_size = self.config["min_sizes"][source_id]
                s_k = min_size / image_size
                boxes.append([cx, cy, s_k, s_k])

                # 大きめ正方形
                max_size = self.config["max_sizes"][source_id]
                s_k_prime = s_k * np.sqrt(max_size / min_size)
                boxes.append([cx, cy, s_k_prime, s_k_prime])

                # 長方形
                for aspect_ratio in self.config["aspect_ratios"][source_id]:
                    sqrt_aspect_ratio = np.sqrt(aspect_ratio)
                    boxes.append([cx, cy, s_k * sqrt_aspect_ratio, s_k / sqrt_aspect_ratio]) # 横長
                    boxes.append([cx, cy, s_k / sqrt_aspect_ratio, s_k * sqrt_aspect_ratio]) # 縦長

        boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes.clamp_(min=0, max=1)

        return boxes

class Detect(torch.autograd.Function):
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh

    def forward(self, loc, conf, dbox_list):
        num_batch = loc.size(0)
        num_dbox = loc.size(1)
        num_classes = conf.size(2)

        # 全クラスの値の合計が 1 になるように規格化
        # confidences[*][*].sum() が 1 となる
        confidences = torch.nn.Softmax(dim=-1)(conf)

        output = torch.zeros(num_batch, num_classes, self.top_k, 5)
        for i in range(num_batch):
            decoded_boxes = decode(loc[i], dbox_list)

            conf_scores = confidences[i].transpose(0, 1)#.clone()
            for class_id in range(1, num_classes):
                conf_mask = conf_scores[class_id].gt(self.conf_thresh)
                scores = conf_scores[class_id][conf_mask]
                boxes = decoded_boxes[conf_mask]

                if scores.nelement() == 0:
                    continue

                target_ids = nm_suppression(boxes, scores, self.nms_thresh, self.top_k)
                count = target_ids.size(0)

                output[i, class_id, :count] = torch.cat([
                    scores[target_ids].unsqueeze(1),
                    boxes[target_ids]
                ], dim=1)

        return output


def make_vgg():
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "C", 512, 512, 512, "M", 512, 512, 512]

    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(MaxPool2d(kernel_size=2, stride=2))
        elif v == "C":
            layers.append(MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        else:
            layers.append(Conv2d(in_channels, out_channels=v, kernel_size=3, padding=1))
            layers.append(ReLU(inplace=True))
            in_channels = v

    layers.extend([
        MaxPool2d(kernel_size=3, stride=1, padding=1),
        Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
        ReLU(inplace=True),
        Conv2d(1024, 1024, kernel_size=1),
        ReLU(inplace=True)
    ])

    return ModuleList(layers)

def make_extras():
    return ModuleList([
        Conv2d(1024, 256, kernel_size=1),
        Conv2d( 256, 512, kernel_size=3, stride=2, padding=1),
        Conv2d( 512, 128, kernel_size=1),
        Conv2d( 128, 256, kernel_size=3, stride=2, padding=1),
        Conv2d( 256, 128, kernel_size=1),
        Conv2d( 128, 256, kernel_size=3),
        Conv2d( 256, 128, kernel_size=1),
        Conv2d( 128, 256, kernel_size=3)
    ])

def make_loc(n_classes=21):
    return ModuleList([
        Conv2d( 512, 4 * 4, kernel_size=3, padding=1),
        Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
        Conv2d( 512, 6 * 4, kernel_size=3, padding=1),
        Conv2d( 256, 6 * 4, kernel_size=3, padding=1),
        Conv2d( 256, 4 * 4, kernel_size=3, padding=1),
        Conv2d( 256, 4 * 4, kernel_size=3, padding=1)
    ])

def make_conf(n_classes=21):
    return ModuleList([
        Conv2d( 512, 4 * n_classes, kernel_size=3, padding=1),
        Conv2d(1024, 6 * n_classes, kernel_size=3, padding=1),
        Conv2d( 512, 6 * n_classes, kernel_size=3, padding=1),
        Conv2d( 256, 6 * n_classes, kernel_size=3, padding=1),
        Conv2d( 256, 4 * n_classes, kernel_size=3, padding=1),
        Conv2d( 256, 4 * n_classes, kernel_size=3, padding=1)
    ])

class L2Norm(Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()

        self.weight = Parameter(torch.Tensor(input_channels))
        torch.nn.init.constant_(self.weight, scale)

    def forward(self, x):
        EPSILON = 1e-10

        norm = torch.norm(x, dim=1, keepdim=True) + EPSILON
        x_normalized = torch.div(x, norm)

        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x_normalized)
        x_weighted = weights * x_normalized

        return x_weighted

def decode(loc, dbox_list):
    boxes = torch.cat([
        dbox_list[:, :2] + 0.1 * loc[:, :2] * dbox_list[:, 2:], # cx, cy
        dbox_list[:, 2:] * torch.exp(0.2 * loc[:, 2:])
    ], dim=1)

    # 気持ち悪いけど速度優先(新規インスタンスを確保するより速度面で有利)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    target_ids = []

    areas = torch.mul(
        boxes[:, 2] - boxes[:, 0], # xmax - xmin
        boxes[:, 3] - boxes[:, 1]  # ymax - ymin
    )

    sorted_indexes = scores.sort(descending=True)[1][:top_k]

    remaining_indexes = sorted_indexes.clone()
    while remaining_indexes.numel() > 0:
        # 最も信頼度の高い(=先頭の) box を取得
        target_index = remaining_indexes[0]
        target_box = boxes[target_index].detach()

        target_ids.append(target_index)

        # 最も信頼度の高い box を除いた残りのインデックス一覧
        remaining_indexes = remaining_indexes[1:]

        # box が残っていない場合は終了
        if remaining_indexes.size(0) == 0:
            break

        # (現時点で)最も信頼度の高い box とその他の box との交わり部分の座標を算出
        remaining_boxes = torch.index_select(boxes.detach(), dim=0, index=remaining_indexes)
        _x1 = torch.clamp(remaining_boxes[:, 0], min=target_box[0])
        _y1 = torch.clamp(remaining_boxes[:, 1], min=target_box[1])
        _x2 = torch.clamp(remaining_boxes[:, 2], max=target_box[2])
        _y2 = torch.clamp(remaining_boxes[:, 3], max=target_box[3])

        # Intersection
        _w = torch.clamp(_x2 - _x1, min=0.0)
        _h = torch.clamp(_y2 - _y1, min=0.0)
        intersection = _w * _h

        # Union
        remaining_areas = torch.index_select(areas, dim=0, index=remaining_indexes)
        union = (remaining_areas - intersection) + areas[target_index]

        # IoU
        iou = intersection / union

        # IoU が閾値以上の box を排除
        # IoU が閾値以上のケースは各 box が同一の物体を指しているとし、その中で最も信頼度の高い box のみを残す
        remaining_indexes = remaining_indexes[iou.le(overlap)]

    return torch.tensor(target_ids)
