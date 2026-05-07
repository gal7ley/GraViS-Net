
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet, BasicBlock


class HalfResNet18(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=1000)

        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(BasicBlock, 32, 2)
        self.layer2 = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 2, stride=2)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# --- 1. CBAM Module ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 防止除零错误 (如果通道数很小)
        hidden_planes = max(in_planes // ratio, 4)
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channel, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        s_mask = self.sa(x)
        x = x * s_mask
        return x, s_mask


# --- 2. SE-Block ---
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.fc(x)
        return x * y, y


# --- 3. Adapter Module (不变) ---
class FeatureAdapter(nn.Module):
    def __init__(self, in_channel, out_channel=256):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x.view(x.size(0), 1, -1)

class SynergyNetV11(nn.Module):
    def __init__(self, num_classes, fusion_dim=256):
        super().__init__()

        # === RGB Branch: ResNet50 + CBAM (保持不变) ===
        r50 = models.resnet50(weights='IMAGENET1K_V1')
        layers = list(r50.children())
        self.rgb_stem = nn.Sequential(*layers[:4])  # 64

        self.rgb_l1 = layers[4]  # 256
        self.cbam1 = CBAM(256)

        self.rgb_l2 = layers[5]  # 512
        self.cbam2 = CBAM(512)

        self.rgb_l3 = layers[6]  # 1024
        self.cbam3 = CBAM(1024)

        self.rgb_l4 = layers[7]  # 2048
        self.cbam4 = CBAM(2048)

        r18 = HalfResNet18()
        r18 = models.resnet18(weights='IMAGENET1K_V1')


        layers18 = list(r18.children())
        self.ivg_stem = nn.Sequential(*layers18[:4])  # 32
        self.ivg_l1 = layers18[4]  # 32
        self.ivg_l2 = layers18[5]  # 64
        self.ivg_l3 = layers18[6]  # 128
        self.ivg_l4 = layers18[7]  # 256

        # === Adapters (IVG部分输入维度减半) ===
        self.adapt_rgb2 = FeatureAdapter(512, fusion_dim)
        self.adapt_rgb3 = FeatureAdapter(1024, fusion_dim)
        self.adapt_rgb4 = FeatureAdapter(2048, fusion_dim)

        # IVG Adapters 修改: 128->64, 256->128, 512->256
        self.adapt_ivg2 = FeatureAdapter(64, fusion_dim)
        self.adapt_ivg3 = FeatureAdapter(128, fusion_dim)
        self.adapt_ivg4 = FeatureAdapter(256, fusion_dim)

        # === Deep Supervision Head (RGB L3) ===
        self.ds_head = nn.Linear(fusion_dim, num_classes)

        # === Fusion & Head ===
        total_dim = fusion_dim * 6
        self.se_block = SEBlock(total_dim)

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        self._init_new_weights()

    def _init_new_weights(self):
        for m in self.modules():
            if isinstance(m, (FeatureAdapter, SEBlock, CBAM)):
                for sub_m in m.modules():
                    if isinstance(sub_m, nn.Conv2d):
                        nn.init.kaiming_normal_(sub_m.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(sub_m, nn.Linear):
                        nn.init.normal_(sub_m.weight, 0, 0.01)
        nn.init.normal_(self.ds_head.weight, 0, 0.01)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, rgb_img, ivg_img):
        # --- RGB Flow (不变) ---
        x = self.rgb_stem(rgb_img)
        x = self.cbam1(self.rgb_l1(x))[0]

        x = self.cbam2(self.rgb_l2(x))[0]
        f_rgb2 = self.adapt_rgb2(x)

        x = self.cbam3(self.rgb_l3(x))[0]
        f_rgb3 = self.adapt_rgb3(x)

        aux_out = None
        if self.training:
            aux_out = self.ds_head(f_rgb3)

        x, rgb_spatial_mask = self.cbam4(self.rgb_l4(x))
        f_rgb4 = self.adapt_rgb4(x)

        # --- IVG Flow (通道减半，逻辑不变) ---
        y = self.ivg_stem(ivg_img)
        y = self.ivg_l1(y)

        y = self.ivg_l2(y)
        f_ivg2 = self.adapt_ivg2(y)  # Input: 64 -> 256

        y = self.ivg_l3(y)
        f_ivg3 = self.adapt_ivg3(y)  # Input: 128 -> 256

        y = self.ivg_l4(y)
        f_ivg4 = self.adapt_ivg4(y)  # Input: 256 -> 256

        # --- Fusion  ---
        concat_feat = torch.cat([f_rgb2, f_rgb3, f_rgb4, f_ivg2, f_ivg3, f_ivg4], dim=1)
        feat_weighted, se_weights = self.se_block(concat_feat)
        main_out = self.classifier(feat_weighted)

        # --- Debug Info ---
        probe_info = {
            'rgb_l4_mask_mean': rgb_spatial_mask.mean().item(),
            'se_w_mean': se_weights.mean().item(),
            'se_w_rgb': se_weights[:, :256 * 3].mean().item(),
            'se_w_ivg': se_weights[:, 256 * 3:].mean().item()
        }

        return main_out, aux_out, probe_info, feat_weighted

    def get_rgb_params(self):
        return list(self.rgb_stem.parameters()) + \
            list(self.rgb_l1.parameters()) + list(self.cbam1.parameters()) + \
            list(self.rgb_l2.parameters()) + list(self.cbam2.parameters()) + \
            list(self.rgb_l3.parameters()) + list(self.cbam3.parameters()) + \
            list(self.rgb_l4.parameters()) + list(self.cbam4.parameters())

    def get_new_params(self):
        return list(self.ivg_stem.parameters()) + \
            list(self.ivg_l1.parameters()) + \
            list(self.ivg_l2.parameters()) + \
            list(self.ivg_l3.parameters()) + \
            list(self.ivg_l4.parameters()) + \
            list(self.adapt_rgb2.parameters()) + list(self.adapt_rgb3.parameters()) + list(
                self.adapt_rgb4.parameters()) + \
            list(self.adapt_ivg2.parameters()) + list(self.adapt_ivg3.parameters()) + list(
                self.adapt_ivg4.parameters()) + \
            list(self.ds_head.parameters()) + \
            list(self.se_block.parameters()) + \
            list(self.classifier.parameters())