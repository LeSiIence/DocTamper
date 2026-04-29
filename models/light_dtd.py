"""
LightDTD: 轻量化学生网络，用于篡改检测与知识蒸馏。
- VPH_Light: MobileNetV3-Small 前三个 Stage 作为视觉多尺度特征
- FPH_Light: DCT + 量化表嵌入，深度可分离卷积，输出 64 通道
- Adaptation Layer: 融合特征通道对齐到教师融合层 192 通道
- Decoder_Light: 增强型 FPN 解码，DSConv + LightSCSE 注意力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from typing import List, Tuple

TEACHER_FUSION_CHANNELS = 192


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride,
                            padding=kernel_size // 2, groups=in_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn_pw = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_dw(self.dw(x)), inplace=True)
        return F.relu(self.bn_pw(self.pw(x)), inplace=True)


class LightSCSE(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        mid = max(in_channels // reduction, 1)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cSE(x) + x * self.sSE(x)


class AddCoords(nn.Module):
    def __init__(self, with_r: bool = True):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        B, _, x_dim, y_dim = input_tensor.size()
        xx = torch.arange(x_dim, dtype=input_tensor.dtype, device=input_tensor.device)
        yy = torch.arange(y_dim, dtype=input_tensor.dtype, device=input_tensor.device)
        xx_c, yy_c = torch.meshgrid(xx, yy, indexing="ij")
        xx_c = xx_c / (x_dim - 1) * 2 - 1
        yy_c = yy_c / (y_dim - 1) * 2 - 1
        xx_c = xx_c.unsqueeze(0).unsqueeze(0).expand(B, 1, x_dim, y_dim)
        yy_c = yy_c.unsqueeze(0).unsqueeze(0).expand(B, 1, x_dim, y_dim)
        ret = torch.cat((input_tensor, xx_c, yy_c), dim=1)
        if self.with_r:
            rr = torch.sqrt((xx_c - 0.5).pow(2) + (yy_c - 0.5).pow(2))
            ret = torch.cat([ret, rr], dim=1)
        return ret


class VPH_Light(nn.Module):
    STAGE_INDICES = (1, 4, 9)

    def __init__(self, pretrained: bool = False):
        super().__init__()
        backbone = tv_models.mobilenet_v3_small(
            weights="IMAGENET1K_V1" if pretrained else None)
        self.features = backbone.features
        self._stage_indices = list(self.STAGE_INDICES)
        self._out_channels: List[int] = [16, 40, 96]

    @property
    def out_channels(self) -> List[int]:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self._stage_indices:
                outs.append(x)
        return outs


class FPH_Light(nn.Module):
    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.obembed = nn.Embedding(21, 21)
        nn.init.eye_(self.obembed.weight)
        self.qtembed = nn.Embedding(64, 16)
        self.conv1 = nn.Sequential(
            nn.Conv2d(21, 64, kernel_size=3, stride=1, dilation=8, padding=8),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(16, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.addcoords = AddCoords()
        # 35 = 16*2 + 3(coords) → 8x8 stride 下采样 + 2 层 DSConv
        self.conv0 = nn.Sequential(
            nn.Conv2d(35, 64, kernel_size=8, stride=8, padding=0, bias=False),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv2d(64, 64, 3),
            DepthwiseSeparableConv2d(64, out_channels, 3),
        )
        self._out_channels = out_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor, qtable: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.clamp(0, 20).long()
        x = self.conv2(self.conv1(self.obembed(x).permute(0, 3, 1, 2).contiguous()))
        B, C, H, W = x.shape
        qtable = qtable.view(B, -1).long()
        x_blocks = x.reshape(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 3, 5, 2, 4)
        qemb = self.qtembed(qtable.unsqueeze(-1).unsqueeze(-1))
        if qemb.dim() == 5:
            qemb = qemb.squeeze(-1).squeeze(-1)
        qemb = qemb.transpose(1, 2).contiguous().view(B, 16, 8, 8)
        qemb = qemb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, H // 8, W // 8)
        x_weighted = (x_blocks * qemb).permute(0, 1, 4, 2, 5, 3).reshape(B, C, H, W)
        fused = torch.cat([x_weighted, x], dim=1)
        return self.conv0(self.addcoords(fused))


class AdaptationLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = TEACHER_FUSION_CHANNELS):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Decoder_Light(nn.Module):
    """增强型 FPN 解码器：DSConv + LightSCSE 注意力。"""

    def __init__(
        self,
        encoder_channels: List[int],
        fused_channels: int = TEACHER_FUSION_CHANNELS,
        decoder_channels: Tuple[int, ...] = (128, 64, 32),
        out_channels: int = 1,
    ):
        super().__init__()
        # c3 + fused → reduce
        self.reduce_c3 = nn.Sequential(
            nn.Conv2d(encoder_channels[2] + fused_channels, decoder_channels[0], 1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.ReLU(inplace=True),
        )
        # stage 2: upsample + concat c2
        self.reduce_c2 = nn.Conv2d(encoder_channels[1], decoder_channels[1], 1, bias=False)
        self.up_block2 = nn.Sequential(
            DepthwiseSeparableConv2d(decoder_channels[0] + decoder_channels[1], decoder_channels[1]),
            LightSCSE(decoder_channels[1]),
        )
        # stage 1: upsample + concat c1
        self.reduce_c1 = nn.Conv2d(encoder_channels[0], decoder_channels[2], 1, bias=False)
        self.up_block1 = nn.Sequential(
            DepthwiseSeparableConv2d(decoder_channels[1] + decoder_channels[2], decoder_channels[2]),
            LightSCSE(decoder_channels[2]),
        )
        self.head = nn.Conv2d(decoder_channels[2], out_channels, 1)

    def forward(self, feats: List[torch.Tensor], fused: torch.Tensor) -> torch.Tensor:
        c1, c2, c3 = feats[0], feats[1], feats[2]
        if fused.shape[2:] != c3.shape[2:]:
            fused = F.interpolate(fused, size=c3.shape[2:], mode="bilinear", align_corners=False)
        x = self.reduce_c3(torch.cat([c3, fused], dim=1))

        c2_r = self.reduce_c2(c2)
        x = F.interpolate(x, size=c2.shape[2:], mode="bilinear", align_corners=False)
        x = self.up_block2(torch.cat([x, c2_r], dim=1))

        c1_r = self.reduce_c1(c1)
        x = F.interpolate(x, size=c1.shape[2:], mode="bilinear", align_corners=False)
        x = self.up_block1(torch.cat([x, c1_r], dim=1))

        return self.head(x)


class LightDTD(nn.Module):
    def __init__(
        self,
        fph_out_channels: int = 64,
        teacher_fusion_channels: int = TEACHER_FUSION_CHANNELS,
        pretrained_vph: bool = False,
        classes: int = 1,
    ):
        super().__init__()
        self.vph = VPH_Light(pretrained=pretrained_vph)
        self.fph = FPH_Light(out_channels=fph_out_channels)
        visual_ch = self.vph.out_channels[-1]  # 96
        fused_ch = visual_ch + self.fph.out_channels  # 96 + 64 = 160
        self.fuse_conv = nn.Sequential(
            LightSCSE(fused_ch, reduction=8),
            DepthwiseSeparableConv2d(fused_ch, fused_ch, 3),
        )
        self.adapt_layer = AdaptationLayer(fused_ch, teacher_fusion_channels)
        self.decoder = Decoder_Light(
            encoder_channels=self.vph.out_channels,
            fused_channels=teacher_fusion_channels,
            decoder_channels=(128, 64, 32),
            out_channels=classes,
        )

    def forward(
        self,
        x: torch.Tensor,
        dct: torch.Tensor,
        qt: torch.Tensor,
    ) -> torch.Tensor:
        visual_feats = self.vph(x)
        fph_out = self.fph(dct, qt)
        c3 = visual_feats[2]
        if fph_out.shape[2:] != c3.shape[2:]:
            fph_out = F.interpolate(fph_out, size=c3.shape[2:], mode="bilinear", align_corners=False)
        fused = torch.cat([c3, fph_out], dim=1)
        fused = self.fuse_conv(fused)
        fused_adapted = self.adapt_layer(fused)
        logits = self.decoder(visual_feats, fused_adapted)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        return logits

    def get_adapted_fusion(self, x: torch.Tensor, dct: torch.Tensor, qt: torch.Tensor) -> torch.Tensor:
        visual_feats = self.vph(x)
        fph_out = self.fph(dct, qt)
        c3 = visual_feats[2]
        if fph_out.shape[2:] != c3.shape[2:]:
            fph_out = F.interpolate(fph_out, size=c3.shape[2:], mode="bilinear", align_corners=False)
        fused = torch.cat([c3, fph_out], dim=1)
        fused = self.fuse_conv(fused)
        return self.adapt_layer(fused)
