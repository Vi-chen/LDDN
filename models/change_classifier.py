import time
import torchvision
from typing import Tuple
from .layers import Up, Classifier
from torch.nn import Module, ModuleList
import torch
import torch.nn as nn

from models.DWAM import DWAM
from models.MIM import DBlock
from models.Fusion_module import DSFM

class ChangeClassifier(Module):
    def __init__(
        self,
        num_classes,
        num,
        bkbn_name="efficientnet_b4",
        pretrained=True,
        output_layer_bkbn="4",
        freeze_backbone=False,
        backbone_only=False,
        use_frequency=True,
        use_interaction=True,
        use_mdb=True,
    ):
        super().__init__()
        self.backbone_only = backbone_only
        self.use_frequency = use_frequency
        self.use_interaction = use_interaction
        self.use_mdb = use_mdb

        # Load the pretrained backbone according to parameters:
        self.backbone = get_backbone(
            bkbn_name, pretrained, output_layer_bkbn, freeze_backbone
        )

        self.stage_channels = [24, 32, 56, 112]
        self.refined_channels = [14, 28, 56, 112]

        self.tice = ModuleList(
            [
                DSFM(in_ch, out_ch)
                for in_ch, out_ch in zip(self.stage_channels, self.refined_channels)
            ]
        )

        # frequency refinement applied before temporal interaction (per-scale dual-branch)
        self.freq_blocks = ModuleList(
            [DWAM(ch, use_fc=True, se_ratio=8) for ch in self.stage_channels]
        )

        self.mdb_refiners = ModuleList(
            [
                DBlock(ch)
                for ch in self.refined_channels
                # DBlock(ch) if idx >= 2 else nn.Identity()
                # for idx, ch in enumerate(self.refined_channels)
            ]
        )

        self.up = ModuleList(
            [
                Up(self.refined_channels[3], self.refined_channels[2]),
                Up(self.refined_channels[2], self.refined_channels[1]),
                Up(self.refined_channels[1], self.refined_channels[0]),
                Up(self.refined_channels[0], self.refined_channels[0] // 2),
            ]
        )

        self.diff_projection = ModuleList(
            [
                nn.Identity()
                if in_ch == out_ch
                else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
                for in_ch, out_ch in zip(self.stage_channels, self.refined_channels)
            ]
        )

        # Final classification layer:
        self.classify = Classifier(self.refined_channels[0] // 2, num_classes)

    def forward(self, x1, x2):
        # forward backbone resnet
        features_1, features_2 = self.encode(x1, x2)
        if self.backbone_only:
            diff = self._project_features(self._simple_difference(features_1, features_2))
            latents = self.decoder(diff)
            return self.classify(latents)

        feat_a, feat_b = features_1, features_2

        if self.use_frequency:
            feat_a, feat_b = self.frequency(feat_a, feat_b)

        if self.use_interaction:
            diff = self.interaction(feat_a, feat_b)
        else:
            diff = self._project_features(self._simple_difference(feat_a, feat_b))

        if self.use_mdb:
            diff = self.refine(diff)

        latents = self.decoder(diff)
        pred = self.classify(latents)
        return pred

    def interaction(self, x1, x2):
        d = []
        for i in range(0, 4):
            x = self.tice[i](x1[i], x2[i])
            d.append(x)
        return d

    def refine(self, features):
        refined = []
        for idx, feat in enumerate(features):
            refined.append(self.mdb_refiners[idx](feat))
        return refined

    def frequency(self, feats_a, feats_b) -> Tuple[list, list]:
        out_a, out_b = [], []
        for idx, (fa, fb) in enumerate(zip(feats_a, feats_b)):
            ea, eb = self.freq_blocks[idx](fa, fb)
            out_a.append(ea)
            out_b.append(eb)
        return out_a, out_b

    def _simple_difference(self, x1_feats, x2_feats):
        diffs = []
        for feat1, feat2 in zip(x1_feats, x2_feats):
            diffs.append(torch.abs(feat1 - feat2))
        return diffs

    def _project_features(self, features):
        projected = []
        for idx, feat in enumerate(features):
            projected.append(self.diff_projection[idx](feat))
        return projected

    def encode(self, x1, x2):
        x1_downsample = []
        x2_downsample = []
        for num, layer in enumerate(self.backbone):
            x1 = layer(x1)
            x2 = layer(x2)
            if num != 0:
                x1_downsample.append(x1)
                x2_downsample.append(x2)

        return x1_downsample, x2_downsample

    def decoder(self, features):
        x = self.up[0](features[3])
        x = self.up[1](x, features[2])
        x = self.up[2](x, features[1])
        x = self.up[3](x, features[0])
        return x


def get_backbone(
    bkbn_name, pretrained, output_layer_bkbn, freeze_backbone):
    # The whole model:
    entire_model = getattr(torchvision.models, bkbn_name)(pretrained=pretrained).features

    # Slicing it:
    derived_model = ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break

    # Freezing the backbone weights:
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model


if __name__ == '__main__':
    x1 = torch.randn(1, 3, 256, 256).cuda().float()
    x2 = torch.randn(1, 3, 256, 256).cuda().float()
    net1 = ChangeClassifier(num_classes=2, num=3).cuda()
    s1 = net1(x1, x2)
    print(s1.shape)

    # from fvcore.nn import FlopCountAnalysis
    # flops = FlopCountAnalysis(net1, (x1, x2))
    # total = sum([param.nelement() for param in net1.parameters()])
    # print("Params_Num: %.2fM" % (total/1e6))
    # print(flops.total()/1e9)
    with torch.no_grad():
        for _ in range(10):
            _ = net1(x1, x2)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = net1(x1, x2)
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / 100
    print(f"平均推理时间：{avg_inference_time}秒")
