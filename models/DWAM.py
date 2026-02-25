import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def build_wavelet_kernels(device=None, dtype=torch.float32):
    s = 1.0 / math.sqrt(2.0)
    h0 = torch.tensor([s, s], dtype=dtype, device=device)      
    h1 = torch.tensor([-s, s], dtype=dtype, device=device)     
    
    LL = torch.ger(h0, h0) 
    LH = torch.ger(h0, h1)  
    HL = torch.ger(h1, h0)
    HH = torch.ger(h1, h1)  
    filt = torch.stack([LL, LH, HL, HH], dim=0).unsqueeze(1)
    return filt  # (4,1,2,2)


class SE(nn.Module):

    def __init__(self, channels, r=8):
        super().__init__()
        hidden = max(channels // r, 4)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x):
        # x: [B,C,H,W]
        z = F.adaptive_avg_pool2d(x, 1)         # [B,C,1,1]
        z = F.relu(self.fc1(z), inplace=True)
        z = torch.sigmoid(self.fc2(z))          # [B,C,1,1]
        return z


class HFP(nn.Module):

    def __init__(self, channels, use_fc=True, se_ratio=8):
        super().__init__()
        self.channels = channels
        self.use_fc = use_fc
        self.theta = nn.Parameter(torch.zeros(3, channels, 1, 1))
        self.dir_gate = nn.Parameter(torch.zeros(3, channels, 1, 1))
        if use_fc:
            self.fc = nn.Linear(channels, channels, bias=True)
        self.sub_redistribute = nn.Parameter(torch.zeros(channels, 3)) 
        self.se = SE(channels, r=se_ratio)
        self.gamma = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        filt = build_wavelet_kernels()
        self.register_buffer("w_analysis", filt)   # (4,1,2,2)
        self.register_buffer("w_synthesis", filt) 
    def dwt(self, x):
        B, C, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        weight = self.w_analysis.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        y = F.conv2d(x, weight=weight, bias=None, stride=2, padding=0, groups=C)  # (B,4C,H/2,W/2)

        y = y.view(B, C, 4, y.size(-2), y.size(-1)).contiguous()
        LL = y[:, :, 0]  # (B,C,h,w)
        LH = y[:, :, 1]
        HL = y[:, :, 2]
        HH = y[:, :, 3]
        return LH, HL, HH, LL

    def idwt(self, LH, HL, HH, LL):
        B, C, h, w = LL.shape
        y = torch.stack([LL, LH, HL, HH], dim=2).view(B, 4 * C, h, w)
        weight = self.w_synthesis.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        x_rec = F.conv_transpose2d(y, weight=weight, bias=None, stride=2, padding=0, groups=C)
        return x_rec

    @staticmethod
    def soft_threshold(x, thr):
        # soft-shrinkage： sign(x) * relu(|x| - thr)
        return torch.sign(x) * F.relu(torch.abs(x) - thr)

    def forward(self, x):
        B, C, H, W = x.shape
        LH, HL, HH, LL = self.dwt(x)

        eps = 1e-6
        m_LH = LH.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HL = HL.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HH = HH.abs().mean(dim=(2, 3), keepdim=True) + eps

        t = torch.sigmoid(self.theta)
        thr_LH = t[0].unsqueeze(0) * m_LH
        thr_HL = t[1].unsqueeze(0) * m_HL
        thr_HH = t[2].unsqueeze(0) * m_HH

        LH_hat = self.soft_threshold(LH, thr_LH)
        HL_hat = self.soft_threshold(HL, thr_HL)
        HH_hat = self.soft_threshold(HH, thr_HH)

        g = torch.sigmoid(self.dir_gate)  # (3,C,1,1)
        LH_hat = LH_hat * g[0].unsqueeze(0)   
        HL_hat = HL_hat * g[1].unsqueeze(0)   
        HH_hat = HH_hat * g[2].unsqueeze(0)   

        h = LH_hat.size(-2)
        w = LH_hat.size(-1)
        stack3 = torch.stack([LH_hat, HL_hat, HH_hat], dim=2)  # [B,C,3,h,w]
        # a: [B,C,3,h,w] -> softmax over band-dimension
        attn_band = F.softmax(stack3.abs().mean(dim=1, keepdim=True), dim=2)  

        H_mix = (stack3 * attn_band).sum(dim=2)  # [B,C,h,w]

        w_redis = F.softmax(self.sub_redistribute, dim=1)  # [C,3]
        w_redis = w_redis.view(1, C, 3, 1, 1)
        H_mix_exp = H_mix.unsqueeze(2)  # [B,C,1,h,w]
        H_redist = H_mix_exp * w_redis  # [B,C,3,h,w]
        LH2 = H_redist[:, :, 0]
        HL2 = H_redist[:, :, 1]
        HH2 = H_redist[:, :, 2]

        X_re = self.idwt(LH2, HL2, HH2, LL)  

        gap_vec = F.adaptive_avg_pool2d(X_re, 1).view(B, C)  # [B,C]
        if self.use_fc:
            gap_vec = self.fc(gap_vec)                       # [B,C]
        w_softmax = F.softmax(gap_vec, dim=1).view(B, C, 1, 1)  # [B,C,1,1]

        w_se = self.se(X_re)  # [B,C,1,1]

        out = x * w_softmax + self.gamma * (X_re * w_se)
        return out


class DWAM(nn.Module):

    def __init__(
        self,
        channels: int,
        *,
        share_weight: bool = True,
        use_diff: bool = True,
        diff_share_weight: bool = True,
        fusion: str = "gate",
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_diff = use_diff
        self.fusion = fusion

        if share_weight:
            proc = HFP(channels, **kwargs)
            self.proc_a = proc
            self.proc_b = proc
        else:
            self.proc_a = HFP(channels, **kwargs)
            self.proc_b = HFP(channels, **kwargs)

        if self.use_diff:
            if diff_share_weight:
                self.proc_diff = self.proc_a
            else:
                self.proc_diff = HFP(channels, **kwargs)

            if self.fusion == "gate":
                self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
            else:
                raise ValueError(f"Unsupported fusion mode: {self.fusion}")

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_diff:
            out_a = feat_a
            out_b = feat_b
        else:
            out_a = self.proc_a(feat_a)
            out_b = self.proc_b(feat_b)

        if self.use_diff:
            diff = feat_a - feat_b
            diff_freq = self.proc_diff(diff)

            if self.fusion == "gate":
                gate = torch.sigmoid(self.gate_conv(diff_freq))
                out_a = out_a * gate
                out_b = out_b * gate

        return out_a, out_b

