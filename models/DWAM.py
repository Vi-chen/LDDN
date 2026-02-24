import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def build_wavelet_kernels(device=None, dtype=torch.float32):
    """
    返回 2x2 的四个 2D 分析核：LL, LH, HL, HH
    对于 Db1 (Haar)：h0=[1/sqrt2, 1/sqrt2], h1=[-1/sqrt2, 1/sqrt2]
    2D 核是外积：h_row^T * h_col
    """
    s = 1.0 / math.sqrt(2.0)
    h0 = torch.tensor([s, s], dtype=dtype, device=device)      # 低通
    h1 = torch.tensor([-s, s], dtype=dtype, device=device)     # 高频
    # 外积得到 2x2 核
    LL = torch.ger(h0, h0)  # 低-低
    LH = torch.ger(h0, h1)  # 低-高（垂直边更敏感）
    HL = torch.ger(h1, h0)  # 高-低（水平边更敏感）
    HH = torch.ger(h1, h1)  # 高-高（对角边/角点）
    # 形状统一为 (4,1,2,2) 方便后续扩展到 groups=C
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


class DW_GCA(nn.Module):

    def __init__(self, channels, use_fc=True, se_ratio=8):
        super().__init__()
        self.channels = channels
        self.use_fc = use_fc

        # ---------- 高频软阈值参数（3个子带 * C） ----------
        # 用 Sigmoid 将其约束到 [0,1]，再自适应缩放到子带的均值幅度
        self.theta = nn.Parameter(torch.zeros(3, channels, 1, 1))

        # ---------- 方向性门控（3个子带 * C），Sigmoid ----------
        # 用于在阈值后对不同方向的响应进行通道级再加权
        self.dir_gate = nn.Parameter(torch.zeros(3, channels, 1, 1))

        # ---------- 可选的 FC（对 GAP 后的通道向量做线性变换，便于Softmax通道归一化前的可学习投影） ----------
        if use_fc:
            self.fc = nn.Linear(channels, channels, bias=True)

        # ---------- 可学习的子带再分配系数（C x 3），用于将融合后的 H_mix 映射回 LH/HL/HH ----------
        # 用 softmax 保证三者之和为1，提高可解释性与稳定性
        self.sub_redistribute = nn.Parameter(torch.zeros(channels, 3))  # 初始化为0，softmax后约等于均分

        # ---------- 通道注意力（SE） ----------
        self.se = SE(channels, r=se_ratio)

        # ---------- 可学习残差系数 ----------
        self.gamma = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # ---------- 小波核（注册为 buffer，不参与训练） ----------
        filt = build_wavelet_kernels()
        self.register_buffer("w_analysis", filt)   # (4,1,2,2)
        self.register_buffer("w_synthesis", filt)  # Haar 正交：合成=分析

    # ---------- DWT 与 IDWT ----------
    def dwt(self, x):
        """
        x: (B,C,H,W)
        返回：LH, HL, HH, LL
        """
        B, C, H, W = x.shape

        # 填充到偶数尺寸，保证stride=2整除（避免边界丢失）
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        # 组卷积：每个通道使用同一组 4 个滤波器
        weight = self.w_analysis.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        y = F.conv2d(x, weight=weight, bias=None, stride=2, padding=0, groups=C)  # (B,4C,H/2,W/2)

        # 按子带拆分： [LL, LH, HL, HH] 顺序与上面 build 函数保持一致
        y = y.view(B, C, 4, y.size(-2), y.size(-1)).contiguous()
        LL = y[:, :, 0]  # (B,C,h,w)
        LH = y[:, :, 1]
        HL = y[:, :, 2]
        HH = y[:, :, 3]
        return LH, HL, HH, LL

    def idwt(self, LH, HL, HH, LL):
        """
        逆变换：将四个子带重建为 (B,C,H,W)
        """
        B, C, h, w = LL.shape
        # 将 4 个子带 stack 回 (B,4C,h,w)
        y = torch.stack([LL, LH, HL, HH], dim=2).view(B, 4 * C, h, w)

        # 反卷积作为合成滤波器，stride=2
        weight = self.w_synthesis.repeat(C, 1, 1, 1)  # (4C,1,2,2)
        x_rec = F.conv_transpose2d(y, weight=weight, bias=None, stride=2, padding=0, groups=C)
        return x_rec

    # ---------- 高频软阈值 ----------
    @staticmethod
    def soft_threshold(x, thr):
        # 经典 soft-shrinkage： sign(x) * relu(|x| - thr)
        return torch.sign(x) * F.relu(torch.abs(x) - thr)

    def forward(self, x):
        """
        x: [B,C,H,W]
        输出：与输入同形状 [B,C,H,W]
        """
        B, C, H, W = x.shape

        # 1) 小波分解
        LH, HL, HH, LL = self.dwt(x)

        # 2) 高频子带软阈值（自适应按通道尺度化阈值）
        eps = 1e-6
        m_LH = LH.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HL = HL.abs().mean(dim=(2, 3), keepdim=True) + eps
        m_HH = HH.abs().mean(dim=(2, 3), keepdim=True) + eps

        t = torch.sigmoid(self.theta)  # (3,C,1,1), 映射到 0~1
        thr_LH = t[0].unsqueeze(0) * m_LH
        thr_HL = t[1].unsqueeze(0) * m_HL
        thr_HH = t[2].unsqueeze(0) * m_HH

        LH_hat = self.soft_threshold(LH, thr_LH)
        HL_hat = self.soft_threshold(HL, thr_HL)
        HH_hat = self.soft_threshold(HH, thr_HH)

        # 3) 方向性门控（通道级）：为 LH/HL/HH 引入 Sigmoid 门
        g = torch.sigmoid(self.dir_gate)  # (3,C,1,1)
        LH_hat = LH_hat * g[0].unsqueeze(0)   # 加强/抑制对“垂直边”更敏感的通道
        HL_hat = HL_hat * g[1].unsqueeze(0)   # 加强/抑制对“水平边”更敏感的通道
        HH_hat = HH_hat * g[2].unsqueeze(0)   # 加强/抑制对“对角/角点”更敏感的通道

        # 4) 跨子带注意力（逐像素地决定更信任哪个方向）
        # 将三个子带堆成 [B, C, 3, h, w]，对 dim=2 做 softmax，得到跨子带权重
        h = LH_hat.size(-2)
        w = LH_hat.size(-1)
        stack3 = torch.stack([LH_hat, HL_hat, HH_hat], dim=2)  # [B,C,3,h,w]
        # a: [B,C,3,h,w] -> softmax over band-dimension
        attn_band = F.softmax(stack3.abs().mean(dim=1, keepdim=True), dim=2)  # 用跨通道的能量引导（更稳定）
        # 也可以直接对 stack3 做一个 1x1x1 的线性映射后 softmax，这里用能量引导更轻量

        # 按权重加权求和得到融合高频 H_mix（仍为 [B,C,h,w]）
        H_mix = (stack3 * attn_band).sum(dim=2)  # [B,C,h,w]

        # 5) 可学习子带再分配：将 H_mix 映射回 LH2/HL2/HH2，保证可以做 IDWT
        # 对每个通道有3个权重，softmax 到 3 个子带
        w_redis = F.softmax(self.sub_redistribute, dim=1)  # [C,3]
        # reshape 为 [1,C,3,1,1] 便于广播
        w_redis = w_redis.view(1, C, 3, 1, 1)
        # 将 H_mix 拓展成3份，然后乘以通道的再分配系数
        H_mix_exp = H_mix.unsqueeze(2)  # [B,C,1,h,w]
        H_redist = H_mix_exp * w_redis  # [B,C,3,h,w]
        # 拆回三路
        LH2 = H_redist[:, :, 0]
        HL2 = H_redist[:, :, 1]
        HH2 = H_redist[:, :, 2]

        # 6) 逆小波重建，得到重建特征 X_re
        X_re = self.idwt(LH2, HL2, HH2, LL)  # [B,C,H',W'] 与输入 H/W 基本一致

        # 7) 通道注意力：SE（绝对重要性） + Softmax（相对分配）
        # 7.1 Softmax 通道归一化（可选FC增强可分性）
        gap_vec = F.adaptive_avg_pool2d(X_re, 1).view(B, C)  # [B,C]
        if self.use_fc:
            gap_vec = self.fc(gap_vec)                       # [B,C]
        w_softmax = F.softmax(gap_vec, dim=1).view(B, C, 1, 1)  # [B,C,1,1]

        # 7.2 SE 通道权重（Sigmoid）
        w_se = self.se(X_re)  # [B,C,1,1]

        # 8) 残差融合：输入经 softmax 权重强调（更“分配式”），重建分支经 SE 强调（更“绝对式”）
        out = x * w_softmax + self.gamma * (X_re * w_se)
        return out


class DWAM(nn.Module):
    """
    双分支封装：将 DW_GCA 应用于成对特征 (feat_a, feat_b)。
    share_weight=True 时两个时相共享同一组权重，保持对称处理；
    否则各自独立学习，适合跨模态或差异较大的场景。
    可选在内部构造差分分支：diff = feat_a - feat_b，经 DW_GCA 处理后生成 gate，
    用 sigmoid(gate_conv(diff_freq)) 对 out_a/out_b 做调制，仍输出两路特征。
    """

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
            proc = DW_GCA(channels, **kwargs)
            self.proc_a = proc
            self.proc_b = proc
        else:
            self.proc_a = DW_GCA(channels, **kwargs)
            self.proc_b = DW_GCA(channels, **kwargs)

        # 差分分支（默认复用 proc_a 权重，保证对称）
        if self.use_diff:
            if diff_share_weight:
                self.proc_diff = self.proc_a
            else:
                self.proc_diff = DW_GCA(channels, **kwargs)

            if self.fusion == "gate":
                self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
            else:
                raise ValueError(f"Unsupported fusion mode: {self.fusion}")

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_diff:
            # 主干直接走原始特征，不再做单帧频域处理
            out_a = feat_a
            out_b = feat_b
        else:
            out_a = self.proc_a(feat_a)
            out_b = self.proc_b(feat_b)

        if self.use_diff:
            # 差分 -> 频域增强 -> gate
            diff = feat_a - feat_b
            diff_freq = self.proc_diff(diff)

            if self.fusion == "gate":
                gate = torch.sigmoid(self.gate_conv(diff_freq))
                out_a = out_a * gate
                out_b = out_b * gate

        return out_a, out_b

