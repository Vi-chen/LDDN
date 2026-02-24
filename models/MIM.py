from typing import Optional, Sequence
import torch.nn as nn
import torch


def autopad(kernel_size: int, padding: Optional[int] = None, dilation: int = 1) -> int:
    """根据卷积核大小和扩张率计算填充大小。"""
    if padding is None:
        padding = (kernel_size - 1) * dilation // 2  # 如果没有指定填充，计算合适的填充大小
    return padding  # 返回计算的填充大小

def make_divisible(value: int, divisor: int = 8) -> int:
    """将值调整为可被指定除数整除。"""
    return int((value + divisor // 2) // divisor * divisor) # 调整值为最接近的可整除数

class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None):
        super().__init__()
        layers = [] # 初始化层列表
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=(norm_cfg is None)))
        # 添加卷积层，若无归一化层则使用偏置
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg) # 获取归一化层
            layers.append(norm_layer)
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg) # 获取激活层
            layers.append(act_layer)
        self.block = nn.Sequential(*layers) # 使用Sequential封装所有层

    def forward(self, x):
        return self.block(x) # 前向传播

    def _get_norm_layer(self, num_features, norm_cfg):
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    def _get_act_layer(self, act_cfg):
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")

class Poly_Kernel_Inception_Block(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 1.0,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
        if act_cfg is None:
            act_cfg = dict(type='SiLU')
        out_channels = out_channels or in_channels # 如果未指定输出通道，则使用输入通道
        hidden_channels = make_divisible(int(out_channels * expansion), 8) # 计算隐藏通道数并调整为可整除

        # 预处理卷积层
        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # 深度可分离卷积层1
        self.dw_conv = ConvModule(hidden_channels, hidden_channels, kernel_sizes[0], 1,
                                  autopad(kernel_sizes[0], None, dilations[0]),
                                  dilation=dilations[0], groups=hidden_channels,
                                  norm_cfg=None, act_cfg=None)
        # 深度可分离卷积层2
        self.dw_conv1 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[1], 1,
                                   autopad(kernel_sizes[1], None, dilations[1]),
                                   dilation=dilations[1], groups=hidden_channels,
                                   norm_cfg=None, act_cfg=None)
        # 深度可分离卷积层3
        self.dw_conv2 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[2], 1,
                                   autopad(kernel_sizes[2], None, dilations[2]),
                                   dilation=dilations[2], groups=hidden_channels,
                                   norm_cfg=None, act_cfg=None)
        # 深度可分离卷积层4
        self.dw_conv3 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[3], 1,
                                   autopad(kernel_sizes[3], None, dilations[3]),
                                   dilation=dilations[3], groups=hidden_channels,
                                   norm_cfg=None, act_cfg=None)
        # 深度可分离卷积层5
        self.dw_conv4 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[4], 1,
                                   autopad(kernel_sizes[4], None, dilations[4]),
                                   dilation=dilations[4], groups=hidden_channels,
                                   norm_cfg=None, act_cfg=None)

        # 点卷积层
        self.pw_conv = ConvModule(hidden_channels, hidden_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # 后处理卷积层
        self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        identity = x  # 保存输入以用于残差
        x = self.pre_conv(x)  # 通过预处理卷积层

        # 使用同一输入进行多尺度卷积，再聚合
        dw0 = self.dw_conv(x)
        dw1 = self.dw_conv1(x)
        dw2 = self.dw_conv2(x)
        dw3 = self.dw_conv3(x)
        dw4 = self.dw_conv4(x)
        x = dw0 + dw1 + dw2 + dw3 + dw4

        x = self.pw_conv(x)  # 应用点卷积层整合特征
        x = self.post_conv(x)  # 后处理卷积层将特征映射回输出通道
        return identity + x  # 残差连接


class DBlock(Poly_Kernel_Inception_Block):
    """
    与旧版接口兼容的封装，默认保持输入输出通道一致，便于直接替换原 MDBlock。
    """

    def __init__(self, channels: int, **kwargs):
        kwargs.setdefault("out_channels", channels)
        super().__init__(in_channels=channels, **kwargs)

if __name__ == "__main__":
    input_tensor = torch.randn(1, 64, 128, 128) # 创建随机输入张量
    model = Poly_Kernel_Inception_Block(in_channels=64, out_channels=64) # 实例化模型
    output_tensor = model(input_tensor) # 前向传播
    print("Input shape:", input_tensor.shape) # 打印输入形状
    print("Output shape:", output_tensor.shape) # 打印输出形状
