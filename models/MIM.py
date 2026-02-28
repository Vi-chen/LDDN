from typing import Optional, Sequence
import torch.nn as nn
import torch


def autopad(kernel_size: int, padding: Optional[int] = None, dilation: int = 1) -> int:

    if padding is None:
        padding = (kernel_size - 1) * dilation // 2 
    return padding  

def make_divisible(value: int, divisor: int = 8) -> int:
    return int((value + divisor // 2) // divisor * divisor) 

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
        layers = [] 
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=(norm_cfg is None)))
     
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg) 
            layers.append(norm_layer)
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x) 

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
        out_channels = out_channels or in_channels 
        hidden_channels = make_divisible(int(out_channels * expansion), 8) 

    
        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

     
        self.dw_conv = ConvModule(hidden_channels, hidden_channels, kernel_sizes[0], 1,
                                  autopad(kernel_sizes[0], None, dilations[0]),
                                  dilation=dilations[0], groups=hidden_channels,
                                  norm_cfg=None, act_cfg=None)
   
        self.dw_conv1 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[1], 1,
                                   autopad(kernel_sizes[1], None, dilations[1]),
                                   dilation=dilations[1], groups=hidden_channels,
                                   norm_cfg=None, act_cfg=None)
  
        self.dw_conv2 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[2], 1,
                                   autopad(kernel_sizes[2], None, dilations[2]),
                                   dilation=dilations[2], groups=hidden_channels,
                                   norm_cfg=None, act_cfg=None)

        self.dw_conv3 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[3], 1,
                                   autopad(kernel_sizes[3], None, dilations[3]),
                                   dilation=dilations[3], groups=hidden_channels,
                                   norm_cfg=None, act_cfg=None)
   
        self.dw_conv4 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[4], 1,
                                   autopad(kernel_sizes[4], None, dilations[4]),
                                   dilation=dilations[4], groups=hidden_channels,
                                   norm_cfg=None, act_cfg=None)

        self.pw_conv = ConvModule(hidden_channels, hidden_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        identity = x  
        x = self.pre_conv(x) 

        dw0 = self.dw_conv(x)
        dw1 = self.dw_conv1(x)
        dw2 = self.dw_conv2(x)
        dw3 = self.dw_conv3(x)
        dw4 = self.dw_conv4(x)
        x = dw0 + dw1 + dw2 + dw3 + dw4

        x = self.pw_conv(x)  
        x = self.post_conv(x)  
        return identity + x  

class MIM(Poly_Kernel_Inception_Block):
   
    def __init__(self, channels: int, **kwargs):
        kwargs.setdefault("out_channels", channels)
        super().__init__(in_channels=channels, **kwargs)

