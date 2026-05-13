import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class Scale(nn.Module):
    def __init__(self, init=0.0):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(init, dtype=torch.float32))
    
    def forward(self, x):
        return x * self.s

class RCB(nn.Module):
    def __init__(self, in_ch, out_ch, norm=nn.BatchNorm2d):
        super().__init__()
        self.unify = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.residual = nn.Sequential(
            nn.Conv2d(out_ch, out_ch //4, 3, padding=1, bias=False),
            norm(out_ch//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch//4, out_ch, 3, padding=1, bias=False)
        )
        self.norm = norm(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.unify(x)
        x = self.act(self.norm(x + self.residual(x)))
        return x

class Dblock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_ch, in_ch, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_ch, in_ch, 3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(in_ch, in_ch, 3, padding=8, dilation=8)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        out = x + x1 + x2 + x3 + x4
        return out

class DSC(nn.Module):
    def __init__(self, nin, nout, k=3, p=1):
        super().__init__()
        self.dw = nn.Conv2d(nin, nin, k, padding=p, groups=nin)
        self.pw = nn.Conv2d(nin, nout, 1)
    
    def forward(self, x):
        return self.pw(self.dw(x))
    
class CAB(nn.Module):
    def __init__(self, ch,norm=nn.BatchNorm2d):
        super().__init__()
        self.delta1 = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            norm(ch),
            nn.Conv2d(ch, 2, 3, padding=1, bias=False),
        )
        self.delta2 = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            norm(ch),
            nn.Conv2d(ch, 2, 3, padding=1, bias=False),
        )
        nn.init.zeros_(self.delta1[-1].weight)
        nn.init.zeros_(self.delta2[-1].weight)

    def _grid_sample(self, x, size, delta):
        n, c, h, w = x.shape
        out_h, out_w = size
        norm = torch.tensor([[[[w,h]]]], device=x.device, dtype=x.dtype)
        gx = torch.linspace(-1, 1, out_w, device=x.device, dtype=x.dtype)
        gy = torch.linspace(-1, 1, out_h, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0).repeat(n, 1, 1, 1)
        grid = grid + delta.permute(0, 2, 3, 1) / norm
        return F.grid_sample(x, grid, align_corners=False)
    
    def forward(self, low, high):
        h, w = low.shape[-2:]
        high = F.interpolate(high, size=(h, w), mode="bilinear", align_corners=True)
        cat = torch.cat([low, high], dim=1)
        d1 = self.delta1(cat)
        d2 = self.delta2(cat)
        high = self._grid_sample(high, (h, w), d1)
        low = self._grid_sample(low, (h, w), d2)
        return high + low
    
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch, out_ch, 1)
        )
        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.BatchNorm2d(in_ch),
            DSC(in_ch, in_ch),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            DSC(in_ch, out_ch),
            nn.BatchNorm2d(out_ch)
        )
    
    def forward(self, x):
        return self.decode(x) + self.res(x)
    
class AlignDecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm=nn.BatchNorm2d):
        super().__init__()
        self.up = CAB(in_ch, norm)
        self.res = nn.Conv2d(in_ch, out_ch, 1)
        self.decode = nn.Sequential(
            norm(in_ch),
            DSC(in_ch, in_ch),
            norm(in_ch),
            nn.ReLU(inplace=True),
            DSC(in_ch, out_ch),
            norm(out_ch),
        )
    def forward(self, low, high):
        f = self.up(low, high)
        return self.decode(f) + self.res(f)

class GatedFusion(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, y):
        g = self.gate(torch.cat([x, y], dim=1))
        return x * g + y * (1 - g)

class SpatialQKV(nn.Module):
    def __init__(self, in_ch, ch):
        super().__init__()
        self.q = nn.Conv2d(in_ch, ch, 1, bias=False)
        self.k = nn.Conv2d(in_ch, ch, 1, bias=False)
        self.v = nn.Conv2d(in_ch, ch, 1, bias=False)
    
    def forward(self, x):
        n, _, h, w = x.shape
        q = self.q(x).reshape(n, -1, h * w).transpose(1, 2)
        k = self.k(x).reshape(n, -1, h * w)
        v = self.v(x).reshape(n, -1, h * w).transpose(1, 2)
        return q, k ,v

class ChannelQKV(nn.Module):
    def __init__(self, in_ch, ch):
        super().__init__()
        self.q = nn.Conv2d(in_ch, ch, 1, bias=False)
        self.k = nn.Conv2d(in_ch, ch, 1, bias=False)
        self.v = nn.Conv2d(in_ch, ch, 1, bias=False)
        
    def forward(self, x):    
        n, _, h, w = x.shape
        q = self.q(x).reshape(n, -1, h * w)
        k = self.k(x).reshape(n, -1, h * w).transpose(1, 2)
        v = self.v(x).reshape(n, -1, h * w)
        return q, k ,v
    
class SpatialAttnBlock(nn.Module):
    def __init__(self, in_ch, ch, out_ch):
        super().__init__()
        self.qkv_opt = SpatialQKV(in_ch, ch)
        self.qkv_sar = SpatialQKV(in_ch, ch)
        self.gamma_opt = Scale(0.0)
        self.gamma_sar = Scale(0.0)
        self.proj_opt = nn.Conv2d(ch, out_ch, 1, bias=False)
        self.proj_sar = nn.Conv2d(ch, out_ch, 1, bias=False)
    
    def forward(self, x_opt, x_sar):
        q1, k1, v1 = self.qkv_opt(x_opt)
        q2, k2, v2 = self.qkv_sar(x_sar)
        att = torch.softmax(q1 @ k1, dim=-1) @ torch.softmax(q2 @ k2, dim=-1)
        n, _, h, w = x_opt.shape
        s1 = (att @ v1).transpose(1, 2).reshape(n, -1, h, w)
        s2 = (att @ v2).transpose(1, 2).reshape(n, -1, h, w)
        return x_opt + self.gamma_opt(self.proj_opt(s1)), x_sar + self.gamma_sar(self.proj_sar(s2))
    
class ChannelAttnBlock(nn.Module):
    def __init__(self, in_ch, ch, out_ch):
        super().__init__()
        self.qkv_opt = ChannelQKV(in_ch, ch)
        self.qkv_sar = ChannelQKV(in_ch, ch)
        self.gamma_opt = Scale(0.0)
        self.gamma_sar = Scale(0.0)
        self.proj_opt = nn.Conv2d(ch, out_ch, 1, bias=False)
        self.proj_sar = nn.Conv2d(ch, out_ch, 1, bias=False)
    
    def forward(self, x_opt, x_sar):
        q1, k1, v1 = self.qkv_opt(x_opt)
        q2, k2, v2 = self.qkv_sar(x_sar)
        att = torch.softmax(q1 @ k1, dim=-1) @ torch.softmax(q2 @ k2, dim=-1)
        n, _, h, w = x_opt.shape
        s1 = (att @ v1).reshape(n, -1, h, w)
        s2 = (att @ v2).reshape(n, -1, h, w)
        return x_opt + self.gamma_opt(self.proj_opt(s1)), x_sar + self.gamma_sar(self.proj_sar(s2))

class AttFuseBlock(nn.Module):
    def __init__(self, in_ch, ch, out_ch):
        super().__init__()
        self.sab = SpatialAttnBlock(in_ch, ch, out_ch)
        self.cab = ChannelAttnBlock(in_ch, ch, out_ch)
    
    def forward(self, x_opt, x_sar):
        s_opt, s_sar = self.sab(x_opt, x_sar)
        c_opt, c_sar = self.cab(x_opt, x_sar)
        return s_opt + c_opt, s_sar + c_sar

class CMCDNet(nn.Module):
    def __init__(self, side_dim=64, att_dim_factor=2, num_classes=1):
        super().__init__()
        self.backbone_opt = timm.create_model(
            "resnet34",
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            in_chans=3,
            pretrained=True
        )
        self.backbone_sar = timm.create_model(
            "efficientnet_b2",
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
            in_chans=1,
            pretrained=True
        )

        self.enc_opt_dims = [64, 64, 128, 256, 512]
        self.enc_sar_dims = [16, 24, 48, 120, 352]

        self.center_opt = Dblock(self.enc_opt_dims[-1])
        self.center_sar = Dblock(self.enc_sar_dims[-1])

        self.side_rgb = nn.ModuleList([RCB(c, side_dim) for c in self.enc_opt_dims])
        self.side_sar = nn.ModuleList([RCB(c, side_dim) for c in self.enc_sar_dims])

        self.fuse = nn.ModuleList([GatedFusion(side_dim) for _ in range(4)])
        self.fuse_5 = AttFuseBlock(side_dim, side_dim // att_dim_factor, side_dim)

        self.decode_rgb = nn.ModuleList([
            DecoderBlock(side_dim, side_dim),
            AlignDecoderBlock(side_dim, side_dim),
            AlignDecoderBlock(side_dim, side_dim),
            AlignDecoderBlock(side_dim, side_dim),
            AlignDecoderBlock(side_dim, side_dim),
        ])

        self.decode_sar = nn.ModuleList([
            DecoderBlock(side_dim, side_dim),
            AlignDecoderBlock(side_dim, side_dim),
            AlignDecoderBlock(side_dim, side_dim),
            AlignDecoderBlock(side_dim, side_dim),
            AlignDecoderBlock(side_dim, side_dim),
        ])

        self.final_fuse = GatedFusion(side_dim)
        self.head = nn.Sequential(
            nn.Conv2d(side_dim, side_dim, 3, padding=1),
            nn.Conv2d(side_dim, side_dim, 3, padding=1),
            nn.Conv2d(side_dim, num_classes, 1),
        )
    
    def forward(self, x_opt, x_sar):

        opt = self.backbone_opt(x_opt)
        sar = self.backbone_sar(x_sar)

        opt[-1] = self.center_opt(opt[-1])
        sar[-1] = self.center_sar(sar[-1])

        x_side = [m(f) for m, f in zip(self.side_rgb, opt)]
        y_side = [m(f) for m, f in zip(self.side_sar, sar)]


        for i in range(4):
            y_side[i] = self.fuse[i](x_side[i], y_side[i])
        
        _, y5 = self.fuse_5(x_side[4], y_side[4])
        y_side[4] = y5

        out_rgb = self.decode_rgb[4](x_side[3], x_side[4])
        out_sar = self.decode_sar[4](y_side[3], y_side[4])

        out_rgb = self.decode_rgb[3](x_side[2], out_rgb)
        out_sar = self.decode_sar[3](y_side[2], out_sar)

        out_rgb = self.decode_rgb[2](x_side[1], out_rgb)
        out_sar = self.decode_sar[2](y_side[1], out_sar)
        
        out_rgb = self.decode_rgb[1](x_side[0], out_rgb)
        out_sar = self.decode_sar[1](y_side[0], out_sar)

        out_rgb = self.decode_rgb[0](out_rgb)
        out_sar = self.decode_sar[0](out_sar)

        out = self.final_fuse(out_rgb, out_sar)
        logits = self.head(out)
        return logits

