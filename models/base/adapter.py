#!/usr/bin/env python3
# -----------------------------------------------
# Written by Qizhong Tan
# -----------------------------------------------

import torch
import clip
import math
import utils.logging as logging
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange
from collections import OrderedDict
import torchvision

from utils.registry import Registry

HEAD_REGISTRY = Registry("Head")

logger = logging.get_logger(__name__)


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b t h w c')
        x = self.norm(x)
        x = rearrange(x, 'b t h w c -> b c t h w')
        return x


def OTAM_dist(dists, lbda=0.5):
    dists = F.pad(dists, (1, 1), 'constant', 0)
    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        cum_dists[:, :, 0, m] = dists[:, :, 0, m] + cum_dists[:, :, 0, m - 1]

    # remaining rows
    for l in range(1, dists.shape[2]):
        # first non-zero column
        cum_dists[:, :, l, 1] = dists[:, :, l, 1] - lbda * torch.log(torch.exp(- cum_dists[:, :, l - 1, 0] / lbda) + torch.exp(- cum_dists[:, :, l - 1, 1] / lbda) + torch.exp(- cum_dists[:, :, l, 0] / lbda))

        # middle columns
        for m in range(2, dists.shape[3] - 1):
            cum_dists[:, :, l, m] = dists[:, :, l, m] - lbda * torch.log(torch.exp(- cum_dists[:, :, l - 1, m - 1] / lbda) + torch.exp(- cum_dists[:, :, l, m - 1] / lbda))

        # last column
        cum_dists[:, :, l, -1] = dists[:, :, l, -1] - lbda * torch.log(torch.exp(- cum_dists[:, :, l - 1, -2] / lbda) + torch.exp(- cum_dists[:, :, l - 1, -1] / lbda) + torch.exp(- cum_dists[:, :, l, -2] / lbda))

    return cum_dists[:, :, -1, -1]


class ResNet_DeformAttention(nn.Module):
    def __init__(self, dim, heads, groups, kernel_size, stride, padding):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_channels = dim // heads
        self.scale = self.head_channels ** -0.5
        self.groups = groups
        self.group_channels = self.dim // self.groups
        self.group_heads = self.heads // self.groups
        self.factor = 2.0

        self.conv_offset = nn.Sequential(
            nn.Conv3d(in_channels=self.group_channels, out_channels=self.group_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=self.group_channels),
            LayerNormProxy(self.group_channels),
            nn.GELU(),
            nn.Conv3d(in_channels=self.group_channels, out_channels=3, kernel_size=(1, 1, 1), bias=False)
        )

        self.proj_q = nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1, 1, 1))
        self.proj_k = nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1, 1, 1))
        self.proj_v = nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1, 1, 1))
        self.proj_out = nn.Conv3d(in_channels=self.dim, out_channels=self.dim, kernel_size=(1, 1, 1))

    @torch.no_grad()
    def _get_ref_points(self, T, H, W, B, dtype, device):
        ref_z, ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, T - 0.5, T, dtype=dtype, device=device),
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_z, ref_y, ref_x), -1)
        ref[..., 0].div_(T).mul_(2).sub_(1)
        ref[..., 1].div_(H).mul_(2).sub_(1)
        ref[..., 2].div_(W).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.groups, -1, -1, -1, -1)  # B * g T H W 3

        return ref

    def forward(self, x):
        B, C, T, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = rearrange(q, 'b (g c) t h w -> (b g) c t h w', g=self.groups, c=self.group_channels)
        offset = self.conv_offset(q_off)  # B * g 3 Tp Hp Wp
        Tp, Hp, Wp = offset.size(2), offset.size(3), offset.size(4)
        n_sample = Tp * Hp * Wp
        # logger.info('{}x{}x{}={}'.format(Tp, Hp, Wp, n_sample))

        offset_range = torch.tensor([min(1.0, self.factor / Tp), min(1.0, self.factor / Hp), min(1.0, self.factor / Wp)], device=device).reshape(1, 3, 1, 1, 1)
        offset = offset.tanh().mul(offset_range)
        offset = rearrange(offset, 'b p t h w -> b t h w p')
        reference = self._get_ref_points(Tp, Hp, Wp, B, dtype, device)
        pos = offset + reference

        x_sampled = F.grid_sample(input=x.reshape(B * self.groups, self.group_channels, T, H, W),
                                  grid=pos[..., (2, 1, 0)],  # z, y, x -> x, y, z
                                  mode='bilinear', align_corners=True)  # B * g, Cg, Tp, Hp, Wp

        x_sampled = x_sampled.reshape(B, C, 1, 1, n_sample)
        q = q.reshape(B * self.heads, self.head_channels, T * H * W)
        k = self.proj_k(x_sampled).reshape(B * self.heads, self.head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.heads, self.head_channels, n_sample)

        attn = einsum('b c m, b c n -> b m n', q, k)
        attn = attn.mul(self.scale)

        attn = F.softmax(attn, dim=-1)

        out = einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, T, H, W)
        out = self.proj_out(out)

        return out


class ResNet_Vanilla_Adapter(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.args = cfg
        self.in_channels = dim
        self.out_channels = dim
        self.adapter_channels = int(dim * cfg.ADAPTER.ADAPTER_SCALE)

        self.down = nn.Conv3d(in_channels=self.in_channels, out_channels=self.adapter_channels, kernel_size=(1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.out_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        # bt c h w
        x_in = x

        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.args.DATA.NUM_INPUT_FRAMES)
        x = self.down(x)

        x = self.relu(x)

        x = self.up(x)
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        x += x_in
        return x


class ResNet_ST_Adapter(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.args = cfg
        self.in_channels = dim
        self.out_channels = dim
        self.adapter_channels = int(dim * cfg.ADAPTER.ADAPTER_SCALE)

        self.down = nn.Conv3d(in_channels=self.in_channels, out_channels=self.adapter_channels, kernel_size=(1, 1, 1))
        self.conv = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.adapter_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), groups=self.adapter_channels)
        self.up = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.out_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        # bt c h w
        x_in = x

        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.args.DATA.NUM_INPUT_FRAMES)
        x = self.down(x)

        x = self.conv(x)

        x = self.up(x)
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        x += x_in
        return x


class ResNet_DST_Adapter(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.args = cfg
        self.in_channels = dim
        self.out_channels = dim
        self.adapter_channels = int(dim * cfg.ADAPTER.ADAPTER_SCALE)

        self.down = nn.Conv3d(in_channels=self.in_channels, out_channels=self.adapter_channels, kernel_size=(1, 1, 1))

        self.s_conv = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.adapter_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), groups=self.adapter_channels, bias=False)
        self.s_bn = nn.BatchNorm3d(num_features=self.adapter_channels)

        self.t_conv = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.adapter_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), groups=self.adapter_channels, bias=False)
        self.t_bn = nn.BatchNorm3d(num_features=self.adapter_channels)

        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.out_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        # bt c h w
        x_in = x

        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.args.DATA.NUM_INPUT_FRAMES)
        x = self.down(x)

        # Spatial Pathway
        xs = self.s_bn(self.s_conv(x))

        # Temporal Pathway
        xt = self.t_bn(self.t_conv(x))

        x = (xs + xt) / 2
        x = self.relu(x)

        x = self.up(x)
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        x += x_in
        return x


class ResNet_D2ST_Adapter(nn.Module):
    def __init__(self, cfg, dim):
        super().__init__()
        self.args = cfg
        self.in_channels = dim
        self.out_channels = dim
        self.adapter_channels = int(dim * cfg.ADAPTER.ADAPTER_SCALE)
        self.down = nn.Conv3d(in_channels=self.in_channels, out_channels=self.adapter_channels, kernel_size=(1, 1, 1))

        self.pos_embed = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.adapter_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=self.adapter_channels)
        self.s_ln = LayerNormProxy(dim=self.adapter_channels)
        self.t_ln = LayerNormProxy(dim=self.adapter_channels)
        if dim == self.args.ADAPTER.WIDTH // 8:
            self.s_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=1, groups=1, kernel_size=(4, 7, 7), stride=(4, 7, 7), padding=(0, 0, 0))
            self.t_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=1, groups=1, kernel_size=(1, 14, 14), stride=(1, 14, 14), padding=(0, 0, 0))
        elif dim == self.args.ADAPTER.WIDTH // 4:
            self.s_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=2, groups=2, kernel_size=(4, 7, 7), stride=(4, 7, 7), padding=(0, 0, 0))
            self.t_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=2, groups=2, kernel_size=(1, 14, 14), stride=(1, 14, 14), padding=(0, 0, 0))
        elif dim == self.args.ADAPTER.WIDTH // 2:
            self.s_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=4, groups=4, kernel_size=(4, 5, 5), stride=(4, 3, 3), padding=(0, 0, 0))
            self.t_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=4, groups=4, kernel_size=(1, 7, 7), stride=(1, 7, 7), padding=(0, 0, 0))
        else:
            self.s_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=8, groups=8, kernel_size=(4, 4, 4), stride=(4, 3, 3), padding=(0, 0, 0))
            self.t_attn = ResNet_DeformAttention(dim=self.adapter_channels, heads=8, groups=8, kernel_size=(1, 7, 7), stride=(1, 7, 7), padding=(0, 0, 0))
        self.gelu = nn.GELU()

        self.up = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.out_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        # bt c h w
        x_in = x

        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.args.DATA.NUM_INPUT_FRAMES)
        x = self.down(x)

        x = x + self.pos_embed(x)

        # Spatial Deformable Attention
        xs = x + self.s_attn(self.s_ln(x))

        # Temporal Deformable Attention
        xt = x + self.t_attn(self.t_ln(x))

        x = (xs + xt) / 2
        x = self.gelu(x)

        x = self.up(x)
        x = rearrange(x, 'b c t h w -> (b t) c h w')

        x += x_in
        return x


@HEAD_REGISTRY.register()
class ResNet(nn.Module):
    def __init__(self, cfg):
        super(ResNet, self).__init__()
        self.args = cfg
        self.num_frames = cfg.DATA.NUM_INPUT_FRAMES
        if self.args.ADAPTER.LAYERS == 18:
            backbone = torchvision.models.resnet18(pretrained=True)
        elif self.args.ADAPTER.LAYERS == 34:
            backbone = torchvision.models.resnet34(pretrained=True)
        elif self.args.ADAPTER.LAYERS == 50:
            backbone = torchvision.models.resnet50(pretrained=True)
        self.stage1 = nn.Sequential(*list(backbone.children())[:5])
        self.Adapter_1 = ResNet_D2ST_Adapter(cfg, self.args.ADAPTER.WIDTH // 8)
        self.stage2 = nn.Sequential(*list(backbone.children())[5])
        self.Adapter_2 = ResNet_D2ST_Adapter(cfg, self.args.ADAPTER.WIDTH // 4)
        self.stage3 = nn.Sequential(*list(backbone.children())[6])
        self.Adapter_3 = ResNet_D2ST_Adapter(cfg, self.args.ADAPTER.WIDTH // 2)
        self.stage4 = nn.Sequential(*list(backbone.children())[7])
        self.Adapter_4 = ResNet_D2ST_Adapter(cfg, self.args.ADAPTER.WIDTH)
        self.stage5 = nn.Sequential(*list(backbone.children())[8:-1])
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION_VALUE"):
            self.classification_layer = nn.Linear(self.args.ADAPTER.WIDTH, int(self.args.TRAIN.NUM_CLASS))
        self.init_weights()

    def init_weights(self):
        # zero-initialize Adapters
        for n1, m1 in self.named_modules():
            if 'Adapter' in n1:
                for n2, m2 in m1.named_modules():
                    if 'up' in n2:
                        logger.info('init:  {}.{}'.format(n1, n2))
                        nn.init.constant_(m2.weight, 0)
                        nn.init.constant_(m2.bias, 0)

    def get_feat(self, x):
        x = self.stage1(x)
        x = self.Adapter_1(x)
        x = self.stage2(x)
        x = self.Adapter_2(x)
        x = self.stage3(x)
        x = self.Adapter_3(x)
        x = self.stage4(x)
        x = self.Adapter_4(x)
        x = self.stage5(x)
        return x.squeeze()

    def extract_class_indices(self, labels, which_class):
        class_mask = torch.eq(labels, which_class)
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
        return torch.reshape(class_mask_indices, (-1,))

    def forward(self, inputs):
        support_images, query_images = inputs['support_set'], inputs['target_set']
        support_features = self.get_feat(support_images)
        query_features = self.get_feat(query_images)
        support_labels = inputs['support_labels']
        unique_labels = torch.unique(support_labels)

        support_features = support_features.reshape(-1, self.num_frames, self.args.ADAPTER.WIDTH)
        query_features = query_features.reshape(-1, self.num_frames, self.args.ADAPTER.WIDTH)

        class_logits = None
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION_VALUE"):
            class_logits = self.classification_layer(torch.cat([torch.mean(support_features, dim=1), torch.mean(query_features, dim=1)], 0))

        support_features = [torch.mean(torch.index_select(support_features, 0, self.extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        support_features = torch.stack(support_features)

        support_num = support_features.shape[0]
        query_num = query_features.shape[0]

        support_features = support_features.unsqueeze(0).repeat(query_num, 1, 1, 1)
        support_features = rearrange(support_features, 'q s t c -> q (s t) c')

        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(query_features, dim=2).permute(0, 2, 1)).reshape(query_num, support_num, self.num_frames, self.num_frames)
        dist = 1 - frame_sim

        # Bi-MHM
        class_dist = dist.min(3)[0].sum(2) + dist.min(2)[0].sum(2)

        # OTAM
        # class_dist = OTAM_dist(dist) + OTAM_dist(rearrange(dist, 'q s n m -> q s m n'))

        return_dict = {'logits': - class_dist, 'class_logits': class_logits}
        return return_dict


class ViT_DeformAttention(nn.Module):
    def __init__(self, cfg, dim, heads, groups, kernel_size, stride, padding):
        super().__init__()
        self.args = cfg
        self.dim = dim
        self.heads = heads
        self.head_channels = dim // heads
        self.scale = self.head_channels ** -0.5
        self.groups = groups
        self.group_channels = self.dim // self.groups
        self.group_heads = self.heads // self.groups
        self.factor = 2.0

        self.conv_offset = nn.Sequential(
            nn.Conv3d(in_channels=self.group_channels, out_channels=self.group_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=self.group_channels),
            LayerNormProxy(self.group_channels),
            nn.GELU(),
            nn.Conv3d(in_channels=self.group_channels, out_channels=3, kernel_size=(1, 1, 1), bias=False)
        )

        self.proj_q = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.proj_k = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.proj_v = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.proj_out = nn.Linear(in_features=self.dim, out_features=self.dim)

    @torch.no_grad()
    def _get_ref_points(self, T, H, W, B, dtype, device):
        ref_z, ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, T - 0.5, T, dtype=dtype, device=device),
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_z, ref_y, ref_x), -1)
        ref[..., 0].div_(T).mul_(2).sub_(1)
        ref[..., 1].div_(H).mul_(2).sub_(1)
        ref[..., 2].div_(W).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.groups, -1, -1, -1, -1)  # B * g T H W 3

        return ref

    def forward(self, x):
        # hw+1 bt c
        n, BT, C = x.shape
        T = self.args.DATA.NUM_INPUT_FRAMES
        B = BT // T
        H = round(math.sqrt(n - 1))
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = rearrange(q[1:, :, :], '(h w) (b t) c -> b c t h w', h=H, t=T)
        q_off = rearrange(q_off, 'b (g c) t h w -> (b g) c t h w', g=self.groups, c=self.group_channels)
        offset = self.conv_offset(q_off)  # B * g 3 Tp Hp Wp
        Tp, Hp, Wp = offset.size(2), offset.size(3), offset.size(4)
        n_sample = Tp * Hp * Wp
        # logger.info('{}x{}x{}={}'.format(Tp, Hp, Wp, n_sample))

        offset_range = torch.tensor([min(1.0, self.factor / Tp), min(1.0, self.factor / Hp), min(1.0, self.factor / Wp)], device=device).reshape(1, 3, 1, 1, 1)
        offset = offset.tanh().mul(offset_range)
        offset = rearrange(offset, 'b p t h w -> b t h w p')
        reference = self._get_ref_points(Tp, Hp, Wp, B, dtype, device)
        pos = offset + reference

        x_sampled = rearrange(x[1:, :, :], '(h w) (b t) c -> b c t h w', h=H, t=T)
        x_sampled = rearrange(x_sampled, 'b (g c) t h w -> (b g) c t h w', g=self.groups)
        x_sampled = F.grid_sample(input=x_sampled, grid=pos[..., (2, 1, 0)], mode='bilinear', align_corners=True)  # B * g, Cg, Tp, Hp, Wp
        x_sampled = rearrange(x_sampled, '(b g) c t h w -> b (g c) t h w', g=self.groups)
        x_sampled = rearrange(x_sampled, 'b c t h w -> b (t h w) c')

        q = rearrange(q, 'n (b t) c -> b c (t n)', b=B)
        q = rearrange(q, 'b (h c) n -> (b h) c n', h=self.heads)

        k = self.proj_k(x_sampled)
        k = rearrange(k, 'b n (h c) -> (b h) c n', h=self.heads)

        v = self.proj_v(x_sampled)
        v = rearrange(v, 'b n (h c) -> (b h) c n', h=self.heads)

        attn = einsum('b c m, b c n -> b m n', q, k)
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=-1)

        out = einsum('b m n, b c n -> b c m', attn, v)
        out = rearrange(out, '(b h) c n -> b (h c) n', h=self.heads)
        out = rearrange(out, 'b c (t n) -> n (b t) c', t=T)
        out = self.proj_out(out)

        return out


class ViT_D2ST_Adapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.args = cfg
        self.in_channels = cfg.ADAPTER.WIDTH
        self.out_channels = cfg.ADAPTER.WIDTH
        self.adapter_channels = int(cfg.ADAPTER.WIDTH * cfg.ADAPTER.ADAPTER_SCALE)

        self.down = nn.Linear(in_features=self.in_channels, out_features=self.adapter_channels)
        self.gelu1 = nn.GELU()

        self.pos_embed = nn.Conv3d(in_channels=self.adapter_channels, out_channels=self.adapter_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=self.adapter_channels)
        self.s_ln = nn.LayerNorm(normalized_shape=self.adapter_channels)
        self.s_attn = ViT_DeformAttention(cfg=cfg, dim=self.adapter_channels, heads=4, groups=4, kernel_size=(4, 5, 5), stride=(4, 3, 3), padding=(0, 0, 0))
        self.t_ln = nn.LayerNorm(normalized_shape=self.adapter_channels)
        self.t_attn = ViT_DeformAttention(cfg=cfg, dim=self.adapter_channels, heads=4, groups=4, kernel_size=(1, 7, 7), stride=(1, 7, 7), padding=(0, 0, 0))
        self.gelu = nn.GELU()

        self.up = nn.Linear(in_features=self.adapter_channels, out_features=self.out_channels)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        # hw+1 bt c
        n, bt, c = x.shape
        H = round(math.sqrt(n - 1))
        x_in = x

        x = self.down(x)
        x = self.gelu1(x)

        cls = x[0, :, :].unsqueeze(0)
        x = x[1:, :, :]

        x = rearrange(x, '(h w) (b t) c -> b c t h w', t=self.args.DATA.NUM_INPUT_FRAMES, h=H)
        x = x + self.pos_embed(x)
        x = rearrange(x, 'b c t h w -> (h w) (b t) c')

        x = torch.cat([cls, x], dim=0)

        # Spatial Deformable Attention
        xs = x + self.s_attn(self.s_ln(x))

        # Temporal Deformable Attention
        xt = x + self.t_attn(self.t_ln(x))

        x = (xs + xt) / 2
        x = self.gelu(x)

        x = self.up(x)
        x = self.gelu2(x)

        x += x_in
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.ADAPTER.WIDTH
        n_head = cfg.ADAPTER.HEADS
        self.ln_1 = LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.Adapter = ViT_D2ST_Adapter(cfg)

    def attention(self, x):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x):
        # x shape [HW+1, BT, C]
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        x = self.Adapter(x)
        return x


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_checkpoint = hasattr(cfg.TRAIN, "GRADIENT_CHECKPOINT") and cfg.TRAIN.GRADIENT_CHECKPOINT
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(cfg) for _ in range(cfg.ADAPTER.LAYERS)])

    def forward(self, x):
        for block in self.resblocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        return x


class D2STFocusResidual(nn.Module):
    """Conservative FOCUS-style temporal residual branch for D2ST."""

    def __init__(self, dim, heads=8, dropout=0.1, gate_init=-4.0, temporal_gate_init=-4.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.time_proj = nn.Linear(1, dim)
        self.temporal_alpha = nn.Parameter(torch.tensor(float(temporal_gate_init)))
        self.temporal = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        hidden = max(dim // 16, 16)
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.constant_(self.gate[-1].bias, float(gate_init))

    def forward(self, video_feats, timestamps=None):
        if timestamps is not None:
            B, T, _ = video_feats.shape
            timestamps = timestamps.to(device=video_feats.device, dtype=video_feats.dtype)
            if timestamps.shape != (B, T):
                timestamps = timestamps.reshape(B, T)
            uniform_ts = torch.linspace(
                0, 1, T, device=video_feats.device, dtype=video_feats.dtype
            ).unsqueeze(0)
            temporal_deviation = timestamps - uniform_ts
            video_feats = video_feats + torch.sigmoid(self.temporal_alpha) * self.time_proj(
                temporal_deviation.unsqueeze(-1)
            )
        residual = self.temporal(self.norm(video_feats))
        gate = torch.sigmoid(self.gate(video_feats))
        return video_feats + gate * residual


class D2STTaskAwareMatcher(nn.Module):
    """Episode-conditioned soft frame matcher for D2ST frame features.

    The matching matrix is shared across candidate classes for each query, as
    it is generated from the query and the complete support episode. This keeps
    the matcher task-aware without leaking a candidate class into the weights.
    """

    def __init__(self, dim, num_frames, hidden_ratio=0.25, dropout=0.1,
                 temperature=1.0, distance_scale=None):
        super().__init__()
        self.num_frames = int(num_frames)
        self.temperature = max(float(temperature), 1e-6)
        self.distance_scale = float(
            2 * self.num_frames if distance_scale is None else distance_scale
        )
        hidden_dim = max(int(dim * float(hidden_ratio)), 32)

        self.context_norm = nn.LayerNorm(dim)
        self.generator = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.num_frames * self.num_frames),
        )

        # Start from uniform matching. The outer residual gate separately keeps
        # the full model close to the original Bi-MHM decision rule.
        nn.init.constant_(self.generator[-1].weight, 0.0)
        nn.init.constant_(self.generator[-1].bias, 0.0)

    def _class_prototypes(self, support_features, support_labels):
        unique_labels = torch.unique(support_labels, sorted=True)
        prototypes = [
            support_features[support_labels == label].mean(dim=0)
            for label in unique_labels
        ]
        return torch.stack(prototypes, dim=0)

    def forward(self, support_features, query_features, support_labels):
        if support_features.ndim != 3 or query_features.ndim != 3:
            raise ValueError(
                "Task-aware matcher expects support/query features as (B,T,D), "
                f"got {tuple(support_features.shape)} and {tuple(query_features.shape)}"
            )
        if support_features.shape[1] != self.num_frames or query_features.shape[1] != self.num_frames:
            raise ValueError(
                f"Task-aware matcher expects {self.num_frames} frames, got "
                f"{support_features.shape[1]} and {query_features.shape[1]}"
            )

        support_labels = support_labels.to(device=support_features.device)
        support_prototypes = self._class_prototypes(support_features, support_labels)

        # TS-DCM-style task context: each query is conditioned on the complete
        # support episode, not on one candidate class at a time.
        support_context = support_prototypes.mean(dim=(0, 1))
        query_context = query_features.mean(dim=1)
        task_context = self.context_norm(query_context + support_context.unsqueeze(0))

        match_logits = self.generator(task_context)
        match_weights = F.softmax(
            match_logits / self.temperature, dim=-1
        ).view(-1, self.num_frames, self.num_frames)

        query_norm = F.normalize(query_features, dim=-1, eps=1e-6)
        support_norm = F.normalize(support_prototypes, dim=-1, eps=1e-6)
        frame_similarity = torch.einsum(
            'qtd,csd->qcts', query_norm, support_norm
        )
        matched_similarity = (
            frame_similarity * match_weights.unsqueeze(1)
        ).sum(dim=(-1, -2))

        # Match the numerical range of the two T-frame sums used by Bi-MHM.
        return -(1.0 - matched_similarity) * self.distance_scale


class D2STEpisodePrototypeCalibrator(nn.Module):
    """Build class-exclusive D2ST prototypes from the support episode only."""

    def __init__(self, num_frames, temperature=0.2, strength=1.0,
                 mode="weighted_projection", top_k=1, shot_scale_power=0.5):
        super().__init__()
        self.num_frames = int(num_frames)
        self.temperature = max(float(temperature), 1e-6)
        self.strength = float(strength)
        self.mode = str(mode).lower()
        self.top_k = max(int(top_k), 1)
        self.shot_scale_power = max(float(shot_scale_power), 0.0)
        if self.mode not in {"weighted_projection", "nearest_contrastive"}:
            raise ValueError(
                "Prototype calibration mode must be 'weighted_projection' or "
                f"'nearest_contrastive', got {mode!r}."
            )
        self.register_buffer("last_ambiguity", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_shot_scale", torch.tensor(1.0), persistent=False)

    @staticmethod
    def _class_prototypes(support_features, support_labels):
        unique_labels = torch.unique(support_labels, sorted=True)
        prototypes = [
            support_features[support_labels == label].mean(dim=0)
            for label in unique_labels
        ]
        return torch.stack(prototypes, dim=0)

    def forward(self, support_features, query_features, support_labels):
        if support_features.ndim != 3 or query_features.ndim != 3:
            raise ValueError(
                "Prototype calibration expects support/query features as (B,T,D), "
                f"got {tuple(support_features.shape)} and {tuple(query_features.shape)}"
            )
        if support_features.shape[1] != self.num_frames or query_features.shape[1] != self.num_frames:
            raise ValueError(
                f"Prototype calibration expects {self.num_frames} frames, got "
                f"{support_features.shape[1]} and {query_features.shape[1]}"
            )

        support_labels = support_labels.to(device=support_features.device)
        prototypes = self._class_prototypes(support_features, support_labels)
        class_count = prototypes.shape[0]
        shot_counts = torch.stack(
            [(support_labels == label).sum() for label in torch.unique(support_labels, sorted=True)]
        ).to(dtype=prototypes.dtype)
        shot_scale = shot_counts.float().mean().pow(-self.shot_scale_power)
        self.last_shot_scale.copy_(shot_scale.detach().to(self.last_shot_scale))

        if class_count > 1:
            class_descriptors = F.normalize(prototypes.mean(dim=1), dim=-1, eps=1e-6)
            pair_similarity = torch.matmul(class_descriptors, class_descriptors.transpose(0, 1))
            diagonal = torch.eye(class_count, dtype=torch.bool, device=prototypes.device)
            prototype_directions = F.normalize(prototypes, dim=-1, eps=1e-6)

            if self.mode == "weighted_projection":
                # Original EPC v1, retained for an explicit ablation.
                other_weights = F.softmax(
                    pair_similarity.masked_fill(diagonal, float("-inf")) / self.temperature,
                    dim=-1,
                )
                other_context = F.normalize(
                    torch.matmul(other_weights, class_descriptors), dim=-1, eps=1e-6
                )
                ambiguity = (
                    other_weights * pair_similarity.masked_fill(diagonal, 0.0)
                ).sum(dim=-1).clamp_min(0.0)
                shared_projection = (
                    prototype_directions * other_context[:, None, :]
                ).sum(dim=-1, keepdim=True).clamp_min(0.0)
                prototypes = prototype_directions - (
                    self.strength
                    * ambiguity[:, None, None]
                    * shared_projection
                    * other_context[:, None, :]
                )
            else:
                # Adapt only against the most confusable support classes. The
                # relative gap suppresses calibration when all classes are
                # similarly related, which is common in SSv2 episodes.
                masked_similarity = pair_similarity.masked_fill(diagonal, float("-inf"))
                top_k = min(self.top_k, class_count - 1)
                top_values, top_indices = masked_similarity.topk(top_k, dim=-1)
                top_weights = F.softmax(top_values / self.temperature, dim=-1)
                competitors = class_descriptors[top_indices]
                competitor_context = F.normalize(
                    (top_weights[..., None] * competitors).sum(dim=1), dim=-1, eps=1e-6
                )
                nearest_similarity = (top_weights * top_values).sum(dim=-1)
                other_mean = pair_similarity.masked_fill(diagonal, 0.0).sum(dim=-1)
                other_mean = other_mean / float(class_count - 1)
                ambiguity = (
                    (nearest_similarity - other_mean).clamp_min(0.0)
                    / (1.0 - other_mean).clamp_min(1e-3)
                ).clamp_max(1.0)
                exclusive_direction = F.normalize(
                    class_descriptors - competitor_context, dim=-1, eps=1e-6
                )
                prototypes = prototype_directions + (
                    self.strength
                    * shot_scale
                    * ambiguity[:, None, None]
                    * exclusive_direction[:, None, :]
                )

            self.last_ambiguity.copy_(
                ambiguity.detach().mean().to(self.last_ambiguity)
            )
        else:
            self.last_ambiguity.zero_()

        query_norm = F.normalize(query_features, dim=-1, eps=1e-6)
        prototype_norm = F.normalize(prototypes, dim=-1, eps=1e-6)
        frame_similarity = torch.einsum(
            'qtd,csd->qcts', query_norm, prototype_norm
        )
        dist = 1.0 - frame_similarity
        class_dist = dist.min(dim=3)[0].sum(dim=2) + dist.min(dim=2)[0].sum(dim=2)
        return -class_dist

    def get_diagnostics(self):
        return {
            "ambiguity": self.last_ambiguity.detach().item(),
            "shot_scale": self.last_shot_scale.detach().item(),
        }


class D2STMultiVelocityMatcher(nn.Module):
    """Match ordered multi-velocity motion features from a support episode.

    Bi-MHM is intentionally order-insensitive. This branch complements it with
    short- and long-range feature differences followed by bidirectional OTAM
    alignment, while keeping the original appearance matcher untouched.
    """

    def __init__(self, dim, num_frames, velocities=(1, 2), hidden_ratio=0.25,
                 dropout=0.1, distance_scale=None, otam_lambda=0.5):
        super().__init__()
        self.num_frames = int(num_frames)
        self.velocities = tuple(
            sorted({int(v) for v in velocities if 0 < int(v) < self.num_frames})
        )
        if not self.velocities:
            raise ValueError(
                "Multi-velocity matcher requires at least one velocity in "
                f"[1, {self.num_frames - 1}]."
            )
        self.distance_scale = float(
            2 * self.num_frames if distance_scale is None else distance_scale
        )
        self.otam_lambda = max(float(otam_lambda), 1e-4)
        hidden_dim = max(int(dim * float(hidden_ratio)), 32)
        self.motion_norm = nn.LayerNorm(dim)
        self.motion_projection = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.constant_(self.motion_projection[-1].weight, 0.0)
        nn.init.constant_(self.motion_projection[-1].bias, 0.0)
        self.velocity_logits = nn.Parameter(
            torch.zeros(len(self.velocities), dtype=torch.float32)
        )
        self.register_buffer("last_motion_delta", torch.tensor(0.0), persistent=False)

    def _motion_features(self, features, velocity):
        # A larger temporal offset captures slower action changes. The
        # zero-initialized residual starts from the raw D2ST motion direction.
        differences = features[:, velocity:] - features[:, :-velocity]
        differences = differences + self.motion_projection(
            self.motion_norm(differences)
        )
        return F.normalize(differences, dim=-1, eps=1e-6)

    def _ordered_distance(self, query_features, support_features):
        query_norm = F.normalize(query_features, dim=-1, eps=1e-6)
        support_norm = F.normalize(support_features, dim=-1, eps=1e-6)
        pair_distance = 1.0 - torch.einsum(
            "qtd,csd->qcts", query_norm, support_norm
        )
        forward_distance = OTAM_dist(pair_distance, lbda=self.otam_lambda)
        reverse_distance = OTAM_dist(
            pair_distance.transpose(-1, -2), lbda=self.otam_lambda
        )
        return (forward_distance + reverse_distance) * 0.5

    def _class_prototypes(self, support_features, support_labels):
        unique_labels = torch.unique(support_labels, sorted=True)
        prototypes = [
            support_features[support_labels == label].mean(dim=0)
            for label in unique_labels
        ]
        return torch.stack(prototypes, dim=0)

    def forward(self, support_features, query_features, support_labels):
        if support_features.ndim != 3 or query_features.ndim != 3:
            raise ValueError(
                "Multi-velocity matcher expects support/query features as (B,T,D), "
                f"got {tuple(support_features.shape)} and {tuple(query_features.shape)}"
            )
        if support_features.shape[1] != self.num_frames or query_features.shape[1] != self.num_frames:
            raise ValueError(
                f"Multi-velocity matcher expects {self.num_frames} frames, got "
                f"{support_features.shape[1]} and {query_features.shape[1]}"
            )

        support_labels = support_labels.to(device=support_features.device)
        support_prototypes = self._class_prototypes(support_features, support_labels)
        velocity_weights = F.softmax(self.velocity_logits, dim=0).to(
            dtype=query_features.dtype
        )
        velocity_distances = []
        motion_delta = []
        for velocity in self.velocities:
            query_motion = self._motion_features(query_features, velocity)
            support_motion = self._motion_features(support_prototypes, velocity)
            velocity_distances.append(
                self._ordered_distance(query_motion, support_motion)
                / float(query_motion.shape[1])
            )
            motion_delta.append(query_motion.abs().mean())

        self.last_motion_delta.copy_(torch.stack(motion_delta).mean().detach())
        class_distance = torch.stack(velocity_distances, dim=0)
        class_distance = (class_distance * velocity_weights[:, None, None]).sum(dim=0)
        return -class_distance * self.distance_scale

    def get_diagnostics(self):
        return {
            "motion_delta": self.last_motion_delta.detach().item(),
            "velocity_weights": [
                value for value in F.softmax(self.velocity_logits.detach(), dim=0).tolist()
            ],
        }


@HEAD_REGISTRY.register()
class ViT_CLIP(nn.Module):
    def __init__(self, cfg):
        super(ViT_CLIP, self).__init__()
        self.args = cfg
        self.pretrained = cfg.ADAPTER.PRETRAINED
        self.width = cfg.ADAPTER.WIDTH
        self.patch_size = cfg.ADAPTER.PATCH_SIZE
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.width, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        scale = self.width ** -0.5
        self.layers = cfg.ADAPTER.LAYERS
        self.class_embedding = nn.Parameter(scale * torch.randn(self.width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((cfg.DATA.TRAIN_CROP_SIZE // self.patch_size) ** 2 + 1, self.width))
        self.ln_pre = LayerNorm(self.width)
        self.num_frames = cfg.DATA.NUM_INPUT_FRAMES
        self.temporal_embedding = nn.Parameter(torch.zeros(1, self.num_frames, self.width))
        self.transformer = Transformer(cfg)
        self.ln_post = LayerNorm(self.width)
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION_VALUE"):
            self.classification_layer = nn.Linear(self.width, int(self.args.TRAIN.NUM_CLASS))
        self.focus_enable = hasattr(cfg, "FOCUS") and getattr(cfg.FOCUS, "ENABLE", False)
        if self.focus_enable:
            self.focus_branch = D2STFocusResidual(
                dim=self.width,
                heads=int(getattr(cfg.FOCUS, "HEADS", 8)),
                dropout=float(getattr(cfg.FOCUS, "DROPOUT", 0.1)),
                gate_init=float(getattr(cfg.FOCUS, "GATE_INIT", -4.0)),
                temporal_gate_init=float(getattr(cfg.FOCUS, "TEMPORAL_GATE_INIT", -4.0)),
            )
            self.focus_alpha = nn.Parameter(
                torch.tensor(float(getattr(cfg.FOCUS, "ALPHA_INIT", -4.0)))
            )
            self.focus_detach_input = bool(
                getattr(cfg.FOCUS, "DETACH_INPUT", False)
            )
        self.task_match_enable = (
            hasattr(cfg, "TASK_MATCH") and getattr(cfg.TASK_MATCH, "ENABLE", False)
        )
        if self.task_match_enable:
            self.task_matcher = D2STTaskAwareMatcher(
                dim=self.width,
                num_frames=self.num_frames,
                hidden_ratio=float(getattr(cfg.TASK_MATCH, "HIDDEN_RATIO", 0.25)),
                dropout=float(getattr(cfg.TASK_MATCH, "DROPOUT", 0.1)),
                temperature=float(getattr(cfg.TASK_MATCH, "TEMPERATURE", 1.0)),
                distance_scale=float(
                    getattr(cfg.TASK_MATCH, "DISTANCE_SCALE", 2 * self.num_frames)
                ),
            )
            self.task_match_alpha = nn.Parameter(
                torch.tensor(float(getattr(cfg.TASK_MATCH, "ALPHA_INIT", -4.0)))
            )
            self.task_match_use_focus = bool(
                getattr(cfg.TASK_MATCH, "USE_FOCUS_FEATURES", True)
            )
        self.proto_calib_enable = (
            hasattr(cfg, "PROTO_CALIB") and getattr(cfg.PROTO_CALIB, "ENABLE", False)
        )
        if self.proto_calib_enable:
            self.proto_calibrator = D2STEpisodePrototypeCalibrator(
                num_frames=self.num_frames,
                temperature=float(getattr(cfg.PROTO_CALIB, "TEMPERATURE", 0.2)),
                strength=float(getattr(cfg.PROTO_CALIB, "STRENGTH", 1.0)),
                mode=str(getattr(cfg.PROTO_CALIB, "MODE", "weighted_projection")),
                top_k=int(getattr(cfg.PROTO_CALIB, "TOP_K", 1)),
                shot_scale_power=float(
                    getattr(cfg.PROTO_CALIB, "SHOT_SCALE_POWER", 0.5)
                ),
            )
            self.proto_calib_alpha = nn.Parameter(
                torch.tensor(float(getattr(cfg.PROTO_CALIB, "ALPHA_INIT", -4.0)))
            )
            self.register_buffer(
                "proto_calib_logit_delta", torch.tensor(0.0), persistent=False
            )
        self.multi_velocity_enable = (
            hasattr(cfg, "MULTI_VELOCITY")
            and getattr(cfg.MULTI_VELOCITY, "ENABLE", False)
        )
        if self.multi_velocity_enable:
            velocities = getattr(cfg.MULTI_VELOCITY, "VELOCITIES", [1, 2])
            self.multi_velocity_matcher = D2STMultiVelocityMatcher(
                dim=self.width,
                num_frames=self.num_frames,
                velocities=velocities,
                hidden_ratio=float(
                    getattr(cfg.MULTI_VELOCITY, "HIDDEN_RATIO", 0.25)
                ),
                dropout=float(getattr(cfg.MULTI_VELOCITY, "DROPOUT", 0.1)),
                distance_scale=float(
                    getattr(
                        cfg.MULTI_VELOCITY,
                        "DISTANCE_SCALE",
                        2 * self.num_frames,
                    )
                ),
                otam_lambda=float(getattr(cfg.MULTI_VELOCITY, "OTAM_LAMBDA", 0.5)),
            )
            self.multi_velocity_alpha = nn.Parameter(
                torch.tensor(float(getattr(cfg.MULTI_VELOCITY, "ALPHA_INIT", -4.0)))
            )
            self.register_buffer(
                "multi_velocity_logit_delta", torch.tensor(0.0), persistent=False
            )
        self.init_weights()

    def get_fusion_weights(self):
        weights = {}
        if self.focus_enable:
            weights["focus"] = torch.sigmoid(self.focus_alpha).detach().item()
        if self.task_match_enable:
            weights["task_match"] = torch.sigmoid(
                self.task_match_alpha
            ).detach().item()
        if self.proto_calib_enable:
            weights["proto_calib"] = torch.sigmoid(
                self.proto_calib_alpha
            ).detach().item()
        if self.multi_velocity_enable:
            weights["multi_velocity"] = torch.sigmoid(
                self.multi_velocity_alpha
            ).detach().item()
        return weights

    def get_calibration_diagnostics(self):
        if not self.proto_calib_enable:
            return {}
        diagnostics = self.proto_calibrator.get_diagnostics()
        diagnostics["logit_delta"] = self.proto_calib_logit_delta.detach().item()
        return diagnostics

    def get_multi_velocity_diagnostics(self):
        if not self.multi_velocity_enable:
            return {}
        diagnostics = self.multi_velocity_matcher.get_diagnostics()
        diagnostics["logit_delta"] = self.multi_velocity_logit_delta.detach().item()
        return diagnostics

    def init_weights(self):
        logger.info(f'load model from: {self.pretrained}')
        # Load OpenAI CLIP pretrained weights
        clip_model, _ = clip.load(self.pretrained, device="cpu")
        pretrain_dict = clip_model.visual.state_dict()
        del clip_model
        del pretrain_dict['proj']
        msg = self.load_state_dict(pretrain_dict, strict=False)
        logger.info('Missing keys: {}'.format(msg.missing_keys))
        logger.info('Unexpected keys: {}'.format(msg.unexpected_keys))
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        torch.cuda.empty_cache()
        # zero-initialize Adapters
        for n1, m1 in self.named_modules():
            if 'Adapter' in n1:
                for n2, m2 in m1.named_modules():
                    if 'up' in n2:
                        logger.info('init:  {}.{}'.format(n1, n2))
                        nn.init.constant_(m2.weight, 0)
                        nn.init.constant_(m2.bias, 0)

    def extract_class_indices(self, labels, which_class):
        class_mask = torch.eq(labels, which_class)
        class_mask_indices = torch.nonzero(class_mask, as_tuple=False)
        return torch.reshape(class_mask_indices, (-1,))

    def _bimhm_class_dist(self, support_features, query_features, support_labels):
        unique_labels = torch.unique(support_labels)
        support_features = [
            torch.mean(
                torch.index_select(
                    support_features,
                    0,
                    self.extract_class_indices(support_labels, c),
                ),
                dim=0,
            )
            for c in unique_labels
        ]
        support_features = torch.stack(support_features)

        support_num = support_features.shape[0]
        query_num = query_features.shape[0]
        support_features = support_features.unsqueeze(0).repeat(query_num, 1, 1, 1)
        support_features = rearrange(support_features, 'q s t c -> q (s t) c')

        frame_sim = torch.matmul(
            F.normalize(support_features, dim=2),
            F.normalize(query_features, dim=2).permute(0, 2, 1),
        ).reshape(query_num, support_num, self.num_frames, self.num_frames)
        dist = 1 - frame_sim
        return dist.min(3)[0].sum(2) + dist.min(2)[0].sum(2)

    def get_feat(self, x):
        x = self.conv1(x)  # b*t c h w
        x = rearrange(x, 'b c h w -> b (h w) c')
        # b*t h*w+1 c
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        # n = h*w+1
        n = x.shape[1]

        x = rearrange(x, '(b t) n c -> (b n) t c', t=self.num_frames)
        x = x + self.temporal_embedding
        x = rearrange(x, '(b n) t c -> (b t) n c', n=n)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        x = x[:, 0, :]
        return x

    def forward(self, inputs):
        support_images, query_images = inputs['support_set'], inputs['target_set']
        support_features = self.get_feat(support_images)
        query_features = self.get_feat(query_images)
        support_labels = inputs['support_labels']
        unique_labels = torch.unique(support_labels)

        support_features = support_features.reshape(-1, self.num_frames, self.args.ADAPTER.WIDTH)
        query_features = query_features.reshape(-1, self.num_frames, self.args.ADAPTER.WIDTH)
        support_features_raw = support_features
        query_features_raw = query_features
        support_timestamps = inputs.get("support_timestamps", None)
        target_timestamps = inputs.get("target_timestamps", None)
        if support_timestamps is not None:
            support_timestamps = support_timestamps.reshape(-1, self.num_frames)
        if target_timestamps is not None:
            target_timestamps = target_timestamps.reshape(-1, self.num_frames)

        class_logits = None
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION_VALUE"):
            class_logits = self.classification_layer(torch.cat([torch.mean(support_features, dim=1), torch.mean(query_features, dim=1)], 0))

        support_features = [torch.mean(torch.index_select(support_features, 0, self.extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        support_features = torch.stack(support_features)

        support_num = support_features.shape[0]
        query_num = query_features.shape[0]

        support_features = support_features.unsqueeze(0).repeat(query_num, 1, 1, 1)
        support_features = rearrange(support_features, 'q s t c -> q (s t) c')

        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(query_features, dim=2).permute(0, 2, 1)).reshape(query_num, support_num, self.num_frames, self.num_frames)
        dist = 1 - frame_sim

        # Bi-MHM
        class_dist = dist.min(3)[0].sum(2) + dist.min(2)[0].sum(2)

        # OTAM
        # class_dist = OTAM_dist(dist) + OTAM_dist(rearrange(dist, 'q s n m -> q s m n'))

        base_logits = -class_dist
        task_support = support_features_raw
        task_query = query_features_raw
        focus_logits = None
        if self.focus_enable:
            focus_support_input = support_features_raw
            focus_query_input = query_features_raw
            if self.focus_detach_input:
                focus_support_input = focus_support_input.detach()
                focus_query_input = focus_query_input.detach()
            focus_support = self.focus_branch(focus_support_input, support_timestamps)
            focus_query = self.focus_branch(focus_query_input, target_timestamps)
            focus_class_dist = self._bimhm_class_dist(
                focus_support,
                focus_query,
                support_labels,
            )
            focus_logits = -focus_class_dist
            if self.task_match_enable and self.task_match_use_focus:
                task_support = focus_support
                task_query = focus_query

        task_logits = None
        if self.task_match_enable:
            task_logits = self.task_matcher(
                task_support,
                task_query,
                support_labels,
            )

        proto_calib_logits = None
        if self.proto_calib_enable:
            proto_calib_logits = self.proto_calibrator(
                support_features_raw,
                query_features_raw,
                support_labels,
            )
            self.proto_calib_logit_delta.copy_(
                (proto_calib_logits - base_logits).detach().abs().mean().to(
                    self.proto_calib_logit_delta
                )
            )

        multi_velocity_logits = None
        if self.multi_velocity_enable:
            multi_velocity_logits = self.multi_velocity_matcher(
                support_features_raw,
                query_features_raw,
                support_labels,
            )
            self.multi_velocity_logit_delta.copy_(
                (multi_velocity_logits - base_logits).detach().abs().mean().to(
                    self.multi_velocity_logit_delta
                )
            )

        # Each optional branch learns an independent residual from the same
        # D2ST decision. This preserves single-branch behavior and avoids
        # forcing FOCUS and TASK_MATCH to compete for a softmax budget.
        logits = base_logits
        if focus_logits is not None:
            focus_weight = torch.sigmoid(self.focus_alpha)
            logits = logits + focus_weight * (focus_logits - base_logits.detach())
        if task_logits is not None:
            task_weight = torch.sigmoid(self.task_match_alpha)
            logits = logits + task_weight * (task_logits - base_logits.detach())
        if proto_calib_logits is not None:
            proto_calib_weight = torch.sigmoid(self.proto_calib_alpha)
            logits = logits + proto_calib_weight * (
                proto_calib_logits - base_logits.detach()
            )
        if multi_velocity_logits is not None:
            multi_velocity_weight = torch.sigmoid(self.multi_velocity_alpha)
            logits = logits + multi_velocity_weight * (
                multi_velocity_logits - base_logits.detach()
            )

        return_dict = {'logits': logits, 'class_logits': class_logits}
        return return_dict
