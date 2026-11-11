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


class D2STSpatialPatternMatcher(nn.Module):
    """Match spatial pattern prototypes extracted from ViT patch tokens.

    Inspired by spatial prototype matching in DiST (arXiv:2602.18043), but
    not a reproduction of its semantic-guided Spatial Knowledge Compensator.
    Regional grid averages are not learned object prototypes. The ViT already
    produces a 14x14 patch grid, so fixed grid pooling gives four regional
    tokens per frame without adding a second backbone, text encoder, or point
    tracker. A token-level bidirectional match is computed inside each pair of
    frames, followed by the original Bi-MHM temporal aggregation.
    """

    def __init__(self, num_frames, grid_size=2, distance_scale=None):
        super().__init__()
        self.num_frames = int(num_frames)
        self.grid_size = max(int(grid_size), 1)
        self.distance_scale = float(
            1.0 if distance_scale is None else distance_scale
        )

    def _pool_spatial_patterns(self, patch_features):
        if patch_features.ndim != 4:
            raise ValueError(
                "Spatial pattern matcher expects patch features as (B,T,P,D), "
                f"got {tuple(patch_features.shape)}"
            )
        batch, frames, patch_count, dim = patch_features.shape
        spatial_size = int(math.sqrt(patch_count))
        if spatial_size * spatial_size != patch_count:
            raise ValueError(
                "Spatial pattern matcher requires a square patch grid, "
                f"got {patch_count} patches"
            )
        if spatial_size % self.grid_size != 0:
            raise ValueError(
                f"Patch grid {spatial_size} is not divisible by pattern grid "
                f"{self.grid_size}"
            )

        cell_size = spatial_size // self.grid_size
        patterns = patch_features.reshape(
            batch,
            frames,
            self.grid_size,
            cell_size,
            self.grid_size,
            cell_size,
            dim,
        )
        patterns = patterns.mean(dim=(3, 5))
        return patterns.reshape(batch, frames, self.grid_size * self.grid_size, dim)

    @staticmethod
    def _class_prototypes(spatial_features, support_labels):
        unique_labels = torch.unique(support_labels, sorted=True)
        prototypes = [
            spatial_features[support_labels == label].mean(dim=0)
            for label in unique_labels
        ]
        return torch.stack(prototypes)

    def forward(self, support_patch_features, query_patch_features, support_labels):
        if support_patch_features.ndim != 4 or query_patch_features.ndim != 4:
            raise ValueError(
                "Spatial pattern matcher expects support/query patch features as "
                f"(B,T,P,D), got {tuple(support_patch_features.shape)} and "
                f"{tuple(query_patch_features.shape)}"
            )
        if (
            support_patch_features.shape[1] != self.num_frames
            or query_patch_features.shape[1] != self.num_frames
        ):
            raise ValueError(
                f"Spatial pattern matcher expects {self.num_frames} frames, got "
                f"{support_patch_features.shape[1]} and "
                f"{query_patch_features.shape[1]}"
            )

        support_labels = support_labels.to(device=support_patch_features.device)
        support_patterns = self._pool_spatial_patterns(support_patch_features)
        query_patterns = self._pool_spatial_patterns(query_patch_features)
        support_prototypes = self._class_prototypes(support_patterns, support_labels)

        query_norm = F.normalize(query_patterns, dim=-1, eps=1e-6)
        support_norm = F.normalize(support_prototypes, dim=-1, eps=1e-6)
        token_similarity = torch.einsum(
            "qtkd,sujd->qstukj", query_norm, support_norm
        )
        token_distance = 1.0 - token_similarity

        # Match regional tokens in both directions before matching frames.
        query_to_support = token_distance.min(dim=-1).values.mean(dim=-1)
        support_to_query = token_distance.min(dim=-2).values.mean(dim=-1)
        frame_distance = (query_to_support + support_to_query) * 0.5

        class_distance = (
            frame_distance.min(dim=3)[0].sum(dim=2)
            + frame_distance.min(dim=2)[0].sum(dim=2)
        )
        return -class_distance * self.distance_scale

    def get_diagnostics(self):
        return {"grid_size": self.grid_size, "tokens_per_frame": self.grid_size ** 2}


class D2STTrajectoryGeometryMatcher(nn.Module):
    """Approximate patch trajectories and match their local motion geometry.

    This is a D2ST-native research prototype, not a GeoTRAM reproduction: patch
    correspondences are inferred from CLIP features, without an external point
    tracker, monocular depth, or DINOv2-guided seed selection.
    """

    def __init__(self, dim, num_frames, hidden_ratio=0.25, dropout=0.1,
                 temperature=0.07, distance_scale=None, local_neighbors=8,
                 point_grid_size=7):
        super().__init__()
        self.num_frames = int(num_frames)
        self.temperature = max(float(temperature), 1e-4)
        self.distance_scale = float(
            2 * self.num_frames if distance_scale is None else distance_scale
        )
        self.local_neighbors = max(int(local_neighbors), 1)
        self.point_grid_size = max(int(point_grid_size), 1)
        hidden_dim = max(int(dim * float(hidden_ratio)), 32)
        self.motion_projection = nn.Sequential(
            nn.LayerNorm(dim + 4),
            nn.Linear(dim + 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.xavier_uniform_(self.motion_projection[-1].weight)
        nn.init.zeros_(self.motion_projection[-1].bias)
        self.register_buffer("last_track_confidence", torch.tensor(0.0), persistent=False)

    def _neighbor_indices(self, patch_count, device, dtype):
        side = int(math.sqrt(patch_count))
        if side * side != patch_count:
            raise ValueError(
                f"Trajectory geometry requires a square patch grid, got {patch_count}"
            )
        coords_float = torch.stack(torch.meshgrid(
            torch.linspace(-1.0, 1.0, side, device=device, dtype=torch.float32),
            torch.linspace(-1.0, 1.0, side, device=device, dtype=torch.float32),
            indexing="ij",
        ), dim=-1).reshape(patch_count, 2)
        distances = (
            coords_float[:, None, :] - coords_float[None, :, :]
        ).square().sum(dim=-1)
        distances.fill_diagonal_(float("inf"))
        if patch_count < 2:
            raise ValueError("Trajectory geometry requires at least two points")
        k = min(self.local_neighbors, patch_count - 1)
        return coords_float.to(dtype=dtype), distances.topk(k, largest=False, dim=-1).indices

    def _trajectory_descriptors(self, patch_features):
        if patch_features.ndim != 4:
            raise ValueError(
                "Trajectory geometry expects features as (B,T,P,D), got "
                f"{tuple(patch_features.shape)}"
            )
        batch, frames, patch_count, dim = patch_features.shape
        if frames != self.num_frames:
            raise ValueError(
                f"Trajectory geometry expects {self.num_frames} frames, got {frames}"
            )
        patch_side = int(math.sqrt(patch_count))
        if patch_side * patch_side != patch_count:
            raise ValueError(
                f"Trajectory geometry requires a square patch grid, got {patch_count}"
            )
        if patch_side % self.point_grid_size != 0:
            raise ValueError(
                f"Patch grid {patch_side} is not divisible by trajectory point grid "
                f"{self.point_grid_size}"
            )
        cell_size = patch_side // self.point_grid_size
        patch_features = patch_features.reshape(
            batch,
            frames,
            self.point_grid_size,
            cell_size,
            self.point_grid_size,
            cell_size,
            dim,
        ).mean(dim=(3, 5)).reshape(
            batch, frames, self.point_grid_size ** 2, dim
        )
        coords, neighbors = self._neighbor_indices(
            self.point_grid_size ** 2,
            patch_features.device,
            patch_features.dtype,
        )
        normalized = F.normalize(patch_features, dim=-1, eps=1e-6)
        tracked_features = patch_features[:, 0]
        tracked_coords = coords.unsqueeze(0).expand(batch, -1, -1)
        frame_descriptors = []
        confidence_values = []

        for frame_index in range(1, frames):
            current_features = patch_features[:, frame_index]
            affinity = torch.bmm(
                F.normalize(tracked_features, dim=-1, eps=1e-6),
                normalized[:, frame_index].transpose(1, 2),
            )
            correspondence = F.softmax(affinity / self.temperature, dim=-1)
            next_coords = torch.matmul(correspondence, coords)
            next_features = torch.matmul(correspondence, current_features)
            displacement = next_coords - tracked_coords
            neighbor_displacement = displacement[:, neighbors]
            relative_motion = (
                displacement.unsqueeze(2) - neighbor_displacement
            ).mean(dim=2)
            feature_change = next_features - tracked_features
            states = torch.cat(
                [feature_change, displacement, relative_motion], dim=-1
            )
            patch_motion = self.motion_projection(states)
            quality = affinity.max(dim=-1).values.clamp_min(0.05)
            descriptor = (
                patch_motion * quality.unsqueeze(-1)
            ).sum(dim=1) / quality.sum(dim=1, keepdim=True).clamp_min(1e-6)
            frame_descriptors.append(descriptor)
            confidence_values.append(quality.mean())
            tracked_features = next_features
            tracked_coords = next_coords

        self.last_track_confidence.copy_(torch.stack(confidence_values).mean().detach())
        return torch.stack(frame_descriptors, dim=1)

    @staticmethod
    def _class_prototypes(video_descriptors, support_labels):
        unique_labels = torch.unique(support_labels, sorted=True)
        return torch.stack([
            video_descriptors[support_labels == label].mean(dim=0)
            for label in unique_labels
        ])

    def forward(self, support_patch_features, query_patch_features, support_labels):
        if support_patch_features.shape[-1] != query_patch_features.shape[-1]:
            raise ValueError("Support/query patch feature dimensions must match")
        support_labels = support_labels.to(device=support_patch_features.device)
        support_descriptors = self._trajectory_descriptors(support_patch_features)
        query_descriptors = self._trajectory_descriptors(query_patch_features)
        support_prototypes = self._class_prototypes(support_descriptors, support_labels)
        query_norm = F.normalize(query_descriptors, dim=-1, eps=1e-6)
        support_norm = F.normalize(support_prototypes, dim=-1, eps=1e-6)
        frame_distance = 1.0 - torch.einsum(
            "qtd,csd->qcts", query_norm, support_norm
        )
        class_distance = (
            frame_distance.min(dim=3).values.sum(dim=2)
            + frame_distance.min(dim=2).values.sum(dim=2)
        )
        return -class_distance * self.distance_scale

    def get_diagnostics(self):
        return {"track_confidence": self.last_track_confidence.detach().item()}


class D2STQueryAdaptiveFusion(nn.Module):
    """Fuse matcher residuals with a query-specific uncertainty gate."""

    def __init__(self, num_branches=2, hidden_ratio=0.5, dropout=0.1,
                 gate_init=0.0, temperature=1.0):
        super().__init__()
        self.num_branches = int(num_branches)
        self.temperature = max(float(temperature), 1e-4)
        feature_dim = 2 + 3 * self.num_branches
        hidden_dim = max(int(feature_dim * float(hidden_ratio)), 16)
        self.gate_network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.num_branches),
        )
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.constant_(self.gate_network[-1].bias, float(gate_init))
        self.register_buffer("last_gate_mean", torch.zeros(self.num_branches), persistent=False)
        self.register_buffer("last_gate_std", torch.zeros(self.num_branches), persistent=False)
        self.register_buffer("last_base_entropy", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_disagreement", torch.zeros(self.num_branches), persistent=False)

    @staticmethod
    def _margin(logits):
        top2 = logits.topk(k=min(2, logits.shape[-1]), dim=-1).values
        if top2.shape[-1] == 1:
            return top2[..., 0]
        return top2[..., 0] - top2[..., 1]

    def _entropy(self, logits):
        probabilities = F.softmax(logits / self.temperature, dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        return entropy / max(math.log(max(logits.shape[-1], 2)), 1e-6)

    @staticmethod
    def _calibrate_to_reference(branch_logits, reference_logits):
        branch_mean = branch_logits.mean(dim=-1, keepdim=True)
        reference_mean = reference_logits.mean(dim=-1, keepdim=True)
        branch_centered = branch_logits - branch_mean
        reference_centered = reference_logits - reference_mean
        branch_scale = branch_centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
        reference_scale = reference_centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
        return reference_mean + branch_centered * (reference_scale / branch_scale)

    def forward(self, reference_logits, branch_logits, branch_weights=None):
        if len(branch_logits) != self.num_branches:
            raise ValueError(
                f"Adaptive fusion expects {self.num_branches} branches, "
                f"got {len(branch_logits)}"
            )
        if any(branch.shape != reference_logits.shape for branch in branch_logits):
            raise ValueError("Adaptive fusion logits must have identical shapes")
        if branch_weights is None:
            branch_weights = [1.0] * self.num_branches
        if len(branch_weights) != self.num_branches:
            raise ValueError("Adaptive fusion requires one global weight per branch")

        reference_probabilities = F.softmax(
            reference_logits / self.temperature, dim=-1
        )
        reference_scale = (
            reference_logits - reference_logits.mean(dim=-1, keepdim=True)
        ).square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-4)
        reference_entropy = self._entropy(reference_logits)
        reference_margin = self._margin(reference_logits)
        features = [reference_margin, reference_entropy]
        calibrated_branches = []
        disagreements = []
        for branch in branch_logits:
            calibrated = self._calibrate_to_reference(branch, reference_logits)
            calibrated_branches.append(calibrated)
            branch_probabilities = F.softmax(
                calibrated / self.temperature, dim=-1
            )
            disagreements.append(
                (branch_probabilities - reference_probabilities).abs().mean(dim=-1)
            )
            features.extend([
                self._margin(calibrated),
                (calibrated - reference_logits).abs().mean(dim=-1)
                / reference_scale.squeeze(-1),
                -(branch_probabilities * branch_probabilities.clamp_min(1e-8).log()).sum(dim=-1),
            ])
        gate_features = torch.stack(features, dim=-1)
        gates = torch.sigmoid(self.gate_network(gate_features))
        fused_logits = reference_logits
        static_logits = reference_logits
        for branch, gate, weight in zip(
            calibrated_branches, gates.unbind(dim=-1), branch_weights
        ):
            residual = branch - reference_logits.detach()
            static_logits = static_logits + weight * residual
            fused_logits = fused_logits + gate.unsqueeze(-1) * (
                2.0 * weight * residual
            )

        self.last_gate_mean.copy_(gates.detach().mean(dim=0))
        self.last_gate_std.copy_(gates.detach().std(dim=0, unbiased=False))
        self.last_base_entropy.copy_(reference_entropy.detach().mean())
        self.last_disagreement.copy_(torch.stack(disagreements).detach().mean(dim=1))
        return fused_logits, static_logits

    def get_diagnostics(self):
        return {
            "gate_mean": self.last_gate_mean.detach().cpu().tolist(),
            "gate_std": self.last_gate_std.detach().cpu().tolist(),
            "base_entropy": self.last_base_entropy.detach().item(),
            "branch_disagreement": self.last_disagreement.detach().cpu().tolist(),
        }


class D2STRelationalTemporalMatcher(nn.Module):
    """Match within-video temporal relations instead of isolated frames."""

    def __init__(self, num_frames, distance_scale=None, temperature=1.0):
        super().__init__()
        self.num_frames = int(num_frames)
        self.distance_scale = float(
            2 * self.num_frames if distance_scale is None else distance_scale
        )
        self.temperature = max(float(temperature), 1e-6)
        self.register_buffer("last_relation_delta", torch.tensor(0.0), persistent=False)

    @staticmethod
    def _class_prototypes(features, support_labels):
        unique_labels = torch.unique(support_labels, sorted=True)
        return torch.stack([
            features[support_labels == label].mean(dim=0)
            for label in unique_labels
        ])

    @staticmethod
    def _relation_matrix(features):
        normalized = F.normalize(features, dim=-1, eps=1e-6)
        return torch.bmm(normalized, normalized.transpose(1, 2))

    def forward(self, support_features, query_features, support_labels):
        if support_features.ndim != 3 or query_features.ndim != 3:
            raise ValueError(
                "Relational matcher expects support/query features as (B,T,D), "
                f"got {tuple(support_features.shape)} and {tuple(query_features.shape)}"
            )
        if support_features.shape[1] != self.num_frames or query_features.shape[1] != self.num_frames:
            raise ValueError(
                f"Relational matcher expects {self.num_frames} frames, got "
                f"{support_features.shape[1]} and {query_features.shape[1]}"
            )
        support_labels = support_labels.to(device=support_features.device)
        support_relations = self._relation_matrix(support_features)
        query_relations = self._relation_matrix(query_features)
        relation_prototypes = self._class_prototypes(support_relations, support_labels)
        relation_distance = (
            query_relations[:, None] - relation_prototypes[None]
        ).abs().mean(dim=(-1, -2))
        self.last_relation_delta.copy_(
            relation_distance.detach().mean().to(self.last_relation_delta)
        )
        return -relation_distance * self.distance_scale / self.temperature

    def get_diagnostics(self):
        return {"relation_delta": self.last_relation_delta.detach().item()}


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
        self.support_aggregation_enable = (
            hasattr(cfg, "SUPPORT_AGGREGATION")
            and getattr(cfg.SUPPORT_AGGREGATION, "ENABLE", False)
        )
        self.bimhm_softmin_temperature = float(
            getattr(
                getattr(cfg, "BIMHM_MATCHING", None),
                "SOFTMIN_TEMPERATURE",
                0.0,
            )
        )
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
        self.spatial_pattern_enable = (
            hasattr(cfg, "SPATIAL_PATTERN")
            and getattr(cfg.SPATIAL_PATTERN, "ENABLE", False)
        )
        if self.spatial_pattern_enable:
            self.spatial_pattern_matcher = D2STSpatialPatternMatcher(
                num_frames=self.num_frames,
                grid_size=int(getattr(cfg.SPATIAL_PATTERN, "GRID_SIZE", 2)),
                distance_scale=float(
                    getattr(
                        cfg.SPATIAL_PATTERN,
                        "DISTANCE_SCALE",
                        1.0,
                    )
                ),
            )
            self.spatial_pattern_alpha = nn.Parameter(
                torch.tensor(
                    float(getattr(cfg.SPATIAL_PATTERN, "ALPHA_INIT", -4.0))
                )
            )
            self.spatial_pattern_detach_input = bool(
                getattr(cfg.SPATIAL_PATTERN, "DETACH_INPUT", True)
            )
            self.register_buffer(
                "spatial_pattern_logit_delta", torch.tensor(0.0), persistent=False
            )
        self.trajectory_geometry_enable = (
            hasattr(cfg, "TRAJECTORY_GEOMETRY")
            and getattr(cfg.TRAJECTORY_GEOMETRY, "ENABLE", False)
        )
        if self.trajectory_geometry_enable:
            self.trajectory_geometry_matcher = D2STTrajectoryGeometryMatcher(
                dim=self.width,
                num_frames=self.num_frames,
                hidden_ratio=float(
                    getattr(cfg.TRAJECTORY_GEOMETRY, "HIDDEN_RATIO", 0.25)
                ),
                dropout=float(getattr(cfg.TRAJECTORY_GEOMETRY, "DROPOUT", 0.1)),
                temperature=float(
                    getattr(cfg.TRAJECTORY_GEOMETRY, "TEMPERATURE", 0.07)
                ),
                distance_scale=float(
                    getattr(cfg.TRAJECTORY_GEOMETRY, "DISTANCE_SCALE", 2 * self.num_frames)
                ),
                local_neighbors=int(
                    getattr(cfg.TRAJECTORY_GEOMETRY, "LOCAL_NEIGHBORS", 8)
                ),
                point_grid_size=int(
                    getattr(cfg.TRAJECTORY_GEOMETRY, "POINT_GRID_SIZE", 7)
                ),
            )
            self.trajectory_geometry_alpha = nn.Parameter(
                torch.tensor(
                    float(getattr(cfg.TRAJECTORY_GEOMETRY, "ALPHA_INIT", -4.0))
                )
            )
            self.trajectory_geometry_detach_input = bool(
                getattr(cfg.TRAJECTORY_GEOMETRY, "DETACH_INPUT", True)
            )
            self.register_buffer(
                "trajectory_geometry_logit_delta", torch.tensor(0.0), persistent=False
            )
        self.adaptive_fusion_enable = (
            hasattr(cfg, "ADAPTIVE_FUSION")
            and getattr(cfg.ADAPTIVE_FUSION, "ENABLE", False)
        )
        if self.adaptive_fusion_enable:
            self.adaptive_fusion = D2STQueryAdaptiveFusion(
                num_branches=int(getattr(cfg.ADAPTIVE_FUSION, "NUM_BRANCHES", 2)),
                hidden_ratio=float(
                    getattr(cfg.ADAPTIVE_FUSION, "HIDDEN_RATIO", 0.5)
                ),
                dropout=float(getattr(cfg.ADAPTIVE_FUSION, "DROPOUT", 0.1)),
                gate_init=float(getattr(cfg.ADAPTIVE_FUSION, "GATE_INIT", 0.0)),
                temperature=float(
                    getattr(cfg.ADAPTIVE_FUSION, "TEMPERATURE", 1.0)
                ),
            )
            self.register_buffer(
                "adaptive_fusion_logit_delta", torch.tensor(0.0), persistent=False
            )
        self.relation_match_enable = (
            hasattr(cfg, "RELATION_MATCH")
            and getattr(cfg.RELATION_MATCH, "ENABLE", False)
        )
        if self.relation_match_enable:
            self.relation_matcher = D2STRelationalTemporalMatcher(
                num_frames=self.num_frames,
                distance_scale=float(getattr(cfg.RELATION_MATCH, "DISTANCE_SCALE", 16.0)),
                temperature=float(getattr(cfg.RELATION_MATCH, "TEMPERATURE", 1.0)),
            )
            self.relation_match_alpha = nn.Parameter(torch.tensor(float(
                getattr(cfg.RELATION_MATCH, "ALPHA_INIT", -4.0)
            )))
            self.register_buffer(
                "relation_match_logit_delta", torch.tensor(0.0), persistent=False
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
        if self.spatial_pattern_enable:
            weights["spatial_pattern"] = torch.sigmoid(
                self.spatial_pattern_alpha
            ).detach().item()
        if self.trajectory_geometry_enable:
            weights["trajectory_geometry"] = torch.sigmoid(
                self.trajectory_geometry_alpha
            ).detach().item()
        if self.relation_match_enable:
            weights["relation_match"] = torch.sigmoid(
                self.relation_match_alpha
            ).detach().item()
        return weights

    def get_relation_match_diagnostics(self):
        if not self.relation_match_enable:
            return {}
        diagnostics = self.relation_matcher.get_diagnostics()
        diagnostics["logit_delta"] = self.relation_match_logit_delta.detach().item()
        diagnostics["gate_trainable"] = self.relation_match_alpha.requires_grad
        return diagnostics

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

    def get_spatial_pattern_diagnostics(self):
        if not self.spatial_pattern_enable:
            return {}
        diagnostics = self.spatial_pattern_matcher.get_diagnostics()
        diagnostics["last_episode_raw_logit_delta"] = (
            self.spatial_pattern_logit_delta.detach().item()
        )
        diagnostics["gate_trainable"] = self.spatial_pattern_alpha.requires_grad
        diagnostics["detach_input"] = self.spatial_pattern_detach_input
        return diagnostics

    def get_trajectory_geometry_diagnostics(self):
        if not self.trajectory_geometry_enable:
            return {}
        diagnostics = self.trajectory_geometry_matcher.get_diagnostics()
        diagnostics["logit_delta"] = (
            self.trajectory_geometry_logit_delta.detach().item()
        )
        diagnostics["gate_trainable"] = self.trajectory_geometry_alpha.requires_grad
        diagnostics["detach_input"] = self.trajectory_geometry_detach_input
        return diagnostics

    def get_adaptive_fusion_diagnostics(self):
        if not self.adaptive_fusion_enable:
            return {}
        diagnostics = self.adaptive_fusion.get_diagnostics()
        diagnostics["logit_delta"] = self.adaptive_fusion_logit_delta.detach().item()
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
        return (
            self._bimhm_reduce(dist, dim=3).sum(2)
            + self._bimhm_reduce(dist, dim=2).sum(2)
        )

    def _bimhm_reduce(self, distances, dim):
        temperature = getattr(self, "bimhm_softmin_temperature", 0.0)
        if temperature <= 0.0:
            return distances.min(dim=dim).values
        weights = F.softmax(-distances / temperature, dim=dim)
        return (weights * distances).sum(dim=dim)

    def _bimhm_class_dist_per_support(
        self, support_features, query_features, support_labels
    ):
        """Match each support video independently, then average class distances."""
        support_norm = F.normalize(support_features, dim=-1)
        query_norm = F.normalize(query_features, dim=-1)
        frame_sim = torch.einsum("qtd,sud->qstu", query_norm, support_norm)
        dist = 1 - frame_sim
        support_dist = (
            self._bimhm_reduce(dist, dim=3).sum(2)
            + self._bimhm_reduce(dist, dim=2).sum(2)
        )

        class_distances = [
            support_dist[:, support_labels == label].mean(dim=1)
            for label in torch.unique(support_labels)
        ]
        return torch.stack(class_distances, dim=1)

    def get_feat(self, x, return_patch_tokens=False):
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
        cls_features = x[:, 0, :]
        if not return_patch_tokens:
            return cls_features
        return cls_features, x[:, 1:, :]

    def forward(self, inputs):
        support_images, query_images = inputs['support_set'], inputs['target_set']
        return_patch_tokens = (
            self.spatial_pattern_enable or self.trajectory_geometry_enable
        )
        if return_patch_tokens:
            support_features, support_patch_features = self.get_feat(
                support_images, return_patch_tokens=True
            )
            query_features, query_patch_features = self.get_feat(
                query_images, return_patch_tokens=True
            )
        else:
            support_features = self.get_feat(support_images)
            query_features = self.get_feat(query_images)
        support_labels = inputs['support_labels']
        unique_labels = torch.unique(support_labels)

        support_features = support_features.reshape(-1, self.num_frames, self.args.ADAPTER.WIDTH)
        query_features = query_features.reshape(-1, self.num_frames, self.args.ADAPTER.WIDTH)
        if return_patch_tokens:
            support_patch_features = support_patch_features.reshape(
                -1,
                self.num_frames,
                support_patch_features.shape[1],
                support_patch_features.shape[2],
            )
            query_patch_features = query_patch_features.reshape(
                -1,
                self.num_frames,
                query_patch_features.shape[1],
                query_patch_features.shape[2],
            )
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

        if self.support_aggregation_enable:
            class_dist = self._bimhm_class_dist_per_support(
                support_features, query_features, support_labels
            )
        else:
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

        spatial_pattern_logits = None
        if self.spatial_pattern_enable:
            spatial_support_input = support_patch_features
            spatial_query_input = query_patch_features
            if self.spatial_pattern_detach_input:
                spatial_support_input = spatial_support_input.detach()
                spatial_query_input = spatial_query_input.detach()
            spatial_pattern_logits = self.spatial_pattern_matcher(
                spatial_support_input,
                spatial_query_input,
                support_labels,
            )
            self.spatial_pattern_logit_delta.copy_(
                (spatial_pattern_logits - base_logits).detach().abs().mean().to(
                    self.spatial_pattern_logit_delta
                )
            )

        trajectory_geometry_logits = None
        if self.trajectory_geometry_enable:
            trajectory_support_input = support_patch_features
            trajectory_query_input = query_patch_features
            if self.trajectory_geometry_detach_input:
                trajectory_support_input = trajectory_support_input.detach()
                trajectory_query_input = trajectory_query_input.detach()
            trajectory_geometry_logits = self.trajectory_geometry_matcher(
                trajectory_support_input,
                trajectory_query_input,
                support_labels,
            )
            self.trajectory_geometry_logit_delta.copy_(
                (trajectory_geometry_logits - base_logits).detach().abs().mean().to(
                    self.trajectory_geometry_logit_delta
                )
            )

        relation_match_logits = None
        if self.relation_match_enable:
            relation_match_logits = self.relation_matcher(
                support_features_raw,
                query_features_raw,
                support_labels,
            )
            self.relation_match_logit_delta.copy_(
                (relation_match_logits - base_logits).detach().abs().mean().to(
                    self.relation_match_logit_delta
                )
            )

        # Each optional branch learns an independent residual from the same
        # D2ST decision unless query-wise adaptive fusion is enabled.
        logits = base_logits
        if focus_logits is not None:
            focus_weight = torch.sigmoid(self.focus_alpha)
            logits = logits + focus_weight * (focus_logits - base_logits.detach())
        logits_without_adaptive_fusion = None
        adaptive_fusion_logits = None
        legacy_global_fusion_logits = None
        if self.adaptive_fusion_enable:
            if task_logits is None or multi_velocity_logits is None:
                raise RuntimeError(
                    "Adaptive fusion requires TASK_MATCH and MULTI_VELOCITY branches"
                )
            adaptive_fusion_logits, logits_without_adaptive_fusion = self.adaptive_fusion(
                logits,
                [task_logits, multi_velocity_logits],
                [
                    torch.sigmoid(self.task_match_alpha),
                    torch.sigmoid(self.multi_velocity_alpha),
                ],
            )
            legacy_global_fusion_logits = logits
            legacy_global_fusion_logits = legacy_global_fusion_logits + (
                torch.sigmoid(self.task_match_alpha)
                * (task_logits - base_logits.detach())
            )
            legacy_global_fusion_logits = legacy_global_fusion_logits + (
                torch.sigmoid(self.multi_velocity_alpha)
                * (multi_velocity_logits - base_logits.detach())
            )
            logits = adaptive_fusion_logits
            self.adaptive_fusion_logit_delta.copy_(
                (adaptive_fusion_logits - base_logits).detach().abs().mean().to(
                    self.adaptive_fusion_logit_delta
                )
            )
        else:
            if task_logits is not None:
                task_weight = torch.sigmoid(self.task_match_alpha)
                logits = logits + task_weight * (task_logits - base_logits.detach())
        if proto_calib_logits is not None:
            proto_calib_weight = torch.sigmoid(self.proto_calib_alpha)
            logits = logits + proto_calib_weight * (
                proto_calib_logits - base_logits.detach()
            )
        if not self.adaptive_fusion_enable and multi_velocity_logits is not None:
            multi_velocity_weight = torch.sigmoid(self.multi_velocity_alpha)
            logits = logits + multi_velocity_weight * (
                multi_velocity_logits - base_logits.detach()
            )
        logits_without_relation_match = logits
        if relation_match_logits is not None:
            relation_match_weight = torch.sigmoid(self.relation_match_alpha)
            logits = logits + relation_match_weight * (
                relation_match_logits - base_logits.detach()
            )
        logits_without_spatial = logits
        if spatial_pattern_logits is not None:
            spatial_pattern_weight = torch.sigmoid(self.spatial_pattern_alpha)
            logits = logits + spatial_pattern_weight * (
                spatial_pattern_logits - base_logits.detach()
            )
        logits_without_trajectory_geometry = logits
        if trajectory_geometry_logits is not None:
            trajectory_geometry_weight = torch.sigmoid(
                self.trajectory_geometry_alpha
            )
            logits = logits + trajectory_geometry_weight * (
                trajectory_geometry_logits - base_logits.detach()
            )

        return_dict = {'logits': logits, 'class_logits': class_logits}
        if adaptive_fusion_logits is not None and not self.training:
            return_dict['logits_without_adaptive_fusion'] = (
                logits_without_adaptive_fusion.detach()
            )
            return_dict['adaptive_fusion_logits'] = adaptive_fusion_logits.detach()
            return_dict['legacy_global_fusion_logits'] = (
                legacy_global_fusion_logits.detach()
            )
        if spatial_pattern_logits is not None and not self.training:
            # Evaluate the marginal contribution on identical episodes without
            # another backbone pass. Query labels are used only by the meter.
            return_dict['logits_without_spatial'] = logits_without_spatial.detach()
            return_dict['spatial_pattern_logits'] = spatial_pattern_logits.detach()
        if relation_match_logits is not None and not self.training:
            return_dict['logits_without_relation_match'] = logits_without_relation_match.detach()
            return_dict['relation_match_logits'] = relation_match_logits.detach()
        if trajectory_geometry_logits is not None and not self.training:
            return_dict['logits_without_trajectory_geometry'] = (
                logits_without_trajectory_geometry.detach()
            )
            return_dict['trajectory_geometry_logits'] = trajectory_geometry_logits.detach()
        return return_dict
