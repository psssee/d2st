#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited.

"""
OTAM Frame Selector 鈥?learnable frame selection module.

Given T frames (extracted features), scores each frame with a lightweight MLP,
then selects K frames via hard segment partitioning 鈥?guarantees temporal
diversity with zero overhead. Selected indices are SORTED TEMPORALLY to
preserve OTAM sequence alignment.

鉁?瀵归綈鏍稿績鐩爣 v2: OTAM 涓夊厓缁勬崯澶辨槸**鍞竴鏍稿績璁粌淇″彿**锛?
   姊害浠?OTAM 绱Н璺濈**瀹屾暣鍥炰紶**鍒?score_net 鐨勬瘡涓潈閲嶃€?
   - 纭笁鍏冪粍鎸栨帢锛堟渶闅炬/璐熸牱鏈級鎻愬崌璁粌鏁堢巼
   - 鍒嗘暟闂撮殧鎹熷け榛樿绂佺敤锛坵eight=0锛夛紝涓嶅弬涓庢搴?
   - 鍙井鍒?OTAM锛堟棤 in-place 鎿嶄綔锛宼orch.logsumexp 鏁板€肩ǔ瀹氾級
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 鉁?瀵归綈鏍稿績鐩爣锛氬鍏ョЩ鍒版枃浠堕《閮紝绂佹鍑芥暟鍐呴噸澶嶅鍏?
def cos_sim(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return torch.matmul(x, y.t())
# ============================================================================
# Feature dimension lookup table
# ============================================================================
BACKBONE_FEAT_DIM_MAP = {
    "RN50": 1024, "RN101": 512, "RN50x4": 640, "RN50x16": 768,
    "RN50x64": 1024, "ViT-B/32": 512, "ViT-B/16": 512,
    "ViT-L/14": 768, "ViT-L/14@336px": 768,
}


# ============================================================================
# 鉁?瀵归綈鏍稿績鐩爣锛氬彲寰垎 OTAM 绱Н璺濈
#  - 鏃?in-place 鎿嶄綔锛堜笉閲嶅鍐欏叆鍚屼竴 buffer锛屼笉瑙﹀彂 PyTorch 鐗堟湰璁℃暟鍣ㄩ€掑锛?
#  - 浣跨敤 torch.logsumexp 淇濊瘉鏁板€肩ǔ瀹氭€?
#  - 姊害浠庤緭鍑哄畬鏁村洖浼犲埌 dists 杈撳叆
# ============================================================================
def otam_cum_dist_diff(dists, lbda=0.5):
    """
    Differentiable OTAM cumulative distance with numerically stable
    log-sum-exp.  Gradient flows from output back to `dists`.

    淇: 鍘熷 OTAM_cum_dist_v2 鍦ㄥ惊鐜腑瀵?cum_dists 鍋?in-place 鍐欏叆锛?
    瀵艰嚧 PyTorch autograd 鏃犳硶杩借釜姊害锛堢増鏈鏁板櫒閫掑锛夈€?
    鏈増鏈瘡姝ュ垱寤烘柊寮犻噺 + torch.logsumexp锛岃绠楀浘瀹屾暣銆佹暟鍊肩ǔ瀹氥€?

    Args:
        dists: (B, Q, K, K+2) 甯ч棿璺濈鐭╅樀锛堢粡 padding 鍚庯級
        lbda: soft-min 骞虫粦鍙傛暟 位
    Returns:
        (B, Q) 姣忓瑙嗛鐨勭疮绉?OTAM 璺濈锛屼繚鎸佹搴?
    """
    B, Q, K, M = dists.shape
    # M = K + 2 (棣栧熬 padding)

    # 绗?0 琛岋細鏍囧噯鍓嶇紑鍜岋紙torch.cumsum 鍘熺敓鍙井锛?
    # D[0, m] = sum_{i=1}^{m} dist[0, i]
    row0 = torch.cumsum(dists[:, :, 0, :], dim=-1)  # (B, Q, M)
    rows = [row0]

    for l in range(1, K):
        prev = rows[-1]                     # D[l-1, :]  (B, Q, M)锛屾湁姊害
        steps = [torch.zeros(B, Q, 1, device=dists.device)]  # D[l, 0] = 0 padding

        # 鈹€鈹€ m = 1锛氱涓€闈為浂鍒?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        # D[l,1] = dist[l,1] - 位路log(exp(-D[l-1,0]/位)+exp(-D[l-1,1]/位)+exp(-D[l,0]/位))
        terms_1 = torch.stack([
            -prev[:, :, 0:1] / lbda,           # -D[l-1,0]/位
            -prev[:, :, 1:2] / lbda,           # -D[l-1,1]/位
            torch.zeros(B, Q, 1, device=dists.device),  # -D[l,0]/位 = 0
        ], dim=0)
        s1 = dists[:, :, l, 1:2] - lbda * torch.logsumexp(terms_1, dim=0)
        steps.append(s1)

        # 鈹€鈹€ m = 2 鍒?M-2锛氫腑闂村垪 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        # D[l,m] = dist[l,m] - 位路log(exp(-D[l-1,m-1]/位)+exp(-D[l,m-1]/位))
        for m in range(2, M - 1):
            terms = torch.stack([
                -prev[:, :, m-1:m] / lbda,     # -D[l-1,m-1]/位
                -steps[m - 1] / lbda,           # -D[l,m-1]/位
            ], dim=0)
            sm = dists[:, :, l, m:m+1] - lbda * torch.logsumexp(terms, dim=0)
            steps.append(sm)

        # 鈹€鈹€ m = M-1锛氭渶鍚庝竴鍒?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        # D[l,-1] = dist[l,-1] - 位路log(exp(-D[l-1,-2]/位)+exp(-D[l-1,-1]/位)+exp(-D[l,-2]/位))
        terms_last = torch.stack([
            -prev[:, :, -2:-1] / lbda,         # -D[l-1,-2]/位
            -prev[:, :, -1:] / lbda,            # -D[l-1,-1]/位
            -steps[-1] / lbda,                  # -D[l,-2]/位
        ], dim=0)
        slast = dists[:, :, l, -1:] - lbda * torch.logsumexp(terms_last, dim=0)
        steps.append(slast)

        rows.append(torch.cat(steps, dim=-1))

    # 杩斿洖鏈€鍙充笅瑙掔疮绉窛绂?(B, Q)
    return rows[-1][:, :, -1]


# ============================================================================
# Core Frame Selector
# ============================================================================
class OTAMFrameSelector(nn.Module):
    """
    Hard segment-based frame selector.  Each segment independently picks
    per_seg frames via topk, then all selected indices are SORTED
    TEMPORALLY to preserve sequence order for OTAM alignment.
    """
    def __init__(self, feat_dim=512, total_frames=32, select_frames=8, segments=4):
        super().__init__()
        self.T = total_frames
        self.K = select_frames
        self.segments = segments

        assert self.T % self.segments == 0
        assert self.K % self.segments == 0
        self.per_seg = self.K // self.segments
        self.segment_size = self.T // self.segments

        # 鈿?Lightweight scoring network
        self.score_net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.GELU(),
            nn.Linear(feat_dim // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.score_net:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, video_feats, return_scores=False):
        """
        Args:
            video_feats: (B, T, D) frame features
            return_scores: if True, also return scores
        Returns:
            selected_indices: (B, K) indices sorted temporally
            frame_scores (optional): (B, T)
        """
        B, T, C = video_feats.shape
        if T != self.T:
            raise ValueError(f"Expected {self.T} frames, got {T}")

        # 1. Score each frame
        frame_scores = self.score_net(video_feats).squeeze(-1)  # (B, T)

        # 2. Hard segment partitioning
        selected_indices = []
        for i in range(self.segments):
            start = i * self.segment_size
            end = (i + 1) * self.segment_size
            seg_scores = frame_scores[:, start:end]
            _, seg_idx = torch.topk(seg_scores, self.per_seg, dim=-1)
            seg_idx = seg_idx + start  # global indices
            selected_indices.append(seg_idx)

        selected_indices = torch.cat(selected_indices, dim=-1)  # (B, K)
        selected_indices, _ = selected_indices.sort(dim=-1)     # (B, K)

        if return_scores:
            return selected_indices, frame_scores
        return selected_indices

    def select_features(self, video_feats, return_scores=True):
        """One-stop: select frame indices, gather features, sort temporally."""
        indices, scores = self.forward(video_feats, return_scores=True)
        B, K = indices.shape
        batch_idx = torch.arange(B, device=indices.device).unsqueeze(1).expand(-1, K)
        selected_feats = video_feats[batch_idx, indices]
        if return_scores:
            return selected_feats, indices, scores
        return selected_feats, indices

    def extra_repr(self):
        return (f"T={self.T}, K={self.K}, segments={self.segments}, "
                f"per_seg={self.per_seg}")


# ============================================================================
# 鉁?瀵归綈鏍稿績鐩爣锛歄TAM 涓夊厓缁勬崯澶憋紙鍙井鍒?+ 纭笁鍏冪粍鎸栨帢锛?
# ============================================================================
def otam_triplet_loss(selected_feats, labels, frame_scores=None,
                      margin=0.3, index_compact_weight=0.05,
                      score_margin_weight=0.0,  # 鉁?榛樿 0锛氳緟鍔╂崯澶遍粯璁ょ鐢?
                      indices=None, segments=4, total_frames=32):
    """
    OTAM 涓夊厓缁勬崯澶?鈥斺€?鍞竴鏍稿績璁粌淇″彿銆?

    鉁?瀵归綈鏍稿績鐩爣锛?
    1. selected_feats 淇濇寔姊害锛堢Щ闄?detach()锛夛紝OTAM 璺濈璁＄畻浣跨敤 otam_cum_dist_diff
    2. 纭笁鍏冪粍鎸栨帢锛氭瘡 anchor 鍙栧悓绫绘渶杩滄鏍锋湰 + 寮傜被鏈€杩戣礋鏍锋湰
    3. 鎬绘崯澶?= OTAM 涓夊厓缁?脳 1.0 + 绱у噾鎬?脳 0.05锛堝垎鏁伴棿闅旈粯璁ょ鐢級
    4. 鎵€鏈夊垎娈靛弬鏁版潵鑷叆鍙?segments / total_frames锛屾棤纭紪鐮?

    Args:
        selected_feats: (B, K, D) 鈥?鏈夋搴?猸?浠?STE 杈撳嚭
        labels:  (B,)
        frame_scores: (B, T) 鈥?杈呭姪鎹熷け鐢紙榛樿 weight=0 涓嶅惎鐢級
        margin:  triplet margin
        index_compact_weight: 绱у噾鎬ф潈閲?(鈮?.05)
        score_margin_weight: 鍒嗘暟闂撮殧鏉冮噸锛堥粯璁?0 = 绂佺敤锛?
        indices: (B, K) 鈥?绱у噾鎬ф崯澶辩敤
        segments: 鍒嗘鏁帮紙涓?FRAME_SELECTOR.SEGMENTS 涓€鑷达級
        total_frames: 鎬诲抚鏁帮紙涓?FRAME_SELECTOR.TOTAL_FRAMES 涓€鑷达級

    Returns:
        loss_dict:
            loss_otam_total    鈥?鍔犳潈鎬绘崯澶憋紙璁粌鐢紝鍚搴︼級
            loss_otam_triplet  鈥?OTAM 涓夊厓缁勬崯澶憋紙涓绘崯澶憋紝鏈夋搴︼級
            loss_otam_pos_dist 鈥?姝ｆ牱鏈窛绂伙紙鐩戞帶锛?
            loss_otam_neg_dist 鈥?璐熸牱鏈窛绂伙紙鐩戞帶锛?
            loss_score_margin  鈥?鍒嗘暟闂撮殧锛堥粯璁?0锛?
            loss_compact       鈥?绱㈠紩绱у噾鎬э紙灏忔潈閲嶅厹搴曪級
    """
    loss_dict = {}
    B, K, D = selected_feats.shape
    device = selected_feats.device

    if B < 2:
        loss_dict["loss_otam_total"] = torch.tensor(0.0, device=device)
        return loss_dict

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    #  Part A: OTAM 涓夊厓缁勬崯澶?鈥?鍞竴鏍稿績璁粌淇″彿
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    # 鉁?鍏抽敭锛歴elected_feats **鏃?* detach()锛屾搴︿竴璺洖浼犲埌 score_net
    flat = selected_feats.reshape(B * K, D)
    sim = cos_sim(flat, flat)                               # (B*K, B*K)锛屾湁姊害
    dists_4d = (1 - sim).reshape(B, K, B, K).permute(0, 2, 1, 3).contiguous()  # (B,B,K,K)
    otam_all = otam_cum_dist_diff(dists_4d)                 # (B,B)锛屾湁姊害 鉁?

    # 鈹€鈹€ 纭笁鍏冪粍鎸栨帢 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # 姣?anchor i: hardest_pos = max 鍚岀被璺濈锛堟渶闅炬锛?
    #               hardest_neg = min 寮傜被璺濈锛堟渶闅捐礋锛?
    #               triplet = relu(hardest_pos - hardest_neg + margin)
    triplet_loss = torch.tensor(0.0, device=device)
    valid_anchors = 0

    for i in range(B):
        same = (labels == labels[i]).nonzero(as_tuple=True)[0]
        same = same[same != i]                        # 鎺掗櫎鑷韩锛堣窛绂?鈮?0锛?
        diff = (labels != labels[i]).nonzero(as_tuple=True)[0]

        if len(same) > 0 and len(diff) > 0:
            hardest_pos = otam_all[i, same].max()     # 鍚岀被鏈€杩?= 鏈€闅?
            hardest_neg = otam_all[i, diff].min()     # 寮傜被鏈€杩?= 鏈€闅?
            triplet_loss = triplet_loss + F.relu(hardest_pos - hardest_neg + margin)
            valid_anchors += 1

    if valid_anchors > 0:
        triplet_loss = triplet_loss / valid_anchors

    loss_dict["loss_otam_triplet"] = triplet_loss      # 鉁?鏈夋搴︼紝鍙備笌璁粌

    # 鈹€鈹€ 闅忔満閲囨牱姝?璐熻窛绂伙紙鐩戞帶鐢紝淇濇寔鏃ュ織鍚戝悗鍏煎锛夆攢鈹€
    with torch.no_grad():
        pos_list, neg_list = [], []
        for i in range(B):
            same = (labels == labels[i]).nonzero(as_tuple=True)[0]
            same = same[same != i]
            if len(same) > 0:
                j = same[torch.randint(len(same), (1,)).item()]
                pos_list.append(otam_all[i, j])
            diff = (labels != labels[i]).nonzero(as_tuple=True)[0]
            if len(diff) > 0:
                j = diff[torch.randint(len(diff), (1,)).item()]
                neg_list.append(otam_all[i, j])
        if pos_list and neg_list:
            loss_dict["loss_otam_pos_dist"] = torch.stack(pos_list).mean()
            loss_dict["loss_otam_neg_dist"] = torch.stack(neg_list).mean()
        else:
            loss_dict["loss_otam_pos_dist"] = torch.tensor(0.0, device=device)
            loss_dict["loss_otam_neg_dist"] = torch.tensor(0.0, device=device)

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    #  Part B: 鍒嗘暟闂撮殧杈呭姪鎹熷け锛堥粯璁ゆ潈閲?0 = 绂佺敤锛?
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    # 鉁?榛樿鍏抽棴锛屼粎鍦ㄦ樉寮忚缃?score_margin_weight > 0 鏃剁敓鏁?
    score_loss = torch.tensor(0.0, device=device)
    if frame_scores is not None and indices is not None and score_margin_weight > 0:
        T = frame_scores.shape[1]
        per_seg = K // segments
        seg_sz = T // segments
        for s in range(segments):
            start = s * seg_sz
            end = (s + 1) * seg_sz
            seg_sc = frame_scores[:, start:end]
            seg_idx = indices[:, s * per_seg:(s + 1) * per_seg]
            mask = torch.zeros_like(seg_sc)
            for p in range(per_seg):
                bidx = torch.arange(B, device=device)
                mask[bidx, seg_idx[:, p].clamp(0, seg_sz - 1)] = 1.0
            sel_scores = (seg_sc * mask).sum(-1) / mask.sum(-1).clamp(min=1)
            unsel_scores = (seg_sc * (1 - mask)).sum(-1) / (1 - mask).sum(-1).clamp(min=1)
            score_loss = score_loss + F.relu(unsel_scores - sel_scores + 0.1).mean()
        score_loss = score_loss / segments
    loss_dict["loss_score_margin"] = score_loss

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    #  Part C: 绱㈠紩绱у噾鎬ф儵缃?鈥?浠呯洃鎺ч」锛屾棤姊害 鈿狅笍
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    # 璇ユ崯澶卞熀浜?torch.topk 杈撳嚭鐨?*鏁村瀷绱㈠紩**璁＄畻锛堥潪鍙井鎿嶄綔锛夛紝
    # `indices` 娌℃湁姊害锛屽洜姝?compact_loss **涓嶅弬涓庡弬鏁版洿鏂?*銆?
    # 鐪熸鐨勬椂搴忓鏍锋€х害鏉熺敱纭垎娈佃鍒欙紙segment 鈫?topk per segment锛変繚璇侊紝
    # 姝ゆ崯澶变粎鍋?鈮?.05 鐨勬瀬灏忔潈閲嶅厹搴曪紝閬垮厤閫夊抚鏋佺涓嶅潎鍖€銆?
    # 鍗充娇璁剧疆 index_compact_weight=0 涔熶笉褰卞搷鏍稿績璁粌鐩爣銆? 鉁?
    compact_loss = torch.tensor(0.0, device=device)
    if indices is not None and index_compact_weight > 0 and segments is not None:
        per_seg = K // segments
        seg_sz = total_frames // segments
        for s in range(segments):
            sidx = indices[:, s * per_seg:(s + 1) * per_seg]
            if sidx.shape[1] >= 2:
                sidx_sorted, _ = sidx.sort(dim=-1)
                spread = sidx_sorted[:, 1] - sidx_sorted[:, 0]
                ideal = float(seg_sz) / (per_seg + 1)
                compact_loss = compact_loss + F.relu(ideal - spread.float()).mean()
        compact_loss = compact_loss / segments
    loss_dict["loss_compact"] = index_compact_weight * compact_loss

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    #  Total锛歄TAM 涓夊厓缁勶紙涓伙級+ 绱у噾鎬э紙鍏滃簳锛? 鍒嗘暟闂撮殧锛堥粯璁ょ鐢級
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    total = triplet_loss + score_margin_weight * score_loss + index_compact_weight * compact_loss

    # 鉁?纭繚 total 濮嬬粓鏈?grad_fn锛堝嵆浣?valid_anchors=0锛?
    #    PyTorch 绂佹瀵规棤 grad_fn 鐨勭函甯搁噺璋冪敤 .backward()
    #    selected_feats.sum()*0.0 杩炴帴璁＄畻鍥句絾姊害鎭掍负 0锛屼笉褰卞搷鍙傛暟鏇存柊
    if not total.requires_grad:
        total = total + (selected_feats.sum() * 0.0)

    loss_dict["loss_otam_total"] = total
    return loss_dict


# ============================================================================
# STE Per-Segment Feature Selection (with temporal sorting)
# ============================================================================
def ste_select_features(selector, frame_scores, feats, tau=0.5):
    """
    Per-segment STE selection with TEMPORAL SORTING of output indices.

    Forward: hard topk per segment.   Backward: softmax approximation.
    Output indices are sorted to preserve OTAM sequence alignment.

    Args:
        selector:     OTAMFrameSelector instance
        frame_scores: (B, T) scores from score_net
        feats:        (B, T, D) frame features
        tau:          softmax temperature for STE gradient

    Returns:
        selected_feats: (B, K, D) STE-differentiable, temporally sorted
        indices:        (B, K) hard indices, temporally sorted
    """
    B, T, D = feats.shape
    K = selector.K
    segs = selector.segments
    per_seg = selector.per_seg
    seg_sz = selector.segment_size

    sel_list, idx_list = [], []
    for s in range(segs):
        start, end = s * seg_sz, (s + 1) * seg_sz
        seg_sc = frame_scores[:, start:end]       # (B, seg_sz)
        seg_fe = feats[:, start:end, :]           # (B, seg_sz, D)

        # Hard topk (forward path)
        _, h_idx = torch.topk(seg_sc, per_seg, dim=-1)
        bidx = torch.arange(B, device=feats.device).unsqueeze(1).expand(-1, per_seg)
        hard = seg_fe[bidx, h_idx]                # (B, per_seg, D)

        # Soft weights for gradient path
        sm = F.softmax(seg_sc / tau, dim=-1)
        sw, si = sm.sort(dim=-1, descending=True)
        topk_w = sw[:, :per_seg]
        topk_f = seg_fe.gather(1, si[:, :per_seg].unsqueeze(-1).expand(-1, -1, D))
        soft = topk_w.unsqueeze(-1) * topk_f

        # STE: forward = hard, backward 鈮?soft
        seg_sel = hard + (soft - soft.detach())
        sel_list.append(seg_sel)
        idx_list.append(h_idx + start)

    # 鈹€鈹€ Sort temporally across all selected frames 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    all_idx = torch.cat(idx_list, dim=-1)          # (B, K)
    sorted_idx = all_idx.argsort(dim=-1)           # (B, K) permutation
    batch_perm = torch.arange(B, device=feats.device).unsqueeze(1).expand(-1, K)
    all_feats = torch.cat(sel_list, dim=1)         # (B, K, D)
    sorted_feats = all_feats[batch_perm, sorted_idx]
    sorted_indices = all_idx[batch_perm, sorted_idx]

    return sorted_feats, sorted_indices


def get_feat_dim_from_backbone(backbone_name):
    return BACKBONE_FEAT_DIM_MAP.get(backbone_name, 512)


