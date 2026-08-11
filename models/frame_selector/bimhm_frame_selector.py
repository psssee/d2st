#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited.

"""Bi-MHM frame selector and losses.

This module keeps the same hard segment-wise top-k selector contract as the
historical OTAM selector, but changes the training signal to better fit the
D2ST matching head.

The default Bi-MHM objective is still available.  A new match-aware variant
adds frame-level matchability supervision:

    frame_matchability = best_negative_distance - best_positive_distance

where best positive is the closest same-class frame from another video and best
negative is the closest frame from a different class.

That gives the selector a direct target for your intuition:
frames should be chosen because they are easy to pair with the same action and
hard to pair with other actions, while segment-wise top-k still protects
coverage.
"""

import torch
import torch.nn.functional as F

from models.frame_selector.otam_frame_selector import OTAMFrameSelector


class BiMHMFrameSelector(OTAMFrameSelector):
    """Segment-wise top-k selector optimized with Bi-MHM-style losses."""

    pass


def bimhm_distance(query_feats, support_feats, normalize=True,
                   normalize_by_frames=False):
    """Compute D2ST-style bidirectional minimum-matching distance."""
    if query_feats.ndim != 3 or support_feats.ndim != 3:
        raise ValueError(
            "bimhm_distance expects (B,T,D) tensors, got "
            f"{tuple(query_feats.shape)} and {tuple(support_feats.shape)}"
        )
    if query_feats.shape[1] != support_feats.shape[1]:
        raise ValueError(
            "Query/support frame counts must match, got "
            f"{query_feats.shape[1]} and {support_feats.shape[1]}"
        )

    if normalize:
        query_feats = F.normalize(query_feats, dim=-1)
        support_feats = F.normalize(support_feats, dim=-1)

    # (Bq, Bs, T, T): query frame i against support frame j.
    frame_dist = 1.0 - torch.einsum("qtd,bsd->qbst", query_feats, support_feats)

    # query -> support and support -> query.
    query_to_support = frame_dist.min(dim=3).values.sum(dim=2)
    support_to_query = frame_dist.min(dim=2).values.sum(dim=2)
    distances = query_to_support + support_to_query

    if normalize_by_frames:
        distances = distances / float(query_feats.shape[1])
    return distances


def _pairwise_bimhm_distance(selected_feats, normalize_by_frames=False):
    return bimhm_distance(
        selected_feats,
        selected_feats,
        normalize_by_frames=normalize_by_frames,
    )


def _hard_triplet_from_pairwise(pairwise_dist, labels, margin=0.3):
    device = pairwise_dist.device
    labels = labels.to(device=device)
    batch_size = labels.numel()
    losses = []
    pos_values = []
    neg_values = []

    for i in range(batch_size):
        same = labels == labels[i]
        same[i] = False
        different = labels != labels[i]

        if same.any() and different.any():
            hardest_positive = pairwise_dist[i, same].max()
            hardest_negative = pairwise_dist[i, different].min()
            losses.append(F.relu(hardest_positive - hardest_negative + margin))
            pos_values.append(hardest_positive)
            neg_values.append(hardest_negative)

    if losses:
        loss = torch.stack(losses).mean()
        pos_dist = torch.stack(pos_values).mean()
        neg_dist = torch.stack(neg_values).mean()
    else:
        loss = pairwise_dist.sum() * 0.0
        pos_dist = pairwise_dist.detach().new_zeros(())
        neg_dist = pairwise_dist.detach().new_zeros(())

    return loss, pos_dist, neg_dist, len(losses)


def _prototype_classification_loss(selected_feats, labels, temperature=1.0,
                                   normalize_by_frames=False):
    """D2ST-aligned class-prototype CE loss."""
    device = selected_feats.device
    labels = labels.to(device=device)
    unique_labels = torch.unique(labels, sorted=True)
    logits = []
    targets = []

    for i in range(selected_feats.shape[0]):
        class_distances = []
        valid_anchor = True
        for class_id in unique_labels:
            members = labels == class_id
            if class_id == labels[i]:
                members[i] = False
            if not members.any():
                valid_anchor = False
                break

            prototype = selected_feats[members].mean(dim=0, keepdim=True)
            query = selected_feats[i:i + 1]
            distance = bimhm_distance(
                query,
                prototype,
                normalize_by_frames=normalize_by_frames,
            )[0, 0]
            class_distances.append(distance)

        if valid_anchor and len(class_distances) == len(unique_labels):
            logits.append(-torch.stack(class_distances) / max(float(temperature), 1e-6))
            targets.append((unique_labels == labels[i]).nonzero(as_tuple=False)[0, 0])

    if not logits:
        return selected_feats.sum() * 0.0, 0

    return F.cross_entropy(torch.stack(logits), torch.stack(targets)), len(logits)


def _frame_matchability_targets(all_feats, labels, normalize=True,
                                exclude_same_video=True):
    """Build frame-level matchability targets from the current batch.

    A frame is considered better when it is closer to same-class frames from
    other videos and farther from frames of different classes.
    """
    if all_feats is None:
        raise ValueError("all_feats is required for match-aware training")
    if all_feats.ndim != 3:
        raise ValueError(
            f"Expected all_feats as (B,T,D), got {tuple(all_feats.shape)}"
        )

    device = all_feats.device
    B, T, D = all_feats.shape
    flat = all_feats.reshape(B * T, D)
    if normalize:
        flat = F.normalize(flat, dim=-1)

    dist = 1.0 - torch.matmul(flat, flat.t())
    frame_labels = labels.to(device=device).repeat_interleave(T)
    video_ids = torch.arange(B, device=device).repeat_interleave(T)

    same_class = frame_labels[:, None] == frame_labels[None, :]
    same_video = video_ids[:, None] == video_ids[None, :]
    if exclude_same_video:
        pos_mask = same_class & (~same_video)
    else:
        pos_mask = same_class
    pos_mask.fill_diagonal_(False)
    neg_mask = ~same_class
    neg_mask.fill_diagonal_(False)

    inf = torch.finfo(dist.dtype).max
    pos = dist.masked_fill(~pos_mask, inf).min(dim=1).values
    neg = dist.masked_fill(~neg_mask, inf).min(dim=1).values
    valid = torch.isfinite(pos) & torch.isfinite(neg)

    target = (neg - pos).masked_fill(~valid, 0.0)
    return target.view(B, T), valid.view(B, T)


def _matchability_regression_loss(frame_scores, all_feats, labels,
                                  normalize_targets=True):
    """Supervise frame scores with frame-level matchability targets."""
    if frame_scores is None and all_feats is None:
        device = labels.device if hasattr(labels, "device") else "cpu"
        return torch.zeros((), device=device), 0
    if frame_scores is None:
        return all_feats.sum() * 0.0, 0
    if all_feats is None:
        return frame_scores.sum() * 0.0, 0

    target, valid = _frame_matchability_targets(all_feats, labels)
    if not valid.any():
        return frame_scores.sum() * 0.0, 0

    scores = frame_scores.float()
    target = target.float()
    mask = valid.float()

    if normalize_targets:
        score_count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        target_count = score_count
        score_mean = (scores * mask).sum(dim=1, keepdim=True) / score_count
        target_mean = (target * mask).sum(dim=1, keepdim=True) / target_count
        score_var = ((scores - score_mean) ** 2 * mask).sum(dim=1, keepdim=True) / score_count
        target_var = ((target - target_mean) ** 2 * mask).sum(dim=1, keepdim=True) / target_count
        scores = (scores - score_mean) / torch.sqrt(score_var + 1e-6)
        target = (target - target_mean) / torch.sqrt(target_var + 1e-6)

    loss = F.smooth_l1_loss(scores[valid], target[valid])
    return loss, int(valid.sum().item())


def bimhm_loss(
    selected_feats,
    labels,
    frame_scores=None,
    all_feats=None,
    margin=0.3,
    temperature=1.0,
    mode="triplet",
    triplet_weight=1.0,
    class_ce_weight=1.0,
    match_weight=1.0,
    normalize_by_frames=False,
    match_normalize_targets=True,
):
    """Bi-MHM selector loss.

    Modes:
        triplet      -> hard positive/negative margin loss.
        class_ce     -> D2ST-style class-prototype cross entropy.
        match_aware  -> triplet + frame-level matchability regression.
        combined     -> triplet + class_ce + matchability regression.
    """
    mode = str(mode).lower()
    if mode not in {"triplet", "class_ce", "match_aware", "combined"}:
        raise ValueError(
            f"Unknown Bi-MHM loss mode '{mode}'. "
            "Use triplet, class_ce, match_aware, or combined."
        )

    pairwise_dist = _pairwise_bimhm_distance(
        selected_feats,
        normalize_by_frames=normalize_by_frames,
    )
    triplet, pos_dist, neg_dist, valid_triplets = _hard_triplet_from_pairwise(
        pairwise_dist, labels, margin=margin
    )

    class_ce = selected_feats.sum() * 0.0
    valid_class_anchors = 0
    if mode in {"class_ce", "combined"}:
        class_ce, valid_class_anchors = _prototype_classification_loss(
            selected_feats,
            labels,
            temperature=temperature,
            normalize_by_frames=normalize_by_frames,
        )

    match_loss = selected_feats.sum() * 0.0
    valid_match_frames = 0
    if mode in {"match_aware", "combined"} and float(match_weight) > 0:
        match_loss, valid_match_frames = _matchability_regression_loss(
            frame_scores,
            all_feats,
            labels,
            normalize_targets=match_normalize_targets,
        )

    if mode == "triplet":
        total = float(triplet_weight) * triplet
    elif mode == "class_ce":
        total = float(class_ce_weight) * class_ce
    elif mode == "match_aware":
        total = float(triplet_weight) * triplet + float(match_weight) * match_loss
    else:
        total = (
            float(triplet_weight) * triplet
            + float(class_ce_weight) * class_ce
            + float(match_weight) * match_loss
        )

    if not total.requires_grad:
        total = total + selected_feats.sum() * 0.0

    return {
        "loss_bimhm_total": total,
        "loss_bimhm_triplet": triplet,
        "loss_bimhm_class_ce": class_ce,
        "loss_bimhm_match": match_loss,
        "loss_bimhm_pos_dist": pos_dist,
        "loss_bimhm_neg_dist": neg_dist,
        "valid_bimhm_triplets": valid_triplets,
        "valid_bimhm_class_anchors": valid_class_anchors,
        "valid_bimhm_match_frames": valid_match_frames,
    }

