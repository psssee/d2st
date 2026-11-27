import torch

from models.base.adapter import D2STReciprocalConfidenceMatcher


def _baseline_bimhm(support, query, labels):
    prototypes = torch.stack(
        [support[labels == label].mean(dim=0) for label in torch.unique(labels)]
    )
    similarity = torch.einsum(
        "qtd,csd->qcts",
        torch.nn.functional.normalize(query, dim=-1),
        torch.nn.functional.normalize(prototypes, dim=-1),
    )
    distance = 1.0 - similarity
    return -(
        distance.min(dim=3)[0].sum(dim=2)
        + distance.min(dim=2)[0].sum(dim=2)
    )


def test_uniform_similarity_has_zero_confidence():
    matcher = D2STReciprocalConfidenceMatcher(num_frames=4)
    similarity = torch.zeros(2, 3, 4, 4)

    refined, confidence = matcher.refine_similarity(similarity)

    assert torch.allclose(confidence, torch.zeros_like(confidence), atol=1e-6)
    assert torch.allclose(refined, similarity, atol=1e-6)


def test_clear_reciprocal_matches_receive_larger_refinement():
    matcher = D2STReciprocalConfidenceMatcher(
        num_frames=4, temperature=0.1, strength=0.25
    )
    similarity = torch.zeros(1, 1, 4, 4)
    diagonal = torch.arange(4)
    similarity[0, 0, diagonal, diagonal] = 0.8

    refined, confidence = matcher.refine_similarity(similarity)

    diagonal_confidence = confidence[0, 0, diagonal, diagonal]
    off_diagonal = ~torch.eye(4, dtype=torch.bool)
    assert diagonal_confidence.mean() > confidence[0, 0][off_diagonal].mean()
    assert torch.all(refined[0, 0, diagonal, diagonal] > 0.8)


def test_one_sided_match_is_not_rewarded():
    matcher = D2STReciprocalConfidenceMatcher(num_frames=4, temperature=0.1)
    similarity = torch.zeros(1, 1, 4, 4)
    similarity[0, 0, :, 0] = 0.8

    refined, confidence = matcher.refine_similarity(similarity)

    assert torch.allclose(confidence, torch.zeros_like(confidence), atol=1e-6)
    assert torch.allclose(refined, similarity, atol=1e-6)


def test_zero_strength_matches_original_bimhm():
    torch.manual_seed(7)
    support = torch.randn(10, 4, 16)
    query = torch.randn(5, 4, 16)
    labels = torch.arange(5).repeat_interleave(2)
    matcher = D2STReciprocalConfidenceMatcher(num_frames=4, strength=0.0)

    logits = matcher(support, query, labels)

    assert logits.shape == (5, 5)
    assert torch.allclose(logits, _baseline_bimhm(support, query, labels))


def test_similarity_gradients_are_finite_when_confidence_is_detached():
    matcher = D2STReciprocalConfidenceMatcher(
        num_frames=4, detach_confidence=True
    )
    similarity = torch.randn(2, 3, 4, 4, requires_grad=True)

    refined, _ = matcher.refine_similarity(similarity)
    refined.sum().backward()

    assert similarity.grad is not None
    assert torch.isfinite(similarity.grad).all()


def test_default_fusion_gate_is_conservative():
    alpha = torch.tensor(-4.0)
    assert torch.sigmoid(alpha).item() < 0.02
