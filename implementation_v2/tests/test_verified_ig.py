"""Intégration Captum réelle sur modèle jouet ; aucune qualité humaine revendiquée."""
from types import SimpleNamespace
import numpy as np
import pytest


def fixture_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("captum")
    class Model(torch.nn.Module):
        config = SimpleNamespace(model_type="xlm-roberta")
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding.from_pretrained(torch.tensor([[0.], [.2], [.4]]), freeze=False)
        def get_input_embeddings(self):
            return self.embedding
        def forward(self, inputs_embeds, attention_mask=None, position_ids=None):
            assert position_ids[0].tolist() == [1, 2]
            score = inputs_embeds.sum((1, 2))
            return SimpleNamespace(logits=torch.stack((score, -score), dim=1))
    class Tokenizer:
        pad_token_id = 0
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]]),
                    "offset_mapping": torch.tensor([[[0, 3], [4, 7]]])}
    return torch, Model(), Tokenizer()


def test_native_ig_probability_completeness_and_two_checks():
    from helm.explainers import VerifiedIGExplainer
    torch, model, tokenizer = fixture_model()
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        p0 = 1/(1+np.exp(-1.2))
        result = VerifiedIGExplainer(model, tokenizer, max_nodes=1152).explain(
            "mot mot", lambda texts: np.tile([p0, 1-p0], (len(texts), 1)))
    finally:
        torch.set_num_threads(old_threads)
    m = result.metadata
    assert m["valid"] is True and m["target_class"] == 0
    assert len(m["checks"]) == 2
    assert [c["total_nodes_used"] for c in m["checks"]] == [768, 1152]
    assert [c["quadrature_nodes_pair"] for c in m["checks"]] == [[16, 32], [24, 48]]
    assert sum(m["scores_by_position"]) == pytest.approx(p0-.5, abs=1e-6)
    assert result.token_scores["mot"] == pytest.approx(p0-.5, abs=1e-6)
    assert m["scores_by_position"][1] == pytest.approx(2*m["scores_by_position"][0], abs=1e-6)


def test_forward_mismatch_fails_without_fallback():
    from helm.explainers import VerifiedIGExplainer
    _, model, tokenizer = fixture_model()
    with pytest.raises(ValueError, match="forward"):
        VerifiedIGExplainer(model, tokenizer).explain("mot mot", lambda _: [[.6, .4]])


def test_numerical_failure_keeps_diagnostics_without_token_explanation(monkeypatch):
    from helm.explainers import VerifiedIGExplainer
    torch, model, tokenizer = fixture_model()
    from captum.attr import IntegratedGradients
    def broken(self, inputs, **kwargs):
        return torch.full_like(inputs, float("nan")), torch.tensor(float("nan"))
    monkeypatch.setattr(IntegratedGradients, "attribute", broken)
    p0 = 1/(1+np.exp(-1.2))
    result = VerifiedIGExplainer(model, tokenizer).explain("mot mot", lambda _: [[p0, 1-p0]])
    assert result.metadata["valid"] is False
    assert not result.token_scores
    assert result.metadata["effective_method"] == "integrated_gradients"
