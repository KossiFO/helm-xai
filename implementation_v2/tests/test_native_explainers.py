"""Intégrations réelles des extras, sans téléchargement de modèle.

Les modèles sont synthétiques ; ces tests vérifient les appels aux bibliothèques,
la classe expliquée et les gradients, pas la qualité XAI sur le corpus de thèse.
"""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("with_mask", [True, False])
def test_captum_attributions_match_linear_model(with_mask):
    torch = pytest.importorskip("torch")
    pytest.importorskip("captum")
    from helm.explainers import IGExplainer

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            inputs = {"input_ids": torch.tensor([[1, 2]])}
            if with_mask:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            return inputs

        def convert_ids_to_tokens(self, ids):
            return [["[PAD]", "idiot", "merci"][i] for i in ids]

    class LinearClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding.from_pretrained(
                torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]]), freeze=False,
            )

        def forward(self, input_ids, attention_mask=None):
            values = self.embedding(input_ids).sum(dim=-1)
            if attention_mask is not None:
                assert attention_mask.shape == input_ids.shape
                values = values * attention_mask
            score = values.sum(dim=-1)
            return SimpleNamespace(logits=torch.stack((-score, score), dim=1))

    model = LinearClassifier()
    result = IGExplainer(model=model, embedding_layer=model.embedding,
                         tokenizer=Tokenizer()).explain(
        "idiot merci", predict_fn=None, num_features=8,
    )
    assert result.metadata["mode"] == "captum"
    assert result.token_scores == pytest.approx({"idiot": 1.0, "merci": 2.0})
    assert abs(result.convergence_delta) < 1e-6


@pytest.mark.parametrize("text", ["idiot gentil", "merci gentil"])
def test_native_anchors_explains_predicted_class(text):
    pytest.importorskip("anchor.anchor_text")
    pytest.importorskip("spacy")
    from helm.explainers import AnchorsExplainer

    def predict(texts):
        p = np.array([0.9 if "idiot" in t.split() else 0.1 for t in texts])
        return np.column_stack((1 - p, p))

    state = np.random.get_state()
    try:
        np.random.seed(42)
        result = AnchorsExplainer(use_native=True, num_samples=20).explain(
            text, predict, num_features=8,
        )
    finally:
        np.random.set_state(state)
    assert result.metadata["mode"] == "native"
    assert result.metadata["precision"] >= 0.95
    if "idiot" in text:
        assert "idiot" in result.metadata["anchor_words"]


def test_internal_anchors_can_be_selected_with_native_installed():
    from helm.explainers import AnchorsExplainer

    def predict(texts):
        p = np.array([0.9 if "idiot" in t.split() else 0.1 for t in texts])
        return np.column_stack((1 - p, p))

    result = AnchorsExplainer(use_native=False).explain("idiot gentil", predict)
    assert result.metadata["mode"] == "beam_search"
