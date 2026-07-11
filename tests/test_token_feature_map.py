"""Self-check for SAESteering.build_token_feature_map()."""

import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, "drrik")


def test_build_token_feature_map():
    """Verify build_token_feature_map returns correct structure."""
    mock_sae = MagicMock()
    mock_sae.eval.return_value = mock_sae
    mock_sae.activation_dim = 4
    mock_sae.hidden_dim = 8

    with patch("drrik.steering.LanguageModel") as mock_lm_cls:
        mock_model = MagicMock()
        mock_lm_instance = MagicMock()
        mock_lm_instance._module = mock_model
        mock_lm_cls.return_value = mock_lm_instance

        with patch("drrik.steering.AutoTokenizer"):
            tokenizer_mock = MagicMock()
            tokenizer_mock.all_special_ids = [100, 101]
            tokenizer_mock.vocab_size = 50000

            from drrik.steering import SAESteering

            steering = SAESteering.__new__(SAESteering)
            steering.sae = mock_sae
            steering.model = MagicMock()
            steering.model._module = mock_model
            steering.tokenizer = tokenizer_mock
            steering.target_layer = 5

            result = steering.build_token_feature_map(tokens=[10, 20, 30])

    assert set(result.keys()) == {
        10,
        20,
        30,
    }, f"Expected {{10,20,30}}, got {set(result.keys())}"
    for tid in (10, 20, 30):
        info = result[tid]
        assert "active_features" in info
        assert "mean_activations" in info
        assert isinstance(info["active_features"], list)
        assert isinstance(info["mean_activations"], dict)
    print("PASS: build_token_feature_map returns correct structure")


if __name__ == "__main__":
    test_build_token_feature_map()
