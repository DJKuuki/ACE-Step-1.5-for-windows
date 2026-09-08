"""Unit tests for AceStepLyricEncoder float16 numerical stability guard.

Validates that AceStepLyricEncoder automatically upcasts to float32 during forward
passes when inputs are float16 (preventing MLP dot-product overflow on pre-Ampere GPUs),
properly restores weights to float16 afterwards, and acts as a passthrough for bfloat16
and float32 inputs.
"""

from __future__ import annotations

import unittest

import torch
from transformers import PretrainedConfig

from acestep.models.turbo.modeling_acestep_v15_turbo import AceStepLyricEncoder


def _create_test_config() -> PretrainedConfig:
    """Create a minimal PretrainedConfig for AceStepLyricEncoder testing."""
    return PretrainedConfig(
        text_hidden_dim=32,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        num_hidden_layers=1,
        num_lyric_encoder_hidden_layers=1,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        output_attentions=False,
        output_hidden_states=False,
        _attn_implementation="eager",
        use_sliding_window=False,
        sliding_window=512,
        attention_bias=False,
        attention_dropout=0.0,
        rope_scaling=None,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        initializer_range=0.02,
        is_turbo=True,
        layer_types=["full_attention"],
    )


class LyricEncoderFp16SafetyTests(unittest.TestCase):
    """Test suite for AceStepLyricEncoder FP16 safety guard and dtype preservation."""

    def test_lyric_encoder_fp16_upcasts_and_restores(self) -> None:
        """Verify FP16 inputs execute safely and model parameters restore to FP16."""
        cfg = _create_test_config()
        encoder = AceStepLyricEncoder(cfg).half()

        inp = torch.randn(1, 4, cfg.text_hidden_dim, dtype=torch.float16)
        mask = torch.ones(1, 4, dtype=torch.long)

        out = encoder(inputs_embeds=inp, attention_mask=mask)

        self.assertEqual(out.last_hidden_state.dtype, torch.float16)
        self.assertFalse(torch.isnan(out.last_hidden_state).any().item())
        self.assertFalse(torch.isinf(out.last_hidden_state).any().item())

        for param in encoder.parameters():
            self.assertEqual(param.dtype, torch.float16)

    def test_lyric_encoder_fp16_large_activations_do_not_produce_nan(self) -> None:
        """Verify activations that would overflow FP16 (>65504) do not yield NaN."""
        cfg = _create_test_config()
        encoder = AceStepLyricEncoder(cfg).half()

        # Scale weights to induce large hidden activations
        with torch.no_grad():
            for param in encoder.parameters():
                param.data.fill_(10.0)

        inp = torch.ones(1, 4, cfg.text_hidden_dim, dtype=torch.float16) * 10.0
        mask = torch.ones(1, 4, dtype=torch.long)

        out = encoder(inputs_embeds=inp, attention_mask=mask)

        self.assertEqual(out.last_hidden_state.dtype, torch.float16)
        self.assertFalse(
            torch.isnan(out.last_hidden_state).any().item(),
            "Output contains NaN despite FP32 guard",
        )

    def test_lyric_encoder_bfloat16_passthrough(self) -> None:
        """Verify bfloat16 inputs preserve bfloat16 without forced conversion."""
        cfg = _create_test_config()
        encoder = AceStepLyricEncoder(cfg).bfloat16()

        inp = torch.randn(1, 4, cfg.text_hidden_dim, dtype=torch.bfloat16)
        mask = torch.ones(1, 4, dtype=torch.long)

        out = encoder(inputs_embeds=inp, attention_mask=mask)

        self.assertEqual(out.last_hidden_state.dtype, torch.bfloat16)
        for param in encoder.parameters():
            self.assertEqual(param.dtype, torch.bfloat16)

    def test_lyric_encoder_float32_passthrough(self) -> None:
        """Verify float32 inputs preserve float32 without forced conversion."""
        cfg = _create_test_config()
        encoder = AceStepLyricEncoder(cfg).float()

        inp = torch.randn(1, 4, cfg.text_hidden_dim, dtype=torch.float32)
        mask = torch.ones(1, 4, dtype=torch.long)

        out = encoder(inputs_embeds=inp, attention_mask=mask)

        self.assertEqual(out.last_hidden_state.dtype, torch.float32)
        for param in encoder.parameters():
            self.assertEqual(param.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
