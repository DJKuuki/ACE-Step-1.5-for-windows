"""Tests for DCW FP16 safety clamping and environment variable gating.

Ensures that DCW operations do not overflow to Inf or produce NaN when operating
in float16 precision on pre-Ampere GPUs, and verifies that ACESTEP_DCW_ENABLED
controls the corrector lifecycle.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from acestep.models.common.dcw_correction import DCWCorrector
from acestep.models.common.dcw_primitives import dcw_double, dcw_low, dcw_pix


class DcwFp16SafetyTests(unittest.TestCase):
    """Test suite for DCW numerical safety and configuration gating."""

    def test_dcw_pix_fp16_clamping_prevents_inf(self) -> None:
        """Verify dcw_pix clamps extreme values to avoid float16 Inf/NaN."""
        x = torch.tensor([[[60000.0]]], dtype=torch.float16)
        y = torch.tensor([[[-60000.0]]], dtype=torch.float16)
        # Without clamping, 60000 + 1.0 * (60000 - (-60000)) = 180000 -> Inf in fp16
        out = dcw_pix(x, y, scaler=1.0)
        self.assertFalse(torch.isinf(out).any(), "Expected no Inf values in output")
        self.assertFalse(torch.isnan(out).any(), "Expected no NaN values in output")
        self.assertLessEqual(out.max().item(), 65000.0)

    def test_dcw_corrector_respects_env_var_disabled(self) -> None:
        """Verify DCWCorrector is disabled when ACESTEP_DCW_ENABLED=false."""
        with patch.dict(os.environ, {"ACESTEP_DCW_ENABLED": "false"}):
            corrector = DCWCorrector(enabled=True, mode="double", scaler=0.05)
            self.assertFalse(corrector.enabled)
            self.assertFalse(corrector.is_active)

    def test_dcw_corrector_respects_env_var_enabled(self) -> None:
        """Verify DCWCorrector is enabled when ACESTEP_DCW_ENABLED=true."""
        with patch.dict(os.environ, {"ACESTEP_DCW_ENABLED": "true"}):
            corrector = DCWCorrector(enabled=True, mode="double", scaler=0.05)
            self.assertTrue(corrector.enabled)
            self.assertTrue(corrector.is_active)

    def test_dcw_corrector_no_op_when_disabled(self) -> None:
        """Verify apply() returns x_next untouched when corrector is inactive."""
        corrector = DCWCorrector(enabled=False)
        x = torch.randn(1, 10, 64)
        y = torch.randn(1, 10, 64)
        out = corrector.apply(x, y, t_curr=0.1)
        self.assertTrue(torch.equal(out, x))


if __name__ == "__main__":
    unittest.main()
