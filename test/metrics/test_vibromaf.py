"""Haptic Multi-Method Assessment Fusion Tests."""

from pathlib import Path

import numpy as np
from pyfakefs.fake_filesystem_unittest import TestCase

from vibromaf.metrics.vibromaf import vibro_maf
from vibromaf.util import model


class VibroMAFTest(TestCase):
    """VibroMAF Test."""

    def setUp(self) -> None:
        self.setUpPyfakefs()
        self.dummy_model_path = Path("test-model.pickle")
        regressor = model.make_vibromaf_pipeline()
        regressor.fit([[0, 0, 0], [1, 1, 1]], [1, 0])
        model.save_model(regressor, self.dummy_model_path)

    def test_vibromaf_wrapper__sample_signals__some_output_value(self):
        sample_reference_signal = np.ones(1000) * 1000 + np.random.randn(1000)
        sample_distorted_signal = sample_reference_signal + np.random.randn(1000)

        result = vibro_maf(
            sample_distorted_signal, sample_reference_signal, Path("test-model.pickle")
        )

        self.assertGreaterEqual(1, result)
        self.assertGreaterEqual(result, 0)
