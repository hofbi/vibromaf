"""Tests for model helper."""

from pathlib import Path

from pyfakefs.fake_filesystem_unittest import TestCase

from vibromaf.util import model


class ModelTest(TestCase):
    """Model Test."""

    def setUp(self) -> None:
        self.setUpPyfakefs()

    def test_model_serialization__model_should_be_same_after_save_and_load(self):
        pipe = model.make_vibromaf_pipeline()

        model.save_model(pipe, Path("test.pikle"))
        loaded = model.load_model(Path("test.pikle"))

        self.assertEqual(str(pipe.get_params()), str(loaded.get_params()))

    def test_make_vibromaf_pipeline__standard_scaler_should_scale_column_wise(self):
        """As we have SNR in one column and other metrics with range of 0 to 1 in the
        other columns then scaler should not scale them based on the SNR range."""
        pipe = model.make_vibromaf_pipeline()
        data = [[0, 0], [0, 0], [10, 1], [10, 1]]

        scaler = pipe["standardscaler"]
        scaler.fit(data)

        self.assertListEqual([5, 0.5], list(scaler.mean_))
        self.assertListEqual([25, 0.25], list(scaler.var_))
