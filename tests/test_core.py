import unittest

import numpy as np
from landlab import RasterModelGrid

from landlab_grid_validation import FieldSpec, compare_fields_on_grid, field_to_node_array


class TestCoreComparison(unittest.TestCase):
    def test_probability_vs_binary_inventory_metrics(self):
        grid = RasterModelGrid((3, 3), xy_spacing=1.0)
        probability = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.1, 0.0, 0.4, 0.6])
        observed = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1])

        result = compare_fields_on_grid(
            grid,
            probability,
            observed,
            FieldSpec(
                name="landslide_probability",
                kind="continuous",
                variable="landslide_probability",
                units="probability",
                threshold=0.5,
            ),
            FieldSpec(
                name="landslide_inventory",
                kind="binary",
                variable="landslide_presence",
                positive_class="landslide",
            ),
        )

        self.assertEqual(result.comparison_type, "probability_vs_binary")
        self.assertEqual(result.metrics["true_positive"], 4)
        self.assertEqual(result.metrics["false_positive"], 0)
        self.assertAlmostEqual(result.metrics["roc_auc"], 1.0)
        self.assertGreaterEqual(result.metrics["brier_score"], 0.0)

    def test_continuous_vs_continuous_metrics_with_nodata(self):
        grid = RasterModelGrid((2, 3), xy_spacing=1.0)
        predicted = np.array([1.0, 2.0, -9999.0, 4.0, 5.0, 6.0])
        observed = np.array([1.5, 1.5, 3.0, 4.5, 5.5, 5.5])

        result = compare_fields_on_grid(
            grid,
            predicted,
            observed,
            FieldSpec(name="predicted_dz", kind="continuous", units="m", nodata=-9999),
            FieldSpec(name="observed_dz", kind="continuous", units="m"),
        )

        self.assertEqual(result.comparison_type, "continuous_vs_continuous")
        self.assertEqual(result.metrics["n"], 5)
        self.assertAlmostEqual(result.metrics["mae"], 0.5)

    def test_binary_vs_binary_metrics_from_landlab_field_names(self):
        grid = RasterModelGrid((2, 3), xy_spacing=1.0)
        grid.add_field("predicted_erosion", np.array([1, 0, 1, 0, 1, 0]), at="node")
        grid.add_field("mapped_erosion", np.array([1, 0, 0, 0, 1, 1]), at="node")

        result = compare_fields_on_grid(
            grid,
            "predicted_erosion",
            "mapped_erosion",
            FieldSpec(name="predicted_erosion", kind="binary", variable="erosion"),
            FieldSpec(name="mapped_erosion", kind="binary", variable="erosion"),
        )

        self.assertEqual(result.comparison_type, "binary_vs_binary")
        self.assertEqual(result.metrics["true_positive"], 2)
        self.assertEqual(result.metrics["false_positive"], 1)
        self.assertEqual(result.metrics["false_negative"], 1)

    def test_categorical_vs_categorical_metrics(self):
        grid = RasterModelGrid((2, 4), xy_spacing=1.0)
        predicted = np.array([0, 1, 1, 2, 0, 2, 2, 1])
        observed = np.array([0, 1, 2, 2, 0, 2, 1, 1])
        classes = {0: "stable", 1: "erosion", 2: "deposition"}

        result = compare_fields_on_grid(
            grid,
            predicted,
            observed,
            FieldSpec(name="predicted_mwr", kind="categorical", classes=classes),
            FieldSpec(name="observed_mwr", kind="categorical", classes=classes),
        )

        self.assertEqual(result.comparison_type, "categorical_vs_categorical")
        self.assertAlmostEqual(result.metrics["overall_accuracy"], 6 / 8)
        self.assertEqual(result.metrics["confusion_matrix"]["2"]["1"], 1)

    def test_field_to_node_array_rejects_wrong_shape(self):
        grid = RasterModelGrid((2, 3), xy_spacing=1.0)
        with self.assertRaises(ValueError):
            field_to_node_array(grid, np.ones((3, 2)))


if __name__ == "__main__":
    unittest.main()
