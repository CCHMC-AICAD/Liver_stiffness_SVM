import csv
import json
import tempfile
import unittest
from pathlib import Path

from Liver_stiffness_SVM import (
    load_csv_dataset,
    make_demo_dataset,
    run_experiment,
)


class LiverStiffnessSVMTests(unittest.TestCase):
    def test_demo_experiment_is_reproducible_and_has_expected_shape(self):
        features, labels = make_demo_dataset(n_samples=120, random_state=7)
        result = run_experiment(features, labels, folds=4, random_state=7)

        self.assertEqual(result.n_samples, 120)
        self.assertEqual(result.n_features, 12)
        self.assertEqual(len(result.confusion_matrix), 2)
        self.assertGreaterEqual(result.accuracy, 0.0)
        self.assertLessEqual(result.accuracy, 1.0)
        self.assertGreaterEqual(result.roc_auc, 0.0)
        self.assertLessEqual(result.roc_auc, 1.0)

    def test_csv_loader_reads_numeric_features_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["feature_a", "feature_b", "label"])
                writer.writerows([
                    [0.0, 1.0, 0],
                    [1.0, 0.0, 1],
                    [0.2, 0.8, 0],
                    [0.8, 0.2, 1],
                ])

            features, labels = load_csv_dataset(path, "label")

        self.assertEqual(features.shape, (4, 2))
        self.assertEqual(labels.tolist(), [0, 1, 0, 1])

    def test_result_can_be_serialized_as_json(self):
        features, labels = make_demo_dataset(n_samples=80)
        result = run_experiment(features, labels, folds=4)
        serialized = json.dumps(result.__dict__)
        self.assertIn("roc_auc", serialized)


if __name__ == "__main__":
    unittest.main()
