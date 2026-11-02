"""Data-contract tests: leakage, pairing, budgets, and reproducibility."""
import unittest
import numpy as np
from data import construct_data


class DataTests(unittest.TestCase):
    def test_paired_designs(self):
        one, audit_one, meta_one = construct_data("one_sample", 913000)
        two, audit_two, meta_two = construct_data("two_sample", 913000)
        self.assertEqual(meta_one["population_prior_supplied"], meta_two["population_prior_supplied"])
        np.testing.assert_array_equal(one["test"]["source_row"], two["test"]["source_row"])
        np.testing.assert_array_equal(one["test"]["y"], two["test"]["y"])
        for splits, audit in ((one, audit_one), (two, audit_two)):
            self.assertTrue(audit.source_row.is_unique)
            for name in ("train", "val"):
                split = splits[name]
                self.assertTrue(np.all(split["y"][split["s"] == 1] == 1))
                self.assertEqual(len(split["source_row"]), len(set(split["source_row"])))
                self.assertFalse(set(split["source_row"]) & set(splits["test"]["source_row"]))
                self.assertTrue(np.isfinite(split["X"]).all())
                self.assertEqual(len(one[name]["s"]), len(two[name]["s"]))
                self.assertEqual(one[name]["s"].sum(), two[name]["s"].sum())
        positive_two = audit_two[audit_two.observed_s == 1]
        self.assertTrue((positive_two.reservoir == "positive_source").all())
        unlabeled_two = audit_two[(audit_two.observed_s == 0) & audit_two.selected]
        self.assertTrue((unlabeled_two.reservoir == "background").all())

    def test_reproducible(self):
        first, _, _ = construct_data("one_sample", 913000)
        second, _, _ = construct_data("one_sample", 913000)
        for name in first:
            for key in first[name]:
                np.testing.assert_array_equal(first[name][key], second[name][key])


if __name__ == "__main__":
    unittest.main()
