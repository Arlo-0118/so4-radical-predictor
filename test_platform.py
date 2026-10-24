from __future__ import annotations

import json
from pathlib import Path
import unittest

from rdkit import Chem

from predictor_core import SO4Predictor


ROOT = Path(__file__).resolve().parent


class PlatformContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = SO4Predictor()

    def test_manifest_and_feature_contract(self) -> None:
        manifest_path = ROOT / "artifacts" / "big_qsar_so4_v2_mixed" / "deployment_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(self.predictor.feature_names), 1249)
        self.assertEqual(manifest["feature_dimensions"]["total"], 1249)
        self.assertAlmostEqual(self.predictor.formal_ad_threshold, 0.7005319)
        self.assertAlmostEqual(self.predictor.severe_similarity_threshold, 0.40)
        self.assertEqual(self.predictor.missing_sentinel, -999.0)
        self.assertGreater(len(self.predictor.reference_label_spans), 0)
        features = self.predictor._build_features(Chem.MolFromSmiles("CCO"), None, None, None)
        self.assertEqual(features.columns.tolist(), self.predictor.feature_names)

    def test_missing_conditions_are_explicit(self) -> None:
        report = self.predictor.predict("CCO")
        self.assertEqual(report["condition_applicability_domain"]["status"], "not-assessed-missing")
        self.assertEqual(report["input"]["conditions"], {"pH": None, "temp_C": None, "pH_over_TK": None})
        self.assertEqual(report["input"]["missing_continuous_encoding"], -999.0)

    def test_condition_ratio_and_range_warning(self) -> None:
        in_range = self.predictor.predict("CCO", p_h=7.0, temp_c=25.0)
        self.assertTrue(in_range["input"]["pH_over_TK_derived_from_pH_and_temperature"])
        self.assertEqual(in_range["condition_applicability_domain"]["status"], "within-observed-range")

        out_of_range = self.predictor.predict("CCO", p_h=12.0, temp_c=50.0)
        self.assertEqual(out_of_range["condition_applicability_domain"]["status"], "out-of-range")
        self.assertEqual(out_of_range["overall_risk"], "high")
        self.assertGreaterEqual(len(out_of_range["condition_applicability_domain"]["warnings"]), 2)

    def test_exact_reference_is_inside_structural_ad(self) -> None:
        reference_smiles = str(self.predictor.reference.iloc[0]["smiles_rdkit"])
        report = self.predictor.predict(reference_smiles)
        structure = report["structure_applicability_domain"]
        self.assertAlmostEqual(structure["nearest_neighbor_similarity"], 1.0)
        self.assertEqual(structure["status"], "in-domain")

    def test_severe_novelty_triggers_high_risk(self) -> None:
        report = self.predictor.predict("C1CCCCCCCCCCCC1")
        structure = report["structure_applicability_domain"]
        self.assertLess(structure["nearest_neighbor_similarity"], 0.40)
        self.assertEqual(structure["status"], "severe-extrapolation")
        self.assertEqual(report["overall_risk"], "high")

    def test_invalid_smiles_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.predictor.predict("not_a_smiles")

    def test_physically_invalid_ph_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 14"):
            self.predictor.predict("CCO", p_h=22.0)

    def test_valid_but_training_out_of_range_ph_is_flagged(self) -> None:
        report = self.predictor.predict("CCO", p_h=12.0)
        self.assertEqual(report["condition_applicability_domain"]["status"], "out-of-range")
        self.assertEqual(report["overall_risk"], "high")


if __name__ == "__main__":
    unittest.main(verbosity=2)
