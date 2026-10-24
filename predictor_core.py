from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class PredictorPaths:
    root_dir: Path = Path(__file__).resolve().parent

    @property
    def artifact_dir(self) -> Path:
        return self.root_dir / "artifacts" / "big_qsar_so4_v2_mixed"


def standardize_molecule(smiles: str):
    if smiles is None or not str(smiles).strip():
        return None, None, None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None, None, None
        mol = rdMolStandardize.Cleanup(mol)
        Chem.SanitizeMol(mol)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        parent = rdMolStandardize.Cleanup(rdMolStandardize.FragmentParent(mol))
        parent_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
        return canonical, parent_smiles, Chem.MolToInchiKey(mol)
    except Exception:
        return None, None, None


def _descriptor_names() -> List[str]:
    return [name for name, _ in Descriptors._descList]


def _descriptor_block(mol: Chem.Mol) -> np.ndarray:
    values = []
    for _, function in Descriptors._descList:
        try:
            value = float(function(mol))
            values.append(value if np.isfinite(value) else 0.0)
        except Exception:
            values.append(0.0)
    return np.asarray(values, dtype=np.float32)


def _fingerprint(mol: Chem.Mol, n_bits: int = 1024):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)


def _fingerprint_array(mol: Chem.Mol, n_bits: int = 1024) -> np.ndarray:
    fp = _fingerprint(mol, n_bits)
    values = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, values)
    return values


def _raw_so4_prior(descriptors: Dict[str, float]) -> float:
    return float(
        0.18 * descriptors["NumAromaticRings"]
        + 0.16 * descriptors["NumHeteroatoms"]
        + 0.14 * descriptors["MolMR"]
        + 0.10 * descriptors["BertzCT"]
        - 0.12 * descriptors["FractionCSP3"]
        - 0.10 * descriptors["NumRotatableBonds"]
        - 0.10 * descriptors["TPSA"]
        + 0.08 * abs(descriptors["MaxAbsPartialCharge"])
    )


class SO4Predictor:
    def __init__(self, paths: Optional[PredictorPaths] = None):
        self.paths = paths or PredictorPaths()
        artifact = self.paths.artifact_dir
        self.manifest = json.loads((artifact / "deployment_manifest.json").read_text(encoding="utf-8"))
        self.feature_names = list(self.manifest["feature_names"])
        self.meta_indices = np.asarray(self.manifest["meta_feature_indices"], dtype=int)
        self.base_order = list(self.manifest["base_model_order"])
        self.base_models = {
            "catboost": joblib.load(artifact / "catboost_model.joblib"),
            "catboost_b": joblib.load(artifact / "catboost_b_model.joblib"),
            "lightgbm": joblib.load(artifact / "lightgbm_model.joblib"),
            "xgboost": joblib.load(artifact / "xgboost_model.joblib"),
            "xgboost_b": joblib.load(artifact / "xgboost_b_model.joblib"),
        }
        self.meta_cat = joblib.load(artifact / "stacked_cat_model.joblib")
        self.meta_ridge = joblib.load(artifact / "stacked_ridge_model.joblib")
        self.reference = pd.read_csv(artifact / "deployment_reference.csv")
        span_table = pd.read_csv(artifact / "cross_source_same_structure_label_span.csv")
        self.reference_label_spans = dict(
            zip(span_table["parent_smiles"].astype(str), span_table["logK_span"].astype(float))
        )
        self.reference_mols = [Chem.MolFromSmiles(value) for value in self.reference["smiles_rdkit"]]
        self.reference_fps = [_fingerprint(mol) for mol in self.reference_mols]
        self.q_hat = float(self.manifest["calibration"]["q_hat"])
        self.beta = float(self.manifest["calibration"]["beta"])
        ad_policy = self.manifest["ad_policy"]
        self.formal_ad_threshold = float(ad_policy["formal_structure_ad_threshold"])
        self.severe_similarity_threshold = float(ad_policy["severe_structure_warning_threshold"])
        self.condition_ranges = self.manifest["condition_policy"]["observed_training_ranges"]
        self.missing_sentinel = float(self.manifest["condition_policy"]["missing_continuous_encoding"])
        prior = self.manifest["so4_prior"]
        self.prior_mean = float(prior["training_mean"])
        self.prior_std = float(prior["training_std"])
        warning_policy = self.manifest["failure_warning_policy"]
        self.label_span_threshold = float(warning_policy["adjacent_reference_label_span_threshold"])
        self.electronic_thresholds = {
            "MaxAbsPartialCharge": float(warning_policy["max_abs_partial_charge_threshold"]),
            "TPSA": float(warning_policy["tpsa_threshold"]),
        }

        expected_descriptors = int(self.manifest["feature_dimensions"]["rdkit_2d"])
        if len(_descriptor_names()) != expected_descriptors:
            raise RuntimeError(
                f"RDKit descriptor mismatch: expected {expected_descriptors}, got {len(_descriptor_names())}. "
                "Install the versions pinned in requirements.txt."
            )

    @staticmethod
    def _normalize_conditions(p_h, temp_c, p_h_over_tk):
        p_h = None if p_h is None else float(p_h)
        temp_c = None if temp_c is None else float(temp_c)
        p_h_over_tk = None if p_h_over_tk is None else float(p_h_over_tk)
        if p_h is not None and not 0.0 <= p_h <= 14.0:
            raise ValueError(
                "pH must be between 0 and 14 for this environmental aqueous-phase tool. "
                "Values outside this physical input guardrail are not accepted."
            )
        derived = False
        if p_h_over_tk is None and p_h is not None and temp_c is not None:
            p_h_over_tk = p_h / (temp_c + 273.15)
            derived = True
        return p_h, temp_c, p_h_over_tk, derived

    def _build_features(self, mol: Chem.Mol, p_h, temp_c, p_h_over_tk) -> pd.DataFrame:
        descriptor_values = _descriptor_block(mol)
        descriptor_map = dict(zip(_descriptor_names(), descriptor_values))
        prior_raw = _raw_so4_prior(descriptor_map)
        prior_z = (prior_raw - self.prior_mean) / (self.prior_std + 1e-8)
        missing = {"pH": p_h is None, "temp_C": temp_c is None, "pH_over_TK": p_h_over_tk is None}
        feature_map: Dict[str, float] = dict(descriptor_map)
        feature_map.update({f"ECFP4_{i}": float(value) for i, value in enumerate(_fingerprint_array(mol))})
        feature_map.update(
            {
                "so4_prior": float(prior_z),
                "pH": self.missing_sentinel if p_h is None else p_h,
                "temp_C": self.missing_sentinel if temp_c is None else temp_c,
                "pH_over_TK": self.missing_sentinel if p_h_over_tk is None else p_h_over_tk,
                "all_conditions_missing": float(all(missing.values())),
                "pH_missing": float(missing["pH"]),
                "temp_C_missing": float(missing["temp_C"]),
                "pH_over_TK_missing": float(missing["pH_over_TK"]),
            }
        )
        frame = pd.DataFrame([[feature_map[name] for name in self.feature_names]], columns=self.feature_names)
        return frame.astype(np.float32)

    def _condition_assessment(self, p_h, temp_c, p_h_over_tk) -> Dict[str, Any]:
        supplied = {"pH": p_h, "temp_C": temp_c, "pH_over_TK": p_h_over_tk}
        warnings = []
        for name, value in supplied.items():
            if value is None:
                continue
            limits = self.condition_ranges[name]
            if value < limits["min"] or value > limits["max"]:
                warnings.append(f"{name}={value:.4g} is outside the training range [{limits['min']}, {limits['max']}]")
        return {
            "status": "out-of-range" if warnings else ("not-assessed-missing" if all(v is None for v in supplied.values()) else "within-observed-range"),
            "warnings": warnings,
            "observed_training_ranges": self.condition_ranges,
        }

    def _structure_assessment(self, nn_similarity: float) -> Dict[str, Any]:
        if nn_similarity >= self.formal_ad_threshold:
            status = "in-domain"
            risk_tier = "low"
            message = "The query is within the formal empirical structural AD."
        elif nn_similarity >= self.severe_similarity_threshold:
            status = "out-of-domain"
            risk_tier = "moderate"
            message = "The query is outside the formal structural AD; interpret the estimate as extrapolative."
        else:
            status = "severe-extrapolation"
            risk_tier = "high"
            message = "Severe structural extrapolation warning; experimental verification is strongly recommended."
        return {
            "status": status,
            "risk_tier": risk_tier,
            "nearest_neighbor_similarity": nn_similarity,
            "formal_ad_threshold": self.formal_ad_threshold,
            "severe_warning_threshold": self.severe_similarity_threshold,
            "message": message,
        }

    @staticmethod
    def _multisite_reactivity(mol: Chem.Mol) -> bool:
        azo = len(mol.GetSubstructMatches(Chem.MolFromSmarts("[N;X2]=[N;X2]")))
        sulfonate = len(mol.GetSubstructMatches(Chem.MolFromSmarts("S(=O)(=O)[O-,OH]")))
        heteroatoms = sum(atom.GetAtomicNum() not in {1, 6} for atom in mol.GetAtoms())
        aromatic_rings = int(Descriptors.NumAromaticRings(mol))
        formal_charge = abs(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
        return bool(
            (azo >= 1 and sulfonate >= 1)
            or (heteroatoms >= 6 and formal_charge >= 1)
            or (aromatic_rings >= 2 and heteroatoms >= 5)
        )

    @staticmethod
    def _max_abs_gasteiger_charge(mol: Chem.Mol) -> float:
        try:
            Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
            values = [abs(float(atom.GetProp("_GasteigerCharge"))) for atom in mol.GetAtoms()]
            finite = [value for value in values if np.isfinite(value)]
            return max(finite, default=0.0)
        except Exception:
            return 0.0

    def _neighbors(self, query_fp, top_k: int):
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, self.reference_fps), dtype=float)
        order = np.argsort(-similarities)[:top_k]
        rows = []
        for idx in order:
            ref = self.reference.iloc[int(idx)]
            rows.append(
                {
                    "compound_name": str(ref.get("compound_name", "")),
                    "smiles": str(ref["smiles_rdkit"]),
                    "logK": float(ref["logK"]),
                    "similarity": float(similarities[idx]),
                }
            )
        return similarities, rows

    def _warning_assessment(self, mol, similarities):
        score = 0
        flags = []
        nn_similarity = float(np.max(similarities))
        if nn_similarity < self.severe_similarity_threshold:
            score += 2
            flags.append("severe_structural_novelty")

        nearest_idx = int(np.argmax(similarities))
        nearest_parent = str(self.reference.iloc[nearest_idx]["parent_smiles"])
        label_span = float(self.reference_label_spans.get(nearest_parent, 0.0))
        if label_span > self.label_span_threshold:
            score += 2
            flags.append("adjacent_analogue_disagreement")

        if self._multisite_reactivity(mol):
            score += 1
            flags.append("multisite_reactivity")

        max_charge = self._max_abs_gasteiger_charge(mol)
        tpsa = float(Descriptors.TPSA(mol))
        if max_charge >= self.electronic_thresholds["MaxAbsPartialCharge"] or tpsa >= self.electronic_thresholds["TPSA"]:
            score += 1
            flags.append("electronic_or_surface_extreme")

        risk_tier = "low" if score == 0 else ("moderate" if score <= 2 else "high")
        return {
            "score": score,
            "risk_tier": risk_tier,
            "flagged": bool(score >= 3),
            "flags": flags,
            "nearest_analogue_label_span": label_span,
            "nearest_analogue_parent_smiles": nearest_parent,
            "action": "experimental verification required" if score >= 3 else (
                "interpret with caution" if score >= 1 else "screening use within stated uncertainty"
            ),
        }

    def predict(self, smiles: str, p_h=None, temp_c=None, p_h_over_tk=None, top_k_neighbors: int = 5) -> Dict[str, Any]:
        canonical, parent_smiles, inchi_key = standardize_molecule(smiles)
        if canonical is None:
            raise ValueError("The SMILES could not be parsed and standardized by RDKit.")
        mol = Chem.MolFromSmiles(canonical)
        p_h, temp_c, p_h_over_tk, ratio_derived = self._normalize_conditions(p_h, temp_c, p_h_over_tk)
        features = self._build_features(mol, p_h, temp_c, p_h_over_tk)
        x = features.to_numpy(dtype=np.float32)
        base_predictions = np.asarray(
            [float(self.base_models[name].predict(features)[0]) for name in self.base_order], dtype=np.float32
        )
        sigma = float(np.std(base_predictions))
        meta_cat_input = np.concatenate([base_predictions.reshape(1, -1), x[:, self.meta_indices]], axis=1)
        meta_cat = float(self.meta_cat.predict(meta_cat_input)[0])
        meta_ridge = float(self.meta_ridge.predict(base_predictions.reshape(1, -1))[0])
        prediction = 0.7 * meta_cat + 0.3 * meta_ridge
        half_width = self.q_hat * (1.0 + self.beta * sigma)

        query_fp = _fingerprint(mol)
        similarities, neighbors = self._neighbors(query_fp, max(3, int(top_k_neighbors)))
        nn_similarity = float(np.max(similarities))
        structure = self._structure_assessment(nn_similarity)
        condition = self._condition_assessment(p_h, temp_c, p_h_over_tk)
        warning = self._warning_assessment(mol, similarities)
        if structure["risk_tier"] == "high" or warning["risk_tier"] == "high" or condition["status"] == "out-of-range":
            overall_risk = "high"
        elif structure["risk_tier"] == "moderate" or warning["risk_tier"] == "moderate":
            overall_risk = "moderate"
        else:
            overall_risk = "low"

        return {
            "input": {
                "smiles": smiles,
                "canonical_smiles": canonical,
                "parent_smiles": parent_smiles,
                "inchi_key": inchi_key,
                "conditions": {"pH": p_h, "temp_C": temp_c, "pH_over_TK": p_h_over_tk},
                "pH_over_TK_derived_from_pH_and_temperature": ratio_derived,
                "missing_continuous_encoding": self.missing_sentinel,
            },
            "prediction": {
                "logK": prediction,
                "k_M-1_s-1": float(10.0 ** prediction),
                "ci_lower": prediction - half_width,
                "ci_upper": prediction + half_width,
                "interval_half_width": half_width,
                "ensemble_disagreement_sigma": sigma,
                "nominal_coverage": 0.90,
            },
            "base_model_predictions": dict(zip(self.base_order, map(float, base_predictions))),
            "structure_applicability_domain": structure,
            "condition_applicability_domain": condition,
            "failure_warning": warning,
            "overall_risk": overall_risk,
            "neighbors": neighbors,
            "deployment": {
                "feature_dimension": len(self.feature_names),
                "model_purpose": self.manifest["model_purpose"],
                "target_definition": self.manifest["target_definition"],
            },
        }


def build_download_bundle(report: Dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("prediction_report.json", json.dumps(report, ensure_ascii=False, indent=2))
        archive.writestr("nearest_neighbors.csv", pd.DataFrame(report["neighbors"]).to_csv(index=False))
    return buffer.getvalue()
