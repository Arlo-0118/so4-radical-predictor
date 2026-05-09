from __future__ import annotations

import io
import json
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


@dataclass
class PredictorPaths:
    root_dir: Path = Path(__file__).resolve().parent

    @property
    def artifact_dir(self) -> Path:
        return self.root_dir / "artifacts" / "big_qsar_so4_v2_mixed"

    @property
    def enhancement_dir(self) -> Path:
        return self.artifact_dir / "esandt_enhancements"


def standardize_molecule(smiles: str):
    if smiles is None or pd.isna(smiles):
        return None, None, None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None, None, None
        mol = rdMolStandardize.Cleanup(mol)
        Chem.SanitizeMol(mol)
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        parent = rdMolStandardize.FragmentParent(mol)
        parent = rdMolStandardize.Cleanup(parent)
        parent_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
        inchi_key = Chem.MolToInchiKey(mol)
        return can, parent_smiles, inchi_key
    except Exception:
        return None, None, None


def descriptor_names() -> List[str]:
    return [name for name, _ in Descriptors._descList]


def compute_descriptor_block(mol):
    vals = []
    for _, fn in Descriptors._descList:
        try:
            v = float(fn(mol))
            if np.isnan(v) or np.isinf(v):
                v = 0.0
        except Exception:
            v = 0.0
        vals.append(v)
    return np.asarray(vals, dtype=np.float32)


def compute_fp_block(mol, n_bits: int):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_so4_prior(desc_df: pd.DataFrame):
    cols = desc_df.columns

    def col(name):
        return desc_df[name].to_numpy() if name in cols else np.zeros(len(desc_df), dtype=np.float32)

    prior = (
        0.18 * col("NumAromaticRings")
        + 0.16 * col("NumHeteroatoms")
        + 0.14 * col("MolMR")
        + 0.10 * col("BertzCT")
        - 0.12 * col("FractionCSP3")
        - 0.10 * col("NumRotatableBonds")
        + 0.10 * col("TPSA")
        + 0.10 * col("MaxAbsPartialCharge")
    )
    return prior.astype(np.float32)


class SO4Predictor:
    def __init__(self, paths: Optional[PredictorPaths] = None):
        self.paths = paths or PredictorPaths()
        self.fp_bits = 1024
        self.conformal_alpha = 0.10
        self.conformal_beta = 1.0
        self.nominal_coverage = 1.0 - self.conformal_alpha

        self.metrics = json.loads((self.paths.artifact_dir / "metrics.json").read_text(encoding="utf-8"))
        self.q_hat = float(self.metrics["metrics"]["conformal"]["q_hat"])
        self.base_models = {
            "catboost": joblib.load(self.paths.artifact_dir / "catboost_model.joblib"),
            "catboost_b": joblib.load(self.paths.artifact_dir / "catboost_b_model.joblib"),
            "lightgbm": joblib.load(self.paths.artifact_dir / "lightgbm_model.joblib"),
            "xgboost": joblib.load(self.paths.artifact_dir / "xgboost_model.joblib"),
            "xgboost_b": joblib.load(self.paths.artifact_dir / "xgboost_b_model.joblib"),
        }
        self.meta_cat = joblib.load(self.paths.artifact_dir / "stacked_cat_model.joblib")
        self.meta_ridge = joblib.load(self.paths.artifact_dir / "stacked_ridge_model.joblib")

        self.agg_df = pd.read_csv(self.paths.artifact_dir / "cleaned_rows_aggregated.csv")
        self.model_df = self.agg_df[self.agg_df["kept_for_model"].astype(str).str.lower() == "true"].copy().reset_index(drop=True)
        self.feature_names, self.meta_feature_idx = self._prepare_reference_features()

        self.reference_mols = [Chem.MolFromSmiles(s) for s in self.model_df["smiles_rdkit"]]
        self.reference_fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=self.fp_bits) for m in self.reference_mols]

        self.ad_threshold = float(json.loads((self.paths.artifact_dir / "esandt_extension_summary.json").read_text(encoding="utf-8"))["applicability_domain_threshold"])
        self.random_bins = pd.read_csv(self.paths.enhancement_dir / "random_similarity_bin_error_summary.csv")
        self.scaffold_bins = pd.read_csv(self.paths.enhancement_dir / "scaffold_similarity_bin_error_summary.csv")

    def _prepare_reference_features(self):
        mols = [Chem.MolFromSmiles(s) for s in self.model_df["smiles_rdkit"]]
        desc = np.vstack([compute_descriptor_block(m) for m in mols]).astype(np.float32)
        desc_df = pd.DataFrame(desc, columns=descriptor_names())
        num = self.model_df[["pH", "temp_C", "pH_over_TK", "rep_n", "rep_std", "n_source_blocks", "all_conditions_missing"]].copy()
        for col in ["pH", "temp_C", "pH_over_TK"]:
            num[f"{col}_missing"] = num[col].isna().astype(np.float32)
        num_df = num.fillna(-999.0)
        src_cols = [
            "src_A_raw_k_with_smiles",
            "src_B_logk_name_only",
            "src_C_logk_with_pH_T",
            "src_D_logk_with_pH_over_TK",
        ]
        feature_names = (
            descriptor_names()
            + [f"ECFP4_{i}" for i in range(self.fp_bits)]
            + ["so4_prior"]
            + list(num_df.columns)
            + src_cols
        )
        meta_feature_idx = np.concatenate(
            [
                np.arange(len(descriptor_names()), dtype=int),
                np.array([len(descriptor_names()) + self.fp_bits], dtype=int),
                np.arange(len(descriptor_names()) + self.fp_bits + 1, len(feature_names), dtype=int),
            ]
        )
        return feature_names, meta_feature_idx

    def _source_flags(self, p_h: Optional[float], temp_c: Optional[float], p_h_over_tk: Optional[float]) -> Dict[str, int]:
        flags = {
            "src_A_raw_k_with_smiles": 0,
            "src_B_logk_name_only": 0,
            "src_C_logk_with_pH_T": 0,
            "src_D_logk_with_pH_over_TK": 0,
        }
        if p_h_over_tk is not None:
            flags["src_D_logk_with_pH_over_TK"] = 1
            surrogate = "condition-template:D"
        elif p_h is not None or temp_c is not None:
            flags["src_C_logk_with_pH_T"] = 1
            surrogate = "condition-template:C"
        else:
            flags["src_B_logk_name_only"] = 1
            surrogate = "condition-template:B"
        flags["source_surrogate"] = surrogate
        return flags

    def _build_query_df(self, smiles: str, p_h=None, temp_c=None, p_h_over_tk=None) -> pd.DataFrame:
        smiles_rdkit, parent_smiles, _ = standardize_molecule(smiles)
        if smiles_rdkit is None:
            raise ValueError("SMILES could not be parsed or standardized by RDKit.")
        flags = self._source_flags(p_h, temp_c, p_h_over_tk)
        all_missing = int(p_h is None and temp_c is None and p_h_over_tk is None)
        return pd.DataFrame([
            {
                "parent_smiles": parent_smiles,
                "smiles_rdkit": smiles_rdkit,
                "compound_name": "QUERY_MOLECULE",
                "pH": np.nan if p_h is None else float(p_h),
                "temp_C": np.nan if temp_c is None else float(temp_c),
                "pH_over_TK": np.nan if p_h_over_tk is None else float(p_h_over_tk),
                "rep_n": 1.0,
                "rep_std": 0.0,
                "n_source_blocks": 1.0,
                "all_conditions_missing": float(all_missing),
                "logK": 0.0,
                "source_block": flags["source_surrogate"],
                "src_A_raw_k_with_smiles": flags["src_A_raw_k_with_smiles"],
                "src_B_logk_name_only": flags["src_B_logk_name_only"],
                "src_C_logk_with_pH_T": flags["src_C_logk_with_pH_T"],
                "src_D_logk_with_pH_over_TK": flags["src_D_logk_with_pH_over_TK"],
            }
        ])

    def _build_query_features(self, query_df: pd.DataFrame) -> pd.DataFrame:
        mol = Chem.MolFromSmiles(query_df.iloc[0]["smiles_rdkit"])
        desc = compute_descriptor_block(mol)[None, :]
        fp = compute_fp_block(mol, self.fp_bits)[None, :]
        desc_df = pd.DataFrame(desc, columns=descriptor_names())
        so4_prior = compute_so4_prior(desc_df)[:, None]
        num = query_df[["pH", "temp_C", "pH_over_TK", "rep_n", "rep_std", "n_source_blocks", "all_conditions_missing"]].copy()
        for col in ["pH", "temp_C", "pH_over_TK"]:
            num[f"{col}_missing"] = num[col].isna().astype(np.float32)
        num_df = num.fillna(-999.0).to_numpy(dtype=np.float32)
        src = query_df[["src_A_raw_k_with_smiles", "src_B_logk_name_only", "src_C_logk_with_pH_T", "src_D_logk_with_pH_over_TK"]].astype(np.float32).to_numpy(dtype=np.float32)
        x = np.concatenate([desc, fp, so4_prior, num_df, src], axis=1).astype(np.float32)
        return pd.DataFrame(x, columns=self.feature_names)

    def _similarity_bin(self, sim: float) -> str:
        if sim < 0.4:
            return "<0.4"
        if sim < 0.6:
            return "0.4-0.6"
        return ">=0.6"

    def _lookup_expected_error(self, sim: float) -> Dict[str, float]:
        idx = 0 if sim < 0.4 else 1 if sim < 0.6 else 2
        return {
            "random_mean_abs_error": float(self.random_bins.iloc[idx]["mean_abs_error"]),
            "random_median_abs_error": float(self.random_bins.iloc[idx]["median_abs_error"]),
            "scaffold_mean_abs_error": float(self.scaffold_bins.iloc[idx]["mean_abs_error"]),
            "scaffold_median_abs_error": float(self.scaffold_bins.iloc[idx]["median_abs_error"]),
            "scaffold_coverage": float(self.scaffold_bins.iloc[idx]["coverage"]),
        }

    def _risk_flags(self, mol: Chem.Mol, nn_similarity: float) -> List[str]:
        flags: List[str] = []
        mw = float(Descriptors.MolWt(mol))
        tpsa = float(Descriptors.TPSA(mol))
        hetero = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
        has_charge = any(a.GetFormalCharge() != 0 for a in mol.GetAtoms())
        acid = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1,O-]"))
        amine = mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H2,H1,H0;+0,+1]"))
        if has_charge:
            flags.append("high_ionicity")
        if acid and mw < 120:
            flags.append("small_acid")
        if tpsa > 90 and hetero >= 4:
            flags.append("polar_multifunctional")
        if amine and tpsa > 70 and hetero >= 3:
            flags.append("strongly_polar_n_containing")
        if nn_similarity < 0.4:
            flags.append("novel_scaffold_or_chemotype")
        return flags

    def _recommended_use(self, nn_similarity: float, half_width: float, risk_flags: List[str]) -> str:
        if nn_similarity >= 0.6 and half_width <= 0.40 and not any(f in risk_flags for f in ["high_ionicity", "small_acid", "polar_multifunctional", "novel_scaffold_or_chemotype"]):
            return "screening-ready"
        if nn_similarity >= 0.4:
            return "use-with-caution"
        return "coarse-screen-only"

    def _domain_level(self, nn_similarity: float, half_width: float) -> str:
        if nn_similarity >= 0.6 and half_width <= 0.40:
            return "In-domain"
        if nn_similarity >= 0.4:
            return "Borderline"
        return "Out-of-domain"

    def _confidence_level(self, nn_similarity: float, half_width: float, risk_flags: List[str]) -> str:
        major_risks = {"high_ionicity", "small_acid", "polar_multifunctional", "novel_scaffold_or_chemotype"}
        risk_count = sum(1 for f in risk_flags if f in major_risks)
        if nn_similarity >= 0.7 and half_width <= 0.35 and risk_count == 0:
            return "high-confidence"
        if nn_similarity >= 0.4 and half_width <= 0.65 and risk_count <= 1:
            return "medium-confidence"
        return "low-confidence"

    def predict(self, smiles: str, p_h=None, temp_c=None, p_h_over_tk=None, top_k_neighbors: int = 5) -> Dict[str, Any]:
        query_df = self._build_query_df(smiles=smiles, p_h=p_h, temp_c=temp_c, p_h_over_tk=p_h_over_tk)
        x_query_df = self._build_query_features(query_df)
        base_preds = np.asarray([float(model.predict(x_query_df)[0]) for model in self.base_models.values()], dtype=np.float32)
        sigma = float(np.std(base_preds))
        meta_array = np.concatenate([base_preds[None, :], x_query_df.to_numpy(dtype=np.float32)[:, self.meta_feature_idx]], axis=1)
        pred_cat = float(self.meta_cat.predict(meta_array)[0])
        pred_ridge = float(self.meta_ridge.predict(base_preds[None, :])[0])
        pred_logk = float(0.7 * pred_cat + 0.3 * pred_ridge)
        half_width = float(self.q_hat * (1.0 + self.conformal_beta * sigma))
        ci_lower = pred_logk - half_width
        ci_upper = pred_logk + half_width
        pred_k = float(10 ** pred_logk)

        mol = Chem.MolFromSmiles(query_df.iloc[0]["smiles_rdkit"])
        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.fp_bits)
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(query_fp, self.reference_fps), dtype=np.float32)
        order = np.argsort(similarities)[::-1]
        top_idx = order[:top_k_neighbors]
        nn_similarity = float(similarities[top_idx[0]])
        empirical_in_domain = bool(nn_similarity >= self.ad_threshold)

        neighbors = []
        for idx in top_idx:
            row = self.model_df.iloc[int(idx)]
            neighbors.append({
                "compound_name": str(row["compound_name"]),
                "parent_smiles": str(row["parent_smiles"]),
                "logK": float(row["logK"]),
                "similarity": float(similarities[int(idx)]),
            })

        risk_flags = self._risk_flags(mol, nn_similarity)
        expected_error = self._lookup_expected_error(nn_similarity)
        domain_level = self._domain_level(nn_similarity, half_width)
        confidence_level = self._confidence_level(nn_similarity, half_width, risk_flags)
        recommended_use = self._recommended_use(nn_similarity, half_width, risk_flags)

        return {
            "input": {
                "smiles": smiles,
                "canonical_smiles": str(query_df.iloc[0]["smiles_rdkit"]),
                "parent_smiles": str(query_df.iloc[0]["parent_smiles"]),
                "pH": None if p_h is None else float(p_h),
                "temp_C": None if temp_c is None else float(temp_c),
                "pH_over_TK": None if p_h_over_tk is None else float(p_h_over_tk),
                "source_surrogate": str(query_df.iloc[0]["source_block"]),
            },
            "prediction": {
                "pred_logK": pred_logk,
                "pred_kSO4": pred_k,
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "ci_half_width": half_width,
                "ensemble_disagreement_sigma": sigma,
                "nominal_coverage": self.nominal_coverage,
            },
            "applicability": {
                "nn_similarity": nn_similarity,
                "similarity_bin": self._similarity_bin(nn_similarity),
                "empirical_ad_threshold": self.ad_threshold,
                "empirical_in_domain": empirical_in_domain,
                "domain_level": domain_level,
                "confidence_level": confidence_level,
            },
            "expected_error_reference": expected_error,
            "risk_flags": risk_flags,
            "recommended_use": recommended_use,
            "neighbors": neighbors,
            "base_model_predictions": {name: float(pred) for name, pred in zip(self.base_models.keys(), base_preds)},
        }


def translations(lang: str) -> Dict[str, str]:
    zh = {
        "report_title": "SO4 radical 单分子预测报告",
        "input_smiles": "输入 SMILES",
        "canonical_smiles": "规范化 SMILES",
        "pred_logk": "预测 logK",
        "pred_k": "预测 kSO4",
        "interval": "预测区间",
        "half_width": "区间半宽",
        "sigma": "集成分歧 sigma",
        "nn_similarity": "最近邻相似度",
        "ad_threshold": "经验适用域阈值",
        "domain_level": "适用域等级",
        "confidence": "预测可信度",
        "random_err": "预期绝对误差（随机分层参考）",
        "scaffold_err": "预期绝对误差（新骨架参考）",
        "risk_flags": "风险标签",
        "recommendation": "使用建议",
        "neighbors": "最相近训练样本",
        "screening-ready": "适合优先级筛选和速率常数快速估计。",
        "use-with-caution": "可作为定量参考，但应结合可信区间与结构风险标签谨慎解释。",
        "coarse-screen-only": "更适合作为粗筛线索，不建议直接替代实验测定。",
        "high-confidence": "高可信度",
        "medium-confidence": "中等可信度",
        "low-confidence": "低可信度",
        "none": "无",
    }
    en = {
        "report_title": "SO4 radical Single-Molecule Prediction Report",
        "input_smiles": "Input SMILES",
        "canonical_smiles": "Canonical SMILES",
        "pred_logk": "Predicted logK",
        "pred_k": "Predicted kSO4",
        "interval": "Prediction interval",
        "half_width": "Interval half-width",
        "sigma": "Ensemble disagreement sigma",
        "nn_similarity": "Nearest-neighbor similarity",
        "ad_threshold": "Empirical AD threshold",
        "domain_level": "Domain level",
        "confidence": "Confidence",
        "random_err": "Expected abs error (random-like reference)",
        "scaffold_err": "Expected abs error (novel-scaffold reference)",
        "risk_flags": "Risk flags",
        "recommendation": "Recommendation",
        "neighbors": "Nearest reference neighbors",
        "screening-ready": "Suitable for prioritization and rapid rate-constant estimation.",
        "use-with-caution": "Can be used as a quantitative reference, but interpret together with the interval and structural risk tags.",
        "coarse-screen-only": "Use mainly as a coarse screening clue; do not directly replace experiments.",
        "high-confidence": "High confidence",
        "medium-confidence": "Medium confidence",
        "low-confidence": "Low confidence",
        "none": "none",
    }
    return zh if lang == "zh" else en


def format_prediction_report(report: Dict[str, Any], lang: str = "en") -> str:
    t = translations(lang)
    pred = report["prediction"]
    app = report["applicability"]
    err = report["expected_error_reference"]
    risks = report["risk_flags"] or [t["none"]]
    neighbor_lines = [f"  {i}. {nb['compound_name']} | sim={nb['similarity']:.3f} | logK={nb['logK']:.3f}" for i, nb in enumerate(report["neighbors"], start=1)]
    lines = [
        t["report_title"],
        "=" * 40,
        f"{t['input_smiles']}: {report['input']['smiles']}",
        f"{t['canonical_smiles']}: {report['input']['canonical_smiles']}",
        f"{t['pred_logk']}: {pred['pred_logK']:.4f}",
        f"{t['pred_k']}: {pred['pred_kSO4']:.4e}",
        f"{t['interval']} ({int(pred['nominal_coverage']*100)}% nominal): [{pred['ci_lower']:.4f}, {pred['ci_upper']:.4f}]",
        f"{t['half_width']}: {pred['ci_half_width']:.4f}",
        f"{t['sigma']}: {pred['ensemble_disagreement_sigma']:.4f}",
        f"{t['nn_similarity']}: {app['nn_similarity']:.3f} ({app['similarity_bin']})",
        f"{t['ad_threshold']}: {app['empirical_ad_threshold']:.3f}",
        f"{t['domain_level']}: {app['domain_level']} | empirical_in_domain={app['empirical_in_domain']}",
        f"{t['confidence']}: {t.get(app['confidence_level'], app['confidence_level'])}",
        f"{t['random_err']}: {err['random_mean_abs_error']:.3f}",
        f"{t['scaffold_err']}: {err['scaffold_mean_abs_error']:.3f}",
        f"{t['risk_flags']}: {', '.join(risks)}",
        f"{t['recommendation']}: {t.get(report['recommended_use'], report['recommended_use'])}",
        f"{t['neighbors']}:",
        *neighbor_lines,
    ]
    return "\n".join(lines)


def build_download_bundle(report: Dict[str, Any], lang: str = "en") -> bytes:
    import zipfile

    txt = format_prediction_report(report, lang=lang).encode("utf-8")
    json_bytes = json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")
    neighbor_csv = pd.DataFrame(report["neighbors"]).to_csv(index=False).encode("utf-8-sig")
    base_csv = pd.DataFrame([report["base_model_predictions"]]).to_csv(index=False).encode("utf-8-sig")
    input_csv = pd.DataFrame([report["input"]]).to_csv(index=False).encode("utf-8-sig")
    pred_csv = pd.DataFrame([
        report["prediction"] | report["applicability"] | report["expected_error_reference"] | {"recommended_use": report["recommended_use"], "risk_flags": ";".join(report["risk_flags"])}
    ]).to_csv(index=False).encode("utf-8-sig")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.txt", txt)
        zf.writestr("report.json", json_bytes)
        zf.writestr("input.csv", input_csv)
        zf.writestr("prediction_summary.csv", pred_csv)
        zf.writestr("nearest_neighbors.csv", neighbor_csv)
        zf.writestr("base_model_predictions.csv", base_csv)
    return buf.getvalue()
