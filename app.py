from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from predictor_core import SO4Predictor, build_download_bundle


st.set_page_config(
    page_title="SO4 radical kinetics predictor",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #17324d;
        --muted: #60758a;
        --line: #d9e3ea;
        --paper: #f7fafb;
        --accent: #007c78;
    }
    .stApp {
        background:
            radial-gradient(circle at 92% 4%, rgba(0,124,120,.10), transparent 24rem),
            linear-gradient(180deg, #ffffff 0%, var(--paper) 100%);
    }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,.88);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 8px 22px rgba(23,50,77,.06);
    }
    [data-testid="stMetricValue"] {
        font-size: clamp(1.35rem, 2vw, 1.8rem);
        white-space: nowrap;
    }
    .policy-card {
        background: rgba(255,255,255,.86);
        border: 1px solid var(--line);
        border-left: 5px solid var(--accent);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0 18px 0;
        color: var(--ink);
    }
    .small-note { color: var(--muted); font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> SO4Predictor:
    return SO4Predictor()


def optional_float(value: str, field_name: str) -> Optional[float]:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a number or left blank.") from exc


FLAG_LABELS = {
    "severe_structural_novelty": "Severe structural novelty (similarity < 0.40)",
    "adjacent_analogue_disagreement": "Adjacent-reference experimental disagreement",
    "multisite_reactivity": "Multiple competing reactive motifs",
    "electronic_or_surface_extreme": "Extreme charge or polar-surface proxy",
}

RISK_LABELS = {"low": "Low", "moderate": "Moderate", "high": "High"}
STRUCTURE_STATUS_LABELS = {
    "in-domain": "Within formal AD",
    "out-of-domain": "Outside formal AD",
    "severe-extrapolation": "Severe extrapolation",
}
CONDITION_STATUS_LABELS = {
    "within-observed-range": "Within observed ranges",
    "out-of-range": "Outside observed ranges",
    "not-assessed-missing": "Not assessed (missing)",
}


predictor = load_predictor()

st.title("SO₄•⁻ reaction-rate predictor")
st.caption(
    "Audited prospective model for organic compounds in water: point prediction, 90% interval, "
    "structural AD, condition AD, and a failure-warning index."
)

st.markdown(
    f"""
    <div class="policy-card">
      <b>Decision rule.</b> Similarity ≥ {predictor.formal_ad_threshold:.2f} is within the formal empirical structural AD;
      {predictor.severe_similarity_threshold:.2f}–{predictor.formal_ad_threshold:.2f} is structural extrapolation;
      similarity &lt; {predictor.severe_similarity_threshold:.2f} triggers a severe warning.
      Missing conditions are encoded explicitly rather than replaced by an assumed pH or temperature.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Prediction input")
    smiles = st.text_area(
        "SMILES",
        "CCO",
        height=110,
        help="Enter one valid molecular SMILES. The input is canonicalized with the same RDKit workflow used for training.",
    )
    p_h_text = st.text_input(
        "pH (optional)",
        help="Accepted input: 0–14. Observed training range: 0.5–9.2. Leave blank if unknown.",
    )
    temperature_text = st.text_input(
        "Temperature, °C (optional)",
        help="Observed training range: 9.1–31.85 °C. Leave blank if unknown.",
    )
    st.caption("When both pH and temperature are supplied, pH/T(K) is calculated automatically.")
    top_k = st.slider("Nearest reference molecules", 3, 10, 5)
    run = st.button("Predict", type="primary", use_container_width=True)

    with st.expander("Model and input policy"):
        st.write("Target: log₁₀[k/(M⁻¹ s⁻¹)]")
        st.write(f"Model: prospective {len(predictor.feature_names):,}-feature OOF stack")
        st.write("Unavailable source and replicate variables are excluded.")
        st.write("Missing continuous conditions use the fixed training sentinel (−999) plus missingness indicators.")
        st.write("pH: 380/1,796 available (21.16%); 78.84% missing.")
        st.write("Temperature: 293/1,796 available (16.31%); 83.69% missing.")
        st.write("pH/T(K): 439/1,796 available (24.44%); 75.56% missing.")
        st.write("Outputs are intended for screening and prioritization, not as substitutes for kinetic experiments.")

if not run:
    st.info("Enter a molecule and select Predict. pH and temperature may be left blank.")
else:
    try:
        p_h = optional_float(p_h_text, "pH")
        temperature = optional_float(temperature_text, "Temperature")
        report = predictor.predict(
            smiles,
            p_h=p_h,
            temp_c=temperature,
            p_h_over_tk=None,
            top_k_neighbors=top_k,
        )

        prediction = report["prediction"]
        structure = report["structure_applicability_domain"]
        condition = report["condition_applicability_domain"]
        warning = report["failure_warning"]
        overall_risk = report["overall_risk"]

        st.subheader("Prediction")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted log k", f"{prediction['logK']:.4f}")
        c2.metric("Predicted k (M⁻¹ s⁻¹)", f"{prediction['k_M-1_s-1']:.3e}")
        c3.metric("Nominal 90% interval", f"[{prediction['ci_lower']:.3f}, {prediction['ci_upper']:.3f}]")
        c4.metric("Ensemble disagreement", f"{prediction['ensemble_disagreement_sigma']:.3f}")

        st.subheader("Automatic risk communication")
        similarity = structure["nearest_neighbor_similarity"]
        if structure["status"] == "severe-extrapolation":
            st.error(
                f"AUTOMATIC EXTRAPOLATION ALERT: nearest-neighbor similarity is {similarity:.3f}, below the "
                f"severe-warning threshold of {structure['severe_warning_threshold']:.2f}. This molecule is far from "
                "the model's reference chemistry. Treat the value as screening-only; do not use it as a reliable "
                "quantitative kinetic constant without experimental verification."
            )
        elif structure["status"] == "out-of-domain":
            st.warning(
                f"OUTSIDE THE FORMAL STRUCTURAL AD: similarity is {similarity:.3f}, below the formal boundary of "
                f"{structure['formal_ad_threshold']:.2f}. This is an extrapolative estimate. Use the interval and "
                "nearest references, and obtain experimental confirmation before consequential use."
            )
        else:
            st.success(
                f"STRUCTURALLY IN DOMAIN: similarity is {similarity:.3f}, at or above the formal boundary of "
                f"{structure['formal_ad_threshold']:.2f}. Other warning and condition checks still apply."
            )

        if condition["status"] == "out-of-range":
            st.error(
                "CONDITION-DOMAIN ALERT: at least one supplied condition lies outside the observed training range. "
                "The prediction is high risk and requires experimental verification."
            )
        elif condition["status"] == "not-assessed-missing":
            st.info(
                "CONDITION AD NOT ASSESSED: pH and temperature were not supplied. The model uses −999 plus explicit "
                "missingness indicators; it does not assume pH 7 or 25 °C."
            )

        st.markdown(f"**Overall automatic risk tier: {RISK_LABELS[overall_risk]}.**")
        if overall_risk == "high":
            st.error("Required action: screening use only; prioritize direct kinetic measurement before environmental decisions.")
        elif overall_risk == "moderate":
            st.warning("Recommended action: interpret cautiously and confirm experimentally before consequential use.")

        st.subheader("Applicability domains and warning index")
        a1, a2, a3 = st.columns(3)
        a1.metric("Structural AD", STRUCTURE_STATUS_LABELS[structure["status"]])
        a1.caption("Structural AD asks whether the query resembles molecules represented in the training data.")
        a1.caption(
            f"Similarity = {structure['nearest_neighbor_similarity']:.3f}; "
            f"formal AD = {structure['formal_ad_threshold']:.2f}; "
            f"severe warning = {structure['severe_warning_threshold']:.2f}."
        )
        a1.write(structure["message"])

        a2.metric("Condition AD", CONDITION_STATUS_LABELS[condition["status"]])
        a2.caption("Condition AD checks supplied pH, temperature, and derived pH/T(K) against observed ranges.")
        if condition["warnings"]:
            for message in condition["warnings"]:
                a2.warning(message)
        elif condition["status"] == "not-assessed-missing":
            a2.caption("No condition was supplied. Fixed sentinel values and explicit missingness indicators were used.")
        else:
            a2.caption("All supplied or derived conditions fall within their observed training ranges.")

        a3.metric("Failure-warning tier", RISK_LABELS[warning["risk_tier"]])
        a3.caption(f"Score = {warning['score']}; {warning['action']}.")
        a3.caption(f"Adjacent-reference label span = {warning['nearest_analogue_label_span']:.3f} log units.")

        if warning["flags"]:
            st.markdown("**Triggered warning rules**")
            for flag in warning["flags"]:
                st.write(f"- {FLAG_LABELS.get(flag, flag)}")
        else:
            st.write("No failure-warning rule was triggered.")

        with st.expander("Condition handling and observed ranges", expanded=False):
            normalized = report["input"]["conditions"]
            rows = []
            display_names = {"pH": "pH", "temp_C": "Temperature (°C)", "pH_over_TK": "pH/T(K)"}
            for key, limits in condition["observed_training_ranges"].items():
                total_n = int(predictor.manifest["n_samples"])
                available_n = int(limits["n_available"])
                missing_n = total_n - available_n
                rows.append(
                    {
                        "Variable": display_names[key],
                        "Input used": normalized[key],
                        "Available n": available_n,
                        "Available (%)": round(100.0 * available_n / total_n, 2),
                        "Missing n": missing_n,
                        "Missing (%)": round(100.0 * missing_n / total_n, 2),
                        "Observed minimum": limits["min"],
                        "Observed maximum": limits["max"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            if report["input"]["pH_over_TK_derived_from_pH_and_temperature"]:
                st.caption("pH/T(K) was derived automatically from the supplied pH and temperature.")

        st.subheader("Nearest reference molecules")
        neighbors = pd.DataFrame(report["neighbors"]).rename(
            columns={
                "compound_name": "Compound",
                "smiles": "Reference SMILES",
                "logK": "Experimental log k",
                "similarity": "Tanimoto similarity",
            }
        )
        st.dataframe(neighbors, use_container_width=True, hide_index=True)

        st.download_button(
            "Download prediction report",
            build_download_bundle(report),
            file_name="so4_prediction_report.zip",
            mime="application/zip",
            use_container_width=True,
        )

        with st.expander("Machine-readable output"):
            st.json(report)

        st.markdown(
            '<p class="small-note">The conformal interval is nominal and should not be interpreted as an unconditional '
            "coverage guarantee for structurally novel compounds. Interpret it jointly with both AD indicators and the warning index.</p>",
            unsafe_allow_html=True,
        )
    except ValueError as exc:
        st.error(f"Input rejected: {exc}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
