# Reviewer-ready SO4 radical predictor

This package deploys the audited prospective model used in the revised manuscript. The model has 1,249 inputs and excludes literature-source indicators and replicate-derived variables that are unavailable for a newly submitted molecule.

## Reported outputs

- Point prediction of `log10[k/(M^-1 s^-1)]` and the corresponding rate constant.
- Nominal 90% conformal interval with an explicit extrapolation disclaimer.
- Structural applicability domain (AD), using ECFP4 Tanimoto similarity.
- Separate condition-domain assessment for pH, temperature, and derived pH/T(K).
- Failure-warning score, risk tier, nearest references, and a downloadable audit report.

## Frozen decision rules

- Similarity >= 0.70: within the formal empirical structural AD.
- 0.40 <= similarity < 0.70: outside the formal AD; extrapolative prediction.
- Similarity < 0.40: severe structural-novelty warning.
- Failure-warning tiers: low = 0, moderate = 1-2, high >= 3.
- High-risk results are for screening only and require experimental verification.

Missing condition values are encoded with the fixed training sentinel (`-999`) and explicit missingness indicators; the app does not silently assume neutral pH or room temperature. When both pH and temperature are supplied, pH/T(K) is derived automatically.

The interface applies a physical pH guardrail. Values from 0 to 14 are accepted; values outside the observed training range of 0.5-9.2 are retained only as condition-domain extrapolations and trigger a high-risk warning. Values below 0 or above 14 are rejected before prediction.

Condition reporting in the 1,796-sample training set was sparse: pH was available for 380 samples (21.16%; range 0.5-9.2), temperature for 293 samples (16.31%; range 9.1-31.85 degrees C), and pH/T(K) for 439 samples (24.44%; range 0-0.031). The interface reports structural AD and condition AD as independent indicators. A query below similarity 0.40 triggers an automatic red extrapolation alert with plain-language restrictions and an experimental-verification recommendation.

## Run locally

```powershell
streamlit run app.py
```

Use the pinned Python and package versions. Changing RDKit or model-library versions can change descriptor order or prevent model deserialization; do not upgrade dependencies without retraining and revalidation.

The current Community Cloud deployment uses Python 3.11. Before uploading, run `python -m unittest -v test_platform.py`; the complete audit record is in `VALIDATION_REPORT.md`.
