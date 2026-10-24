# Deployment validation report

Validation date: 2026-10-24

## Frozen model contract

- Prospective deployment model: 1,249 features.
- RDKit 2D descriptors: 217.
- ECFP4 bits: 1,024.
- Mechanistic auxiliary feature: 1 (`so4_prior`).
- Condition and missingness variables: 7.
- Literature-source and replicate-derived variables: excluded.
- Formal structural AD threshold: 0.7005319 (reported as 0.70).
- Severe structural-novelty threshold: 0.40.
- Missing continuous-condition sentinel: -999 with explicit missingness indicators.

## Automated checks

The standard-library test suite verifies:

1. Manifest, feature-count, feature-order, threshold, and missing-value contracts.
2. Successful deserialization and inference for all frozen base and meta models.
3. Explicit handling of completely missing conditions.
4. Automatic derivation of pH/T(K) when pH and temperature are supplied.
5. Detection of pH and temperature values outside observed training ranges.
6. In-domain status for an exact reference structure.
7. High-risk status for a query below the severe-novelty threshold.
8. Safe rejection of invalid SMILES.
9. Rejection of pH values outside the physical input guardrail of 0-14.
10. High-risk condition-domain warning for valid pH values outside the observed training range of 0.5-9.2.

Run:

```powershell
python -m unittest -v test_platform.py
```

Expected result: all tests report `ok` and the final line reports `OK`.

## Streamlit interface check

The page was executed with Streamlit's application test harness. Both initial rendering and a complete prediction interaction were free of application exceptions. The prediction view produced seven metrics, separate structural and condition AD sections, the failure-warning output, nearest-reference data, and a downloadable report.

## Interpretation boundary

The reported 90% interval is nominal and is not presented as an unconditional guarantee for new scaffolds. It must be interpreted jointly with the formal structural AD, condition-domain status, and failure-warning tier. High-risk output is explicitly labeled as screening-only and requires experimental verification.
