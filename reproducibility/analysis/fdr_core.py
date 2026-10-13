"""Correct complete internal-cohort hypothesis families without selecting by p-value."""
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

FAMILIES = {"clinical_correlations": 396, "gait_comparisons": 19}

def correct_family(frame, family, alpha=0.05):
    if family not in FAMILIES:
        raise ValueError("Unknown hypothesis family")
    expected = FAMILIES[family]
    if len(frame) != expected:
        raise ValueError(f"{family}: require all {expected} tests, including nonsignificant tests; received {len(frame)}")
    required = ["test_id", "p_raw"]
    labels = ["error_metric", "clinical_measure"] if family == "clinical_correlations" else ["gait_parameter"]
    required += labels
    absent = sorted(set(required) - set(frame.columns))
    if absent:
        raise ValueError(f"Missing columns: {absent}")
    result = frame.copy()
    for column in ["test_id", *labels]:
        result[column] = result[column].astype("string").str.strip()
        if result[column].isna().any() or result[column].eq("").any():
            raise ValueError(f"Provide every {column}; do not invent hypothesis labels")
    if result.test_id.duplicated().any() or result.duplicated(labels).any():
        raise ValueError("Duplicate tests: supply each hypothesis once")
    if "cohort" in result:
        cohorts = result.cohort.astype("string").str.strip().str.lower()
        if cohorts.isna().any() or not cohorts.isin(["internal", "shida"]).all():
            raise ValueError("These two correction families are for the Shida internal cohort only")
    if family == "clinical_correlations":
        if result.error_metric.nunique() != 18 or result.clinical_measure.nunique() != 22:
            raise ValueError("Require exactly 18 error metrics x 22 clinical measures, one row per combination")
    p = pd.to_numeric(result.p_raw, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Every p_raw must be an exact numeric p-value in [0,1]; blanks, '<0.05' and '<0.001' are not accepted")
    if np.any(p == 0):
        raise ValueError("p_raw contains zero. Export full-precision p-values; investigate rounding or numerical underflow before correction")
    for column in ("n_participants", "r"):
        if column in result:
            raw = result[column]
            provided = raw.notna() & raw.astype(str).str.strip().ne("")
            values = pd.to_numeric(raw, errors="coerce")
            if not np.isfinite(values[provided]).all():
                raise ValueError(f"Invalid numeric {column}")
            if column == "n_participants" and ((values[provided] < 3) | (values[provided] % 1 != 0)).any():
                raise ValueError("n_participants must contain integer participant counts >=3")
            if column == "r" and (values[provided].abs() > 1).any():
                raise ValueError("Correlation coefficients must be between -1 and 1")
    result["p_raw"] = p
    result["family"] = family
    result["family_size"] = expected
    result["q_bh"] = false_discovery_control(p, method="bh")
    result["q_by"] = false_discovery_control(p, method="by")
    result["nominal_p_lt_0_05"] = p < alpha
    result["bh_q_lt_0_05"] = result.q_bh < alpha
    result["by_q_lt_0_05"] = result.q_by < alpha
    return result

def summarize(result):
    return {
        "family": result.family.iloc[0],
        "n_tests": len(result),
        "nominal_p_lt_0_05": int(result.nominal_p_lt_0_05.sum()),
        "bh_q_lt_0_05": int(result.bh_q_lt_0_05.sum()),
        "by_q_lt_0_05": int(result.by_q_lt_0_05.sum()),
        "smallest_p": float(result.p_raw.min()),
        "smallest_q_bh": float(result.q_bh.min()),
        "smallest_q_by": float(result.q_by.min()),
    }
