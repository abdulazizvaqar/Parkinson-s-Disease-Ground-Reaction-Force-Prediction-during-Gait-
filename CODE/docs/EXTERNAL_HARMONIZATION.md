# External harmonization contract

The original external inference code is archived in `legacy/ExternalValidation_GRF(1).py`. The processed workbook and saved outputs do not establish all original axis, participant, visit or unit transformations. The revised importer is reproducible as code, but its verification template is incomplete and it has not generated a validated new external result.

## Revised transformations, after source verification

1. Supply a mapping for every side-specific workbook column, including participant, trial, cycle, visit, inclusion/exclusion and source mass when needed. Side/component column pairing must be independently verified.
2. Read 101 samples as data with header=None; do not drop the first sample as a spreadsheet header.
3. Choose the verified ankle X/Y/Z axis for each AP/Vertical/ML task and apply recorded ankle/GRF signs (+1 or -1). Selection is source-based, not chosen to maximize external performance.
4. If force is in N, divide by verified cycle-associated participant mass in kg. Already N/kg force is not renormalized.
5. Interpolate both ankle/force signals linearly from a 101-point to a 100-point 0–100% phase grid, preserving endpoints. This resampling is not a substitute for event alignment or genuine time intervals.
6. At external evaluation, `harmonize_input_units` in revision_data.py converts known m/mm ankle units to the frozen internal unit system: m→mm ×1000 or mm→m ×0.001. Apply coordinate mode and frozen training scaler without fitting on external data.
7. Preserve participant/trial/visit labels and source hashes. Analyze baseline and post-intervention visits separately when available; do not infer visits from column positions or count repeated model outputs as new participants.

Verification gates require known units, source axes/signs, normalized phase grid, column pairing, participant disjointness, identity/visit mapping and complete context. The templates intentionally leave these values unconfirmed. The source cohort was previously evaluated in the manuscript, so future use must not describe it as untouched throughout the study.

Pelvis coordinates, direction vectors and body scaling require additional synchronized inputs. The scalar workbook cannot supply a verified 3D rotation/alignment transformation. Missing information remains a documented blocker; no laboratory-to-anatomical transform is fabricated.
