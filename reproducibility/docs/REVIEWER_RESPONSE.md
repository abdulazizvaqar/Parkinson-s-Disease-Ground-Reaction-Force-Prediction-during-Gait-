# Draft response to code and reproducibility request

We thank the reviewer for emphasizing that data availability alone is insufficient for reproducibility. We have assembled a reproducibility package containing the supplied original training and analysis scripts, revised preprocessing and participant-wise validation code, executable Colab notebooks, dataset split identifiers, available fold-specific selected hyperparameters, architecture and stopping-rule documentation, software records, and external harmonization code with explicit verification requirements. The package also contains the completed repeated-seed and statistical-analysis summaries, source hashes and checks for data/participant separation.

We distinguish the original study from the subsequent methodological revision. Complete per-trial records and selected settings for all original 3,600 reported runs were not available in the recovered materials. Accordingly, we do not claim that the current package reconstructs every original run exactly. Available revision configurations and completed analyses are identified explicitly, together with missing historical training and harmonization records. This allows the supplied code and completed analyses to be inspected and rerun while making the remaining reproducibility limitations transparent.

The package is ready for deposit on GitHub. Insert the actual repository URL and immutable release/commit after publication; do not state that a public repository exists before it is published.

## Manuscript action after publication

Add a CODE AVAILABILITY section: “The available original scripts and revised analysis workflows, split identifiers, selected configuration records and software documentation are available at [actual repository URL], release [actual tag or commit]. The release documents incomplete historical training and external harmonization records; it does not constitute a complete archived record of all original 3,600 runs.”

Use author action in past tense only after repository publication and manuscript insertion. This package addresses code disclosure but the missing original per-run ledger remains unresolved.
