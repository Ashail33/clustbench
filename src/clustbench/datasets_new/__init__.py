"""Round 14 dataset generators discovered via multi-agent workflow.

Three general-purpose regimes plus two adversarial datasets designed to
break specific algorithm classes (see docs/ALGORITHM_ANALYSIS.md Round 14).
"""

from .hierarchical_nested import gen_hierarchical_nested
from .imbalanced_blobs import gen_imbalanced_blobs
from .heavy_tailed_mixture import gen_heavy_tailed_mixture
from .PowerLawStudentT import gen_PowerLawStudentT
from .VariableDensityBridges import gen_VariableDensityBridges

DATASETS_R14 = {
    "hierarchical_nested": gen_hierarchical_nested,
    "imbalanced_blobs": gen_imbalanced_blobs,
    "heavy_tailed_mixture": gen_heavy_tailed_mixture,
    "PowerLawStudentT": gen_PowerLawStudentT,
    "VariableDensityBridges": gen_VariableDensityBridges,
}
