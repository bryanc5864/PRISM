"""
System configuration for PRISM.

Defines the SystemConfig dataclass that encapsulates all biological system-specific
parameters, enabling PRISM to be applied to any perturbation system (skin, pancreas,
cortex, HSC, etc.) with only a config file change.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml


@dataclass
class SystemConfig:
    """Configuration for a biological system.

    All system-specific constants (condition keys, fate names, marker genes, etc.)
    are encapsulated here. The pipeline code reads from this config instead of
    using hardcoded values.
    """
    name: str                                          # "skin", "pancreas", etc.
    condition_key: str                                 # adata.obs column for perturbation
    conditions: Dict[str, int]                         # {"WT": 0, "En1-cKO": 1}
    fate_names: List[str]                              # ["uncommitted", "eccrine", "hair"]
    known_fate_threshold: int = 2                      # label >= this means "known fate"
    forced_genes: List[str] = field(default_factory=list)
    known_markers: Dict[str, List[str]] = field(default_factory=dict)
    sample_condition_map: Dict[str, str] = field(default_factory=dict)
    root_cluster: str = "Epi0"                         # trajectory root
    cluster_key: str = "cluster"                       # obs column for clusters
    geo_accession: str = ""                            # GEO accession for download
    download_function: str = ""                        # module.function path for data download

    # Label assignment parameters
    label_strategy: str = "hierarchical"                   # "hierarchical" (skin) or "flat" (pancreas, cortex, hsc)
    fate_categories: List[str] = field(default_factory=lambda: ["non_appendage", "undetermined", "eccrine", "hair"])
    marker_scores: Dict[str, List[str]] = field(default_factory=dict)

    # Annotation-based label assignment (e.g., Weinreb cell_type_annotation)
    annotation_key: str = ""                               # obs column with published annotations
    annotation_fate_map: Dict[str, str] = field(default_factory=dict)  # annotation → fate name

    # QC overrides (for Smart-seq2 or other non-10X data)
    max_genes: int = 0  # 0 means use default from training config

    # Branch analysis parameters
    branch_names: Dict[str, str] = field(default_factory=lambda: {"branch_a": "eccrine_branch", "branch_b": "hair_branch"})
    condition_branch_map: Dict[str, str] = field(default_factory=lambda: {"WT": "eccrine_branch", "En1-cKO": "hair_branch"})

    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        """Load system config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        system_data = data.get("system", data)
        return cls(**{k: v for k, v in system_data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str):
        """Save system config to a YAML file."""
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump({"system": asdict(self)}, f, default_flow_style=False, sort_keys=False)


# Default skin config matching current hardcoded behavior
SKIN_CONFIG = SystemConfig(
    name="skin",
    condition_key="genotype",
    conditions={"WT": 0, "En1-cKO": 1},
    fate_names=["uncommitted", "eccrine", "hair"],
    known_fate_threshold=2,
    forced_genes=["En1", "Trpv6", "Dkk4", "Lgr6", "S100a4", "Foxi1", "Defb6"],
    known_markers={
        "eccrine": ["En1", "Trpv6", "Dkk4", "Foxi1", "Defb6"],
        "hair": ["Lhx2", "Sox9", "Wnt10b", "Shh", "Edar"],
    },
    sample_condition_map={
        "GSM6833478": "WT",
        "GSM6833479": "WT",
        "GSM6833480": "WT",
        "GSM6833481": "WT",
        "GSM6833482": "En1-cKO",
        "GSM6833483": "En1-cKO",
    },
    root_cluster="Epi0",
    cluster_key="cluster",
    geo_accession="GSE220977",
    download_function="prism.data.download.download_gse220977",
    label_strategy="hierarchical",
    fate_categories=["non_appendage", "undetermined", "eccrine", "hair"],
    marker_scores={
        "basal": ["Krt14", "Krt5", "Tp63"],
        "diff": ["Krt1", "Krt10", "Ivl", "Lor"],
        "appendage": ["Edar", "Ctnnb1", "Lef1", "Wnt10b"],
        "eccrine": ["En1", "Lgr6", "Dkk4", "Wif1", "Sfrp1"],
        "hair": ["Lhx2", "Sox9", "Shh"],
        "dermal": ["Col1a1", "Col3a1", "Pdgfra", "Dcn"],
        "eden": ["S100a4"],
    },
    branch_names={"branch_a": "eccrine_branch", "branch_b": "hair_branch"},
    condition_branch_map={"WT": "eccrine_branch", "En1-cKO": "hair_branch"},
)
