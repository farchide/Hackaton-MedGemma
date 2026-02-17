"""Data generation, loading, and public dataset integration for Digital Twin Tumor.

Provides realistic longitudinal oncology patient data including lesion
measurements, RECIST classifications, growth model fits, and 3D volumes.

Also includes loaders for real public datasets (PROTEAS, RIDER, QIN)
to demonstrate the system works with clinical data.
"""

from digital_twin_tumor.data.datasets import (
    DATASETS,
    DatasetRegistry,
    PROTEASLoader,
    RIDERLoader,
    convert_public_dataset_to_demo,
    dataset_download_instructions,
)
from digital_twin_tumor.data.demo_loader import DemoLoader
from digital_twin_tumor.data.synthetic import (
    generate_all_demo_data,
    generate_demo_volumes,
)

__all__ = [
    "DATASETS",
    "DatasetRegistry",
    "DemoLoader",
    "PROTEASLoader",
    "RIDERLoader",
    "convert_public_dataset_to_demo",
    "dataset_download_instructions",
    "generate_all_demo_data",
    "generate_demo_volumes",
]
