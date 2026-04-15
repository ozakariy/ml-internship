from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def get_project_root(start: Path | None = None) -> Path:
    """
    Find the repository root by walking upward until pyproject.toml is found.
    """
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate project root. Make sure you are running inside the repo."
    )


def _first_existing_path(candidates: list[Path]) -> Path:
    """
    Return the first existing path from a list of candidates.
    """
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of the expected data files were found:\n"
        + "\n".join(str(p) for p in candidates)
    )


def load_task1_inputs(project_root: Path | None = None) -> Dict[str, pd.DataFrame]:
    """
    Load Task 1 input tables.

    Expected files:
      - data/input/internship_data_matrix.csv
      - data/input/internship_feature_metadata.csv
      - data/input/intership_acquisition_list.csv  # note: file name typo in dataset
      - data/input/exogenous_standards.csv
    """
    root = get_project_root(project_root)
    data_dir = root / "data" / "input"

    data_matrix_path = _first_existing_path(
        [
            data_dir / "internship_data_matrix.csv",
        ]
    )
    feature_metadata_path = _first_existing_path(
        [
            data_dir / "internship_feature_metadata.csv",
        ]
    )
    acquisition_list_path = _first_existing_path(
        [
            data_dir / "intership_acquisition_list.csv",   # actual dataset name
            data_dir / "internship_acquisition_list.csv",  # fallback if renamed later
        ]
    )
    standards_path = _first_existing_path(
        [
            data_dir / "exogenous_standards.csv",
        ]
    )

    data_matrix = pd.read_csv(data_matrix_path)
    feature_metadata = pd.read_csv(feature_metadata_path)
    acquisition_list = pd.read_csv(acquisition_list_path)
    exogenous_standards = pd.read_csv(standards_path)

    return {
        "data_matrix": data_matrix,
        "feature_metadata": feature_metadata,
        "acquisition_list": acquisition_list,
        "exogenous_standards": exogenous_standards,
        "paths": pd.DataFrame(
            {
                "table": [
                    "data_matrix",
                    "feature_metadata",
                    "acquisition_list",
                    "exogenous_standards",
                ],
                "path": [
                    str(data_matrix_path),
                    str(feature_metadata_path),
                    str(acquisition_list_path),
                    str(standards_path),
                ],
            }
        ),
    }
