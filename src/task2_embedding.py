from __future__ import annotations

from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# ---------------------------------------------------------------------
# Column candidates
# ---------------------------------------------------------------------

SEQUENCE_CANDIDATES = [
    "glycan_sequence",
    "glycan",
    "sequence",
    "iupac",
    "iupac_condensed",
    "glycan_iupac",
    "structure",
    "glycan_structure",
]

COMPOSITION_CANDIDATES = [
    "composition",
    "glycan_composition",
    "comp",
]

TISSUE_SAMPLE_CANDIDATES = [
    "tissue_sample",
    "tissue",
    "sample_type",
    "sample_origin",
    "sample",
]

TISSUE_SPECIES_CANDIDATES = [
    "tissue_species",
    "species",
    "organism",
    "taxonomy",
    "host",
]

DISEASE_CANDIDATES = [
    "disease",
    "disease_association",
    "condition",
    "phenotype",
]

PROTEIN_CANDIDATES = [
    "protein",
    "lectin",
    "target",
    "gene",
    "name",
    "protein_name",
]

TASK1_ID_CANDIDATES = [
    "feature",
    "feature_id",
    "candidate_id",
]

TASK1_TIER_CANDIDATES = [
    "target_tier",
    "biomarker_tier",
    "priority_tier",
    "tier",
]

TASK1_CLUSTER_CANDIDATES = [
    "redundancy_cluster",
    "redundancy_cluster_id",
    "cluster_id",
]

TASK1_PROFILE_CANDIDATES = [
    "clinical_profile",
    "profile",
    "feature_profile",
]


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def get_project_root(start: Path | None = None) -> Path:
    """
    Find the repo root by walking upward until pyproject.toml is found.
    """
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate project root. Make sure you are running inside the repo."
    )


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find any of the expected files:\n" + "\n".join(str(p) for p in paths)
    )


def _read_pickle(path: Path) -> Any:
    try:
        return pd.read_pickle(path)
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def _coerce_to_dataframe(obj: Any, source_name: str) -> pd.DataFrame:
    """
    Coerce common pickle payloads into a DataFrame.
    Handles DataFrame, Series, list, tuple, set, numpy array, dict.
    """
    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    if isinstance(obj, pd.Series):
        df = obj.to_frame(name="value").reset_index()
        df.columns = ["index", "value"]
        return df

    if isinstance(obj, dict):
        # Best effort: try normal DataFrame construction
        try:
            return pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame({"value": list(obj.values())})

    if isinstance(obj, (list, tuple, set)):
        values = list(obj)
        if len(values) == 0:
            return pd.DataFrame({"value": []})
        if isinstance(values[0], dict):
            return pd.DataFrame(values)
        return pd.DataFrame({"value": values})

    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            return pd.DataFrame({"value": obj.tolist()})
        return pd.DataFrame(obj)

    raise TypeError(
        f"Could not coerce object from {source_name!r} into a pandas DataFrame. "
        f"Got type: {type(obj)}"
    )


def _normalize_string_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan})
    )


def _find_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if required:
        raise KeyError(f"Could not find required column among: {candidates}")
    return None


def _looks_like_glycan_text(series: pd.Series, min_fraction: float = 0.3) -> bool:
    """
    Heuristic for whether a text series probably contains glycan strings.
    """
    s = _normalize_string_series(series).dropna()
    if s.empty:
        return False

    patterns = [
        r"GlcNAc", r"Gal", r"Man", r"Fuc", r"Neu5Ac", r"Neu5Gc",
        r"\(", r"\)", r"\[", r"\]", r";N", r"WURCS", r"RES", r"GlyTouCan",
    ]
    hits = 0
    for value in s.head(200):
        text = str(value)
        if any(re.search(p, text) for p in patterns):
            hits += 1

    return (hits / min(len(s.head(200)), 200)) >= min_fraction


def _extract_sequence_series(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """
    Extract the best sequence-like column from a DataFrame.
    Uses explicit candidate names first, then falls back to heuristics.
    """
    seq_col = _find_col(df, SEQUENCE_CANDIDATES, required=False)
    if seq_col is not None:
        return _normalize_string_series(df[seq_col]), str(seq_col)

    # Heuristic fallback over object-like columns
    object_cols = [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])
    ]
    for col in object_cols:
        if _looks_like_glycan_text(df[col]):
            return _normalize_string_series(df[col]), str(col)

    # Heuristic fallback to index
    index_series = pd.Series(df.index, name="index")
    if _looks_like_glycan_text(index_series):
        return _normalize_string_series(index_series), "index"

    # Last fallback for simple one-column containers
    if df.shape[1] == 1:
        only_col = df.columns[0]
        return _normalize_string_series(df[only_col]), str(only_col)

    return pd.Series(np.nan, index=df.index, dtype="object"), "missing"


def _first_not_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------

def load_task2_inputs(project_root: Path | None = None) -> dict[str, Any]:
    """
    Load Task 2 input tables from data/glycan_embedding.
    """
    root = get_project_root(project_root)
    data_dir = root / "data" / "glycan_embedding"

    glycan_list_path = _first_existing([data_dir / "glycan_list.csv"])
    df_glycan_path = _first_existing([data_dir / "df_glycan.pkl"])
    glycan_binding_path = _first_existing([data_dir / "glycan_binding.pkl"])
    n_glycans_path = _first_existing(
        [
            data_dir / "N_glycans_df.pkl",
            data_dir / "N-Glycans.pkl",
            data_dir / "N_Glycans.pkl",
        ]
    )

    glycan_list = pd.read_csv(glycan_list_path)
    df_glycan = _coerce_to_dataframe(_read_pickle(df_glycan_path), "df_glycan")
    glycan_binding = _coerce_to_dataframe(_read_pickle(glycan_binding_path), "glycan_binding")
    n_glycans_df = _coerce_to_dataframe(_read_pickle(n_glycans_path), "n_glycans_df")

    return {
        "glycan_list": glycan_list,
        "df_glycan": df_glycan,
        "glycan_binding": glycan_binding,
        "n_glycans_df": n_glycans_df,
        "paths": {
            "glycan_list": glycan_list_path,
            "df_glycan": df_glycan_path,
            "glycan_binding": glycan_binding_path,
            "n_glycans_df": n_glycans_path,
        },
    }


def load_task1_targets(project_root: Path | None = None) -> pd.DataFrame | None:
    """
    Best-effort loader for a Task 1 -> Task 2 handoff table if one already exists.

    This is intentionally tolerant: it searches a few plausible processed locations.
    If nothing is found, returns None and Step 1 can still proceed in provisional mode.
    """
    root = get_project_root(project_root)

    candidates = [
        root / "data" / "processed" / "task1_partc_final" / "task1_task2_handoff.csv",
        root / "data" / "processed" / "task1_partc_final" / "final_biomarker_targets.csv",
        root / "data" / "processed" / "task1_partc_final" / "feature_priority_table.csv",
        root / "data" / "processed" / "task1_partc_final" / "cluster_aware_feature_ranking.csv",
        root / "data" / "processed" / "task1_partb_final" / "retained_feature_manifest_main_annotated.csv",
        root / "data" / "processed" / "task1_partb_final" / "retained_feature_manifest_main.csv",
    ]

    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None

    return pd.read_csv(existing[0])


# ---------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------

def standardize_glycan_table(
    df: pd.DataFrame,
    *,
    source_name: str,
    require_sequence: bool = False,
    mark_as_n_glycan: bool = False,
) -> pd.DataFrame:
    """
    Standardize any glycan-related table to a canonical schema.

    Output columns:
    - glycan_sequence
    - composition
    - tissue_sample
    - tissue_species
    - disease
    - source_name
    - is_n_glycan
    - original_sequence_column
    """
    out = pd.DataFrame(index=df.index)

    sequence_series, seq_source = _extract_sequence_series(df)
    if require_sequence and sequence_series.notna().sum() == 0:
        raise KeyError(
            f"{source_name}: no glycan sequence-like column could be identified. "
            f"Available columns: {list(df.columns)}"
        )

    comp_col = _find_col(df, COMPOSITION_CANDIDATES, required=False)
    sample_col = _find_col(df, TISSUE_SAMPLE_CANDIDATES, required=False)
    species_col = _find_col(df, TISSUE_SPECIES_CANDIDATES, required=False)
    disease_col = _find_col(df, DISEASE_CANDIDATES, required=False)

    out["glycan_sequence"] = sequence_series
    out["composition"] = df[comp_col] if comp_col is not None else np.nan
    out["tissue_sample"] = df[sample_col] if sample_col is not None else np.nan
    out["tissue_species"] = df[species_col] if species_col is not None else np.nan
    out["disease"] = df[disease_col] if disease_col is not None else np.nan
    out["source_name"] = source_name
    out["is_n_glycan"] = int(mark_as_n_glycan)
    out["original_sequence_column"] = seq_source

    # preserve useful ID columns if present
    for col in TASK1_ID_CANDIDATES + TASK1_TIER_CANDIDATES + TASK1_CLUSTER_CANDIDATES + TASK1_PROFILE_CANDIDATES:
        if col in df.columns and col not in out.columns:
            out[col] = df[col]

    out = out.loc[out["glycan_sequence"].notna()].copy()
    out["glycan_sequence"] = _normalize_string_series(out["glycan_sequence"])
    out = out.loc[out["glycan_sequence"].notna()].reset_index(drop=True)
    return out


def standardize_binding_table(glycan_binding: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize glycan_binding into a long table with at least:
    - glycan_sequence
    - protein
    - binding_value

    Supports:
    1) long format with glycan/protein columns
    2) wide numeric format with proteins on rows and glycans as numeric columns
    """
    seq_col = _find_col(glycan_binding, SEQUENCE_CANDIDATES, required=False)
    protein_col = _find_col(glycan_binding, PROTEIN_CANDIDATES, required=False)

    # Case 1: already long-ish
    if seq_col is not None and protein_col is not None:
        numeric_cols = [
            c for c in glycan_binding.columns
            if c not in {seq_col, protein_col} and pd.api.types.is_numeric_dtype(glycan_binding[c])
        ]
        value_col = numeric_cols[0] if numeric_cols else None

        out = pd.DataFrame({
            "glycan_sequence": _normalize_string_series(glycan_binding[seq_col]),
            "protein": _normalize_string_series(glycan_binding[protein_col]),
            "binding_value": glycan_binding[value_col].astype(float) if value_col else 1.0,
        })
        out = out.loc[out["glycan_sequence"].notna() & out["protein"].notna()].copy()
        return out.reset_index(drop=True)

    # Case 2: wide numeric format
    numeric_cols = [c for c in glycan_binding.columns if pd.api.types.is_numeric_dtype(glycan_binding[c])]
    if protein_col is not None and len(numeric_cols) > 0:
        protein_ids = _normalize_string_series(glycan_binding[protein_col])
        wide = glycan_binding[numeric_cols].copy()
        wide.index = protein_ids

        long = (
            wide.reset_index()
            .rename(columns={"index": "protein"})
            .melt(id_vars="protein", var_name="glycan_sequence", value_name="binding_value")
        )
        long["protein"] = _normalize_string_series(long["protein"])
        long["glycan_sequence"] = _normalize_string_series(long["glycan_sequence"])
        long = long.loc[
            long["protein"].notna()
            & long["glycan_sequence"].notna()
            & long["binding_value"].notna()
        ].copy()
        return long.reset_index(drop=True)

    raise ValueError(
        "Could not standardize glycan_binding. "
        "Expected either a long format with glycan/protein columns or a wide numeric format."
    )


# ---------------------------------------------------------------------
# Step 1 — Task 1 -> Task 2 handoff
# ---------------------------------------------------------------------

def _pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_task1_task2_handoff(
    glycan_list: pd.DataFrame,
    task1_targets: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build the handoff table requested in Step 1.

    If task1_targets is provided, we try:
    1) merge on feature-like IDs
    2) fallback merge on glycan sequence if available in both
    Otherwise, we return a provisional handoff from glycan_list alone.
    """
    query_std = standardize_glycan_table(
        glycan_list,
        source_name="glycan_list",
        require_sequence=True,
        mark_as_n_glycan=False,
    )

    # Preserve original glycan_list columns too
    query = glycan_list.copy()

    if task1_targets is None:
        provisional = query_std.copy()
        provisional["feature_id"] = (
            provisional["feature"]
            if "feature" in provisional.columns
            else np.nan
        )
        provisional["biomarker_tier"] = "unassigned_from_task1"
        provisional["redundancy_cluster"] = np.nan
        provisional["clinical_profile"] = np.nan

        handoff = provisional[
            [
                "feature_id",
                "biomarker_tier",
                "redundancy_cluster",
                "clinical_profile",
                "glycan_sequence",
                "composition",
            ]
        ].drop_duplicates().reset_index(drop=True)

        info = {
            "merge_mode": "provisional_from_glycan_list_only",
            "n_task1_rows": 0,
            "n_glycan_list_rows": len(glycan_list),
            "n_handoff_rows": len(handoff),
            "n_mapped_rows": int(handoff["glycan_sequence"].notna().sum()),
        }
        return handoff, info

    t1 = task1_targets.copy()

    # Rename useful Task 1 columns to canonical names
    id_col = _pick_first_existing(t1, TASK1_ID_CANDIDATES)
    tier_col = _pick_first_existing(t1, TASK1_TIER_CANDIDATES)
    cluster_col = _pick_first_existing(t1, TASK1_CLUSTER_CANDIDATES)
    profile_col = _pick_first_existing(t1, TASK1_PROFILE_CANDIDATES)
    t1_seq_col = _pick_first_existing(t1, SEQUENCE_CANDIDATES)

    merge_mode = None
    merged = None

    # 1) Prefer ID-based merge if both sides have an ID
    glycan_id_col = _pick_first_existing(query, TASK1_ID_CANDIDATES)
    if id_col is not None and glycan_id_col is not None:
        merged = t1.merge(
            query,
            left_on=id_col,
            right_on=glycan_id_col,
            how="left",
            suffixes=("_task1", "_glycan"),
        )
        merge_mode = f"id_based:{id_col}->{glycan_id_col}"

    # 2) Fallback to sequence merge if possible
    if merged is None or merged["glycan_sequence"].notna().sum() == 0:
        if t1_seq_col is not None:
            t1_tmp = t1.copy()
            t1_tmp["glycan_sequence"] = _normalize_string_series(t1_tmp[t1_seq_col])
            merged = t1_tmp.merge(
                query_std[["glycan_sequence", "composition"]].drop_duplicates(),
                on="glycan_sequence",
                how="left",
            )
            merge_mode = f"sequence_based:{t1_seq_col}"

    # 3) Conservative fallback
    if merged is None:
        merged = t1.copy()
        merged["glycan_sequence"] = np.nan
        merged["composition"] = np.nan
        merge_mode = "task1_only_no_mapping"

    out = pd.DataFrame()
    out["feature_id"] = merged[id_col] if id_col is not None else np.nan
    out["biomarker_tier"] = merged[tier_col] if tier_col is not None else "missing_tier"
    out["redundancy_cluster"] = merged[cluster_col] if cluster_col is not None else np.nan
    out["clinical_profile"] = merged[profile_col] if profile_col is not None else np.nan
    out["glycan_sequence"] = merged["glycan_sequence"] if "glycan_sequence" in merged.columns else np.nan
    out["composition"] = merged["composition"] if "composition" in merged.columns else np.nan

    out = out.drop_duplicates().reset_index(drop=True)

    info = {
        "merge_mode": merge_mode,
        "n_task1_rows": len(t1),
        "n_glycan_list_rows": len(glycan_list),
        "n_handoff_rows": len(out),
        "n_mapped_rows": int(out["glycan_sequence"].notna().sum()),
        "mapping_rate": float(out["glycan_sequence"].notna().mean()) if len(out) > 0 else np.nan,
    }
    return out, info


# ---------------------------------------------------------------------
# Step 2 — Dataset audit
# ---------------------------------------------------------------------

def build_canonical_master_table(
    glycan_list: pd.DataFrame,
    df_glycan: pd.DataFrame,
    n_glycans_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build:
    - query_std   : discovered glycans / glycan_list
    - reference   : df_glycan + N-glycan controls
    - master      : canonical union table
    """
    query_std = standardize_glycan_table(
        glycan_list,
        source_name="glycan_list",
        require_sequence=True,
        mark_as_n_glycan=False,
    )

    ref_main = standardize_glycan_table(
        df_glycan,
        source_name="df_glycan",
        require_sequence=True,
        mark_as_n_glycan=False,
    )

    ref_ng = standardize_glycan_table(
        n_glycans_df,
        source_name="N_glycans_df",
        require_sequence=True,
        mark_as_n_glycan=True,
    )

    reference = pd.concat([ref_main, ref_ng], ignore_index=True)

    reference = (
        reference.groupby("glycan_sequence", as_index=False)
        .agg(
            composition=("composition", _first_not_null),
            tissue_sample=("tissue_sample", _first_not_null),
            tissue_species=("tissue_species", _first_not_null),
            disease=("disease", _first_not_null),
            is_n_glycan=("is_n_glycan", "max"),
            sources=("source_name", lambda s: "|".join(sorted(set(s.astype(str))))),
        )
        .reset_index(drop=True)
    )
    reference["in_reference"] = 1

    query_unique = (
        query_std.groupby("glycan_sequence", as_index=False)
        .agg(
            composition=("composition", _first_not_null),
            tissue_sample=("tissue_sample", _first_not_null),
            tissue_species=("tissue_species", _first_not_null),
            disease=("disease", _first_not_null),
        )
        .reset_index(drop=True)
    )
    query_unique["in_query"] = 1

    master = reference.merge(
        query_unique,
        on="glycan_sequence",
        how="outer",
        suffixes=("_ref", "_query"),
    )

    master["composition"] = master["composition_ref"].combine_first(master["composition_query"])
    master["tissue_sample"] = master["tissue_sample_ref"].combine_first(master["tissue_sample_query"])
    master["tissue_species"] = master["tissue_species_ref"].combine_first(master["tissue_species_query"])
    master["disease"] = master["disease_ref"].combine_first(master["disease_query"])
    master["is_n_glycan"] = master["is_n_glycan"].fillna(0).astype(int)
    master["in_reference"] = master["in_reference"].fillna(0).astype(int)
    master["in_query"] = master["in_query"].fillna(0).astype(int)

    keep_cols = [
        "glycan_sequence",
        "composition",
        "tissue_sample",
        "tissue_species",
        "disease",
        "is_n_glycan",
        "in_reference",
        "in_query",
        "sources",
    ]
    master = master[keep_cols].copy()
    master = master.sort_values(["in_query", "in_reference", "glycan_sequence"], ascending=[False, False, True])
    master = master.reset_index(drop=True)

    return query_std, reference, master


def audit_task2_sources(
    glycan_list: pd.DataFrame,
    df_glycan: pd.DataFrame,
    glycan_binding: pd.DataFrame,
    n_glycans_df: pd.DataFrame,
    task1_handoff: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Full Step 2 audit.

    Returns a dict of compact audit tables:
    - source_shapes
    - source_uniques
    - label_coverage
    - overlap_summary
    - binding_summary
    - discovered_match_summary
    - master_table
    """
    query_std, reference, master = build_canonical_master_table(
        glycan_list=glycan_list,
        df_glycan=df_glycan,
        n_glycans_df=n_glycans_df,
    )

    ref_main = standardize_glycan_table(df_glycan, source_name="df_glycan", require_sequence=True)
    ref_ng = standardize_glycan_table(n_glycans_df, source_name="N_glycans_df", require_sequence=True, mark_as_n_glycan=True)
    binding_long = standardize_binding_table(glycan_binding)

    query_set = set(query_std["glycan_sequence"].dropna().unique())
    ref_set = set(ref_main["glycan_sequence"].dropna().unique())
    ng_set = set(ref_ng["glycan_sequence"].dropna().unique())
    binding_set = set(binding_long["glycan_sequence"].dropna().unique())

    source_shapes = pd.DataFrame(
        [
            {"source": "glycan_list", "n_rows": len(glycan_list), "n_cols": glycan_list.shape[1]},
            {"source": "df_glycan", "n_rows": len(df_glycan), "n_cols": df_glycan.shape[1]},
            {"source": "glycan_binding", "n_rows": len(glycan_binding), "n_cols": glycan_binding.shape[1]},
            {"source": "N_glycans_df", "n_rows": len(n_glycans_df), "n_cols": n_glycans_df.shape[1]},
        ]
    )

    source_uniques = pd.DataFrame(
        [
            {"source": "glycan_list", "n_unique_glycans": len(query_set)},
            {"source": "df_glycan", "n_unique_glycans": len(ref_set)},
            {"source": "glycan_binding", "n_unique_glycans": len(binding_set)},
            {"source": "N_glycans_df", "n_unique_glycans": len(ng_set)},
            {"source": "reference_union", "n_unique_glycans": len(set(reference["glycan_sequence"]))},
            {"source": "master_union", "n_unique_glycans": len(set(master["glycan_sequence"]))},
        ]
    )

    label_coverage = pd.DataFrame(
        [
            {
                "source": "glycan_list",
                "fraction_with_composition": query_std["composition"].notna().mean(),
                "fraction_with_tissue_sample": query_std["tissue_sample"].notna().mean(),
                "fraction_with_tissue_species": query_std["tissue_species"].notna().mean(),
                "fraction_with_disease": query_std["disease"].notna().mean(),
            },
            {
                "source": "df_glycan",
                "fraction_with_composition": ref_main["composition"].notna().mean(),
                "fraction_with_tissue_sample": ref_main["tissue_sample"].notna().mean(),
                "fraction_with_tissue_species": ref_main["tissue_species"].notna().mean(),
                "fraction_with_disease": ref_main["disease"].notna().mean(),
            },
            {
                "source": "N_glycans_df",
                "fraction_with_composition": ref_ng["composition"].notna().mean(),
                "fraction_with_tissue_sample": ref_ng["tissue_sample"].notna().mean(),
                "fraction_with_tissue_species": ref_ng["tissue_species"].notna().mean(),
                "fraction_with_disease": ref_ng["disease"].notna().mean(),
            },
            {
                "source": "master_union",
                "fraction_with_composition": master["composition"].notna().mean(),
                "fraction_with_tissue_sample": master["tissue_sample"].notna().mean(),
                "fraction_with_tissue_species": master["tissue_species"].notna().mean(),
                "fraction_with_disease": master["disease"].notna().mean(),
            },
        ]
    )

    overlap_summary = pd.DataFrame(
        [
            {
                "comparison": "query ∩ df_glycan",
                "n_overlap": len(query_set & ref_set),
                "fraction_of_query": len(query_set & ref_set) / max(len(query_set), 1),
            },
            {
                "comparison": "query ∩ N_glycans_df",
                "n_overlap": len(query_set & ng_set),
                "fraction_of_query": len(query_set & ng_set) / max(len(query_set), 1),
            },
            {
                "comparison": "query ∩ glycan_binding",
                "n_overlap": len(query_set & binding_set),
                "fraction_of_query": len(query_set & binding_set) / max(len(query_set), 1),
            },
            {
                "comparison": "df_glycan ∩ glycan_binding",
                "n_overlap": len(ref_set & binding_set),
                "fraction_of_df_glycan": len(ref_set & binding_set) / max(len(ref_set), 1),
            },
            {
                "comparison": "N_glycans_df ∩ df_glycan",
                "n_overlap": len(ng_set & ref_set),
                "fraction_of_N_glycans_df": len(ng_set & ref_set) / max(len(ng_set), 1),
            },
        ]
    )

    binding_summary = pd.DataFrame(
        [
            {
                "metric": "n_binding_rows_long",
                "value": int(len(binding_long)),
            },
            {
                "metric": "n_unique_binding_glycans",
                "value": int(binding_long["glycan_sequence"].nunique()),
            },
            {
                "metric": "n_unique_binding_proteins",
                "value": int(binding_long["protein"].nunique()),
            },
            {
                "metric": "fraction_query_with_binding",
                "value": float(len(query_set & binding_set) / max(len(query_set), 1)),
            },
            {
                "metric": "fraction_reference_with_binding",
                "value": float(len(set(reference["glycan_sequence"]) & binding_set) / max(len(set(reference["glycan_sequence"])), 1)),
            },
        ]
    )

    if task1_handoff is not None:
        discovered_match_summary = pd.DataFrame(
            [
                {
                    "metric": "n_task1_handoff_rows",
                    "value": int(len(task1_handoff)),
                },
                {
                    "metric": "n_task1_rows_with_mapped_sequence",
                    "value": int(task1_handoff["glycan_sequence"].notna().sum()),
                },
                {
                    "metric": "fraction_task1_rows_with_mapped_sequence",
                    "value": float(task1_handoff["glycan_sequence"].notna().mean()),
                },
                {
                    "metric": "n_task1_rows_present_in_df_glycan",
                    "value": int(task1_handoff["glycan_sequence"].isin(ref_set).sum()),
                },
                {
                    "metric": "n_task1_rows_with_binding_support",
                    "value": int(task1_handoff["glycan_sequence"].isin(binding_set).sum()),
                },
            ]
        )
    else:
        discovered_match_summary = pd.DataFrame(
            [
                {"metric": "n_task1_handoff_rows", "value": 0},
                {"metric": "n_task1_rows_with_mapped_sequence", "value": 0},
                {"metric": "fraction_task1_rows_with_mapped_sequence", "value": np.nan},
                {"metric": "n_task1_rows_present_in_df_glycan", "value": np.nan},
                {"metric": "n_task1_rows_with_binding_support", "value": np.nan},
            ]
        )

    return {
        "source_shapes": source_shapes,
        "source_uniques": source_uniques,
        "label_coverage": label_coverage,
        "overlap_summary": overlap_summary,
        "binding_summary": binding_summary,
        "discovered_match_summary": discovered_match_summary,
        "query_standardized": query_std,
        "reference_standardized": reference,
        "binding_standardized": binding_long,
        "master_table": master,
    }


# ---------------------------------------------------------------------
# Simple plots for Steps 1–2
# ---------------------------------------------------------------------

def plot_task2_label_coverage(label_coverage: pd.DataFrame) -> plt.Figure:
    df = label_coverage.copy()
    long_df = df.melt(id_vars="source", var_name="label", value_name="fraction")

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    sns.barplot(data=long_df, x="source", y="fraction", hue="label", ax=ax)
    ax.set_title("Task 2 metadata coverage by source")
    ax.set_xlabel("Source")
    ax.set_ylabel("Fraction non-missing")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    return fig


def plot_task2_overlap_summary(overlap_summary: pd.DataFrame) -> plt.Figure:
    df = overlap_summary.copy()

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.barplot(data=df, x="comparison", y="fraction_of_query", ax=ax)
    ax.set_title("Key overlap rates for Task 2 sources")
    ax.set_xlabel("")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    return fig


def plot_task2_source_sizes(source_uniques: pd.DataFrame) -> plt.Figure:
    df = source_uniques.copy()

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.barplot(data=df, x="source", y="n_unique_glycans", ax=ax)
    ax.set_title("Unique glycan counts by source")
    ax.set_xlabel("")
    ax.set_ylabel("Number of unique glycans")
    ax.tick_params(axis="x", rotation=20)
    return fig
import ast
from collections import Counter

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------
# Step 3 — Canonical notation + canonical master table
# ---------------------------------------------------------------------

MONOSACCHARIDE_VOCAB = [
    "Neu5Ac", "Neu5Gc", "Kdn",
    "GalNAc", "GlcNAc", "ManNAc",
    "Gal", "Glc", "Man", "Fuc", "Xyl",
    "HexNAc", "Hex", "dHex",
]

COMPOSITION_KEYS_PRIORITY = [
    "Hex", "HexNAc", "dHex", "Neu5Ac", "Neu5Gc", "Kdn",
    "Fuc", "Xyl", "Gal", "Glc", "Man", "GalNAc", "GlcNAc",
]


def _safe_literal_eval(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return value
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == "":
        return np.nan
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def _normalize_collection_cell(value: Any) -> list[str]:
    """
    Convert list-like metadata cells into a clean sorted list of strings.
    Empty collections are treated as semantically missing.
    """
    value = _safe_literal_eval(value)

    # Handle list-like objects first
    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, pd.Series):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        cleaned = []
        for v in value:
            # scalar missing check only at element level
            if v is None:
                continue
            try:
                if pd.isna(v):
                    continue
            except Exception:
                pass

            text = str(v).strip()
            if text not in {"", "nan", "None", "NA", "[]", "{}"}:
                cleaned.append(text)

        return sorted(set(cleaned))

    # Scalar case
    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    text = str(value).strip()
    if text in {"", "[]", "{}", "nan", "None", "NA"}:
        return []

    return [text]


def _semantic_nonmissing_fraction(series: pd.Series) -> float:
    return float(series.apply(lambda x: len(_normalize_collection_cell(x)) > 0).mean())


def _join_collection(value: Any) -> str | float:
    vals = _normalize_collection_cell(value)
    if len(vals) == 0:
        return np.nan
    return " | ".join(vals)


def _parse_composition(value: Any) -> dict[str, int]:
    """
    Robust composition parser for dict-like strings or loose text.
    """
    value = _safe_literal_eval(value)

    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
        return out

    if pd.isna(value):
        return {}

    text = str(value)

    # patterns like Hex:5, 'Hex': 5, Hex 5
    matches = re.findall(r"([A-Za-z][A-Za-z0-9]*)\s*[:=]?\s*([0-9]+)", text)
    if not matches:
        return {}

    out = {}
    for key, count in matches:
        out[key] = out.get(key, 0) + int(count)
    return out


def canonicalize_sequence_exact(sequence: Any) -> str | float:
    """
    Conservative exact canonicalization:
    - strip spaces
    - normalize repeated whitespace
    - keep linkage information intact
    """
    if pd.isna(sequence):
        return np.nan
    seq = str(sequence).strip()
    seq = re.sub(r"\s+", "", seq)
    if seq == "":
        return np.nan
    return seq


def canonicalize_sequence_topology(sequence: Any) -> str | float:
    """
    Relaxed topology key:
    - remove linkage annotations in parentheses
    - keep branch structure
    - lowercase for robust matching
    This is NOT the primary scientific identifier; it is a secondary matching key.
    """
    seq = canonicalize_sequence_exact(sequence)
    if pd.isna(seq):
        return np.nan

    seq = re.sub(r"\([^)]+\)", "", seq)
    seq = seq.lower()
    seq = re.sub(r"[^a-z0-9\[\]]+", "", seq)
    if seq == "":
        return np.nan
    return seq


def infer_likely_n_glycan(sequence: Any) -> bool:
    """
    Simple structural heuristic for likely N-glycans based on the canonical core.
    """
    seq = canonicalize_sequence_exact(sequence)
    if pd.isna(seq):
        return False

    required_patterns = [
        "GlcNAc(b1-4)GlcNAc",
        "Man(b1-4)GlcNAc",
        "Man(a1-3)",
        "Man(a1-6)",
    ]
    return all(p in seq for p in required_patterns)


def extract_sequence_statistics(sequence: Any) -> dict[str, float]:
    seq = canonicalize_sequence_exact(sequence)
    if pd.isna(seq):
        return {
            "sequence_length_chars": np.nan,
            "branch_count": np.nan,
            "unknown_token_count": np.nan,
            "monosaccharide_token_count": np.nan,
            "fucose_token_count": np.nan,
            "sialic_token_count": np.nan,
        }

    mono_pattern = r"Neu5Ac|Neu5Gc|Kdn|GalNAc|GlcNAc|ManNAc|Gal|Glc|Man|Fuc|Xyl"
    mono_tokens = re.findall(mono_pattern, seq)

    return {
        "sequence_length_chars": float(len(seq)),
        "branch_count": float(seq.count("[")),
        "unknown_token_count": float(seq.count("?")),
        "monosaccharide_token_count": float(len(mono_tokens)),
        "fucose_token_count": float(sum(t == "Fuc" for t in mono_tokens)),
        "sialic_token_count": float(sum(t in {"Neu5Ac", "Neu5Gc", "Kdn"} for t in mono_tokens)),
    }


def build_step3_canonical_master(
    master_table: pd.DataFrame,
    binding_standardized: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the canonical glycan master table for downstream modeling.
    """
    master = master_table.copy()

    master["canonical_sequence"] = master["glycan_sequence"].apply(canonicalize_sequence_exact)
    master["topology_key"] = master["glycan_sequence"].apply(canonicalize_sequence_topology)

    master["composition_dict"] = master["composition"].apply(_parse_composition)
    master["tissue_sample_norm"] = master["tissue_sample"].apply(_join_collection)
    master["tissue_species_norm"] = master["tissue_species"].apply(_join_collection)
    master["disease_norm"] = master["disease"].apply(_join_collection)

    master["has_tissue_sample"] = master["tissue_sample"].apply(lambda x: len(_normalize_collection_cell(x)) > 0)
    master["has_tissue_species"] = master["tissue_species"].apply(lambda x: len(_normalize_collection_cell(x)) > 0)
    master["has_disease"] = master["disease"].apply(lambda x: len(_normalize_collection_cell(x)) > 0)

    master["likely_n_glycan"] = master["glycan_sequence"].apply(infer_likely_n_glycan).astype(int)

    stats_df = pd.DataFrame(
        master["glycan_sequence"].apply(extract_sequence_statistics).tolist(),
        index=master.index,
    )
    master = pd.concat([master, stats_df], axis=1)

    # composition-derived features
    for key in COMPOSITION_KEYS_PRIORITY:
        master[f"comp_{key}"] = master["composition_dict"].apply(lambda d: float(d.get(key, 0)))

    master["comp_total_residues"] = master["composition_dict"].apply(
        lambda d: float(sum(v for v in d.values()))
    )
    master["comp_fucose_total"] = master.apply(
        lambda r: float(r.get("comp_dHex", 0) + r.get("comp_Fuc", 0)),
        axis=1,
    )
    master["comp_sialic_total"] = master.apply(
        lambda r: float(r.get("comp_Neu5Ac", 0) + r.get("comp_Neu5Gc", 0) + r.get("comp_Kdn", 0)),
        axis=1,
    )
    master["comp_fucose_ratio"] = np.where(
        master["comp_total_residues"] > 0,
        master["comp_fucose_total"] / master["comp_total_residues"],
        0.0,
    )
    master["comp_sialic_ratio"] = np.where(
        master["comp_total_residues"] > 0,
        master["comp_sialic_total"] / master["comp_total_residues"],
        0.0,
    )
    master["comp_hexnac_to_hex_ratio"] = np.where(
        master["comp_Hex"] > 0,
        master["comp_HexNAc"] / master["comp_Hex"],
        0.0,
    )

    # binding support: exact and relaxed/topology
    binding = binding_standardized.copy()
    binding["canonical_sequence"] = binding["glycan_sequence"].apply(canonicalize_sequence_exact)
    binding["topology_key"] = binding["glycan_sequence"].apply(canonicalize_sequence_topology)

    exact_binding = (
        binding.groupby("canonical_sequence", dropna=False)
        .agg(
            n_binding_rows=("protein", "size"),
            n_binding_proteins=("protein", "nunique"),
            mean_binding_value=("binding_value", "mean"),
            max_binding_value=("binding_value", "max"),
        )
        .reset_index()
    )

    topology_binding = (
        binding.groupby("topology_key", dropna=False)
        .agg(
            n_binding_rows_topology=("protein", "size"),
            n_binding_proteins_topology=("protein", "nunique"),
        )
        .reset_index()
    )

    master = master.merge(
        exact_binding,
        on="canonical_sequence",
        how="left",
        validate="one_to_one",
    )
    master = master.merge(
        topology_binding,
        on="topology_key",
        how="left",
        validate="many_to_one",
    )

    for col in [
        "n_binding_rows",
        "n_binding_proteins",
        "mean_binding_value",
        "max_binding_value",
        "n_binding_rows_topology",
        "n_binding_proteins_topology",
    ]:
        if col in master.columns:
            master[col] = master[col].fillna(0)

    master["has_binding_exact"] = (master["n_binding_rows"] > 0).astype(int)
    master["has_binding_topology"] = (master["n_binding_rows_topology"] > 0).astype(int)

    # dataset membership
    master["discovered_flag"] = master["in_query"].astype(int)
    master["reference_flag"] = master["in_reference"].astype(int)

    master = master.sort_values(
        ["discovered_flag", "reference_flag", "canonical_sequence"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return master


def summarize_step3_master(canonical_master: pd.DataFrame) -> pd.DataFrame:
    """
    Compact Step 3 summary with semantically correct metadata coverage.
    """
    summary = pd.DataFrame(
        [
            {
                "metric": "n_unique_canonical_glycans",
                "value": int(canonical_master["canonical_sequence"].nunique()),
            },
            {
                "metric": "n_discovered_glycans",
                "value": int(canonical_master["discovered_flag"].sum()),
            },
            {
                "metric": "n_exact_binding_supported",
                "value": int(canonical_master["has_binding_exact"].sum()),
            },
            {
                "metric": "fraction_with_tissue_sample_semantic",
                "value": float(canonical_master["has_tissue_sample"].mean()),
            },
            {
                "metric": "fraction_with_tissue_species_semantic",
                "value": float(canonical_master["has_tissue_species"].mean()),
            },
            {
                "metric": "fraction_with_disease_semantic",
                "value": float(canonical_master["has_disease"].mean()),
            },
            {
                "metric": "fraction_likely_n_glycan",
                "value": float(canonical_master["likely_n_glycan"].mean()),
            },
        ]
    )
    return summary


def plot_step3_query_match_status(canonical_master: pd.DataFrame) -> plt.Figure:
    """
    Plot discovered glycan support under exact vs topology-based matching.
    """
    q = canonical_master.loc[canonical_master["discovered_flag"] == 1].copy()
    plot_df = pd.DataFrame(
        {
            "match_type": [
                "exact_reference_match",
                "topology_reference_match",
                "exact_binding_match",
                "topology_binding_match",
            ],
            "count": [
                int(q["reference_flag"].sum()),
                int(q["topology_key"].isin(
                    canonical_master.loc[canonical_master["reference_flag"] == 1, "topology_key"]
                ).sum()),
                int(q["has_binding_exact"].sum()),
                int(q["has_binding_topology"].sum()),
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    sns.barplot(data=plot_df, x="match_type", y="count", ax=ax)
    ax.set_title("Discovered glycan support after canonicalization")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    return fig


# ---------------------------------------------------------------------
# Step 4 — Interpretable baseline feature spaces
# ---------------------------------------------------------------------

def build_composition_baseline(canonical_master: pd.DataFrame) -> pd.DataFrame:
    """
    Composition-only interpretable baseline.
    """
    feature_cols = [
        "comp_Hex", "comp_HexNAc", "comp_dHex", "comp_Neu5Ac", "comp_Neu5Gc", "comp_Kdn",
        "comp_Fuc", "comp_Xyl", "comp_Gal", "comp_Glc", "comp_Man",
        "comp_total_residues", "comp_fucose_total", "comp_sialic_total",
        "comp_fucose_ratio", "comp_sialic_ratio", "comp_hexnac_to_hex_ratio",
        "branch_count", "unknown_token_count", "monosaccharide_token_count",
        "fucose_token_count", "sialic_token_count",
    ]
    out = canonical_master[["canonical_sequence", *feature_cols]].copy()
    out = out.set_index("canonical_sequence").fillna(0.0)
    return out.astype(float)


def _sequence_to_interpretable_tokens(sequence: str) -> list[str]:
    """
    Build interpretable sequence tokens:
    - residue tokens
    - linkage tokens
    - branch tokens
    """
    if pd.isna(sequence):
        return []

    seq = canonicalize_sequence_exact(sequence)
    tokens = []

    residues = re.findall(r"Neu5Ac|Neu5Gc|Kdn|GalNAc|GlcNAc|ManNAc|Gal|Glc|Man|Fuc|Xyl", seq)
    linkages = re.findall(r"\(([^)]+)\)", seq)

    tokens.extend([f"RES_{r}" for r in residues])
    tokens.extend([f"LNK_{l}" for l in linkages])

    tokens.extend(["BRANCH_OPEN"] * seq.count("["))
    tokens.extend(["BRANCH_CLOSE"] * seq.count("]"))

    # simple residue bigrams in sequence order
    for a, b in zip(residues[:-1], residues[1:]):
        tokens.append(f"PAIR_{a}__{b}")

    return tokens


def build_sequence_token_baseline(
    canonical_master: pd.DataFrame,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
) -> pd.DataFrame:
    """
    Sequence-token baseline analogous to interpretable n-gram features.
    """
    token_text = canonical_master["glycan_sequence"].apply(
        lambda s: " ".join(_sequence_to_interpretable_tokens(s))
    )

    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b\S+\b",
        lowercase=False,
        ngram_range=ngram_range,
        min_df=min_df,
    )
    X = vectorizer.fit_transform(token_text)

    cols = [f"tok__{c}" for c in vectorizer.get_feature_names_out()]
    out = pd.DataFrame.sparse.from_spmatrix(
        X,
        index=canonical_master["canonical_sequence"],
        columns=cols,
    )
    return out


def build_motif_baseline(
    canonical_master: pd.DataFrame,
    feature_set: list[str] | None = None,
) -> pd.DataFrame:
    """
    Motif baseline using glycowork annotation.

    Important:
    - only annotate rows that pass a conservative glycan-sequence validator
    - return a full matrix aligned to canonical_master index
    - invalid/unannotatable rows receive all-zero motif vectors
    """
    if feature_set is None:
        feature_set = ["known", "terminal", "exhaustive"]

    from glycowork.motif.annotate import annotate_dataset

    valid_df, rejected_df = sanitize_motif_input_sequences(canonical_master)

    if valid_df.empty:
        return pd.DataFrame(index=canonical_master["canonical_sequence"].tolist())

    seqs = valid_df["canonical_sequence"].tolist()

    try:
        motif_valid = annotate_dataset(
            seqs,
            feature_set=feature_set,
            condense=True,
        )
    except Exception as e:
        # Fallback: annotate one-by-one and keep only successful rows
        rows = []
        kept_index = []
        first_error = e

        for seq in seqs:
            try:
                tmp = annotate_dataset([seq], feature_set=feature_set, condense=True)
                rows.append(tmp.iloc[0])
                kept_index.append(seq)
            except Exception:
                continue

        if len(rows) == 0:
            raise ValueError(
                f"Motif annotation failed for all candidate sequences. First error was: {first_error}"
            )

        motif_valid = pd.DataFrame(rows, index=kept_index)

    motif_valid.index = valid_df["canonical_sequence"].tolist()[: len(motif_valid)]
    motif_valid = motif_valid.fillna(0)

    keep_cols = motif_valid.columns[(motif_valid != 0).any(axis=0)]
    motif_valid = motif_valid[keep_cols].astype(float)

    # reindex back to full canonical master and fill rejected rows with zeros
    motif_full = motif_valid.reindex(canonical_master["canonical_sequence"].tolist()).fillna(0.0)

    return motif_full


def build_step4_baseline_spaces(
    canonical_master: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Build the three baseline spaces requested in Step 4.
    """
    blocks = {
        "composition": build_composition_baseline(canonical_master),
        "sequence_token": build_sequence_token_baseline(canonical_master),
        "motif": build_motif_baseline(canonical_master),
    }
    return blocks


def _matrix_density(df: pd.DataFrame) -> float:
    if df.shape[0] == 0 or df.shape[1] == 0:
        return np.nan
    arr = df.sparse.to_coo() if hasattr(df, "sparse") else df.to_numpy()
    if hasattr(arr, "nnz"):
        return float(arr.nnz / (arr.shape[0] * arr.shape[1]))
    return float((arr != 0).sum() / arr.size)


def _row_nonzero_mean(df: pd.DataFrame) -> float:
    if df.shape[0] == 0 or df.shape[1] == 0:
        return np.nan
    if hasattr(df, "sparse"):
        counts = (df != 0).sum(axis=1)
        return float(np.asarray(counts).mean())
    return float((df.to_numpy() != 0).sum(axis=1).mean())


def _reduce_for_evaluation(df: pd.DataFrame, n_components: int = 20) -> np.ndarray:
    if hasattr(df, "sparse"):
        X = df.sparse.to_coo().tocsr()
    else:
        X = df.to_numpy(dtype=float)

    max_comp = min(n_components, X.shape[0] - 1, X.shape[1] - 1)
    if max_comp < 2:
        return X.toarray() if hasattr(X, "toarray") else X

    svd = TruncatedSVD(n_components=max_comp, random_state=42)
    return svd.fit_transform(X)


def _balanced_eval_sample(labels: np.ndarray, max_per_class: int = 1000, random_state: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_state)

    idx_pos = np.where(labels == 1)[0]
    idx_neg = np.where(labels == 0)[0]

    take_pos = idx_pos if len(idx_pos) <= max_per_class else rng.choice(idx_pos, size=max_per_class, replace=False)
    take_neg = idx_neg if len(idx_neg) <= max_per_class else rng.choice(idx_neg, size=max_per_class, replace=False)

    idx = np.concatenate([take_pos, take_neg])
    rng.shuffle(idx)
    return idx


def evaluate_step4_baselines(
    baseline_spaces: dict[str, pd.DataFrame],
    canonical_master: pd.DataFrame,
    k_neighbors: int = 5,
) -> pd.DataFrame:
    """
    Basic validation for Step 4:
    - N-glycan silhouette on balanced sample
    - N-glycan kNN purity
    This is not the final validation framework, but enough to compare baselines.
    """
    master = canonical_master.copy()
    eval_label = master["is_n_glycan"].astype(int).to_numpy()

    rows = []
    prevalence = float(eval_label.mean())

    interpretability_map = {
        "composition": "high",
        "sequence_token": "medium-high",
        "motif": "high",
    }

    for name, block in baseline_spaces.items():
        X_red = _reduce_for_evaluation(block)

        # silhouette on balanced sample
        sil = np.nan
        if eval_label.sum() >= 10 and (len(eval_label) - eval_label.sum()) >= 10:
            idx = _balanced_eval_sample(eval_label, max_per_class=1000, random_state=42)
            if len(np.unique(eval_label[idx])) == 2:
                sil = float(silhouette_score(X_red[idx], eval_label[idx]))

        # kNN purity among positive (N-glycan) class
        purity = np.nan
        pos_idx = np.where(eval_label == 1)[0]
        if len(pos_idx) > k_neighbors + 1:
            nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="cosine")
            nn.fit(X_red)
            neigh = nn.kneighbors(X_red[pos_idx], return_distance=False)
            neigh = neigh[:, 1:]  # drop self
            purity = float(eval_label[neigh].mean())

        rows.append(
            {
                "baseline_name": name,
                "n_rows": int(block.shape[0]),
                "n_features": int(block.shape[1]),
                "density": _matrix_density(block),
                "mean_active_features_per_glycan": _row_nonzero_mean(block),
                "interpretability": interpretability_map.get(name, "unknown"),
                "n_glycan_prevalence": prevalence,
                "n_glycan_silhouette": sil,
                "n_glycan_knn_purity_at5": purity,
                "purity_minus_prevalence": np.nan if pd.isna(purity) else purity - prevalence,
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["n_glycan_knn_purity_at5", "n_glycan_silhouette"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    return out


def plot_step4_baseline_comparison(baseline_comparison: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    sns.barplot(
        data=baseline_comparison,
        x="baseline_name",
        y="n_features",
        ax=axes[0],
    )
    axes[0].set_title("Baseline dimensionality")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Number of features")

    sns.barplot(
        data=baseline_comparison,
        x="baseline_name",
        y="n_glycan_knn_purity_at5",
        ax=axes[1],
    )
    axes[1].set_title("Baseline N-glycan kNN purity @5")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Purity")

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)

    return fig


def build_step4_modeling_subset(
    canonical_master: pd.DataFrame,
    n_reference_sample: int = 3000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build a manageable subset for Step 4 baseline benchmarking.

    Keep:
    - all discovered glycans
    - all likely / labeled N-glycan controls
    - a random reference background sample

    This keeps the benchmark computationally tractable while preserving
    the main biological anchors needed for early representation testing.
    """
    master = canonical_master.copy()

    discovered = master.loc[master["discovered_flag"] == 1].copy()

    n_glycan_controls = master.loc[
        (master["is_n_glycan"] == 1) | (master["likely_n_glycan"] == 1)
    ].copy()

    background_pool = master.loc[
        ~master["canonical_sequence"].isin(
            pd.concat([discovered["canonical_sequence"], n_glycan_controls["canonical_sequence"]]).drop_duplicates()
        )
    ].copy()

    if len(background_pool) > n_reference_sample:
        background_pool = background_pool.sample(
            n=n_reference_sample,
            random_state=random_state,
            replace=False,
        )

    subset = pd.concat(
        [discovered, n_glycan_controls, background_pool],
        ignore_index=True,
    ).drop_duplicates(subset=["canonical_sequence"])

    subset = subset.sort_values(
        ["discovered_flag", "is_n_glycan", "likely_n_glycan", "canonical_sequence"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    return subset


def summarize_step4_modeling_subset(step4_subset: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "n_total", "value": int(len(step4_subset))},
            {"metric": "n_discovered", "value": int(step4_subset["discovered_flag"].sum())},
            {"metric": "n_is_n_glycan", "value": int(step4_subset["is_n_glycan"].sum())},
            {"metric": "n_likely_n_glycan", "value": int(step4_subset["likely_n_glycan"].sum())},
            {"metric": "n_exact_binding_supported", "value": int(step4_subset["has_binding_exact"].sum())},
        ]
    )


def is_plausible_glycan_sequence(sequence: Any) -> bool:
    """
    Conservative validator for whether a string plausibly looks like a glycan sequence.

    Accepts common glycan notations and rejects obvious metadata/text contamination.
    """
    if pd.isna(sequence):
        return False

    seq = str(sequence).strip()
    if seq == "":
        return False

    # obvious metadata contamination
    forbidden_signals = [
        "Homo_sapiens",
        "Mus_musculus",
        "Bos_taurus",
        "Arabidopsis",
        "virus",
        "cell_line",
        "blood",
        "plasma",
        "mucosa",
        "seminal_fluid",
    ]
    lowered = seq.lower()
    if any(tok.lower() in lowered for tok in forbidden_signals):
        return False

    # metadata-like list explosions
    if seq.startswith("[") and "][" in seq and "GlcNAc" not in seq and "Gal" not in seq and "Man" not in seq:
        return False

    # supported / plausible glycan notation indicators
    glycan_signals = [
        "GlcNAc", "GalNAc", "Neu5Ac", "Neu5Gc", "Kdn",
        "Gal", "Glc", "Man", "Fuc", "Xyl",
        "WURCS=", "RES", ";N",
    ]
    if any(sig in seq for sig in glycan_signals):
        return True

    # also allow compact topology strings with glycan-like bracket/linkage syntax
    if ("(" in seq and ")" in seq) or ("[" in seq and "]" in seq):
        return True

    return False


def sanitize_motif_input_sequences(
    canonical_master: pd.DataFrame,
    sequence_col: str = "canonical_sequence",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only rows with plausible glycan sequences for motif annotation.
    Returns:
    - valid rows
    - rejected rows with reason flag
    """
    df = canonical_master.copy()

    df["motif_input_valid"] = df[sequence_col].apply(is_plausible_glycan_sequence)
    rejected = df.loc[~df["motif_input_valid"]].copy()
    valid = df.loc[df["motif_input_valid"]].copy()

    return valid, rejected


def summarize_motif_input_qc(
    canonical_master: pd.DataFrame,
    valid_for_motif: pd.DataFrame,
    rejected_for_motif: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "n_total_rows", "value": int(len(canonical_master))},
            {"metric": "n_valid_for_motif", "value": int(len(valid_for_motif))},
            {"metric": "n_rejected_for_motif", "value": int(len(rejected_for_motif))},
            {"metric": "fraction_valid_for_motif", "value": float(len(valid_for_motif) / max(len(canonical_master), 1))},
            {"metric": "n_discovered_valid_for_motif", "value": int(valid_for_motif["discovered_flag"].sum()) if "discovered_flag" in valid_for_motif.columns else np.nan},
        ]
    )

def preview_rejected_motif_sequences(
    canonical_master: pd.DataFrame,
    n: int = 20,
) -> pd.DataFrame:
    valid_df, rejected_df = sanitize_motif_input_sequences(canonical_master)
    cols = [c for c in ["canonical_sequence", "glycan_sequence", "tissue_species_norm", "sources"] if c in rejected_df.columns]
    return rejected_df[cols].head(n)


from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------
# Step 5 — Validation tasks
# ---------------------------------------------------------------------

def _split_pipe_labels(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "na"}:
        return []
    return [x.strip() for x in text.split(" | ") if x.strip()]


def summarize_step5_validation_feasibility(
    canonical_master: pd.DataFrame,
    binding_standardized: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = [
        {
            "task": "N-glycan coherence",
            "n_eligible": int(canonical_master["is_n_glycan"].sum()),
            "fraction_eligible": float(canonical_master["is_n_glycan"].mean()),
            "status": "primary",
        },
        {
            "task": "tissue-species neighborhood preservation",
            "n_eligible": int(canonical_master["has_tissue_species"].sum()),
            "fraction_eligible": float(canonical_master["has_tissue_species"].mean()),
            "status": "secondary",
        },
        {
            "task": "tissue-sample neighborhood preservation",
            "n_eligible": int(canonical_master["has_tissue_sample"].sum()),
            "fraction_eligible": float(canonical_master["has_tissue_sample"].mean()),
            "status": "secondary",
        },
        {
            "task": "disease neighborhood preservation",
            "n_eligible": int(canonical_master["has_disease"].sum()),
            "fraction_eligible": float(canonical_master["has_disease"].mean()),
            "status": "weak_secondary",
        },
        {
            "task": "discovered-glycan retrieval",
            "n_eligible": int(canonical_master["discovered_flag"].sum()),
            "fraction_eligible": float(canonical_master["discovered_flag"].mean()),
            "status": "primary",
        },
    ]

    if binding_standardized is not None:
        out.append(
            {
                "task": "binding relevance",
                "n_eligible": int(canonical_master["has_binding_exact"].sum()),
                "fraction_eligible": float(canonical_master["has_binding_exact"].mean()),
                "status": "secondary",
            }
        )

    return pd.DataFrame(out)


def _compute_neighbor_indices(
    embedding_df: pd.DataFrame,
    k: int = 5,
) -> np.ndarray:
    X = embedding_df.to_numpy(dtype=float)
    n_neighbors = min(k + 1, len(embedding_df))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(X)
    neigh = nn.kneighbors(X, return_distance=False)
    if neigh.shape[1] > 1:
        neigh = neigh[:, 1:]
    return neigh


def _label_neighbor_purity(
    embedding_df: pd.DataFrame,
    canonical_master: pd.DataFrame,
    label_col: str,
    k: int = 5,
) -> tuple[float, int]:
    neigh = _compute_neighbor_indices(embedding_df, k=k)
    labels = canonical_master[label_col].apply(_split_pipe_labels)

    scores = []
    n_used = 0
    for i in range(len(canonical_master)):
        anchor_labels = set(labels.iloc[i])
        if len(anchor_labels) == 0:
            continue

        neighbor_hits = []
        for j in neigh[i]:
            other_labels = set(labels.iloc[j])
            neighbor_hits.append(int(len(anchor_labels & other_labels) > 0))

        if len(neighbor_hits) == 0:
            continue

        scores.append(np.mean(neighbor_hits))
        n_used += 1

    return (float(np.mean(scores)) if len(scores) > 0 else np.nan, n_used)


def _binding_jaccard_relevance(
    embedding_df: pd.DataFrame,
    binding_block: pd.DataFrame,
    k: int = 5,
) -> tuple[float, int]:
    neigh = _compute_neighbor_indices(embedding_df, k=k)
    binary = (binding_block.abs() > 0).astype(int)

    scores = []
    n_used = 0
    arr = binary.to_numpy(dtype=bool)

    for i in range(arr.shape[0]):
        a = arr[i]
        if a.sum() == 0:
            continue

        row_scores = []
        for j in neigh[i]:
            b = arr[j]
            union = np.logical_or(a, b).sum()
            if union == 0:
                continue
            inter = np.logical_and(a, b).sum()
            row_scores.append(inter / union)

        if len(row_scores) == 0:
            continue

        scores.append(np.mean(row_scores))
        n_used += 1

    return (float(np.mean(scores)) if len(scores) > 0 else np.nan, n_used)


def evaluate_embedding_validation_tasks(
    embedding_df: pd.DataFrame,
    canonical_master: pd.DataFrame,
    binding_block: pd.DataFrame | None = None,
    k: int = 5,
) -> pd.Series:
    eval_label = canonical_master["is_n_glycan"].astype(int).to_numpy()

    n_glycan_sil = np.nan
    if eval_label.sum() >= 10 and (len(eval_label) - eval_label.sum()) >= 10:
        idx = _balanced_eval_sample(eval_label, max_per_class=1000, random_state=42)
        if len(np.unique(eval_label[idx])) == 2:
            n_glycan_sil = float(
                silhouette_score(embedding_df.to_numpy(dtype=float)[idx], eval_label[idx])
            )

    n_glycan_purity = np.nan
    pos_idx = np.where(eval_label == 1)[0]
    if len(pos_idx) > k + 1:
        neigh = _compute_neighbor_indices(embedding_df, k=k)
        n_glycan_purity = float(eval_label[neigh[pos_idx]].mean())

    species_purity, species_n = _label_neighbor_purity(
        embedding_df, canonical_master, "tissue_species_norm", k=k
    )
    sample_purity, sample_n = _label_neighbor_purity(
        embedding_df, canonical_master, "tissue_sample_norm", k=k
    )
    disease_purity, disease_n = _label_neighbor_purity(
        embedding_df, canonical_master, "disease_norm", k=k
    )

    if binding_block is not None and binding_block.shape[1] > 0:
        binding_relevance, binding_n = _binding_jaccard_relevance(
            embedding_df, binding_block, k=k
        )
    else:
        binding_relevance, binding_n = (np.nan, 0)

    return pd.Series(
        {
            "n_glycan_silhouette": n_glycan_sil,
            "n_glycan_knn_purity_at5": n_glycan_purity,
            "species_neighbor_purity_at5": species_purity,
            "species_labeled_anchors": species_n,
            "sample_neighbor_purity_at5": sample_purity,
            "sample_labeled_anchors": sample_n,
            "disease_neighbor_purity_at5": disease_purity,
            "disease_labeled_anchors": disease_n,
            "binding_jaccard_at5": binding_relevance,
            "binding_labeled_anchors": binding_n,
        }
    )


# ---------------------------------------------------------------------
# Step 6–7 — First-generation linear embeddings + late fusion
# ---------------------------------------------------------------------

def build_metadata_block(
    canonical_master: pd.DataFrame,
    min_count: int = 25,
) -> pd.DataFrame:
    rows = []
    for _, row in canonical_master.iterrows():
        entry = {"canonical_sequence": row["canonical_sequence"]}
        for col in ["tissue_sample_norm", "tissue_species_norm", "disease_norm"]:
            labels = _split_pipe_labels(row[col])
            for label in labels:
                entry[f"{col}::{label}"] = 1.0
        rows.append(entry)

    meta = pd.DataFrame(rows).set_index("canonical_sequence").fillna(0.0)
    if meta.shape[1] == 0:
        return meta

    keep_cols = []
    for col in meta.columns:
        if meta[col].sum() >= min_count:
            keep_cols.append(col)

    return meta[keep_cols].astype(float)


def build_binding_block(
    canonical_master: pd.DataFrame,
    binding_standardized: pd.DataFrame,
    min_protein_count: int = 20,
    top_n_proteins: int = 256,
) -> pd.DataFrame:
    binding = binding_standardized.copy()
    binding["canonical_sequence"] = binding["glycan_sequence"].apply(canonicalize_sequence_exact)

    keep_sequences = set(canonical_master["canonical_sequence"])
    binding = binding.loc[binding["canonical_sequence"].isin(keep_sequences)].copy()

    if binding.empty:
        return pd.DataFrame(index=canonical_master["canonical_sequence"].tolist())

    protein_counts = (
        binding.groupby("protein")["canonical_sequence"]
        .nunique()
        .sort_values(ascending=False)
    )
    keep_proteins = protein_counts.loc[protein_counts >= min_protein_count].head(top_n_proteins).index.tolist()

    binding = binding.loc[binding["protein"].isin(keep_proteins)].copy()
    if binding.empty:
        return pd.DataFrame(index=canonical_master["canonical_sequence"].tolist())

    block = (
        binding.groupby(["canonical_sequence", "protein"])["binding_value"]
        .mean()
        .unstack(fill_value=0.0)
    )
    block = block.reindex(canonical_master["canonical_sequence"].tolist()).fillna(0.0)
    return block.astype(float)


def _is_sparse_df(df: pd.DataFrame) -> bool:
    return any(isinstance(dtype, pd.SparseDtype) for dtype in df.dtypes)


def reduce_feature_block(
    block: pd.DataFrame,
    prefix: str,
    n_components: int,
) -> pd.DataFrame:
    if block.shape[1] == 0:
        return pd.DataFrame(index=block.index)

    if _is_sparse_df(block):
        X = block.sparse.to_coo().tocsr()
    else:
        X = sparse.csr_matrix(block.to_numpy(dtype=float))

    X = MaxAbsScaler().fit_transform(X)

    max_comp = min(n_components, X.shape[0] - 1, X.shape[1] - 1)
    if max_comp < 2:
        arr = X.toarray()
        cols = [f"{prefix}_00"] if arr.shape[1] == 1 else [f"{prefix}_{i:02d}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, index=block.index, columns=cols)

    svd = TruncatedSVD(n_components=max_comp, random_state=42)
    arr = svd.fit_transform(X)
    cols = [f"{prefix}_{i:02d}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=block.index, columns=cols)


def _concat_blocks(blocks: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [b for b in blocks if b is not None and b.shape[1] > 0]
    if len(non_empty) == 0:
        return pd.DataFrame()
    out = pd.concat(non_empty, axis=1)
    return out.loc[:, ~out.columns.duplicated()].copy()


def build_step6_embedding_suite(
    canonical_master: pd.DataFrame,
    binding_standardized: pd.DataFrame,
    metadata_min_count: int = 25,
    binding_min_protein_count: int = 20,
    binding_top_n_proteins: int = 256,
    comp_n_components: int = 12,
    seq_n_components: int = 32,
    meta_n_components: int = 16,
    bind_n_components: int = 16,
) -> dict[str, Any]:
    raw_blocks = {
        "composition": build_composition_baseline(canonical_master),
        "sequence_token": build_sequence_token_baseline(canonical_master),
        "metadata": build_metadata_block(canonical_master, min_count=metadata_min_count),
        "binding": build_binding_block(
            canonical_master,
            binding_standardized,
            min_protein_count=binding_min_protein_count,
            top_n_proteins=binding_top_n_proteins,
        ),
    }

    reduced_blocks = {
        "composition": reduce_feature_block(raw_blocks["composition"], "comp", comp_n_components),
        "sequence_token": reduce_feature_block(raw_blocks["sequence_token"], "seq", seq_n_components),
        "metadata": reduce_feature_block(raw_blocks["metadata"], "meta", meta_n_components),
        "binding": reduce_feature_block(raw_blocks["binding"], "bind", bind_n_components),
    }

    candidate_embeddings = {
        "composition_only": reduced_blocks["composition"],
        "sequence_token_only": reduced_blocks["sequence_token"],
        "structure": _concat_blocks([reduced_blocks["composition"], reduced_blocks["sequence_token"]]),
        "structure_metadata": _concat_blocks([
            reduced_blocks["composition"],
            reduced_blocks["sequence_token"],
            reduced_blocks["metadata"],
        ]),
        "structure_metadata_binding": _concat_blocks([
            reduced_blocks["composition"],
            reduced_blocks["sequence_token"],
            reduced_blocks["metadata"],
            reduced_blocks["binding"],
        ]),
    }

    return {
        "raw_blocks": raw_blocks,
        "reduced_blocks": reduced_blocks,
        "candidate_embeddings": candidate_embeddings,
    }


def evaluate_embedding_suite(
    embedding_suite: dict[str, Any],
    canonical_master: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    binding_block = embedding_suite["raw_blocks"]["binding"]

    rows = []
    for name, emb in embedding_suite["candidate_embeddings"].items():
        metrics = evaluate_embedding_validation_tasks(
            embedding_df=emb,
            canonical_master=canonical_master,
            binding_block=binding_block,
            k=k,
        )
        row = {"embedding_name": name, "n_dimensions": emb.shape[1]}
        row.update(metrics.to_dict())
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["n_glycan_knn_purity_at5", "species_neighbor_purity_at5", "binding_jaccard_at5"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def build_linear_embedding_layout(
    embedding_df: pd.DataFrame,
    canonical_master: pd.DataFrame,
    embedding_name: str,
) -> pd.DataFrame:
    X = embedding_df.to_numpy(dtype=float)
    if X.shape[1] < 2:
        coords = np.c_[X[:, 0], np.zeros(X.shape[0])]
        evr = [np.nan, np.nan]
    else:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X)
        evr = pca.explained_variance_ratio_

    out = canonical_master[
        [
            "canonical_sequence",
            "discovered_flag",
            "is_n_glycan",
            "has_binding_exact",
            "tissue_species_norm",
            "disease_norm",
        ]
    ].copy()
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    out["embedding_name"] = embedding_name
    out["pc1_var"] = evr[0]
    out["pc2_var"] = evr[1]
    return out


def plot_embedding_layout(
    layout_df: pd.DataFrame,
    color_by: str = "is_n_glycan",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    sns.scatterplot(
        data=layout_df,
        x="x",
        y="y",
        hue=color_by,
        size="discovered_flag",
        sizes=(30, 120),
        alpha=0.7,
        ax=ax,
    )

    discovered = layout_df.loc[layout_df["discovered_flag"] == 1]
    ax.scatter(discovered["x"], discovered["y"], s=140, facecolors="none", edgecolors="black", linewidths=1.2)

    ax.set_title(
        f"{layout_df['embedding_name'].iloc[0]} — linear 2D layout "
        f"(PC1 {100*layout_df['pc1_var'].iloc[0]:.1f}%, PC2 {100*layout_df['pc2_var'].iloc[0]:.1f}%)"
    )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    return fig


def plot_embedding_suite_comparison(
    embedding_eval: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    sns.barplot(data=embedding_eval, x="embedding_name", y="n_glycan_knn_purity_at5", ax=axes[0])
    axes[0].set_title("N-glycan kNN purity @5")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=embedding_eval, x="embedding_name", y="species_neighbor_purity_at5", ax=axes[1])
    axes[1].set_title("Species neighbor purity @5")
    axes[1].tick_params(axis="x", rotation=20)

    sns.barplot(data=embedding_eval, x="embedding_name", y="binding_jaccard_at5", ax=axes[2])
    axes[2].set_title("Binding relevance @5")
    axes[2].tick_params(axis="x", rotation=20)

    return fig


def retrieve_discovered_glycan_neighbors(
    embedding_df: pd.DataFrame,
    canonical_master: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    neigh = _compute_neighbor_indices(embedding_df, k=k)

    rows = []
    discovered_idx = canonical_master.index[canonical_master["discovered_flag"] == 1].tolist()

    for i in discovered_idx:
        anchor = canonical_master.iloc[i]
        for rank, j in enumerate(neigh[i], start=1):
            nbr = canonical_master.iloc[j]
            rows.append(
                {
                    "anchor_sequence": anchor["canonical_sequence"],
                    "neighbor_rank": rank,
                    "neighbor_sequence": nbr["canonical_sequence"],
                    "neighbor_is_n_glycan": nbr["is_n_glycan"],
                    "neighbor_has_binding_exact": nbr["has_binding_exact"],
                    "neighbor_tissue_species": nbr["tissue_species_norm"],
                    "neighbor_disease": nbr["disease_norm"],
                    "neighbor_reference_flag": nbr["reference_flag"],
                    "neighbor_discovered_flag": nbr["discovered_flag"],
                }
            )

    return pd.DataFrame(rows)


from collections import Counter


# ---------------------------------------------------------------------
# Step 9 — Final interpretation set + neighborhood engine
# ---------------------------------------------------------------------

def build_final_interpretation_set(
    canonical_master: pd.DataFrame,
    require_reference_or_discovered: bool = True,
    require_plausible_sequence: bool = True,
) -> pd.DataFrame:
    """
    Final cleaned universe used for downstream interpretation.

    Keeps only rows that are plausible glycan sequences and, by default,
    are either discovered glycans or reference-supported glycans.
    """
    df = canonical_master.copy()

    if require_plausible_sequence:
        df = df.loc[df["canonical_sequence"].apply(is_plausible_glycan_sequence)].copy()

    if require_reference_or_discovered:
        df = df.loc[(df["reference_flag"] == 1) | (df["discovered_flag"] == 1)].copy()

    df = df.drop_duplicates(subset=["canonical_sequence"]).reset_index(drop=True)
    return df


def summarize_final_interpretation_set(final_set: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "n_total", "value": int(len(final_set))},
            {"metric": "n_discovered", "value": int(final_set["discovered_flag"].sum())},
            {"metric": "n_reference", "value": int(final_set["reference_flag"].sum())},
            {"metric": "n_n_glycans", "value": int(final_set["is_n_glycan"].sum())},
            {"metric": "n_likely_n_glycans", "value": int(final_set["likely_n_glycan"].sum())},
            {"metric": "n_binding_exact", "value": int(final_set["has_binding_exact"].sum())},
            {"metric": "n_species_labeled", "value": int(final_set["has_tissue_species"].sum())},
            {"metric": "n_sample_labeled", "value": int(final_set["has_tissue_sample"].sum())},
            {"metric": "n_disease_labeled", "value": int(final_set["has_disease"].sum())},
        ]
    )


def filter_embedding_suite_to_universe(
    embedding_suite: dict[str, Any],
    interpretation_set: pd.DataFrame,
) -> dict[str, Any]:
    keep_index = interpretation_set["canonical_sequence"].tolist()

    out = {
        "raw_blocks": {},
        "reduced_blocks": {},
        "candidate_embeddings": {},
    }

    for section in ["raw_blocks", "reduced_blocks", "candidate_embeddings"]:
        for name, df in embedding_suite[section].items():
            out[section][name] = df.reindex(keep_index).fillna(0.0)

    return out


def choose_final_embedding_name(
    embedding_eval: pd.DataFrame,
    preferred: str = "structure_metadata_binding",
) -> str:
    if preferred in embedding_eval["embedding_name"].tolist():
        return preferred
    return embedding_eval.iloc[0]["embedding_name"]


def _neighbor_distance_table(
    embedding_df: pd.DataFrame,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    X = embedding_df.to_numpy(dtype=float)
    n_neighbors = min(k + 1, len(embedding_df))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)
    if indices.shape[1] > 1:
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    return distances, indices


def _top_terms_from_pipe_column(
    values: pd.Series,
    top_n: int = 10,
) -> list[tuple[str, int]]:
    counter = Counter()
    for value in values:
        for item in _split_pipe_labels(value):
            counter[item] += 1
    return counter.most_common(top_n)


def _top_composition_features(
    neighbor_df: pd.DataFrame,
    top_n: int = 8,
) -> list[tuple[str, float]]:
    comp_cols = [c for c in neighbor_df.columns if c.startswith("comp_")]
    keep = [
        c for c in comp_cols
        if c not in {
            "comp_fucose_ratio",
            "comp_sialic_ratio",
            "comp_hexnac_to_hex_ratio",
        }
    ]
    means = neighbor_df[keep].mean().sort_values(ascending=False)
    means = means.loc[means > 0]
    return [(idx, float(val)) for idx, val in means.head(top_n).items()]


def summarize_discovered_glycan_neighborhoods(
    embedding_df: pd.DataFrame,
    canonical_master: pd.DataFrame,
    binding_block: pd.DataFrame | None = None,
    k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
    - summary table: one row per discovered glycan
    - neighbor table: one row per neighbor relation
    """
    distances, indices = _neighbor_distance_table(embedding_df, k=k)

    rows_summary = []
    rows_neighbors = []

    discovered_idx = canonical_master.index[canonical_master["discovered_flag"] == 1].tolist()

    for i in discovered_idx:
        anchor = canonical_master.iloc[i]
        neighbor_idx = indices[i]
        neighbor_dist = distances[i]
        nbr_df = canonical_master.iloc[neighbor_idx].copy()

        # detailed neighbor rows
        for rank, (j, dist) in enumerate(zip(neighbor_idx, neighbor_dist), start=1):
            nbr = canonical_master.iloc[j]
            rows_neighbors.append(
                {
                    "anchor_sequence": anchor["canonical_sequence"],
                    "neighbor_rank": rank,
                    "neighbor_sequence": nbr["canonical_sequence"],
                    "cosine_distance": float(dist),
                    "neighbor_is_n_glycan": int(nbr["is_n_glycan"]),
                    "neighbor_likely_n_glycan": int(nbr["likely_n_glycan"]),
                    "neighbor_has_binding_exact": int(nbr["has_binding_exact"]),
                    "neighbor_tissue_species": nbr["tissue_species_norm"],
                    "neighbor_tissue_sample": nbr["tissue_sample_norm"],
                    "neighbor_disease": nbr["disease_norm"],
                    "neighbor_reference_flag": int(nbr["reference_flag"]),
                    "neighbor_discovered_flag": int(nbr["discovered_flag"]),
                }
            )

        # binding overlap summary if available
        binding_overlap_mean = np.nan
        if binding_block is not None and binding_block.shape[1] > 0:
            arr = (binding_block.abs() > 0).astype(int).to_numpy(dtype=bool)
            a = arr[i]
            local_scores = []
            if a.sum() > 0:
                for j in neighbor_idx:
                    b = arr[j]
                    union = np.logical_or(a, b).sum()
                    if union == 0:
                        continue
                    inter = np.logical_and(a, b).sum()
                    local_scores.append(inter / union)
            if len(local_scores) > 0:
                binding_overlap_mean = float(np.mean(local_scores))

        rows_summary.append(
            {
                "anchor_sequence": anchor["canonical_sequence"],
                "anchor_is_n_glycan": int(anchor["is_n_glycan"]),
                "anchor_likely_n_glycan": int(anchor["likely_n_glycan"]),
                "anchor_has_binding_exact": int(anchor["has_binding_exact"]),
                "mean_cosine_distance_top10": float(np.mean(neighbor_dist)),
                "max_cosine_distance_top10": float(np.max(neighbor_dist)),
                "neighbor_n_glycan_fraction": float(nbr_df["is_n_glycan"].mean()),
                "neighbor_likely_n_glycan_fraction": float(nbr_df["likely_n_glycan"].mean()),
                "neighbor_binding_fraction": float(nbr_df["has_binding_exact"].mean()),
                "top_species": _top_terms_from_pipe_column(nbr_df["tissue_species_norm"], top_n=5),
                "top_samples": _top_terms_from_pipe_column(nbr_df["tissue_sample_norm"], top_n=5),
                "top_diseases": _top_terms_from_pipe_column(nbr_df["disease_norm"], top_n=5),
                "top_composition_features": _top_composition_features(nbr_df, top_n=8),
                "binding_overlap_mean": binding_overlap_mean,
            }
        )

    return pd.DataFrame(rows_summary), pd.DataFrame(rows_neighbors)


def plot_discovered_neighbor_distances(
    neighborhood_summary: pd.DataFrame,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    sns.barplot(
        data=neighborhood_summary,
        x="anchor_sequence",
        y="mean_cosine_distance_top10",
        ax=ax,
    )
    ax.set_title("Discovered glycans — mean cosine distance to top-10 neighbors")
    ax.set_xlabel("")
    ax.set_ylabel("Mean cosine distance")
    ax.tick_params(axis="x", rotation=35)
    return fig


def plot_discovered_neighbor_support(
    neighborhood_summary: pd.DataFrame,
) -> plt.Figure:
    plot_df = neighborhood_summary.copy()
    plot_df = plot_df.melt(
        id_vars="anchor_sequence",
        value_vars=[
            "neighbor_n_glycan_fraction",
            "neighbor_likely_n_glycan_fraction",
            "neighbor_binding_fraction",
        ],
        var_name="metric",
        value_name="value",
    )

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sns.barplot(data=plot_df, x="anchor_sequence", y="value", hue="metric", ax=ax)
    ax.set_title("Discovered glycans — local neighborhood support")
    ax.set_xlabel("")
    ax.set_ylabel("Fraction among top-10 neighbors")
    ax.tick_params(axis="x", rotation=35)
    return fig


def build_neighbor_context_table(
    neighborhood_summary: pd.DataFrame,
) -> pd.DataFrame:
    out = neighborhood_summary.copy()

    for col in ["top_species", "top_samples", "top_diseases", "top_composition_features"]:
        out[col] = out[col].apply(
            lambda items: "; ".join(
                [f"{k} ({v:.2f})" if isinstance(v, float) else f"{k} ({v})" for k, v in items]
            ) if isinstance(items, list) else ""
        )

    return out

import math
import textwrap


# ---------------------------------------------------------------------
# Step 10–14 — Family context, protein context, cancer layer, ranking,
# final deliverables
# ---------------------------------------------------------------------

CANCER_KEYWORDS = [
    "cancer",
    "carcinoma",
    "adenocarcinoma",
    "tumor",
    "tumour",
    "leukemia",
    "leukaemia",
    "lymphoma",
    "myeloma",
    "sarcoma",
    "cholangiocarcinoma",
    "melanoma",
    "glioma",
]


def shorten_glycan_label(sequence: Any, max_len: int = 55) -> str:
    if pd.isna(sequence):
        return "NA"
    seq = str(sequence)
    if len(seq) <= max_len:
        return seq
    return seq[: max_len - 3] + "..."


def _contains_cancer_keyword(text: Any) -> bool:
    if pd.isna(text):
        return False
    low = str(text).lower()
    return any(keyword in low for keyword in CANCER_KEYWORDS)


def classify_structural_family(sequence: Any) -> list[str]:
    seq = canonicalize_sequence_exact(sequence)
    if pd.isna(seq):
        return []

    labels = []

    if infer_likely_n_glycan(seq):
        labels.append("N-glycan-like")
    if "Neu5Ac" in seq or "Neu5Gc" in seq or "Kdn" in seq:
        labels.append("sialylated")
    if "Fuc" in seq or "dHex" in seq:
        labels.append("fucosylated")
    if "GalNAc" in seq:
        labels.append("GalNAc-bearing")
    if "?" in seq:
        labels.append("ambiguous-linkage")
    if seq.count("[") >= 1:
        labels.append("branched")
    if "GlcNAc(b1-4)GlcNAc" in seq:
        labels.append("core-GlcNAc")
    return labels


def add_structural_family_labels(canonical_master: pd.DataFrame) -> pd.DataFrame:
    df = canonical_master.copy()
    df["family_labels"] = df["canonical_sequence"].apply(classify_structural_family)
    df["family_labels_str"] = df["family_labels"].apply(
        lambda vals: " | ".join(vals) if len(vals) > 0 else np.nan
    )
    return df


def summarize_neighbor_context_layers(
    neighborhood_neighbors: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    rows = []

    use_df = neighborhood_neighbors.loc[
        neighborhood_neighbors["neighbor_rank"] <= top_k
    ].copy()

    for anchor, g in use_df.groupby("anchor_sequence"):
        human_fraction = float(
            g["neighbor_tissue_species"].fillna("").str.contains("Homo_sapiens").mean()
        )
        disease_any_fraction = float(g["neighbor_disease"].notna().mean())
        cancer_fraction = float(
            g["neighbor_disease"].fillna("").apply(_contains_cancer_keyword).mean()
        )

        if cancer_fraction >= 0.20 and human_fraction >= 0.50:
            profile = "cancer-associated human neighborhood"
        elif disease_any_fraction >= 0.20 and human_fraction >= 0.50:
            profile = "general disease-associated human neighborhood"
        elif human_fraction >= 0.50:
            profile = "human structural neighborhood"
        else:
            profile = "weakly contextualized structural neighborhood"

        rows.append(
            {
                "anchor_sequence": anchor,
                "neighbor_human_fraction": human_fraction,
                "neighbor_disease_any_fraction": disease_any_fraction,
                "neighbor_cancer_fraction": cancer_fraction,
                "context_profile": profile,
            }
        )

    return pd.DataFrame(rows)


def _sequence_token_presence(sequences: list[str]) -> Counter:
    counter = Counter()
    for seq in sequences:
        toks = set(_sequence_to_interpretable_tokens(seq))
        toks = {
            t for t in toks
            if not t.startswith("BRANCH_")
        }
        counter.update(toks)
    return counter


def summarize_discovered_motif_proxies(
    neighborhood_neighbors: pd.DataFrame,
    top_k: int = 10,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Uses interpretable sequence tokens as robust motif proxies.
    This is intentionally lightweight and stable compared to full glycowork
    motif annotation over noisy heterogeneous inputs.
    """
    rows = []

    use_df = neighborhood_neighbors.loc[
        neighborhood_neighbors["neighbor_rank"] <= top_k
    ].copy()

    for anchor, g in use_df.groupby("anchor_sequence"):
        seqs = [anchor] + g["neighbor_sequence"].dropna().tolist()
        counts = _sequence_token_presence(seqs)

        # Prefer tokens that are informative
        filtered = []
        for token, count in counts.most_common():
            if token.startswith("RES_") or token.startswith("LNK_") or token.startswith("PAIR_"):
                filtered.append((token, count))
            if len(filtered) >= top_n:
                break

        family_counter = Counter()
        for seq in seqs:
            family_counter.update(classify_structural_family(seq))

        rows.append(
            {
                "anchor_sequence": anchor,
                "top_motif_proxies": filtered,
                "top_family_labels": family_counter.most_common(8),
            }
        )

    return pd.DataFrame(rows)


def build_discovered_neighbor_protein_support(
    neighborhood_neighbors: pd.DataFrame,
    binding_standardized: pd.DataFrame,
    top_k: int = 10,
    top_n_proteins: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Infer protein context from the discovered glycan neighborhoods.
    Proteins are aggregated from exact-binding support of neighbor glycans.
    """
    binding = binding_standardized.copy()
    binding["canonical_sequence"] = binding["glycan_sequence"].apply(canonicalize_sequence_exact)

    rows_long = []
    rows_summary = []

    use_df = neighborhood_neighbors.loc[
        neighborhood_neighbors["neighbor_rank"] <= top_k
    ].copy()

    for anchor, g in use_df.groupby("anchor_sequence"):
        weighted_hits = []

        for _, row in g.iterrows():
            seq = row["neighbor_sequence"]
            rank = int(row["neighbor_rank"])
            weight = 1.0 / rank

            tmp = binding.loc[binding["canonical_sequence"] == seq].copy()
            if tmp.empty:
                continue

            protein_scores = (
                tmp.groupby("protein")["binding_value"]
                .agg(["mean", "count"])
                .reset_index()
            )

            for _, p in protein_scores.iterrows():
                weighted_hits.append(
                    {
                        "anchor_sequence": anchor,
                        "protein": p["protein"],
                        "neighbor_rank_weight": weight,
                        "mean_binding_value": float(p["mean"]),
                        "n_binding_rows": int(p["count"]),
                    }
                )

        if len(weighted_hits) == 0:
            rows_summary.append(
                {
                    "anchor_sequence": anchor,
                    "top_proteins": [],
                }
            )
            continue

        long_df = pd.DataFrame(weighted_hits)

        protein_summary = (
            long_df.groupby(["anchor_sequence", "protein"], as_index=False)
            .agg(
                support_weight=("neighbor_rank_weight", "sum"),
                mean_binding_value=("mean_binding_value", "mean"),
                n_binding_rows=("n_binding_rows", "sum"),
            )
            .sort_values(["anchor_sequence", "support_weight"], ascending=[True, False])
        )

        rows_long.append(protein_summary)

        top_prots = protein_summary.head(top_n_proteins)
        rows_summary.append(
            {
                "anchor_sequence": anchor,
                "top_proteins": [
                    (r["protein"], float(r["support_weight"]))
                    for _, r in top_prots.iterrows()
                ],
            }
        )

    protein_long = (
        pd.concat(rows_long, ignore_index=True)
        if len(rows_long) > 0
        else pd.DataFrame(columns=["anchor_sequence", "protein", "support_weight", "mean_binding_value", "n_binding_rows"])
    )
    protein_summary = pd.DataFrame(rows_summary)

    return protein_summary, protein_long


def compare_discovered_neighbor_robustness(
    final_neighbors: pd.DataFrame,
    reference_neighbors: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    rows = []

    a = final_neighbors.loc[final_neighbors["neighbor_rank"] <= top_k].copy()
    b = reference_neighbors.loc[reference_neighbors["neighbor_rank"] <= top_k].copy()

    anchors = sorted(set(a["anchor_sequence"]).union(set(b["anchor_sequence"])))

    for anchor in anchors:
        set_a = set(a.loc[a["anchor_sequence"] == anchor, "neighbor_sequence"])
        set_b = set(b.loc[b["anchor_sequence"] == anchor, "neighbor_sequence"])

        union = len(set_a | set_b)
        inter = len(set_a & set_b)
        jacc = inter / union if union > 0 else np.nan

        rows.append(
            {
                "anchor_sequence": anchor,
                "neighbor_robustness_jaccard": jacc,
            }
        )

    return pd.DataFrame(rows)


def build_similarity_network(
    neighborhood_neighbors: pd.DataFrame,
    top_k: int = 5,
):
    import networkx as nx

    g = nx.Graph()
    use_df = neighborhood_neighbors.loc[
        neighborhood_neighbors["neighbor_rank"] <= top_k
    ].copy()

    for _, row in use_df.iterrows():
        anchor = row["anchor_sequence"]
        nbr = row["neighbor_sequence"]
        weight = 1.0 - float(row["cosine_distance"])

        g.add_node(anchor, discovered=1)
        g.add_node(nbr, discovered=int(row["neighbor_discovered_flag"]))
        g.add_edge(anchor, nbr, weight=weight)

    return g


def plot_similarity_network(
    graph,
    max_label_len: int = 35,
) -> plt.Figure:
    import networkx as nx

    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    pos = nx.spring_layout(graph, seed=42, weight="weight")

    node_sizes = []
    node_colors = []
    labels = {}

    for node, attrs in graph.nodes(data=True):
        if attrs.get("discovered", 0) == 1:
            node_sizes.append(500)
            node_colors.append("tab:red")
        else:
            node_sizes.append(140)
            node_colors.append("tab:blue")
        labels[node] = shorten_glycan_label(node, max_len=max_label_len)

    nx.draw_networkx_edges(graph, pos, alpha=0.25, ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)

    ax.set_title("Discovered glycan similarity network")
    ax.axis("off")
    return fig


def build_anchor_protein_bipartite_network(
    protein_long: pd.DataFrame,
    top_n_edges_per_anchor: int = 5,
):
    import networkx as nx

    g = nx.Graph()

    if protein_long.empty:
        return g

    for anchor, sub in protein_long.groupby("anchor_sequence"):
        top_sub = sub.sort_values("support_weight", ascending=False).head(top_n_edges_per_anchor)
        g.add_node(anchor, kind="glycan")
        for _, row in top_sub.iterrows():
            protein = row["protein"]
            g.add_node(protein, kind="protein")
            g.add_edge(anchor, protein, weight=float(row["support_weight"]))

    return g


def plot_anchor_protein_bipartite_network(
    graph,
    max_label_len: int = 35,
) -> plt.Figure:
    import networkx as nx

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

    glycan_nodes = [n for n, a in graph.nodes(data=True) if a.get("kind") == "glycan"]
    protein_nodes = [n for n, a in graph.nodes(data=True) if a.get("kind") == "protein"]

    pos = {}
    for i, n in enumerate(glycan_nodes):
        pos[n] = (-1, -i)
    for i, n in enumerate(protein_nodes):
        pos[n] = (1, -i)

    nx.draw_networkx_edges(graph, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=glycan_nodes, node_size=500, node_color="tab:red", ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=protein_nodes, node_size=220, node_color="tab:green", ax=ax)

    labels = {n: shorten_glycan_label(n, max_len=max_label_len) for n in glycan_nodes}
    labels.update({n: shorten_glycan_label(n, max_len=max_label_len) for n in protein_nodes})
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)

    ax.set_title("Discovered glycan–protein context network")
    ax.axis("off")
    return fig


def build_embedding_report_table(embedding_eval: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in embedding_eval.iterrows():
        name = row["embedding_name"]

        if name == "sequence_token_only":
            pros = "Best structural coherence; strongest N-glycan control behavior; interpretable structural reference."
            cons = "Weaker metadata and binding context."
            decision = "keep_as_structural_reference"
        elif name == "structure_metadata_binding":
            pros = "Best overall contextual embedding; strongest species/sample/disease/binding neighborhoods."
            cons = "Slight loss in pure structural sharpness relative to sequence-token only."
            decision = "keep_as_final_interpretation_embedding"
        elif name == "structure_metadata":
            pros = "Very strong metadata neighborhoods."
            cons = "Binding context absent; slightly weaker than structure_metadata_binding overall."
            decision = "secondary_backup"
        elif name == "structure":
            pros = "Balanced structural baseline."
            cons = "Adds limited value once metadata/binding fusion is available."
            decision = "drop_after_ablation"
        else:
            pros = "Simple interpretable baseline."
            cons = "Lower overall contextual performance."
            decision = "drop_after_ablation"

        rows.append(
            {
                **row.to_dict(),
                "pros": pros,
                "cons": cons,
                "final_decision": decision,
            }
        )

    return pd.DataFrame(rows)


def _safe_mean_distance_score(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(max(0.0, 1.0 - min(value / 0.30, 1.0)))


def assign_priority_tier(score: float) -> str:
    if score >= 0.65:
        return "Tier A"
    if score >= 0.45:
        return "Tier B"
    return "Tier C"


def _build_interpretation_note(row: pd.Series) -> str:
    parts = []

    if row["neighbor_likely_n_glycan_fraction"] >= 0.8:
        parts.append("strong N-glycan-like neighborhood")
    elif row["neighbor_likely_n_glycan_fraction"] >= 0.3:
        parts.append("partial N-glycan-like neighborhood")
    else:
        parts.append("weak N-glycan-control support")

    if row["neighbor_binding_fraction"] >= 0.3:
        parts.append("good protein-context support")
    elif row["neighbor_binding_fraction"] > 0:
        parts.append("limited protein-context support")
    else:
        parts.append("no direct protein-context support")

    if row["neighbor_cancer_fraction"] >= 0.2:
        parts.append("cancer-associated neighbor context")
    elif row["neighbor_disease_any_fraction"] >= 0.2:
        parts.append("general disease-associated context")

    if row["neighbor_robustness_jaccard"] >= 0.5:
        parts.append("stable across embeddings")
    else:
        parts.append("embedding-sensitive neighborhood")

    return "; ".join(parts)


def build_final_discovered_enrichment_table(
    interpretation_set: pd.DataFrame,
    neighborhood_summary: pd.DataFrame,
    context_layers: pd.DataFrame,
    motif_proxy_summary: pd.DataFrame,
    protein_summary: pd.DataFrame,
    robustness_df: pd.DataFrame,
    final_embedding_name: str,
    embedding_coords: pd.DataFrame | None = None,
    task1_handoff: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = interpretation_set.loc[
        interpretation_set["discovered_flag"] == 1,
        [
            "canonical_sequence",
            "composition",
            "family_labels_str",
            "has_binding_exact",
            "likely_n_glycan",
        ],
    ].copy()

    out = out.merge(neighborhood_summary, left_on="canonical_sequence", right_on="anchor_sequence", how="left")
    out = out.merge(context_layers, on="anchor_sequence", how="left")
    out = out.merge(motif_proxy_summary, on="anchor_sequence", how="left")
    out = out.merge(protein_summary, on="anchor_sequence", how="left")
    out = out.merge(robustness_df, on="anchor_sequence", how="left")

    if embedding_coords is not None:
        coords = embedding_coords[["canonical_sequence", "x", "y"]].rename(
            columns={"x": "embedding_x", "y": "embedding_y"}
        )
        out = out.merge(coords, left_on="canonical_sequence", right_on="canonical_sequence", how="left")

    out["task1_biomarker_tier"] = "unresolved_task1_handoff"
    if task1_handoff is not None and "glycan_sequence" in task1_handoff.columns:
        tmp = task1_handoff.copy()
        tmp["canonical_sequence"] = tmp["glycan_sequence"].apply(canonicalize_sequence_exact)
        tier_col = "biomarker_tier" if "biomarker_tier" in tmp.columns else None
        if tier_col is not None:
            tmp = tmp[["canonical_sequence", tier_col]].drop_duplicates()
            out = out.merge(tmp, on="canonical_sequence", how="left", suffixes=("", "_task1"))
            out["task1_biomarker_tier"] = out[tier_col].fillna(out["task1_biomarker_tier"])
            out = out.drop(columns=[tier_col])

    out["neighborhood_confidence_score"] = out["mean_cosine_distance_top10"].apply(_safe_mean_distance_score)
    out["binding_overlap_mean"] = out["binding_overlap_mean"].fillna(0.0)
    out["neighbor_robustness_jaccard"] = out["neighbor_robustness_jaccard"].fillna(0.0)

    out["task2_priority_score"] = (
        0.30 * out["neighborhood_confidence_score"]
        + 0.20 * out["neighbor_likely_n_glycan_fraction"].fillna(0.0)
        + 0.15 * out["neighbor_binding_fraction"].fillna(0.0)
        + 0.15 * out["neighbor_cancer_fraction"].fillna(0.0)
        + 0.10 * out["binding_overlap_mean"].fillna(0.0)
        + 0.10 * out["neighbor_robustness_jaccard"].fillna(0.0)
    )

    out["priority_tier"] = out["task2_priority_score"].apply(assign_priority_tier)
    out["final_embedding_name"] = final_embedding_name
    out["final_interpretation_note"] = out.apply(_build_interpretation_note, axis=1)

    # stringify list-like columns for notebook readability
    for col in ["top_motif_proxies", "top_family_labels", "top_proteins"]:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda items: "; ".join(
                    [f"{k} ({v:.2f})" if isinstance(v, float) else f"{k} ({v})" for k, v in items]
                ) if isinstance(items, list) else ""
            )

    return out.sort_values(
        ["task2_priority_score", "mean_cosine_distance_top10"],
        ascending=[False, True],
    ).reset_index(drop=True)
