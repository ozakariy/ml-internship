from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from eda import (
    compute_feature_detection_metrics,
    compute_qc_feature_metrics,
    get_class_order,
    get_feature_columns,
    get_wide_matrix,
    make_batch1_operational_subset,
)

BIO_CLASSES = ("Dunn", "French", "LMU")
TECH_CLASSES = ("B", "SS", "dQC", "QC")


def freeze_partb_inputs(
    data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    include_classes: tuple[str, ...] = BIO_CLASSES + TECH_CLASSES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Freeze the operational Part B input:
    - batch 1 only
    - selected technical + biological classes
    """
    dm, acq = make_batch1_operational_subset(
        data_matrix=data_matrix,
        acquisition_list=acquisition_list,
        include_classes=include_classes,
    )

    inventory = (
        acq.groupby("class", dropna=False)
        .size()
        .rename("n_samples")
        .reset_index()
    )
    inventory["class"] = pd.Categorical(
        inventory["class"],
        categories=get_class_order(inventory["class"].astype(str).tolist()),
        ordered=True,
    )
    inventory = inventory.sort_values("class").reset_index(drop=True)
    return dm, acq, inventory


def compute_detection_inputs(
    data_matrix: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    detection_threshold: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute feature-wise detection metrics at the chosen operational threshold,
    plus a class-level summary.
    """
    feature_detection = compute_feature_detection_metrics(
        data_matrix=data_matrix,
        feature_metadata=feature_metadata,
        acquisition_list=acquisition_list,
        detection_threshold=detection_threshold,
    )

    class_cols = [
        c for c in feature_detection.columns
        if c.startswith("detection_rate__")
        and c not in {
            "detection_rate__all",
            "detection_rate__biological",
            "detection_rate__qc_like",
            "detection_rate__blank",
        }
    ]

    class_summary = (
        feature_detection[class_cols]
        .melt(var_name="metric", value_name="detection_rate")
        .assign(class_name=lambda d: d["metric"].str.replace("detection_rate__", "", regex=False))
        .groupby("class_name", dropna=False)["detection_rate"]
        .agg(
            mean_feature_detection_rate="mean",
            median_feature_detection_rate="median",
            fraction_features_ge_70=lambda s: (s >= 0.70).mean(),
            fraction_features_ge_90=lambda s: (s >= 0.90).mean(),
        )
        .reset_index()
        .rename(columns={"class_name": "class"})
    )

    class_summary["class"] = pd.Categorical(
        class_summary["class"],
        categories=get_class_order(class_summary["class"].astype(str).tolist()),
        ordered=True,
    )
    class_summary = class_summary.sort_values("class").reset_index(drop=True)

    return feature_detection, class_summary


def compute_qc_vs_dqc_metrics(
    data_matrix: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    detection_threshold: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare feature-level stability when using:
    - QC only
    - dQC only

    This supports the Part B decision to use QC as the primary technical filter
    and dQC as a secondary concentration-sensitivity check.
    """
    qc_only = compute_qc_feature_metrics(
        data_matrix=data_matrix,
        acquisition_list=acquisition_list,
        feature_metadata=feature_metadata,
        qc_classes=("QC",),
        detection_threshold=detection_threshold,
    ).rename(
        columns={
            "n_qc_samples": "n_qc_samples_qc_only",
            "qc_detection_rate": "qc_detection_rate_qc_only",
            "qc_mean_intensity": "qc_mean_intensity_qc_only",
            "qc_median_intensity": "qc_median_intensity_qc_only",
            "qc_std_intensity": "qc_std_intensity_qc_only",
            "qc_cv_pct": "qc_cv_pct_qc_only",
            "qc_order_spearman_rho": "qc_order_spearman_rho_qc_only",
            "qc_order_spearman_abs": "qc_order_spearman_abs_qc_only",
        }
    )

    dqc_only = compute_qc_feature_metrics(
        data_matrix=data_matrix,
        acquisition_list=acquisition_list,
        feature_metadata=feature_metadata,
        qc_classes=("dQC",),
        detection_threshold=detection_threshold,
    ).rename(
        columns={
            "n_qc_samples": "n_qc_samples_dqc_only",
            "qc_detection_rate": "qc_detection_rate_dqc_only",
            "qc_mean_intensity": "qc_mean_intensity_dqc_only",
            "qc_median_intensity": "qc_median_intensity_dqc_only",
            "qc_std_intensity": "qc_std_intensity_dqc_only",
            "qc_cv_pct": "qc_cv_pct_dqc_only",
            "qc_order_spearman_rho": "qc_order_spearman_rho_dqc_only",
            "qc_order_spearman_abs": "qc_order_spearman_abs_dqc_only",
        }
    )

    keep_cols_qc = [
        "feature",
        "n_qc_samples_qc_only",
        "qc_detection_rate_qc_only",
        "qc_mean_intensity_qc_only",
        "qc_cv_pct_qc_only",
        "qc_order_spearman_rho_qc_only",
        "qc_order_spearman_abs_qc_only",
        "mz",
        "rt",
    ]
    keep_cols_dqc = [
        "feature",
        "n_qc_samples_dqc_only",
        "qc_detection_rate_dqc_only",
        "qc_mean_intensity_dqc_only",
        "qc_cv_pct_dqc_only",
        "qc_order_spearman_rho_dqc_only",
        "qc_order_spearman_abs_dqc_only",
    ]

    merged = qc_only[keep_cols_qc].merge(
        dqc_only[keep_cols_dqc],
        on="feature",
        how="outer",
        validate="one_to_one",
    )

    merged["stable_qc_only_30"] = merged["qc_cv_pct_qc_only"] < 30.0
    merged["stable_dqc_only_30"] = merged["qc_cv_pct_dqc_only"] < 30.0
    merged["stable_in_both_30"] = merged["stable_qc_only_30"] & merged["stable_dqc_only_30"]
    merged["cv_gap_dqc_minus_qc"] = merged["qc_cv_pct_dqc_only"] - merged["qc_cv_pct_qc_only"]
    merged["abs_drift_gap"] = (
        merged["qc_order_spearman_abs_dqc_only"] - merged["qc_order_spearman_abs_qc_only"]
    ).abs()
    merged["potential_concentration_sensitive_flag"] = (
        merged["stable_qc_only_30"] & (~merged["stable_dqc_only_30"])
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "n_features_total",
                "n_features_qc_cv_lt_30_qc_only",
                "fraction_qc_cv_lt_30_qc_only",
                "n_features_qc_cv_lt_30_dqc_only",
                "fraction_qc_cv_lt_30_dqc_only",
                "n_features_stable_in_both",
                "n_features_qc_stable_but_dqc_unstable",
                "median_qc_cv_qc_only",
                "median_qc_cv_dqc_only",
                "median_abs_qc_drift_qc_only",
                "median_abs_qc_drift_dqc_only",
            ],
            "value": [
                merged["feature"].nunique(),
                int((merged["qc_cv_pct_qc_only"] < 30.0).sum()),
                float((merged["qc_cv_pct_qc_only"] < 30.0).mean()),
                int((merged["qc_cv_pct_dqc_only"] < 30.0).sum()),
                float((merged["qc_cv_pct_dqc_only"] < 30.0).mean()),
                int(merged["stable_in_both_30"].sum()),
                int(merged["potential_concentration_sensitive_flag"].sum()),
                float(merged["qc_cv_pct_qc_only"].median(skipna=True)),
                float(merged["qc_cv_pct_dqc_only"].median(skipna=True)),
                float(merged["qc_order_spearman_abs_qc_only"].median(skipna=True)),
                float(merged["qc_order_spearman_abs_dqc_only"].median(skipna=True)),
            ],
        }
    )

    ranked = merged.sort_values(
        ["potential_concentration_sensitive_flag", "cv_gap_dqc_minus_qc"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    return merged, summary, ranked


def _wide_to_data_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    return wide.reset_index().rename(columns={"index": "sample"})


def apply_processing_view(
    data_matrix: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """
    Candidate sample-processing views for Part B.

    Methods:
    - raw
    - log1p
    - tic_log1p       : total-signal scaling, then log1p
    - median_log1p    : median-positive scaling, then log1p
    """
    feature_cols = get_feature_columns(data_matrix)
    wide = get_wide_matrix(data_matrix)[feature_cols].astype(float).copy()

    if method == "raw":
        transformed = wide

    elif method == "log1p":
        transformed = np.log1p(wide)

    elif method == "tic_log1p":
        sample_sums = wide.sum(axis=1)
        target_sum = np.median(sample_sums[sample_sums > 0])
        safe_sums = sample_sums.replace(0, np.nan)
        transformed = wide.div(safe_sums, axis=0) * target_sum
        transformed = transformed.fillna(0.0)
        transformed = np.log1p(transformed)

    elif method == "median_log1p":
        sample_medians = wide.replace(0, np.nan).median(axis=1)
        target_median = np.nanmedian(sample_medians.to_numpy(dtype=float))
        safe_medians = sample_medians.replace(0, np.nan)
        transformed = wide.div(safe_medians, axis=0) * target_median
        transformed = transformed.fillna(0.0)
        transformed = np.log1p(transformed)

    else:
        raise ValueError(
            "Unknown method. Expected one of: "
            "'raw', 'log1p', 'tic_log1p', 'median_log1p'."
        )

    return _wide_to_data_matrix(transformed)


def build_transformation_candidates(
    data_matrix: pd.DataFrame,
    methods: tuple[str, ...] = ("raw", "log1p", "tic_log1p", "median_log1p"),
) -> Dict[str, pd.DataFrame]:
    return {method: apply_processing_view(data_matrix, method) for method in methods}


def _upper_triangle_values(corr: pd.DataFrame) -> np.ndarray:
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    return corr.where(mask).stack().to_numpy(dtype=float)


def _sample_to_class_median_corr(
    transformed_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    classes: Iterable[str],
) -> pd.Series:
    wide = get_wide_matrix(transformed_matrix)
    meta = acquisition_list[["sample", "class"]].drop_duplicates().copy()
    meta = meta.loc[meta["class"].isin(classes)].copy()

    class_medians = {}
    for cls in meta["class"].unique():
        samples = meta.loc[meta["class"] == cls, "sample"].tolist()
        class_medians[cls] = wide.loc[samples].median(axis=0)

    rows = []
    for _, row in meta.iterrows():
        sample_id = row["sample"]
        cls = row["class"]
        sample_vec = wide.loc[sample_id]
        class_vec = class_medians[cls]

        if sample_vec.nunique() <= 1 or class_vec.nunique() <= 1:
            corr = np.nan
        else:
            corr = sample_vec.corr(class_vec, method="spearman")

        rows.append(
            {
                "sample": sample_id,
                "class": cls,
                "corr_to_class_median": corr,
            }
        )

    return pd.DataFrame(rows)["corr_to_class_median"]


def _compute_pc1_order_rho(
    transformed_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    classes: tuple[str, ...],
) -> tuple[float, float]:
    meta = (
        acquisition_list.loc[acquisition_list["class"].isin(classes), ["sample", "class", "batch", "order"]]
        .drop_duplicates()
        .sort_values(["batch", "order"])
        .reset_index(drop=True)
    )
    wide = get_wide_matrix(transformed_matrix).loc[meta["sample"]].copy()

    keep_cols = wide.columns[wide.std(axis=0) > 0]
    wide = wide[keep_cols]

    if wide.shape[0] < 3 or wide.shape[1] < 2:
        return np.nan, np.nan

    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(wide)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)

    pc1 = pd.Series(pcs[:, 0], index=meta["sample"])
    order = meta.set_index("sample")["order"].astype(float)

    if pc1.nunique() <= 1 or order.nunique() <= 1:
        rho = np.nan
    else:
        rho = pc1.corr(order, method="spearman")

    evr = float(pca.explained_variance_ratio_[0])
    return rho, evr


def evaluate_transformation_candidates(
    transformed_candidates: Dict[str, pd.DataFrame],
    acquisition_list: pd.DataFrame,
    qc_classes: tuple[str, ...] = ("QC",),
    bio_classes: tuple[str, ...] = BIO_CLASSES,
) -> pd.DataFrame:
    """
    Score candidate processing views.

    Notes:
    - We use QC-only here because Part B explicitly defines the primary
      technical criterion on QC.
    - This is a comparison stage; final hard filtering is done later.
    """
    rows = []

    qc_samples = (
        acquisition_list.loc[acquisition_list["class"].isin(qc_classes), "sample"]
        .drop_duplicates()
        .tolist()
    )

    for method, dm in transformed_candidates.items():
        wide = get_wide_matrix(dm).astype(float)

        qc_wide = wide.loc[qc_samples].copy()
        qc_corr = qc_wide.T.corr(method="spearman")
        qc_corr_values = _upper_triangle_values(qc_corr)

        qc_pc1_rho, qc_pc1_evr = _compute_pc1_order_rho(
            transformed_matrix=dm,
            acquisition_list=acquisition_list,
            classes=qc_classes,
        )

        bio_corr_to_class = _sample_to_class_median_corr(
            transformed_matrix=dm,
            acquisition_list=acquisition_list,
            classes=bio_classes,
        )

        qc_total_signal = qc_wide.sum(axis=1)
        qc_total_cv_proxy = (
            100.0 * qc_total_signal.std(ddof=1) / qc_total_signal.mean()
            if qc_total_signal.mean() > 0
            else np.nan
        )

        rows.append(
            {
                "method": method,
                "median_qc_pairwise_corr": float(np.nanmedian(qc_corr_values)),
                "min_qc_pairwise_corr": float(np.nanmin(qc_corr_values)),
                "abs_qc_pc1_order_rho": float(abs(qc_pc1_rho)) if pd.notna(qc_pc1_rho) else np.nan,
                "qc_pc1_variance_explained": qc_pc1_evr,
                "median_bio_corr_to_class_median": float(bio_corr_to_class.median(skipna=True)),
                "min_bio_corr_to_class_median": float(bio_corr_to_class.min(skipna=True)),
                "qc_total_signal_cv_proxy": float(qc_total_cv_proxy),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["median_qc_pairwise_corr", "median_bio_corr_to_class_median"],
        ascending=[False, False],
    ).reset_index(drop=True)


def plot_transformation_metric_summary(
    transformation_summary: pd.DataFrame,
) -> plt.Figure:
    df = transformation_summary.copy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    sns.barplot(
        data=df,
        x="method",
        y="median_qc_pairwise_corr",
        ax=axes[0],
    )
    axes[0].set_title("Median QC pairwise correlation")
    axes[0].set_xlabel("Processing view")
    axes[0].set_ylabel("Spearman correlation")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(
        data=df,
        x="method",
        y="abs_qc_pc1_order_rho",
        ax=axes[1],
    )
    axes[1].set_title("|Spearman rho| of QC PC1 vs run order")
    axes[1].set_xlabel("Processing view")
    axes[1].set_ylabel("Absolute rho")
    axes[1].tick_params(axis="x", rotation=20)

    sns.barplot(
        data=df,
        x="method",
        y="median_bio_corr_to_class_median",
        ax=axes[2],
    )
    axes[2].set_title("Median biological corr. to class median")
    axes[2].set_xlabel("Processing view")
    axes[2].set_ylabel("Spearman correlation")
    axes[2].tick_params(axis="x", rotation=20)

    return fig

def plot_detection_rate_by_class_clear(
    feature_detection_metrics: pd.DataFrame,
    classes: tuple[str, ...] | None = None,
    thresholds: tuple[float, float] = (0.70, 0.90),
) -> plt.Figure:
    """
    Clearer alternative to a boxplot for feature-wise detection rates.

    Left panel:
    - jittered points per class
    - horizontal lines at operational thresholds

    Right panel:
    - stacked proportions of features in detection-rate bins
    """
    rate_cols = [
        c for c in feature_detection_metrics.columns
        if c.startswith("detection_rate__")
        and c not in {
            "detection_rate__all",
            "detection_rate__biological",
            "detection_rate__qc_like",
            "detection_rate__blank",
        }
    ]

    long_df = feature_detection_metrics.melt(
        id_vars=["feature"],
        value_vars=rate_cols,
        var_name="rate_col",
        value_name="detection_rate",
    )
    long_df["class"] = long_df["rate_col"].str.replace("detection_rate__", "", regex=False)

    if classes is not None:
        long_df = long_df.loc[long_df["class"].isin(classes)].copy()

    class_order = get_class_order(long_df["class"].dropna().unique().tolist())

    # Detection bins aligned with Part B reasoning
    bins = [-0.001, thresholds[0], thresholds[1], 0.999999, 1.000001]
    labels = [
        f"< {thresholds[0]:.2f}",
        f"{thresholds[0]:.2f}–{thresholds[1]:.2f}",
        f"{thresholds[1]:.2f}–<1.00",
        "1.00",
    ]
    long_df["detection_bin"] = pd.cut(
        long_df["detection_rate"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    bin_summary = (
        long_df.groupby(["class", "detection_bin"], dropna=False)
        .size()
        .rename("n_features")
        .reset_index()
    )

    total_per_class = (
        long_df.groupby("class")
        .size()
        .rename("n_total")
        .reset_index()
    )

    bin_summary = bin_summary.merge(total_per_class, on="class", how="left")
    bin_summary["fraction"] = bin_summary["n_features"] / bin_summary["n_total"]

    pivot = (
        bin_summary.pivot(index="class", columns="detection_bin", values="fraction")
        .fillna(0.0)
        .reindex(class_order)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # Left panel: jittered points
    sns.stripplot(
        data=long_df,
        x="class",
        y="detection_rate",
        order=class_order,
        jitter=0.28,
        alpha=0.45,
        size=4,
        color="tab:blue",
        ax=axes[0],
    )
    for thr in thresholds:
        axes[0].axhline(thr, color="red", linestyle="--", linewidth=1.8)

    axes[0].set_title("Feature-wise detection rates by class")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Detection rate")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.2, axis="y")

    # Right panel: stacked fractions
    bottom = np.zeros(len(pivot))
    color_map = {
        labels[0]: "#d73027",
        labels[1]: "#fdae61",
        labels[2]: "#74add1",
        labels[3]: "#1a9850",
    }

    for label in labels:
        vals = pivot[label].to_numpy() if label in pivot.columns else np.zeros(len(pivot))
        axes[1].bar(
            pivot.index,
            vals,
            bottom=bottom,
            label=label,
            color=color_map[label],
            edgecolor="white",
            linewidth=0.7,
        )
        bottom += vals

    axes[1].set_title("Proportion of features by detection-support bin")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Fraction of features")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(alpha=0.2, axis="y")
    axes[1].legend(title="Detection bin", bbox_to_anchor=(1.02, 1), loc="upper left")

    return fig
