from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from eda import get_class_order, get_feature_columns, get_wide_matrix
from task1_processing import BIO_CLASSES


def _robust_zscore(series: pd.Series) -> pd.Series:
    series = pd.Series(series, copy=False)
    median = series.median(skipna=True)
    mad = (series - median).abs().median(skipna=True)

    if pd.isna(mad) or mad == 0:
        std = series.std(ddof=1)
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=series.index)
        return (series - median) / std

    return 0.6745 * (series - median) / mad


def _safe_spearman(a: pd.Series, b: pd.Series) -> float:
    if a.nunique(dropna=True) <= 1 or b.nunique(dropna=True) <= 1:
        return np.nan
    return float(a.corr(b, method="spearman"))


def _summarize_detected_values(
    values: pd.Series,
    detection_threshold: float,
) -> dict[str, float]:
    values = pd.Series(values, copy=False).astype(float)
    detected = values.loc[values > detection_threshold]

    n_total = len(values)
    n_detected = len(detected)
    detection_rate = n_detected / n_total if n_total > 0 else np.nan

    if n_detected == 0:
        return {
            "mean_intensity": np.nan,
            "median_intensity": np.nan,
            "std_intensity": np.nan,
            "cv_pct": np.nan,
            "log_sd": np.nan,
            "detection_rate": detection_rate,
            "n_detected": n_detected,
        }

    if n_detected == 1:
        x = float(detected.iloc[0])
        return {
            "mean_intensity": x,
            "median_intensity": x,
            "std_intensity": np.nan,
            "cv_pct": np.nan,
            "log_sd": np.nan,
            "detection_rate": detection_rate,
            "n_detected": n_detected,
        }

    mean_intensity = float(detected.mean())
    std_intensity = float(detected.std(ddof=1))
    cv_pct = float(100.0 * std_intensity / mean_intensity) if mean_intensity > 0 else np.nan
    log_sd = float(np.log1p(detected).std(ddof=1))

    return {
        "mean_intensity": mean_intensity,
        "median_intensity": float(detected.median()),
        "std_intensity": std_intensity,
        "cv_pct": cv_pct,
        "log_sd": log_sd,
        "detection_rate": detection_rate,
        "n_detected": n_detected,
    }


def build_biological_sample_review(
    raw_data_matrix: pd.DataFrame,
    transformed_data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    detection_threshold: float = 10.0,
    bio_classes: tuple[str, ...] = BIO_CLASSES,
    robust_z_threshold: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a biological-sample review scorecard.

    Raw matrix is used for total signal / detected-feature counts.
    Transformed matrix is used for correlation-to-class-median and PCA geometry.
    """
    feature_cols = get_feature_columns(raw_data_matrix)

    raw_wide = get_wide_matrix(raw_data_matrix)[feature_cols].astype(float)
    transformed_wide = get_wide_matrix(transformed_data_matrix)[feature_cols].astype(float)

    meta = (
        acquisition_list.loc[acquisition_list["class"].isin(bio_classes), ["sample", "class", "order", "batch"]]
        .drop_duplicates()
        .sort_values(["class", "order"])
        .reset_index(drop=True)
    )

    review = meta.copy()
    review["total_signal_raw"] = review["sample"].map(raw_wide.sum(axis=1))
    review["log10_total_signal_plus1"] = np.log10(review["total_signal_raw"] + 1.0)

    detected_counts = (raw_wide > detection_threshold).sum(axis=1)
    review["detected_features"] = review["sample"].map(detected_counts)
    review["detection_rate"] = review["detected_features"] / len(feature_cols)
    review["zero_fraction"] = 1.0 - review["detection_rate"]

    # Correlation to class median profile in transformed space
    class_medians = {}
    for cls in bio_classes:
        cls_samples = meta.loc[meta["class"] == cls, "sample"].tolist()
        class_medians[cls] = transformed_wide.loc[cls_samples].median(axis=0)

    corrs = []
    for _, row in review.iterrows():
        sample_id = row["sample"]
        cls = row["class"]
        corrs.append(
            _safe_spearman(
                transformed_wide.loc[sample_id],
                class_medians[cls],
            )
        )
    review["corr_to_class_median"] = corrs

    # PCA in transformed biological space
    bio_wide = transformed_wide.loc[review["sample"]].copy()
    keep_cols = bio_wide.columns[bio_wide.std(axis=0) > 0]
    bio_wide = bio_wide[keep_cols]

    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(bio_wide)

    n_components = int(min(5, X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X)

    score_cols = [f"PC{i+1}" for i in range(n_components)]
    pca_scores = pd.DataFrame(scores, columns=score_cols, index=review["sample"])
    pca_scores = pca_scores.reset_index().rename(columns={"index": "sample"})
    pca_scores = pca_scores.merge(meta, on="sample", how="left", validate="one_to_one")

    # Distance to class centroid in the retained PCA space
    centroid_cols = score_cols
    centroids = (
        pca_scores.groupby("class", dropna=False)[centroid_cols]
        .mean()
        .reset_index()
    )
    pca_scores = pca_scores.merge(
        centroids,
        on="class",
        how="left",
        suffixes=("", "_class_centroid"),
    )

    diffs = []
    for col in centroid_cols:
        diffs.append((pca_scores[col] - pca_scores[f"{col}_class_centroid"]) ** 2)
    pca_scores["pca_distance_to_class_centroid"] = np.sqrt(np.sum(diffs, axis=0))

    review = review.merge(
        pca_scores[["sample"] + score_cols + ["pca_distance_to_class_centroid"]],
        on="sample",
        how="left",
        validate="one_to_one",
    )

    # Within-class robust z-scores
    metrics_for_z = [
        "log10_total_signal_plus1",
        "detected_features",
        "zero_fraction",
        "corr_to_class_median",
        "pca_distance_to_class_centroid",
    ]
    for col in metrics_for_z:
        review[f"{col}_robust_z_within_class"] = (
            review.groupby("class", dropna=False)[col]
            .transform(_robust_zscore)
        )

    # Review flags
    review["flag_low_total_signal"] = (
        review["log10_total_signal_plus1_robust_z_within_class"] < -robust_z_threshold
    )
    review["flag_low_detected_features"] = (
        review["detected_features_robust_z_within_class"] < -robust_z_threshold
    )
    review["flag_high_zero_fraction"] = (
        review["zero_fraction_robust_z_within_class"] > robust_z_threshold
    )
    review["flag_low_corr_to_class_median"] = (
        review["corr_to_class_median_robust_z_within_class"] < -robust_z_threshold
    )
    review["flag_high_pca_distance"] = (
        review["pca_distance_to_class_centroid_robust_z_within_class"] > robust_z_threshold
    )

    flag_cols = [
        "flag_low_total_signal",
        "flag_low_detected_features",
        "flag_high_zero_fraction",
        "flag_low_corr_to_class_median",
        "flag_high_pca_distance",
    ]
    review["n_review_flags"] = review[flag_cols].sum(axis=1)

    review["recommend_exclude"] = (
        (review["n_review_flags"] >= 3)
        & (
            review["flag_low_corr_to_class_median"]
            | review["flag_high_pca_distance"]
        )
    )

    review["review_tier"] = np.select(
        [
            review["recommend_exclude"],
            review["n_review_flags"] >= 2,
            review["n_review_flags"] == 1,
        ],
        [
            "strong_review",
            "review",
            "minor_review",
        ],
        default="keep",
    )

    summary = (
        review.groupby(["class", "review_tier"], dropna=False)
        .size()
        .rename("n_samples")
        .reset_index()
    )

    return review.sort_values(["review_tier", "n_review_flags", "class", "order"], ascending=[True, False, True, True]), summary


def plot_biological_sample_review(
    sample_review: pd.DataFrame,
) -> plt.Figure:
    df = sample_review.copy()
    class_order = get_class_order(df["class"].dropna().unique().tolist())
    tier_order = ["keep", "minor_review", "review", "strong_review"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    sns.scatterplot(
        data=df,
        x="log10_total_signal_plus1",
        y="corr_to_class_median",
        hue="class",
        style="review_tier",
        hue_order=class_order,
        style_order=tier_order,
        s=80,
        ax=axes[0],
    )
    axes[0].set_title("Biological sample review: signal vs class-profile coherence")
    axes[0].set_xlabel("log10(total signal + 1)")
    axes[0].set_ylabel("Spearman corr. to class median")

    for _, row in df.loc[df["review_tier"].isin(["review", "strong_review"])].iterrows():
        axes[0].annotate(
            row["sample"].split("-")[-1],
            (row["log10_total_signal_plus1"], row["corr_to_class_median"]),
            fontsize=8,
            alpha=0.8,
        )

    sns.scatterplot(
        data=df,
        x="PC1",
        y="PC2",
        hue="class",
        style="review_tier",
        hue_order=class_order,
        style_order=tier_order,
        s=80,
        ax=axes[1],
    )
    axes[1].set_title("Biological sample PCA with review tiers")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")

    for _, row in df.loc[df["review_tier"].isin(["review", "strong_review"])].iterrows():
        axes[1].annotate(
            row["sample"].split("-")[-1],
            (row["PC1"], row["PC2"]),
            fontsize=8,
            alpha=0.8,
        )

    return fig


def compute_qc_only_technical_biological_metrics(
    raw_data_matrix: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    detection_threshold: float = 10.0,
    qc_class: str = "QC",
    bio_classes: tuple[str, ...] = BIO_CLASSES,
) -> pd.DataFrame:
    """
    Recompute technical vs biological variability using QC-only as the technical reference.
    """
    feature_cols = get_feature_columns(raw_data_matrix)
    wide = get_wide_matrix(raw_data_matrix)[feature_cols].astype(float)

    qc_samples = (
        acquisition_list.loc[acquisition_list["class"] == qc_class, "sample"]
        .drop_duplicates()
        .tolist()
    )
    bio_samples_by_class = {
        cls: acquisition_list.loc[acquisition_list["class"] == cls, "sample"].drop_duplicates().tolist()
        for cls in bio_classes
    }

    rows = []
    for feature in feature_cols:
        qc_summary = _summarize_detected_values(
            wide.loc[qc_samples, feature],
            detection_threshold=detection_threshold,
        )

        row = {
            "feature": feature,
            "qc_mean_intensity": qc_summary["mean_intensity"],
            "qc_std_intensity": qc_summary["std_intensity"],
            "qc_cv_pct": qc_summary["cv_pct"],
            "qc_log_sd": qc_summary["log_sd"],
            "qc_detection_rate": qc_summary["detection_rate"],
        }

        bio_cv_values = []
        bio_log_values = []
        bio_detection_values = []

        for cls in bio_classes:
            cls_summary = _summarize_detected_values(
                wide.loc[bio_samples_by_class[cls], feature],
                detection_threshold=detection_threshold,
            )

            row[f"{cls}_mean_intensity"] = cls_summary["mean_intensity"]
            row[f"{cls}_std_intensity"] = cls_summary["std_intensity"]
            row[f"{cls}_cv_pct"] = cls_summary["cv_pct"]
            row[f"{cls}_log_sd"] = cls_summary["log_sd"]
            row[f"{cls}_detection_rate"] = cls_summary["detection_rate"]

            if pd.notna(cls_summary["cv_pct"]):
                bio_cv_values.append(cls_summary["cv_pct"])
            if pd.notna(cls_summary["log_sd"]):
                bio_log_values.append(cls_summary["log_sd"])
            if pd.notna(cls_summary["detection_rate"]):
                bio_detection_values.append(cls_summary["detection_rate"])

        row["bio_cv_pct_median"] = np.median(bio_cv_values) if bio_cv_values else np.nan
        row["bio_cv_pct_max"] = np.max(bio_cv_values) if bio_cv_values else np.nan
        row["bio_log_sd_median"] = np.median(bio_log_values) if bio_log_values else np.nan
        row["bio_detection_rate_median"] = (
            np.median(bio_detection_values) if bio_detection_values else np.nan
        )
        row["bio_detection_rate_min"] = (
            np.min(bio_detection_values) if bio_detection_values else np.nan
        )

        row["d_ratio_cv"] = (
            row["qc_cv_pct"] / row["bio_cv_pct_median"]
            if pd.notna(row["qc_cv_pct"]) and pd.notna(row["bio_cv_pct_median"]) and row["bio_cv_pct_median"] > 0
            else np.nan
        )
        row["d_ratio_log"] = (
            row["qc_log_sd"] / row["bio_log_sd_median"]
            if pd.notna(row["qc_log_sd"]) and pd.notna(row["bio_log_sd_median"]) and row["bio_log_sd_median"] > 0
            else np.nan
        )

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.merge(feature_metadata, on="feature", how="left", validate="one_to_one")
    return out


def _build_exclusion_reason(row: pd.Series) -> str:
    reasons = []
    if not bool(row["passes_mz_500"]):
        reasons.append("mz<=500")
    if not bool(row["passes_qc_cv_30"]):
        reasons.append("QC_CV>=30")
    if not bool(row["passes_detection_70_any_class"]):
        reasons.append("bio_detection<70_in_all_classes")
    if bool(row["blank_dominant_veto"]):
        reasons.append("blank_dominant")
    return "; ".join(reasons) if reasons else "retained"


def build_feature_filter_table(
    technical_biological_metrics: pd.DataFrame,
    feature_detection_metrics: pd.DataFrame,
    contamination_metrics: pd.DataFrame,
    blank_carryover_proxy: pd.DataFrame | None = None,
    bio_classes: tuple[str, ...] = BIO_CLASSES,
    qc_cv_threshold: float = 30.0,
    bio_detection_threshold: float = 0.70,
    mz_threshold: float = 500.0,
    blank_ratio_veto: float = 0.50,
    blank_detection_veto: float = 0.50,
    d_ratio_log_high_conf: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    det_cols = ["feature"] + [f"detection_rate__{cls}" for cls in bio_classes]
    cont_cols = [
        "feature",
        "blank_mean_intensity",
        "blank_detection_rate",
        "qc_mean_intensity",
        "bio_mean_intensity",
        "blank_to_qc_ratio",
        "blank_to_bio_ratio",
    ]
    if "blank_dominant_flag" in contamination_metrics.columns:
        cont_cols.append("blank_dominant_flag")

    df = technical_biological_metrics.merge(
        feature_detection_metrics[det_cols],
        on="feature",
        how="left",
        validate="one_to_one",
    ).merge(
        contamination_metrics[cont_cols],
        on="feature",
        how="left",
        validate="one_to_one",
        suffixes=("", "_cont"),
    )

    if blank_carryover_proxy is not None:
        df = df.merge(
            blank_carryover_proxy[["feature", "carryover_spearman_rho", "n_blank_contexts"]],
            on="feature",
            how="left",
            validate="one_to_one",
        )
    else:
        df["carryover_spearman_rho"] = np.nan
        df["n_blank_contexts"] = np.nan

    bio_det_cols = [f"detection_rate__{cls}" for cls in bio_classes]
    df["bio_detection_rate_max"] = df[bio_det_cols].max(axis=1)
    df["bio_detection_rate_min"] = df[bio_det_cols].min(axis=1)

    df["passes_mz_500"] = df["mz"] > mz_threshold
    df["passes_qc_cv_30"] = df["qc_cv_pct"] < qc_cv_threshold
    df["passes_detection_70_any_class"] = df["bio_detection_rate_max"] >= bio_detection_threshold
    df["passes_detection_70_all_classes"] = df["bio_detection_rate_min"] >= bio_detection_threshold

    df["blank_dominant_veto"] = (
        (df["blank_detection_rate"].fillna(0.0) >= blank_detection_veto)
        & (
            (df["blank_to_qc_ratio"].fillna(0.0) >= blank_ratio_veto)
            | (df["blank_to_bio_ratio"].fillna(0.0) >= blank_ratio_veto)
            | (df.get("blank_dominant_flag", False).fillna(False))
        )
    )

    df["potential_carryover_flag"] = (
        (df["carryover_spearman_rho"].fillna(-np.inf) >= 0.70)
        & (df["n_blank_contexts"].fillna(0) >= 5)
    )

    df["retained_main"] = (
        df["passes_mz_500"]
        & df["passes_qc_cv_30"]
        & df["passes_detection_70_any_class"]
        & (~df["blank_dominant_veto"])
    )

    df["retained_high_confidence"] = (
        df["retained_main"]
        & df["d_ratio_log"].notna()
        & (df["d_ratio_log"] <= d_ratio_log_high_conf)
    )

    df["exclusion_reason"] = df.apply(_build_exclusion_reason, axis=1)

    df = df.sort_values(
        ["retained_main", "retained_high_confidence", "d_ratio_log", "qc_cv_pct", "bio_detection_rate_max"],
        ascending=[False, False, True, True, False],
        na_position="last",
    ).reset_index(drop=True)

    summary = pd.DataFrame(
        {
            "stage": [
                "start",
                "mz>500",
                "mz>500 & QC_CV<30",
                "mz>500 & QC_CV<30 & bio_detection>=70_in_any_class",
                "main_retained_after_blank_veto",
                "high_confidence_retained_dratio_log<=1.0",
            ],
            "n_features": [
                int(df["feature"].nunique()),
                int(df["passes_mz_500"].sum()),
                int((df["passes_mz_500"] & df["passes_qc_cv_30"]).sum()),
                int((df["passes_mz_500"] & df["passes_qc_cv_30"] & df["passes_detection_70_any_class"]).sum()),
                int(df["retained_main"].sum()),
                int(df["retained_high_confidence"].sum()),
            ],
        }
    )

    return df, summary


def plot_feature_filtering_summary(
    filter_summary: pd.DataFrame,
    feature_filter_table: pd.DataFrame,
    d_ratio_log_threshold: float = 1.0,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    sns.barplot(
        data=filter_summary,
        x="stage",
        y="n_features",
        ax=axes[0],
    )
    axes[0].set_title("Feature counts after sequential Part B filters")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Number of features")
    axes[0].tick_params(axis="x", rotation=25)

    retained_main = feature_filter_table.loc[feature_filter_table["retained_main"]].copy()
    sns.histplot(
        retained_main["d_ratio_log"].dropna(),
        bins=25,
        ax=axes[1],
    )
    axes[1].axvline(d_ratio_log_threshold, color="red", linestyle="--", linewidth=2)
    axes[1].set_title("D-ratio(log) among main-retained features")
    axes[1].set_xlabel("D-ratio(log)")
    axes[1].set_ylabel("Count")

    return fig


def assemble_partb_outputs(
    raw_data_matrix: pd.DataFrame,
    transformed_data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    sample_review_table: pd.DataFrame,
    feature_filter_table: pd.DataFrame,
    bio_classes: tuple[str, ...] = BIO_CLASSES,
) -> dict[str, pd.DataFrame]:
    """
    Build the retained sample / feature manifests and the matrices for Part C.
    """
    bio_meta = (
        acquisition_list.loc[acquisition_list["class"].isin(bio_classes), ["sample", "class", "order", "batch"]]
        .drop_duplicates()
        .sort_values(["class", "order"])
        .reset_index(drop=True)
    )

    retained_samples = sample_review_table.loc[~sample_review_table["recommend_exclude"], "sample"].tolist()
    retained_features_main = feature_filter_table.loc[feature_filter_table["retained_main"], "feature"].tolist()
    retained_features_high = feature_filter_table.loc[
        feature_filter_table["retained_high_confidence"], "feature"
    ].tolist()

    raw_wide = get_wide_matrix(raw_data_matrix).astype(float)
    transformed_wide = get_wide_matrix(transformed_data_matrix).astype(float)

    retained_sample_manifest = sample_review_table.loc[
        sample_review_table["sample"].isin(retained_samples)
    ].copy()

    retained_feature_manifest_main = feature_filter_table.loc[
        feature_filter_table["retained_main"]
    ].copy()

    retained_feature_manifest_high = feature_filter_table.loc[
        feature_filter_table["retained_high_confidence"]
    ].copy()

    partc_matrix_main = (
        transformed_wide.loc[retained_samples, retained_features_main]
        .reset_index()
        .rename(columns={"index": "sample"})
    )
    partc_matrix_high_conf = (
        transformed_wide.loc[retained_samples, retained_features_high]
        .reset_index()
        .rename(columns={"index": "sample"})
    )

    partc_sample_metadata = bio_meta.loc[bio_meta["sample"].isin(retained_samples)].copy()

    summary = pd.DataFrame(
        {
            "artifact": [
                "retained_biological_samples",
                "retained_main_features",
                "retained_high_confidence_features",
            ],
            "n": [
                len(retained_samples),
                len(retained_features_main),
                len(retained_features_high),
            ],
        }
    )

    return {
        "retained_sample_manifest": retained_sample_manifest,
        "retained_feature_manifest_main": retained_feature_manifest_main,
        "retained_feature_manifest_high": retained_feature_manifest_high,
        "partc_matrix_main": partc_matrix_main,
        "partc_matrix_high_conf": partc_matrix_high_conf,
        "partc_sample_metadata": partc_sample_metadata,
        "partb_output_summary": summary,
    }
