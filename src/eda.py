from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PREFERRED_CLASS_ORDER = ["B", "SS", "dQC", "QC", "Dunn", "French", "LMU"]
CLASS_PALETTE = {
    "B": "#8c8c8c",
    "SS": "#e41a1c",
    "dQC": "#4daf4a",
    "QC": "#377eb8",
    "Dunn": "#984ea3",
    "French": "#ff7f00",
    "LMU": "#a65628",
}


def get_class_order(classes: list[str]) -> list[str]:
    present = set(classes)
    ordered = [c for c in PREFERRED_CLASS_ORDER if c in present]
    ordered += sorted([c for c in classes if c not in ordered])
    return ordered


def audit_task1_inputs(
    data_matrix: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    acquisition_list: pd.DataFrame,
) -> Dict[str, object]:
    """
    Step 0 audit:
    - alignment of sample IDs and feature IDs
    - duplicates
    - missing order positions
    - sample inventory
    """
    if "sample" not in data_matrix.columns:
        raise ValueError("data_matrix must contain a 'sample' column.")
    if "sample" not in acquisition_list.columns:
        raise ValueError("acquisition_list must contain a 'sample' column.")
    if "feature" not in feature_metadata.columns:
        raise ValueError("feature_metadata must contain a 'feature' column.")

    data_samples = pd.Index(data_matrix["sample"])
    acq_samples = pd.Index(acquisition_list["sample"])

    data_features = pd.Index([c for c in data_matrix.columns if c != "sample"])
    metadata_features = pd.Index(feature_metadata["feature"])

    sample_id_match = set(data_samples) == set(acq_samples)
    feature_id_match = set(data_features) == set(metadata_features)

    sample_duplicates_data = data_samples[data_samples.duplicated()].tolist()
    sample_duplicates_acq = acq_samples[acq_samples.duplicated()].tolist()
    feature_duplicates_metadata = metadata_features[metadata_features.duplicated()].tolist()

    missing_samples_in_acq = sorted(set(data_samples) - set(acq_samples))
    missing_samples_in_data = sorted(set(acq_samples) - set(data_samples))
    missing_features_in_metadata = sorted(set(data_features) - set(metadata_features))
    missing_features_in_matrix = sorted(set(metadata_features) - set(data_features))

    order_min = int(acquisition_list["order"].min())
    order_max = int(acquisition_list["order"].max())
    observed_orders = set(acquisition_list["order"].tolist())
    missing_orders = [o for o in range(order_min, order_max + 1) if o not in observed_orders]

    sample_inventory = (
        acquisition_list.groupby(["batch", "class"], dropna=False)
        .size()
        .rename("n_samples")
        .reset_index()
        .sort_values(["batch", "class"])
    )

    biological_classes = {"Dunn", "French", "LMU"}
    batch_bio_summary = (
        acquisition_list.assign(is_biological=acquisition_list["class"].isin(biological_classes))
        .groupby("batch", dropna=False)["is_biological"]
        .sum()
        .rename("n_biological_samples")
        .reset_index()
    )

    report = {
        "n_samples_data_matrix": len(data_samples),
        "n_samples_acquisition_list": len(acq_samples),
        "n_features_data_matrix": len(data_features),
        "n_features_feature_metadata": len(metadata_features),
        "sample_id_match": sample_id_match,
        "feature_id_match": feature_id_match,
        "sample_duplicates_data_matrix": sample_duplicates_data,
        "sample_duplicates_acquisition_list": sample_duplicates_acq,
        "feature_duplicates_feature_metadata": feature_duplicates_metadata,
        "missing_samples_in_acquisition_list": missing_samples_in_acq,
        "missing_samples_in_data_matrix": missing_samples_in_data,
        "missing_features_in_feature_metadata": missing_features_in_metadata,
        "missing_features_in_data_matrix": missing_features_in_matrix,
        "missing_order_positions": missing_orders,
        "sample_inventory": sample_inventory,
        "batch_biological_summary": batch_bio_summary,
    }
    return report


def sample_inventory_table(acquisition_list: pd.DataFrame) -> pd.DataFrame:
    """
    Wide table: counts per class and batch.
    """
    table = (
        acquisition_list.groupby(["batch", "class"])
        .size()
        .unstack(fill_value=0)
    )
    class_order = get_class_order(table.columns.tolist())
    return table.reindex(columns=class_order)


def plot_run_timeline(acquisition_list: pd.DataFrame) -> plt.Figure:
    """
    Step 1 plot: run-order timeline per batch.
    """
    df = acquisition_list.copy()
    class_order = get_class_order(df["class"].dropna().unique().tolist())
    class_to_y = {cls: i for i, cls in enumerate(class_order)}
    df["class_y"] = df["class"].map(class_to_y)

    batches = sorted(df["batch"].dropna().unique().tolist())
    fig, axes = plt.subplots(
        nrows=len(batches),
        ncols=1,
        figsize=(16, 2.8 * len(batches)),
        sharex=False,
        constrained_layout=True,
    )

    if len(batches) == 1:
        axes = [axes]

    for ax, batch in zip(axes, batches):
        sub = df[df["batch"] == batch].sort_values("order")
        sns.scatterplot(
            data=sub,
            x="order",
            y="class_y",
            hue="class",
            hue_order=class_order,
            palette=CLASS_PALETTE,
            s=90,
            ax=ax,
            edgecolor="black",
            linewidth=0.3,
        )
        ax.set_title(f"Run-order timeline — batch {batch}")
        ax.set_xlabel("Run order")
        ax.set_ylabel("Sample class")
        ax.set_yticks(range(len(class_order)))
        ax.set_yticklabels(class_order)
        ax.legend(title="class", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(alpha=0.2, axis="x")

    return fig


def compute_sample_metrics(
    data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    detection_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Step 2: sample-level global metrics.
    """
    feature_cols = [c for c in data_matrix.columns if c != "sample"]

    wide = data_matrix.set_index("sample")[feature_cols].copy()

    metrics = pd.DataFrame(index=wide.index)
    metrics["total_signal"] = wide.sum(axis=1)
    metrics["log10_total_signal_plus1"] = np.log10(metrics["total_signal"] + 1.0)
    metrics["detected_features"] = (wide > detection_threshold).sum(axis=1)
    metrics["detection_rate"] = metrics["detected_features"] / len(feature_cols)
    metrics["zero_fraction"] = (wide <= detection_threshold).mean(axis=1)
    metrics["mean_feature_intensity_nonzero"] = wide.replace(0, np.nan).mean(axis=1)

    # Correlation of each sample to its class median profile
    merged = metrics.reset_index().rename(columns={"index": "sample"}).merge(
        acquisition_list, on="sample", how="left", validate="one_to_one"
    )

    wide_with_class = wide.merge(
        acquisition_list[["sample", "class"]].drop_duplicates(),
        left_index=True,
        right_on="sample",
        how="left",
    )

    class_medians = (
        wide_with_class.groupby("class")[feature_cols]
        .median()
    )

    corr_values = []
    for sample_id in wide.index:
        sample_class = merged.loc[merged["sample"] == sample_id, "class"].iloc[0]
        if pd.isna(sample_class) or sample_class not in class_medians.index:
            corr_values.append(np.nan)
            continue

        sample_vec = np.log1p(wide.loc[sample_id].astype(float))
        class_vec = np.log1p(class_medians.loc[sample_class].astype(float))
        if np.isclose(sample_vec.std(), 0) or np.isclose(class_vec.std(), 0):
            corr_values.append(np.nan)
        else:
            corr_values.append(sample_vec.corr(class_vec, method="spearman"))

    merged["spearman_corr_to_class_median"] = corr_values
    return merged.sort_values(["batch", "order"]).reset_index(drop=True)


def flag_sample_outliers(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Simple robust outlier flags at sample level.
    """
    df = sample_metrics.copy()

    for col in [
        "log10_total_signal_plus1",
        "detected_features",
        "zero_fraction",
        "spearman_corr_to_class_median",
    ]:
        median = df[col].median(skipna=True)
        mad = np.median(np.abs(df[col] - median))
        if mad == 0 or np.isnan(mad):
            df[f"{col}_robust_z"] = np.nan
        else:
            df[f"{col}_robust_z"] = 0.6745 * (df[col] - median) / mad

    df["outlier_total_signal"] = df["log10_total_signal_plus1_robust_z"].abs() > 3.5
    df["outlier_detected_features"] = df["detected_features_robust_z"].abs() > 3.5
    df["outlier_zero_fraction"] = df["zero_fraction_robust_z"].abs() > 3.5
    df["outlier_class_profile"] = df["spearman_corr_to_class_median_robust_z"].abs() > 3.5

    df["any_outlier_flag"] = df[
        [
            "outlier_total_signal",
            "outlier_detected_features",
            "outlier_zero_fraction",
            "outlier_class_profile",
        ]
    ].any(axis=1)

    return df


def plot_global_signal_health(sample_metrics: pd.DataFrame) -> plt.Figure:
    """
    Step 2 plots:
    - total signal by class
    - total signal vs run order
    - detected features by class
    - zero fraction by class
    """
    df = sample_metrics.copy()
    class_order = get_class_order(df["class"].dropna().unique().tolist())

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    sns.boxplot(
        data=df,
        x="class",
        y="log10_total_signal_plus1",
        order=class_order,
        palette=CLASS_PALETTE,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Global signal by class")
    axes[0, 0].set_xlabel("Class")
    axes[0, 0].set_ylabel("log10(total signal + 1)")

    sns.scatterplot(
        data=df,
        x="order",
        y="log10_total_signal_plus1",
        hue="class",
        style="batch",
        palette=CLASS_PALETTE,
        s=80,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Global signal over run order")
    axes[0, 1].set_xlabel("Run order")
    axes[0, 1].set_ylabel("log10(total signal + 1)")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.boxplot(
        data=df,
        x="class",
        y="detected_features",
        order=class_order,
        palette=CLASS_PALETTE,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Detected features by class")
    axes[1, 0].set_xlabel("Class")
    axes[1, 0].set_ylabel("Number of detected features")

    sns.boxplot(
        data=df,
        x="class",
        y="zero_fraction",
        order=class_order,
        palette=CLASS_PALETTE,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Zero fraction by class")
    axes[1, 1].set_xlabel("Class")
    axes[1, 1].set_ylabel("Fraction of zero / undetected features")

    return fig


def summarise_sample_metrics(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Compact summary table by class.
    """
    summary = (
        sample_metrics.groupby(["batch", "class"])
        .agg(
            n_samples=("sample", "size"),
            median_total_signal=("total_signal", "median"),
            mean_total_signal=("total_signal", "mean"),
            median_detected_features=("detected_features", "median"),
            mean_detected_features=("detected_features", "mean"),
            median_zero_fraction=("zero_fraction", "median"),
            mean_corr_to_class_median=("spearman_corr_to_class_median", "mean"),
        )
        .reset_index()
        .sort_values(["batch", "class"])
    )
    return summary

# =========================
# Step 3 — QC stability
# Step 4 — Detection
# Step 5 — Contamination
# Step 6 — Standards
# =========================

def get_feature_columns(data_matrix: pd.DataFrame) -> list[str]:
    """Return feature columns from the data matrix."""
    return [c for c in data_matrix.columns if c != "sample"]


def get_wide_matrix(data_matrix: pd.DataFrame) -> pd.DataFrame:
    """Return sample-indexed feature matrix."""
    feature_cols = get_feature_columns(data_matrix)
    return data_matrix.set_index("sample")[feature_cols].copy()


def get_samples_by_class(
    acquisition_list: pd.DataFrame,
    classes: tuple[str, ...] | list[str],
) -> list[str]:
    """Return sample IDs for the requested classes."""
    return (
        acquisition_list.loc[acquisition_list["class"].isin(classes), "sample"]
        .drop_duplicates()
        .tolist()
    )


def compute_qc_feature_metrics(
    data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    feature_metadata: pd.DataFrame | None = None,
    qc_classes: tuple[str, ...] = ("QC", "dQC"),
    detection_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Compute feature-wise QC stability metrics.

    Outputs include:
    - QC mean / std / CV
    - QC detection rate
    - per-batch QC CV
    - Spearman correlation with run order as a drift proxy
    """
    feature_cols = get_feature_columns(data_matrix)
    wide = get_wide_matrix(data_matrix)

    qc_meta = (
        acquisition_list.loc[acquisition_list["class"].isin(qc_classes), ["sample", "batch", "class", "order"]]
        .drop_duplicates()
        .set_index("sample")
        .sort_values(["batch", "order"])
    )
    qc_samples = qc_meta.index.tolist()
    qc_wide = wide.loc[qc_samples, feature_cols].copy()

    out = pd.DataFrame({"feature": feature_cols})
    out["n_qc_samples"] = len(qc_samples)
    out["qc_detection_rate"] = (qc_wide > detection_threshold).mean(axis=0).values
    out["qc_mean_intensity"] = qc_wide.mean(axis=0).values
    out["qc_median_intensity"] = qc_wide.median(axis=0).values
    out["qc_std_intensity"] = qc_wide.std(axis=0, ddof=1).values

    mean_vals = out["qc_mean_intensity"].to_numpy(dtype=float)
    std_vals = out["qc_std_intensity"].to_numpy(dtype=float)
    out["qc_cv_pct"] = np.where(mean_vals > 0, 100.0 * std_vals / mean_vals, np.nan)

    # Per-batch CVs
    for batch in sorted(qc_meta["batch"].dropna().unique().tolist()):
        batch_samples = qc_meta.index[qc_meta["batch"] == batch].tolist()
        batch_wide = qc_wide.loc[batch_samples]
        batch_mean = batch_wide.mean(axis=0).to_numpy(dtype=float)
        batch_std = batch_wide.std(axis=0, ddof=1).to_numpy(dtype=float)
        out[f"qc_mean_intensity_batch{batch}"] = batch_mean
        out[f"qc_cv_pct_batch{batch}"] = np.where(batch_mean > 0, 100.0 * batch_std / batch_mean, np.nan)

    # Drift proxy: Spearman correlation with run order
    order_series = qc_meta["order"].astype(float)
    drift_rho = []
    for feature in feature_cols:
        vals = np.log1p(qc_wide[feature].astype(float))
        if vals.nunique() <= 1 or order_series.nunique() <= 1:
            drift_rho.append(np.nan)
        else:
            drift_rho.append(vals.corr(order_series, method="spearman"))
    out["qc_order_spearman_rho"] = drift_rho
    out["qc_order_spearman_abs"] = out["qc_order_spearman_rho"].abs()

    if feature_metadata is not None:
        out = out.merge(feature_metadata, on="feature", how="left", validate="one_to_one")

    return out.sort_values(["qc_cv_pct", "feature"], na_position="last").reset_index(drop=True)


def summarise_qc_metrics(
    qc_feature_metrics: pd.DataFrame,
    cv_threshold: float = 30.0,
) -> pd.Series:
    """Compact QC summary for reporting later."""
    summary = pd.Series(
        {
            "n_features": int(qc_feature_metrics["feature"].nunique()),
            "median_qc_cv_pct": float(qc_feature_metrics["qc_cv_pct"].median(skipna=True)),
            "mean_qc_cv_pct": float(qc_feature_metrics["qc_cv_pct"].mean(skipna=True)),
            "n_features_qc_cv_below_threshold": int((qc_feature_metrics["qc_cv_pct"] < cv_threshold).sum()),
            "fraction_features_qc_cv_below_threshold": float((qc_feature_metrics["qc_cv_pct"] < cv_threshold).mean()),
            "median_qc_detection_rate": float(qc_feature_metrics["qc_detection_rate"].median(skipna=True)),
        },
        name="value",
    )
    return summary


def compute_qc_sample_correlations(
    data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    qc_classes: tuple[str, ...] = ("QC", "dQC"),
    method: str = "spearman",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute sample-sample correlation matrix among QC samples.
    """
    feature_cols = get_feature_columns(data_matrix)
    wide = np.log1p(get_wide_matrix(data_matrix)[feature_cols])

    qc_meta = (
        acquisition_list.loc[acquisition_list["class"].isin(qc_classes), ["sample", "batch", "class", "order"]]
        .drop_duplicates()
        .sort_values(["batch", "order"])
    )

    qc_samples = qc_meta["sample"].tolist()
    qc_wide = wide.loc[qc_samples, feature_cols].copy()

    corr = qc_wide.T.corr(method=method)
    return corr, qc_meta.reset_index(drop=True)


def plot_qc_correlation_heatmap(
    qc_corr: pd.DataFrame,
    qc_sample_info: pd.DataFrame,
) -> plt.Figure:
    """
    Plot heatmap of QC sample-sample correlations.
    """
    sample_order = qc_sample_info.sort_values(["batch", "order"])["sample"].tolist()
    corr = qc_corr.loc[sample_order, sample_order]

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    sns.heatmap(
        corr,
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"label": "Spearman correlation"},
    )
    ax.set_title("QC sample-sample correlation heatmap")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Sample")
    return fig


def plot_qc_cv_distribution(
    qc_feature_metrics: pd.DataFrame,
    cv_threshold: float = 30.0,
) -> plt.Figure:
    """
    Plot QC CV distribution and QC CV vs mean intensity.
    """
    df = qc_feature_metrics.copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    sns.histplot(
        data=df,
        x="qc_cv_pct",
        bins=40,
        ax=axes[0],
    )
    axes[0].axvline(cv_threshold, color="red", linestyle="--", linewidth=2, label=f"{cv_threshold:.0f}% threshold")
    axes[0].set_title("Distribution of feature-wise QC CV")
    axes[0].set_xlabel("QC CV (%)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    sns.scatterplot(
        data=df,
        x=np.log10(df["qc_mean_intensity"] + 1.0),
        y="qc_cv_pct",
        alpha=0.7,
        s=40,
        ax=axes[1],
    )
    axes[1].axhline(cv_threshold, color="red", linestyle="--", linewidth=2)
    axes[1].set_title("QC CV vs mean QC intensity")
    axes[1].set_xlabel("log10(mean QC intensity + 1)")
    axes[1].set_ylabel("QC CV (%)")

    return fig


def compute_qc_pca(
    data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    qc_classes: tuple[str, ...] = ("QC", "dQC"),
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute PCA on QC-only samples to inspect multivariate drift/stability.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    feature_cols = get_feature_columns(data_matrix)
    wide = np.log1p(get_wide_matrix(data_matrix)[feature_cols])

    qc_meta = (
        acquisition_list.loc[acquisition_list["class"].isin(qc_classes), ["sample", "batch", "class", "order"]]
        .drop_duplicates()
        .sort_values(["batch", "order"])
    )
    qc_samples = qc_meta["sample"].tolist()
    X = wide.loc[qc_samples].copy()

    # Remove constant features
    keep_cols = X.columns[X.std(axis=0) > 0]
    X = X[keep_cols]

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)

    scores = qc_meta.copy()
    scores["PC1"] = pcs[:, 0]
    scores["PC2"] = pcs[:, 1]

    return scores.reset_index(drop=True), pca.explained_variance_ratio_


def plot_qc_pca(
    qc_pca_scores: pd.DataFrame,
    explained_variance_ratio: np.ndarray,
) -> plt.Figure:
    """
    Plot QC PCA scores and QC PC1 over run order.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    sns.scatterplot(
        data=qc_pca_scores,
        x="PC1",
        y="PC2",
        hue="order",
        style="batch",
        palette="viridis",
        s=100,
        ax=axes[0],
    )
    # connect points within batch to visualize trajectory
    for batch, sub in qc_pca_scores.sort_values("order").groupby("batch"):
        axes[0].plot(sub["PC1"], sub["PC2"], alpha=0.5, linewidth=1)

    axes[0].set_title(
        f"QC PCA (PC1 {100*explained_variance_ratio[0]:.1f}%, "
        f"PC2 {100*explained_variance_ratio[1]:.1f}%)"
    )
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.scatterplot(
        data=qc_pca_scores,
        x="order",
        y="PC1",
        hue="class",
        style="batch",
        palette=CLASS_PALETTE,
        s=100,
        ax=axes[1],
    )
    axes[1].set_title("QC PC1 over run order")
    axes[1].set_xlabel("Run order")
    axes[1].set_ylabel("PC1")
    axes[1].grid(alpha=0.2)
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    return fig


def plot_top_qc_drift_features(
    data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    qc_feature_metrics: pd.DataFrame,
    qc_classes: tuple[str, ...] = ("QC", "dQC"),
    top_n: int = 12,
) -> plt.Figure:
    """
    Plot the top features with strongest absolute QC-order drift.
    """
    feature_cols = get_feature_columns(data_matrix)
    wide = np.log1p(get_wide_matrix(data_matrix)[feature_cols])

    qc_meta = (
        acquisition_list.loc[acquisition_list["class"].isin(qc_classes), ["sample", "batch", "class", "order"]]
        .drop_duplicates()
        .sort_values(["batch", "order"])
    )
    qc_samples = qc_meta["sample"].tolist()
    qc_wide = wide.loc[qc_samples].copy()

    top_features = (
        qc_feature_metrics.sort_values("qc_order_spearman_abs", ascending=False)["feature"]
        .head(top_n)
        .tolist()
    )

    ncols = 3
    nrows = int(np.ceil(len(top_features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.2 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for ax, feature in zip(axes, top_features):
        plot_df = qc_meta.copy()
        plot_df["log1p_intensity"] = qc_wide[feature].values
        sns.lineplot(
            data=plot_df,
            x="order",
            y="log1p_intensity",
            hue="batch",
            marker="o",
            ax=ax,
        )
        rho = qc_feature_metrics.loc[qc_feature_metrics["feature"] == feature, "qc_order_spearman_rho"].iloc[0]
        ax.set_title(f"{feature} (rho={rho:.2f})")
        ax.set_xlabel("Run order")
        ax.set_ylabel("log1p intensity")
        ax.grid(alpha=0.2)

    for ax in axes[len(top_features):]:
        ax.axis("off")

    return fig


def plot_positive_intensity_distributions(
    data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    sample_classes: tuple[str, ...] | None = None,
) -> plt.Figure:
    """
    Plot positive intensity distributions by class to help choose a detection threshold.
    """
    feature_cols = get_feature_columns(data_matrix)
    long_df = (
        data_matrix.melt(id_vars="sample", value_vars=feature_cols, var_name="feature", value_name="intensity")
        .merge(acquisition_list[["sample", "class", "batch", "order"]], on="sample", how="left")
    )

    long_df = long_df.loc[long_df["intensity"] > 0].copy()
    if sample_classes is not None:
        long_df = long_df.loc[long_df["class"].isin(sample_classes)].copy()

    long_df["log10_intensity_plus1"] = np.log10(long_df["intensity"] + 1.0)
    class_order = get_class_order(long_df["class"].dropna().unique().tolist())

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    sns.kdeplot(
        data=long_df,
        x="log10_intensity_plus1",
        hue="class",
        hue_order=class_order,
        common_norm=False,
        fill=False,
        ax=ax,
    )
    ax.set_title("Positive intensity distributions by class")
    ax.set_xlabel("log10(intensity + 1)")
    ax.set_ylabel("Density")
    return fig


def threshold_sensitivity_table(
    data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.0, 1.0, 10.0, 100.0, 1000.0),
) -> pd.DataFrame:
    """
    Compare sample-level detection rates across candidate thresholds.
    """
    feature_cols = get_feature_columns(data_matrix)
    wide = get_wide_matrix(data_matrix)[feature_cols]

    rows = []
    for thr in thresholds:
        det = (wide > thr).astype(int)
        sample_det = det.mean(axis=1).rename("sample_detection_rate").reset_index().rename(columns={"index": "sample"})
        merged = sample_det.merge(acquisition_list[["sample", "class", "batch"]], on="sample", how="left")
        grouped = (
            merged.groupby(["class", "batch"], dropna=False)["sample_detection_rate"]
            .agg(["mean", "median", "min", "max"])
            .reset_index()
        )
        grouped["threshold"] = thr
        rows.append(grouped)

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["threshold", "batch", "class"]).reset_index(drop=True)


def compute_feature_detection_metrics(
    data_matrix: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    detection_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Compute feature-wise detection rates overall and by class.
    """
    feature_cols = get_feature_columns(data_matrix)
    wide = get_wide_matrix(data_matrix)[feature_cols]
    det = (wide > detection_threshold).astype(int)

    out = pd.DataFrame({"feature": feature_cols})
    out["detection_rate__all"] = det.mean(axis=0).values

    classes = acquisition_list["class"].dropna().unique().tolist()
    for cls in get_class_order(classes):
        cls_samples = acquisition_list.loc[acquisition_list["class"] == cls, "sample"].drop_duplicates().tolist()
        out[f"detection_rate__{cls}"] = det.loc[cls_samples].mean(axis=0).values

    bio_samples = acquisition_list.loc[acquisition_list["class"].isin(["Dunn", "French", "LMU"]), "sample"].drop_duplicates().tolist()
    qc_samples = acquisition_list.loc[acquisition_list["class"].isin(["QC", "dQC"]), "sample"].drop_duplicates().tolist()
    blank_samples = acquisition_list.loc[acquisition_list["class"] == "B", "sample"].drop_duplicates().tolist()

    out["detection_rate__biological"] = det.loc[bio_samples].mean(axis=0).values if bio_samples else np.nan
    out["detection_rate__qc_like"] = det.loc[qc_samples].mean(axis=0).values if qc_samples else np.nan
    out["detection_rate__blank"] = det.loc[blank_samples].mean(axis=0).values if blank_samples else np.nan

    out = out.merge(feature_metadata, on="feature", how="left", validate="one_to_one")
    return out.sort_values(["detection_rate__all", "feature"], ascending=[False, True]).reset_index(drop=True)


def plot_detection_rate_by_class(
    feature_detection_metrics: pd.DataFrame,
) -> plt.Figure:
    """
    Plot feature-wise detection-rate distributions by class.
    """
    rate_cols = [c for c in feature_detection_metrics.columns if c.startswith("detection_rate__") and c not in {
        "detection_rate__all",
        "detection_rate__biological",
        "detection_rate__qc_like",
        "detection_rate__blank",
    }]

    long_df = feature_detection_metrics.melt(
        id_vars=["feature"],
        value_vars=rate_cols,
        var_name="rate_col",
        value_name="detection_rate",
    )
    long_df["class"] = long_df["rate_col"].str.replace("detection_rate__", "", regex=False)
    class_order = get_class_order(long_df["class"].tolist())

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    sns.boxplot(
        data=long_df,
        x="class",
        y="detection_rate",
        order=class_order,
        palette=CLASS_PALETTE,
        ax=ax,
    )
    ax.set_title("Feature-wise detection rate by class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Detection rate")
    return fig


def plot_detection_landscape(
    feature_detection_metrics: pd.DataFrame,
    rate_col: str = "detection_rate__all",
) -> plt.Figure:
    """
    Plot RT vs m/z colored by detection rate.
    """
    df = feature_detection_metrics.copy()

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    scatter = ax.scatter(
        df["rt"],
        df["mz"],
        c=df[rate_col],
        cmap="viridis",
        s=55,
        alpha=0.85,
        edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(rate_col)
    ax.set_title(f"Detection landscape colored by {rate_col}")
    ax.set_xlabel("Retention time")
    ax.set_ylabel("m/z")
    return fig


def compute_contamination_metrics(
    data_matrix: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    detection_threshold: float = 0.0,
    blank_classes: tuple[str, ...] = ("B",),
    qc_classes: tuple[str, ...] = ("QC", "dQC"),
    bio_classes: tuple[str, ...] = ("Dunn", "French", "LMU"),
    blank_ratio_threshold: float = 1.0,
    min_blank_detection_rate: float = 0.1,
) -> pd.DataFrame:
    """
    Compute blank-dominance contamination metrics feature-wise.
    """
    feature_cols = get_feature_columns(data_matrix)
    wide = get_wide_matrix(data_matrix)[feature_cols]
    det = (wide > detection_threshold).astype(int)

    blank_samples = get_samples_by_class(acquisition_list, blank_classes)
    qc_samples = get_samples_by_class(acquisition_list, qc_classes)
    bio_samples = get_samples_by_class(acquisition_list, bio_classes)

    out = pd.DataFrame({"feature": feature_cols})

    def group_stats(samples: list[str], prefix: str) -> None:
        if samples:
            out[f"{prefix}_mean_intensity"] = wide.loc[samples].mean(axis=0).values
            out[f"{prefix}_median_intensity"] = wide.loc[samples].median(axis=0).values
            out[f"{prefix}_detection_rate"] = det.loc[samples].mean(axis=0).values
        else:
            out[f"{prefix}_mean_intensity"] = np.nan
            out[f"{prefix}_median_intensity"] = np.nan
            out[f"{prefix}_detection_rate"] = np.nan

    group_stats(blank_samples, "blank")
    group_stats(qc_samples, "qc")
    group_stats(bio_samples, "bio")

    out["blank_to_qc_ratio"] = (out["blank_mean_intensity"] + 1.0) / (out["qc_mean_intensity"] + 1.0)
    out["blank_to_bio_ratio"] = (out["blank_mean_intensity"] + 1.0) / (out["bio_mean_intensity"] + 1.0)
    out["log10_blank_to_qc_ratio"] = np.log10(out["blank_to_qc_ratio"])
    out["log10_blank_to_bio_ratio"] = np.log10(out["blank_to_bio_ratio"])

    out["blank_dominant_flag"] = (
        (out["blank_to_qc_ratio"] >= blank_ratio_threshold)
        & (out["blank_to_bio_ratio"] >= blank_ratio_threshold)
        & (out["blank_detection_rate"] >= min_blank_detection_rate)
    )

    out = out.merge(feature_metadata, on="feature", how="left", validate="one_to_one")
    return out.sort_values(["blank_dominant_flag", "log10_blank_to_bio_ratio"], ascending=[False, False]).reset_index(drop=True)


def plot_contamination_overview(
    contamination_metrics: pd.DataFrame,
) -> plt.Figure:
    """
    Plot blank-dominance contamination summaries.
    """
    df = contamination_metrics.copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    sns.histplot(
        data=df,
        x="log10_blank_to_bio_ratio",
        bins=40,
        ax=axes[0, 0],
    )
    axes[0, 0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0, 0].set_title("log10(blank / biological mean intensity)")
    axes[0, 0].set_xlabel("log10(blank / bio)")

    axes[0, 1].scatter(
        np.log10(df["bio_mean_intensity"] + 1.0),
        np.log10(df["blank_mean_intensity"] + 1.0),
        c=df["blank_dominant_flag"].map({True: 1, False: 0}),
        cmap="coolwarm",
        alpha=0.8,
        s=45,
    )
    min_val = min(
        np.log10(df["bio_mean_intensity"] + 1.0).min(),
        np.log10(df["blank_mean_intensity"] + 1.0).min(),
    )
    max_val = max(
        np.log10(df["bio_mean_intensity"] + 1.0).max(),
        np.log10(df["blank_mean_intensity"] + 1.0).max(),
    )
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--")
    axes[0, 1].set_title("Blank vs biological mean intensity")
    axes[0, 1].set_xlabel("log10(biological mean intensity + 1)")
    axes[0, 1].set_ylabel("log10(blank mean intensity + 1)")

    scatter = axes[1, 0].scatter(
        df["rt"],
        df["mz"],
        c=df["log10_blank_to_bio_ratio"],
        cmap="coolwarm",
        s=55,
        alpha=0.85,
        edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=axes[1, 0])
    cbar.set_label("log10(blank / biological ratio)")
    axes[1, 0].set_title("Contamination landscape")
    axes[1, 0].set_xlabel("Retention time")
    axes[1, 0].set_ylabel("m/z")

    axes[1, 1].scatter(
        df["bio_detection_rate"],
        df["blank_detection_rate"],
        alpha=0.8,
        s=45,
    )
    axes[1, 1].plot([0, 1], [0, 1], color="black", linestyle="--")
    axes[1, 1].set_title("Blank vs biological detection rate")
    axes[1, 1].set_xlabel("Biological detection rate")
    axes[1, 1].set_ylabel("Blank detection rate")

    return fig


def compute_blank_carryover_proxy(
    data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    blank_classes: tuple[str, ...] = ("B",),
) -> pd.DataFrame:
    """
    Proxy for carryover:
    For blank injections, correlate blank feature intensity with the total signal
    of the immediately preceding injection within each batch.
    """
    feature_cols = get_feature_columns(data_matrix)
    wide = get_wide_matrix(data_matrix)[feature_cols]
    sample_total = wide.sum(axis=1).rename("sample_total_signal")

    meta = acquisition_list[["sample", "class", "batch", "order"]].drop_duplicates().copy()
    meta = meta.merge(sample_total.reset_index(), on="sample", how="left")
    meta = meta.sort_values(["batch", "order"]).reset_index(drop=True)

    meta["prev_sample"] = meta.groupby("batch")["sample"].shift(1)
    meta["prev_class"] = meta.groupby("batch")["class"].shift(1)
    meta["prev_total_signal"] = meta.groupby("batch")["sample_total_signal"].shift(1)

    blank_meta = meta.loc[meta["class"].isin(blank_classes) & meta["prev_total_signal"].notna()].copy()
    blank_samples = blank_meta["sample"].tolist()

    if len(blank_samples) == 0:
        return pd.DataFrame(columns=["feature", "carryover_spearman_rho", "n_blank_contexts"])

    blank_wide = np.log1p(wide.loc[blank_samples, feature_cols])
    prev_signal = np.log1p(blank_meta.set_index("sample").loc[blank_samples, "prev_total_signal"])

    rows = []
    for feature in feature_cols:
        vals = blank_wide[feature]
        if vals.nunique() <= 1 or prev_signal.nunique() <= 1:
            rho = np.nan
        else:
            rho = vals.corr(prev_signal, method="spearman")
        rows.append(
            {
                "feature": feature,
                "carryover_spearman_rho": rho,
                "n_blank_contexts": len(blank_samples),
            }
        )

    out = pd.DataFrame(rows).sort_values("carryover_spearman_rho", ascending=False, na_position="last")
    return out.reset_index(drop=True)


def match_exogenous_standards(
    feature_metadata: pd.DataFrame,
    exogenous_standards: pd.DataFrame,
    ppm_tolerance: float = 5000.0,
    rt_tolerance: float = 90.0,
) -> pd.DataFrame:
    """
    Match each exogenous standard to the nearest observed feature using mz and RT.

    We use a normalized score based on:
    - absolute ppm delta / ppm_tolerance
    - absolute RT delta / rt_tolerance
    """
    rows = []
    for _, std_row in exogenous_standards.iterrows():
        tmp = feature_metadata.copy()
        tmp["compound_id"] = std_row["compound_id"]
        tmp["standard_mz"] = float(std_row["mz"])
        tmp["standard_rt"] = float(std_row["Retention_time"])

        tmp["mz_delta"] = tmp["mz"] - tmp["standard_mz"]
        tmp["ppm_delta"] = 1e6 * tmp["mz_delta"] / tmp["standard_mz"]
        tmp["rt_delta"] = tmp["rt"] - tmp["standard_rt"]
        tmp["match_score"] = (tmp["ppm_delta"].abs() / ppm_tolerance) + (tmp["rt_delta"].abs() / rt_tolerance)

        best = tmp.sort_values("match_score").iloc[0].copy()
        best["within_ppm_tolerance"] = abs(best["ppm_delta"]) <= ppm_tolerance
        best["within_rt_tolerance"] = abs(best["rt_delta"]) <= rt_tolerance
        rows.append(best)

    cols = [
        "compound_id",
        "feature",
        "standard_mz",
        "mz",
        "mz_delta",
        "ppm_delta",
        "standard_rt",
        "rt",
        "rt_delta",
        "match_score",
        "within_ppm_tolerance",
        "within_rt_tolerance",
    ]
    return pd.DataFrame(rows)[cols].sort_values("compound_id").reset_index(drop=True)


def compute_standard_tracking(
    data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    standard_matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract intensity tracking for matched standard features across all samples.
    """
    feature_cols = get_feature_columns(data_matrix)
    wide = get_wide_matrix(data_matrix)[feature_cols]

    rows = []
    meta = acquisition_list[["sample", "class", "batch", "order"]].drop_duplicates()

    for _, match_row in standard_matches.iterrows():
        feature = match_row["feature"]
        tmp = (
            wide[[feature]]
            .rename(columns={feature: "intensity"})
            .reset_index()
            .merge(meta, on="sample", how="left")
        )
        tmp["compound_id"] = match_row["compound_id"]
        tmp["matched_feature"] = feature
        tmp["log10_intensity_plus1"] = np.log10(tmp["intensity"] + 1.0)
        tmp["ppm_delta"] = match_row["ppm_delta"]
        tmp["rt_delta"] = match_row["rt_delta"]
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True).sort_values(["compound_id", "batch", "order"]).reset_index(drop=True)


def summarise_standard_tracking(
    standard_tracking: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize matched standard-feature intensities by compound, class, and batch.
    """
    return (
        standard_tracking.groupby(["compound_id", "matched_feature", "batch", "class"], dropna=False)
        .agg(
            n_samples=("sample", "size"),
            mean_intensity=("intensity", "mean"),
            median_intensity=("intensity", "median"),
            mean_log10_intensity_plus1=("log10_intensity_plus1", "mean"),
        )
        .reset_index()
        .sort_values(["compound_id", "batch", "class"])
    )



# =========================
# Step 7 — Structural redundancy
# Step 8 — Technical vs biological variance
# Step 9 — Multivariate overview
# Also: clearer standards plotting
# =========================

def find_isomer_candidates(
    feature_metadata: pd.DataFrame,
    ppm_tolerance: float = 20.0,
    min_rt_delta: float = 0.10,
) -> pd.DataFrame:
    """
    Identify putative isomer pairs:
    nearly identical m/z but separated in retention time.
    """
    fm = feature_metadata[["feature", "mz", "rt"]].dropna().copy().reset_index(drop=True)

    rows = []
    for i in range(len(fm)):
        for j in range(i + 1, len(fm)):
            row_i = fm.iloc[i]
            row_j = fm.iloc[j]

            mean_mz = (row_i["mz"] + row_j["mz"]) / 2.0
            ppm_delta = 1e6 * (row_j["mz"] - row_i["mz"]) / mean_mz
            rt_delta = abs(row_j["rt"] - row_i["rt"])

            if abs(ppm_delta) <= ppm_tolerance and rt_delta >= min_rt_delta:
                rows.append(
                    {
                        "feature_a": row_i["feature"],
                        "feature_b": row_j["feature"],
                        "mz_a": row_i["mz"],
                        "mz_b": row_j["mz"],
                        "rt_a": row_i["rt"],
                        "rt_b": row_j["rt"],
                        "ppm_delta": ppm_delta,
                        "rt_delta": rt_delta,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=["feature_a", "feature_b", "mz_a", "mz_b", "rt_a", "rt_b", "ppm_delta", "rt_delta"]
        )
    return out.sort_values(["ppm_delta", "rt_delta"], key=lambda s: s.abs() if s.name == "ppm_delta" else s).reset_index(drop=True)


def find_isotope_adduct_candidates(
    feature_metadata: pd.DataFrame,
    rt_tolerance: float = 0.15,
    delta_tolerance: float = 0.10,
    known_deltas: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Identify putative isotope/adduct pairs:
    nearly identical RT and mass differences compatible with known shifts.
    """
    if known_deltas is None:
        known_deltas = {
            "M+1 isotope": 1.003355,
            "2x M+1 isotope": 2.006710,
            "Na-H adduct shift": 21.981943,
            "K-H adduct shift": 37.955882,
            "NH4-H adduct shift": 16.018724,
        }

    fm = feature_metadata[["feature", "mz", "rt"]].dropna().copy().reset_index(drop=True)

    rows = []
    for i in range(len(fm)):
        for j in range(i + 1, len(fm)):
            row_i = fm.iloc[i]
            row_j = fm.iloc[j]

            rt_delta = abs(row_j["rt"] - row_i["rt"])
            if rt_delta > rt_tolerance:
                continue

            mz_delta = abs(row_j["mz"] - row_i["mz"])

            best_name = None
            best_delta = None
            best_error = None

            for name, expected_delta in known_deltas.items():
                err = mz_delta - expected_delta
                if best_error is None or abs(err) < abs(best_error):
                    best_name = name
                    best_delta = expected_delta
                    best_error = err

            if best_error is not None and abs(best_error) <= delta_tolerance:
                lower, upper = (row_i, row_j) if row_i["mz"] <= row_j["mz"] else (row_j, row_i)
                rows.append(
                    {
                        "feature_a": lower["feature"],
                        "feature_b": upper["feature"],
                        "mz_a": lower["mz"],
                        "mz_b": upper["mz"],
                        "rt_a": lower["rt"],
                        "rt_b": upper["rt"],
                        "rt_delta": rt_delta,
                        "mz_delta": mz_delta,
                        "delta_type": best_name,
                        "expected_delta": best_delta,
                        "delta_error": best_error,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "feature_a", "feature_b", "mz_a", "mz_b", "rt_a", "rt_b",
                "rt_delta", "mz_delta", "delta_type", "expected_delta", "delta_error"
            ]
        )
    return out.sort_values(["rt_delta", "delta_error"], key=lambda s: s.abs() if s.name == "delta_error" else s).reset_index(drop=True)


def compute_candidate_pair_correlations(
    data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    Compute pairwise intensity correlations for candidate feature pairs
    across all samples, QC-like samples, and biological samples.
    """
    if candidate_pairs.empty:
        return candidate_pairs.copy()

    feature_cols = get_feature_columns(data_matrix)
    wide = np.log1p(get_wide_matrix(data_matrix)[feature_cols])

    sample_subsets = {
        "all": acquisition_list["sample"].drop_duplicates().tolist(),
        "qc_like": acquisition_list.loc[acquisition_list["class"].isin(["QC", "dQC"]), "sample"].drop_duplicates().tolist(),
        "biological_all": acquisition_list.loc[acquisition_list["class"].isin(["Dunn", "French", "LMU"]), "sample"].drop_duplicates().tolist(),
        "batch1_biological": acquisition_list.loc[
            (acquisition_list["batch"] == 1) & acquisition_list["class"].isin(["Dunn", "French", "LMU"]),
            "sample"
        ].drop_duplicates().tolist(),
    }

    out = candidate_pairs.copy()

    for subset_name, samples in sample_subsets.items():
        corrs = []
        ns = []

        for _, row in out.iterrows():
            f1 = row["feature_a"]
            f2 = row["feature_b"]

            sub = wide.loc[samples, [f1, f2]].dropna()
            ns.append(len(sub))

            if len(sub) < 3 or sub[f1].nunique() <= 1 or sub[f2].nunique() <= 1:
                corrs.append(np.nan)
            else:
                corrs.append(sub[f1].corr(sub[f2], method=method))

        out[f"n_samples__{subset_name}"] = ns
        out[f"corr_{method}__{subset_name}"] = corrs

    return out


def plot_pair_correlation_distributions(
    candidate_pairs: pd.DataFrame,
    title: str,
    corr_col: str = "corr_spearman__biological_all",
) -> plt.Figure:
    """
    Plot correlation distribution for candidate feature pairs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    sns.histplot(
        data=candidate_pairs,
        x=corr_col,
        bins=30,
        ax=axes[0],
    )
    axes[0].set_title(f"{title} — correlation distribution")
    axes[0].set_xlabel(corr_col)
    axes[0].set_ylabel("Count")

    if "rt_delta" in candidate_pairs.columns and "ppm_delta" in candidate_pairs.columns:
        axes[1].scatter(
            candidate_pairs["rt_delta"],
            candidate_pairs[corr_col],
            alpha=0.7,
            s=40,
        )
        axes[1].set_xlabel("RT delta")
    elif "rt_delta" in candidate_pairs.columns and "delta_error" in candidate_pairs.columns:
        axes[1].scatter(
            candidate_pairs["rt_delta"],
            candidate_pairs[corr_col],
            alpha=0.7,
            s=40,
        )
        axes[1].set_xlabel("RT delta")
    else:
        axes[1].scatter(
            range(len(candidate_pairs)),
            candidate_pairs[corr_col],
            alpha=0.7,
            s=40,
        )
        axes[1].set_xlabel("Pair index")

    axes[1].set_title(f"{title} — correlation vs metadata separation")
    axes[1].set_ylabel(corr_col)

    return fig


def plot_top_pair_intensity_scatter(
    data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    top_n: int = 6,
    corr_col: str = "corr_spearman__batch1_biological",
) -> plt.Figure:
    """
    Plot intensity-intensity scatter for the top correlated candidate pairs
    in batch 1 biological samples.
    """
    feature_cols = get_feature_columns(data_matrix)
    wide = np.log1p(get_wide_matrix(data_matrix)[feature_cols])

    meta = acquisition_list.loc[
        (acquisition_list["batch"] == 1) & acquisition_list["class"].isin(["Dunn", "French", "LMU"]),
        ["sample", "class"]
    ].drop_duplicates()

    top_pairs = (
        candidate_pairs.sort_values(corr_col, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    ncols = 3
    nrows = int(np.ceil(len(top_pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.2 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, top_pairs.iterrows()):
        f1, f2 = row["feature_a"], row["feature_b"]
        plot_df = (
            wide[[f1, f2]]
            .reset_index()
            .merge(meta, on="sample", how="inner")
        )

        sns.scatterplot(
            data=plot_df,
            x=f1,
            y=f2,
            hue="class",
            palette=CLASS_PALETTE,
            s=70,
            ax=ax,
        )
        ax.set_title(f"{f1} vs {f2}\n{corr_col}={row[corr_col]:.2f}")
        ax.set_xlabel(f"log1p({f1})")
        ax.set_ylabel(f"log1p({f2})")
        ax.legend().remove()

    for ax in axes[len(top_pairs):]:
        ax.axis("off")

    return fig


def compute_technical_biological_metrics(
    data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    qc_feature_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-feature technical vs biological variability summaries.

    Technical variation:
    - QC CV
    - QC log-scale SD

    Biological variation:
    - within-class CV and log-scale SD for Dunn/French/LMU
    - median biological CV / log SD across the biological classes

    D-ratio-style metrics:
    - qc_cv / median_bio_cv
    - qc_log_sd / median_bio_log_sd
    """
    feature_cols = get_feature_columns(data_matrix)
    wide_raw = get_wide_matrix(data_matrix)[feature_cols]
    wide_log = np.log1p(wide_raw)

    out = qc_feature_metrics[
        ["feature", "qc_mean_intensity", "qc_cv_pct", "qc_detection_rate", "qc_order_spearman_rho", "qc_order_spearman_abs", "mz", "rt"]
    ].copy()

    qc_samples = acquisition_list.loc[acquisition_list["class"].isin(["QC", "dQC"]), "sample"].drop_duplicates().tolist()
    qc_log_sd = wide_log.loc[qc_samples].std(axis=0, ddof=1).rename("qc_log_sd").reset_index().rename(columns={"index": "feature"})
    out = out.merge(qc_log_sd, on="feature", how="left")

    bio_classes = ["Dunn", "French", "LMU"]

    for cls in bio_classes:
        cls_samples = acquisition_list.loc[
            (acquisition_list["batch"] == 1) & (acquisition_list["class"] == cls),
            "sample"
        ].drop_duplicates().tolist()

        raw_vals = wide_raw.loc[cls_samples]
        log_vals = wide_log.loc[cls_samples]

        class_mean = raw_vals.mean(axis=0).rename(f"{cls}_mean_intensity")
        class_std = raw_vals.std(axis=0, ddof=1).rename(f"{cls}_std_intensity")
        class_cv = (100.0 * class_std / class_mean).rename(f"{cls}_cv_pct")
        class_log_sd = log_vals.std(axis=0, ddof=1).rename(f"{cls}_log_sd")
        class_det = (raw_vals > 0).mean(axis=0).rename(f"{cls}_detection_rate")

        cls_df = pd.concat([class_mean, class_std, class_cv, class_log_sd, class_det], axis=1).reset_index().rename(columns={"index": "feature"})
        out = out.merge(cls_df, on="feature", how="left")

    bio_cv_cols = [f"{cls}_cv_pct" for cls in bio_classes]
    bio_log_sd_cols = [f"{cls}_log_sd" for cls in bio_classes]
    bio_det_cols = [f"{cls}_detection_rate" for cls in bio_classes]

    out["bio_cv_pct_median"] = out[bio_cv_cols].median(axis=1, skipna=True)
    out["bio_cv_pct_max"] = out[bio_cv_cols].max(axis=1, skipna=True)
    out["bio_log_sd_median"] = out[bio_log_sd_cols].median(axis=1, skipna=True)
    out["bio_detection_rate_median"] = out[bio_det_cols].median(axis=1, skipna=True)
    out["bio_detection_rate_min"] = out[bio_det_cols].min(axis=1, skipna=True)

    out["d_ratio_cv"] = out["qc_cv_pct"] / out["bio_cv_pct_median"]
    out["d_ratio_log"] = out["qc_log_sd"] / out["bio_log_sd_median"]

    out["passes_qc_cv_30"] = out["qc_cv_pct"] < 30.0
    out["passes_detection_70"] = out["bio_detection_rate_min"] >= 0.70
    out["passes_mz_500"] = out["mz"] > 500.0
    out["passes_basic_part_b_screen"] = out["passes_qc_cv_30"] & out["passes_detection_70"] & out["passes_mz_500"]

    return out.sort_values(["d_ratio_log", "qc_cv_pct"], na_position="last").reset_index(drop=True)


def compute_acceptance_counts(
    technical_biological_metrics: pd.DataFrame,
    qc_thresholds: tuple[float, ...] = (20.0, 30.0, 40.0),
    d_ratio_thresholds: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25),
    detection_thresholds: tuple[float, ...] = (0.5, 0.7, 0.9),
    mz_min: float = 500.0,
) -> pd.DataFrame:
    """
    Count how many features remain under increasingly strict technical criteria.
    """
    df = technical_biological_metrics.copy()
    rows = []

    for qc_thr in qc_thresholds:
        n = ((df["qc_cv_pct"] < qc_thr) & (df["bio_detection_rate_min"] >= 0.7) & (df["mz"] > mz_min)).sum()
        rows.append({"criterion_type": "qc_cv_threshold", "threshold": qc_thr, "n_features": int(n)})

    for d_thr in d_ratio_thresholds:
        n = ((df["d_ratio_log"] < d_thr) & (df["bio_detection_rate_min"] >= 0.7) & (df["mz"] > mz_min)).sum()
        rows.append({"criterion_type": "d_ratio_log_threshold", "threshold": d_thr, "n_features": int(n)})

    for det_thr in detection_thresholds:
        n = ((df["qc_cv_pct"] < 30.0) & (df["bio_detection_rate_min"] >= det_thr) & (df["mz"] > mz_min)).sum()
        rows.append({"criterion_type": "bio_detection_threshold", "threshold": det_thr, "n_features": int(n)})

    return pd.DataFrame(rows)


def plot_technical_vs_biological_metrics(
    technical_biological_metrics: pd.DataFrame,
    qc_cv_threshold: float = 30.0,
) -> plt.Figure:
    """
    Plot technical vs biological variability and D-ratio distribution.
    """
    df = technical_biological_metrics.copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    scatter = axes[0].scatter(
        df["bio_cv_pct_median"],
        df["qc_cv_pct"],
        c=df["d_ratio_log"],
        cmap="viridis",
        alpha=0.8,
        s=50,
        edgecolors="none",
    )
    axes[0].axhline(qc_cv_threshold, color="red", linestyle="--", linewidth=2)
    axes[0].plot([0, max(df["bio_cv_pct_median"].max(), df["qc_cv_pct"].max())],
                 [0, max(df["bio_cv_pct_median"].max(), df["qc_cv_pct"].max())],
                 color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Technical vs biological CV")
    axes[0].set_xlabel("Median within-class biological CV (%)")
    axes[0].set_ylabel("QC CV (%)")
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label("D-ratio (log SD)")

    sns.histplot(
        data=df,
        x="d_ratio_log",
        bins=40,
        ax=axes[1],
    )
    axes[1].axvline(1.0, color="red", linestyle="--", linewidth=2)
    axes[1].set_title("Distribution of D-ratio (technical / biological, log SD)")
    axes[1].set_xlabel("D-ratio (log SD)")
    axes[1].set_ylabel("Count")

    return fig


def plot_acceptance_counts(
    acceptance_counts: pd.DataFrame,
) -> plt.Figure:
    """
    Plot number of retained features under varying technical criteria.
    """
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    sns.lineplot(
        data=acceptance_counts,
        x="threshold",
        y="n_features",
        hue="criterion_type",
        marker="o",
        ax=ax,
    )
    ax.set_title("Retained features under increasingly strict criteria")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Number of retained features")
    ax.grid(alpha=0.2)
    return fig


def compute_global_pca(
    data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    classes: tuple[str, ...] | list[str] | None = None,
    batches: tuple[int, ...] | list[int] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute PCA for a chosen subset of samples.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    meta = acquisition_list[["sample", "class", "batch", "order"]].drop_duplicates().copy()

    if classes is not None:
        meta = meta.loc[meta["class"].isin(classes)].copy()
    if batches is not None:
        meta = meta.loc[meta["batch"].isin(batches)].copy()

    meta = meta.sort_values(["batch", "order"]).reset_index(drop=True)

    feature_cols = get_feature_columns(data_matrix)
    wide = np.log1p(get_wide_matrix(data_matrix)[feature_cols])
    X = wide.loc[meta["sample"]].copy()

    keep_cols = X.columns[X.std(axis=0) > 0]
    X = X[keep_cols]

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)

    scores = meta.copy()
    scores["PC1"] = pcs[:, 0]
    scores["PC2"] = pcs[:, 1]

    return scores, pca.explained_variance_ratio_


def flag_pca_outliers(
    pca_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flag samples far from the global PCA center using a robust distance z-score.
    """
    df = pca_scores.copy()
    pc1_med = df["PC1"].median()
    pc2_med = df["PC2"].median()
    dist = np.sqrt((df["PC1"] - pc1_med) ** 2 + (df["PC2"] - pc2_med) ** 2)
    med = np.median(dist)
    mad = np.median(np.abs(dist - med))

    df["pca_distance"] = dist
    if mad == 0 or np.isnan(mad):
        df["pca_distance_robust_z"] = np.nan
    else:
        df["pca_distance_robust_z"] = 0.6745 * (dist - med) / mad
    df["pca_outlier_flag"] = df["pca_distance_robust_z"].abs() > 3.5
    return df


def plot_global_pca(
    pca_scores: pd.DataFrame,
    explained_variance_ratio: np.ndarray,
    title_prefix: str = "PCA",
) -> plt.Figure:
    """
    Plot PCA scores and PC1 over run order.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    sns.scatterplot(
        data=pca_scores,
        x="PC1",
        y="PC2",
        hue="class",
        style="batch",
        palette=CLASS_PALETTE,
        s=100,
        ax=axes[0],
    )
    axes[0].set_title(
        f"{title_prefix} — PC1 {100*explained_variance_ratio[0]:.1f}%, "
        f"PC2 {100*explained_variance_ratio[1]:.1f}%"
    )
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    sns.scatterplot(
        data=pca_scores,
        x="order",
        y="PC1",
        hue="class",
        style="batch",
        palette=CLASS_PALETTE,
        s=100,
        ax=axes[1],
    )
    axes[1].set_title(f"{title_prefix} — PC1 over run order")
    axes[1].set_xlabel("Run order")
    axes[1].set_ylabel("PC1")
    axes[1].grid(alpha=0.2)
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    return fig


# Clearer replacement for the standards plot
def plot_standard_tracking(
    standard_tracking: pd.DataFrame,
    classes: tuple[str, ...] = ("SS", "dQC", "QC", "B"),
    positive_only: bool = True,
    use_within_batch_order: bool = True,
) -> plt.Figure:
    """
    Clear standards plot focused on technical classes.

    Improvements over the previous version:
    - filters to technical classes by default
    - uses within-batch run order, so batch 2 is not compressed
    - shows positive detections only by default
    - writes 'no positive detections' in empty panels
    - keeps one panel per compound x batch for interpretability
    """
    df = standard_tracking.copy()
    df = df.loc[df["class"].isin(classes)].copy()

    if use_within_batch_order:
        df = df.sort_values(["batch", "order"]).copy()
        df["order_within_batch"] = (
            df.groupby("batch")["order"]
            .rank(method="dense")
            .astype(int)
        )
        x_col = "order_within_batch"
        x_label = "Run order within batch"
    else:
        x_col = "order"
        x_label = "Global run order"

    compounds = df["compound_id"].dropna().unique().tolist()
    batches = sorted(df["batch"].dropna().unique().tolist())

    fig, axes = plt.subplots(
        len(compounds),
        len(batches),
        figsize=(7 * len(batches), 4.0 * len(compounds)),
        sharey="row",
        constrained_layout=True,
    )

    if len(compounds) == 1 and len(batches) == 1:
        axes = np.array([[axes]])
    elif len(compounds) == 1:
        axes = np.array([axes])
    elif len(batches) == 1:
        axes = np.array([[ax] for ax in axes])

    for i, compound in enumerate(compounds):
        for j, batch in enumerate(batches):
            ax = axes[i, j]
            sub = df.loc[
                (df["compound_id"] == compound) & (df["batch"] == batch)
            ].copy()

            if sub.empty:
                ax.text(
                    0.5, 0.5, "no data",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=12
                )
                ax.set_axis_off()
                continue

            matched_feature = sub["matched_feature"].iloc[0]
            ppm_delta = sub["ppm_delta"].iloc[0]
            rt_delta = sub["rt_delta"].iloc[0]

            if positive_only:
                sub_plot = sub.loc[sub["intensity"] > 0].copy()
            else:
                sub_plot = sub.copy()

            if sub_plot.empty:
                ax.text(
                    0.5, 0.5, "no positive detections",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=12
                )
                ax.set_title(
                    f"{compound} — batch {batch}\n"
                    f"{matched_feature} (ppm={ppm_delta:.1f}, rt={rt_delta:.2f})"
                )
                ax.set_xlabel(x_label)
                ax.set_ylabel("log10(intensity + 1)")
                ax.grid(alpha=0.2)
                continue

            sns.scatterplot(
                data=sub_plot,
                x=x_col,
                y="log10_intensity_plus1",
                hue="class",
                hue_order=[c for c in PREFERRED_CLASS_ORDER if c in classes],
                palette=CLASS_PALETTE,
                s=85,
                ax=ax,
            )

            # light trend line per class when enough points exist
            for cls, cls_sub in sub_plot.groupby("class"):
                if len(cls_sub) >= 3:
                    cls_sub = cls_sub.sort_values(x_col)
                    ax.plot(
                        cls_sub[x_col],
                        cls_sub["log10_intensity_plus1"],
                        alpha=0.35,
                        linewidth=1.5,
                        color=CLASS_PALETTE.get(cls, "black"),
                    )

            ax.set_title(
                f"{compound} — batch {batch}\n"
                f"{matched_feature} (ppm={ppm_delta:.1f}, rt={rt_delta:.2f})"
            )
            ax.set_xlabel(x_label)
            ax.set_ylabel("log10(intensity + 1)")
            ax.grid(alpha=0.2)

            # remove per-panel legends
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    # shared legend
    handles = []
    labels = []
    for cls in [c for c in PREFERRED_CLASS_ORDER if c in classes]:
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=CLASS_PALETTE[cls],
                markersize=8,
            )
        )
        labels.append(cls)

    fig.legend(
        handles,
        labels,
        title="class",
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
    )

    return fig

# =========================
# Batch-1 operational EDA helpers
# =========================

def subset_task1_inputs(
    data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    batches: tuple[int, ...] | list[int] | None = None,
    classes: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return a subset of the data matrix and acquisition list using sample-level filters.

    The returned acquisition_list is one row per sample and sorted by batch/order.
    The returned data_matrix is reordered to match acquisition_list sample order.
    """
    acq = acquisition_list.copy()

    if batches is not None:
        acq = acq.loc[acq["batch"].isin(batches)].copy()

    if classes is not None:
        acq = acq.loc[acq["class"].isin(classes)].copy()

    acq = (
        acq.drop_duplicates(subset=["sample"])
        .sort_values(["batch", "order"])
        .reset_index(drop=True)
    )

    keep_samples = acq["sample"].tolist()

    dm = data_matrix.loc[data_matrix["sample"].isin(keep_samples)].copy()
    dm = dm.set_index("sample").loc[keep_samples].reset_index()

    return dm, acq


def make_batch1_operational_subset(
    data_matrix: pd.DataFrame,
    acquisition_list: pd.DataFrame,
    include_classes: tuple[str, ...] = ("B", "SS", "dQC", "QC", "Dunn", "French", "LMU"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper for the batch-1-only operational subset used for Part B / C.
    """
    return subset_task1_inputs(
        data_matrix=data_matrix,
        acquisition_list=acquisition_list,
        batches=(1,),
        classes=include_classes,
    )
