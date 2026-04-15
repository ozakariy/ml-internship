from __future__ import annotations

from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from eda import CLASS_PALETTE, get_class_order
from load_data import get_project_root, load_task1_inputs


def _first_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find any of the expected files:\n"
        + "\n".join(str(p) for p in candidates)
    )


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != "sample"]


def load_partc_ready_inputs(
    project_root: Path | None = None,
    processed_subdir: str = "data/processed/task1_partb_final",
    raw_detection_threshold: float = 10.0,
) -> dict[str, pd.DataFrame | Path | float]:
    """
    Load the curated Part B outputs and rebuild raw/transformed matrices needed for Part C.

    Preference order:
    - final annotated manifests if available
    - otherwise fall back to non-annotated manifests
    """
    root = get_project_root(project_root)
    processed_dir = root / processed_subdir

    sample_manifest = pd.read_csv(
        _first_existing_path([processed_dir / "retained_sample_manifest.csv"])
    )

    feature_manifest_main = pd.read_csv(
        _first_existing_path(
            [
                processed_dir / "retained_feature_manifest_main_annotated.csv",
                processed_dir / "retained_feature_manifest_main_annotated_with_redundancy.csv",
                processed_dir / "retained_feature_manifest_main.csv",
            ]
        )
    )

    feature_manifest_high = pd.read_csv(
        _first_existing_path(
            [
                processed_dir / "retained_feature_manifest_high_annotated.csv",
                processed_dir / "retained_feature_manifest_high.csv",
            ]
        )
    )

    partc_matrix_main = pd.read_csv(
        _first_existing_path([processed_dir / "partc_matrix_main.csv"])
    )

    partc_matrix_high = pd.read_csv(
        _first_existing_path(
            [
                processed_dir / "partc_matrix_high_conf.csv",
                processed_dir / "partc_matrix_high_confidence.csv",
            ]
        )
    )

    partc_sample_metadata = pd.read_csv(
        _first_existing_path([processed_dir / "partc_sample_metadata.csv"])
    )

    inputs = load_task1_inputs(root)
    raw_data_matrix = inputs["data_matrix"].copy()
    feature_metadata = inputs["feature_metadata"].copy()
    acquisition_list = inputs["acquisition_list"].copy()

    retained_samples = partc_sample_metadata["sample"].drop_duplicates().tolist()
    main_features = feature_manifest_main["feature"].drop_duplicates().tolist()
    high_features = feature_manifest_high["feature"].drop_duplicates().tolist()

    if "redundancy_representative_flag" in feature_manifest_main.columns:
        representative_features = (
            feature_manifest_main.loc[
                feature_manifest_main["redundancy_representative_flag"].fillna(False),
                "feature",
            ]
            .drop_duplicates()
            .tolist()
        )
        if len(representative_features) == 0:
            representative_features = main_features
    else:
        representative_features = main_features

    raw_wide = raw_data_matrix.set_index("sample")
    raw_partc_main = raw_wide.loc[retained_samples, main_features].reset_index()
    raw_partc_high = raw_wide.loc[retained_samples, high_features].reset_index()
    raw_partc_representative = raw_wide.loc[
        retained_samples, representative_features
    ].reset_index()

    partc_matrix_representative = partc_matrix_main[
        ["sample", *representative_features]
    ].copy()

    feature_manifest_representative = feature_manifest_main.loc[
        feature_manifest_main["feature"].isin(representative_features)
    ].copy()

    sample_manifest_merged = partc_sample_metadata.merge(
        sample_manifest.drop(
            columns=[
                c
                for c in ["class", "order", "batch"]
                if c in sample_manifest.columns
            ]
        ),
        on="sample",
        how="left",
        validate="one_to_one",
    )

    return {
        "processed_dir": processed_dir,
        "raw_detection_threshold": raw_detection_threshold,
        "sample_manifest": sample_manifest_merged,
        "feature_manifest_main": feature_manifest_main,
        "feature_manifest_high": feature_manifest_high,
        "feature_manifest_representative": feature_manifest_representative,
        "partc_matrix_main": partc_matrix_main,
        "partc_matrix_high": partc_matrix_high,
        "partc_matrix_representative": partc_matrix_representative,
        "raw_partc_main": raw_partc_main,
        "raw_partc_high": raw_partc_high,
        "raw_partc_representative": raw_partc_representative,
        "partc_sample_metadata": partc_sample_metadata,
        "feature_metadata": feature_metadata,
        "acquisition_list": acquisition_list,
    }


def build_partc_task_table() -> pd.DataFrame:
    """
    Define the clinical discrimination tasks for Part C.
    """
    tasks = [
        {
            "task_name": "French_vs_Dunn",
            "task_type": "pairwise",
            "positive_class": "French",
            "negative_class": "Dunn",
            "task_label": "Lung cancer vs healthy",
        },
        {
            "task_name": "French_vs_LMU",
            "task_type": "pairwise",
            "positive_class": "French",
            "negative_class": "LMU",
            "task_label": "Lung cancer vs benign disease",
        },
        {
            "task_name": "LMU_vs_Dunn",
            "task_type": "pairwise",
            "positive_class": "LMU",
            "negative_class": "Dunn",
            "task_label": "Benign disease vs healthy",
        },
        {
            "task_name": "three_class_overview",
            "task_type": "multiclass",
            "positive_class": np.nan,
            "negative_class": np.nan,
            "task_label": "Three-class overview",
        },
    ]
    return pd.DataFrame(tasks)


def compute_partc_pca(
    modeling_matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    n_components: int = 3,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    PCA diagnostic for Part C.
    """
    feature_cols = _feature_columns(modeling_matrix)
    X = modeling_matrix.set_index("sample")[feature_cols].astype(float)

    meta = sample_metadata[["sample", "class", "order", "batch"]].drop_duplicates().copy()
    X = X.loc[meta["sample"]]

    keep_cols = X.columns[X.std(axis=0) > 0]
    X = X[keep_cols]

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    n_components = int(min(n_components, X_scaled.shape[0] - 1, X_scaled.shape[1]))
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_scaled)

    score_cols = [f"PC{i+1}" for i in range(n_components)]
    out = pd.DataFrame(scores, columns=score_cols, index=meta["sample"])
    out = out.reset_index().rename(columns={"index": "sample"})
    out = out.merge(meta, on="sample", how="left", validate="one_to_one")

    return out, pca.explained_variance_ratio_


def plot_partc_pca(
    pca_scores: pd.DataFrame,
    explained_variance_ratio: np.ndarray,
    title_prefix: str,
) -> plt.Figure:
    """
    PCA plot + PC1 over run order.
    """
    class_order = get_class_order(pca_scores["class"].dropna().unique().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    sns.scatterplot(
        data=pca_scores,
        x="PC1",
        y="PC2",
        hue="class",
        hue_order=class_order,
        palette=CLASS_PALETTE,
        s=90,
        ax=axes[0],
    )
    axes[0].set_title(
        f"{title_prefix} — PC1 {100*explained_variance_ratio[0]:.1f}%, "
        f"PC2 {100*explained_variance_ratio[1]:.1f}%"
    )

    sns.scatterplot(
        data=pca_scores,
        x="order",
        y="PC1",
        hue="class",
        hue_order=class_order,
        palette=CLASS_PALETTE,
        s=90,
        ax=axes[1],
    )
    axes[1].set_title(f"{title_prefix} — PC1 over run order")
    axes[1].set_xlabel("Run order")
    axes[1].set_ylabel("PC1")
    axes[1].grid(alpha=0.2)

    for ax in axes:
        if ax.get_legend() is not None:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    return fig


def _bh_adjust(p_values: pd.Series) -> pd.Series:
    """
    Benjamini-Hochberg FDR correction.
    """
    p = pd.Series(p_values, copy=True).astype(float)
    out = pd.Series(np.nan, index=p.index, dtype=float)

    valid = p.notna()
    if valid.sum() == 0:
        return out

    pv = p.loc[valid].to_numpy(dtype=float)
    order = np.argsort(pv)
    ranked = pv[order]
    n = len(ranked)

    adjusted = np.empty(n, dtype=float)
    cumulative_min = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = ranked[i] * n / rank
        cumulative_min = min(cumulative_min, value)
        adjusted[i] = cumulative_min

    adjusted = np.clip(adjusted, 0.0, 1.0)

    reverse = np.empty(n, dtype=float)
    reverse[order] = adjusted
    out.loc[valid] = reverse
    return out


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, scores))


def _cliffs_delta(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    if len(x_pos) == 0 or len(x_neg) == 0:
        return np.nan
    diff = x_pos[:, None] - x_neg[None, :]
    return float(
        (np.sum(diff > 0) - np.sum(diff < 0)) / (len(x_pos) * len(x_neg))
    )


def compute_pairwise_univariate_evidence(
    transformed_matrix: pd.DataFrame,
    raw_matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    feature_manifest: pd.DataFrame,
    positive_class: str,
    negative_class: str,
    raw_detection_threshold: float = 10.0,
    task_name: str | None = None,
) -> pd.DataFrame:
    """
    Compute per-feature univariate evidence for one pairwise task.
    """
    transformed_wide = transformed_matrix.set_index("sample")
    raw_wide = raw_matrix.set_index("sample")

    sample_meta = sample_metadata[["sample", "class", "order", "batch"]].drop_duplicates().copy()
    pos_samples = sample_meta.loc[sample_meta["class"] == positive_class, "sample"].tolist()
    neg_samples = sample_meta.loc[sample_meta["class"] == negative_class, "sample"].tolist()

    features = [
        f
        for f in feature_manifest["feature"].tolist()
        if f in transformed_wide.columns and f in raw_wide.columns
    ]

    rows = []
    for feature in features:
        x_pos = transformed_wide.loc[pos_samples, feature].astype(float).to_numpy()
        x_neg = transformed_wide.loc[neg_samples, feature].astype(float).to_numpy()
        r_pos = raw_wide.loc[pos_samples, feature].astype(float).to_numpy()
        r_neg = raw_wide.loc[neg_samples, feature].astype(float).to_numpy()

        try:
            mw = mannwhitneyu(x_pos, x_neg, alternative="two-sided", method="auto")
            u_stat = float(mw.statistic)
            p_value = float(mw.pvalue)
        except ValueError:
            u_stat = np.nan
            p_value = np.nan

        auc = _safe_auc(
            np.r_[np.ones(len(x_pos)), np.zeros(len(x_neg))],
            np.r_[x_pos, x_neg],
        )
        auc_abs = np.nan if pd.isna(auc) else max(auc, 1.0 - auc)

        n_pos = len(x_pos)
        n_neg = len(x_neg)

        rank_biserial = np.nan
        if pd.notna(u_stat) and n_pos > 0 and n_neg > 0:
            rank_biserial = float((2.0 * u_stat) / (n_pos * n_neg) - 1.0)

        raw_median_pos = float(np.median(r_pos))
        raw_median_neg = float(np.median(r_neg))

        rows.append(
            {
                "task_name": task_name or f"{positive_class}_vs_{negative_class}",
                "positive_class": positive_class,
                "negative_class": negative_class,
                "feature": feature,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "median_transformed_positive": float(np.median(x_pos)),
                "median_transformed_negative": float(np.median(x_neg)),
                "median_diff_transformed": float(np.median(x_pos) - np.median(x_neg)),
                "mean_diff_transformed": float(np.mean(x_pos) - np.mean(x_neg)),
                "raw_median_positive": raw_median_pos,
                "raw_median_negative": raw_median_neg,
                "raw_log2_fc": float(
                    np.log2((raw_median_pos + 1.0) / (raw_median_neg + 1.0))
                ),
                "u_statistic": u_stat,
                "p_value": p_value,
                "auc_positive_class": auc,
                "auc_abs": auc_abs,
                "rank_biserial": rank_biserial,
                "cliffs_delta": _cliffs_delta(x_pos, x_neg),
                "positive_detection_rate_raw": float(np.mean(r_pos > raw_detection_threshold)),
                "negative_detection_rate_raw": float(np.mean(r_neg > raw_detection_threshold)),
            }
        )

    out = pd.DataFrame(rows)
    out["detection_rate_diff_raw"] = (
        out["positive_detection_rate_raw"] - out["negative_detection_rate_raw"]
    )
    out["q_value"] = _bh_adjust(out["p_value"])
    out["neglog10_q"] = -np.log10(out["q_value"].clip(lower=1e-300))
    out["effect_direction"] = np.where(
        out["median_diff_transformed"] >= 0,
        f"{positive_class} > {negative_class}",
        f"{positive_class} < {negative_class}",
    )
    out["significant_fdr_05"] = out["q_value"] < 0.05

    out = out.merge(feature_manifest, on="feature", how="left", validate="one_to_one")
    out = out.sort_values(
        ["significant_fdr_05", "q_value", "auc_abs", "cliffs_delta"],
        ascending=[False, True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    out["rank_within_task"] = np.arange(1, len(out) + 1)
    return out


def build_all_pairwise_univariate_evidence(
    partc_inputs: dict[str, pd.DataFrame | Path | float],
    task_table: pd.DataFrame,
    feature_space: str = "main",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Build all pairwise univariate evidence tables for one feature space.
    """
    if feature_space == "main":
        transformed_matrix = partc_inputs["partc_matrix_main"]
        raw_matrix = partc_inputs["raw_partc_main"]
        feature_manifest = partc_inputs["feature_manifest_main"]
    elif feature_space == "high":
        transformed_matrix = partc_inputs["partc_matrix_high"]
        raw_matrix = partc_inputs["raw_partc_high"]
        feature_manifest = partc_inputs["feature_manifest_high"]
    elif feature_space == "representative":
        transformed_matrix = partc_inputs["partc_matrix_representative"]
        raw_matrix = partc_inputs["raw_partc_representative"]
        feature_manifest = partc_inputs["feature_manifest_representative"]
    else:
        raise ValueError("feature_space must be one of: 'main', 'high', 'representative'.")

    sample_metadata = partc_inputs["partc_sample_metadata"]
    raw_detection_threshold = float(partc_inputs["raw_detection_threshold"])

    per_task: dict[str, pd.DataFrame] = {}
    combined = []

    pairwise_tasks = task_table.loc[task_table["task_type"] == "pairwise"].copy()
    for _, task in pairwise_tasks.iterrows():
        table = compute_pairwise_univariate_evidence(
            transformed_matrix=transformed_matrix,
            raw_matrix=raw_matrix,
            sample_metadata=sample_metadata,
            feature_manifest=feature_manifest,
            positive_class=task["positive_class"],
            negative_class=task["negative_class"],
            raw_detection_threshold=raw_detection_threshold,
            task_name=task["task_name"],
        )
        table["feature_space"] = feature_space
        per_task[task["task_name"]] = table
        combined.append(table)

    combined_df = pd.concat(combined, ignore_index=True)
    return per_task, combined_df


def plot_univariate_screen(
    evidence_table: pd.DataFrame,
    title: str,
    top_n: int = 15,
) -> plt.Figure:
    """
    Summary plot for one pairwise task.
    """
    df = evidence_table.copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    scatter = axes[0].scatter(
        df["raw_log2_fc"],
        df["neglog10_q"],
        c=df["auc_abs"],
        cmap="viridis",
        alpha=0.8,
        s=55,
        edgecolors="none",
    )
    axes[0].axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1.8)
    axes[0].set_title(f"{title} — univariate evidence landscape")
    axes[0].set_xlabel("Raw median log2 fold-change")
    axes[0].set_ylabel("-log10(FDR q-value)")
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label("abs(AUC)")

    top = df.sort_values(
        ["q_value", "auc_abs", "cliffs_delta"],
        ascending=[True, False, False],
    ).head(top_n).copy()
    top = top.iloc[::-1]

    axes[1].barh(top["feature"], top["auc_abs"])
    axes[1].set_title(f"{title} — top {top_n} features by FDR/AUC")
    axes[1].set_xlabel("abs(AUC)")
    axes[1].set_ylabel("Feature")

    return fig


def plot_top_univariate_features(
    transformed_matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    evidence_table: pd.DataFrame,
    top_n: int = 6,
) -> plt.Figure:
    """
    Box/strip plots for the strongest univariate features in one task.
    """
    features = (
        evidence_table.sort_values(
            ["q_value", "auc_abs", "cliffs_delta"],
            ascending=[True, False, False],
        )["feature"].head(top_n).tolist()
    )

    long_df = (
        transformed_matrix[["sample", *features]]
        .melt(id_vars="sample", var_name="feature", value_name="value")
        .merge(
            sample_metadata[["sample", "class"]].drop_duplicates(),
            on="sample",
            how="left",
        )
    )
    class_order = get_class_order(long_df["class"].dropna().unique().tolist())

    ncols = 3
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(16, 4.2 * nrows),
        constrained_layout=True,
    )
    axes = np.array(axes).reshape(-1)

    for ax, feature in zip(axes, features):
        sub = long_df.loc[long_df["feature"] == feature].copy()

        sns.boxplot(
            data=sub,
            x="class",
            y="value",
            order=class_order,
            palette=CLASS_PALETTE,
            ax=ax,
            fliersize=0,
        )
        sns.stripplot(
            data=sub,
            x="class",
            y="value",
            order=class_order,
            ax=ax,
            color="black",
            alpha=0.55,
            size=4,
        )
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylabel("Transformed intensity")
        ax.tick_params(axis="x", rotation=20)

    for ax in axes[len(features):]:
        ax.axis("off")

    return fig
def build_cluster_aware_evidence_table(evidence_table: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a feature-level evidence table into one row per redundancy cluster.

    Strategy:
    - keep the cluster representative when available
    - also record the strongest feature in the cluster by q-value / AUC
    """
    df = evidence_table.copy()

    if "redundancy_cluster_id" not in df.columns:
        df["redundancy_cluster_id"] = df["feature"]

    if "redundancy_representative_flag" not in df.columns:
        df["redundancy_representative_flag"] = True

    if "redundancy_cluster_size" not in df.columns:
        cluster_sizes = (
            df.groupby("redundancy_cluster_id")["feature"]
            .nunique()
            .rename("redundancy_cluster_size")
            .reset_index()
        )
        df = df.merge(cluster_sizes, on="redundancy_cluster_id", how="left")

    sort_cols = ["q_value", "auc_abs", "cliffs_delta"]
    sort_ascending = [True, False, False]

    best_feature_per_cluster = (
        df.sort_values(sort_cols, ascending=sort_ascending, na_position="last")
        .groupby("redundancy_cluster_id", as_index=False)
        .first()
        .rename(
            columns={
                "feature": "best_feature_in_cluster",
                "q_value": "best_q_value_in_cluster",
                "auc_abs": "best_auc_abs_in_cluster",
                "cliffs_delta": "best_cliffs_delta_in_cluster",
                "effect_direction": "best_effect_direction_in_cluster",
            }
        )
    )[
        [
            "redundancy_cluster_id",
            "best_feature_in_cluster",
            "best_q_value_in_cluster",
            "best_auc_abs_in_cluster",
            "best_cliffs_delta_in_cluster",
            "best_effect_direction_in_cluster",
        ]
    ]

    representatives = df.loc[df["redundancy_representative_flag"].fillna(False)].copy()
    if representatives.empty:
        representatives = df.copy()

    representative_table = (
        representatives.sort_values(sort_cols, ascending=sort_ascending, na_position="last")
        .groupby("redundancy_cluster_id", as_index=False)
        .first()
    )

    out = representative_table.merge(
        best_feature_per_cluster,
        on="redundancy_cluster_id",
        how="left",
        validate="one_to_one",
    )

    out["representative_matches_best_feature"] = (
        out["feature"] == out["best_feature_in_cluster"]
    )

    out = out.sort_values(
        ["q_value", "best_q_value_in_cluster", "auc_abs"],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)

    out["cluster_rank_within_task"] = np.arange(1, len(out) + 1)
    return out


def _get_feature_space_objects(
    partc_inputs: dict[str, pd.DataFrame | Path | float],
    feature_space: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if feature_space == "main":
        return (
            partc_inputs["partc_matrix_main"],
            partc_inputs["raw_partc_main"],
            partc_inputs["feature_manifest_main"],
        )
    if feature_space == "high":
        return (
            partc_inputs["partc_matrix_high"],
            partc_inputs["raw_partc_high"],
            partc_inputs["feature_manifest_high"],
        )
    if feature_space == "representative":
        return (
            partc_inputs["partc_matrix_representative"],
            partc_inputs["raw_partc_representative"],
            partc_inputs["feature_manifest_representative"],
        )
    raise ValueError("feature_space must be one of: 'main', 'high', 'representative'.")


def prepare_pairwise_modeling_data(
    partc_inputs: dict[str, pd.DataFrame | Path | float],
    positive_class: str,
    negative_class: str,
    feature_space: str = "main",
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """
    Build X/y for one pairwise task and one feature space.
    """
    transformed_matrix, _, _ = _get_feature_space_objects(partc_inputs, feature_space)
    sample_metadata = partc_inputs["partc_sample_metadata"][["sample", "class", "order", "batch"]].drop_duplicates()

    keep_meta = sample_metadata.loc[
        sample_metadata["class"].isin([positive_class, negative_class])
    ].copy()

    feature_cols = _feature_columns(transformed_matrix)
    X_df = transformed_matrix.set_index("sample").loc[keep_meta["sample"], feature_cols].copy()
    y = (keep_meta["class"] == positive_class).astype(int).to_numpy()

    return X_df.to_numpy(dtype=float), y, feature_cols, keep_meta.reset_index(drop=True)


def _build_linear_pipeline(
    model_name: str,
    n_features: int,
    k_best: int = 20,
    random_state: int = 42,
) -> Pipeline:
    k = int(min(max(2, k_best), n_features))

    if model_name == "logreg_l1":
        clf = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        )
    elif model_name == "linear_svm":
        clf = LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        )
    else:
        raise ValueError("model_name must be one of: 'logreg_l1', 'linear_svm'.")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k)),
            ("clf", clf),
        ]
    )


def _get_linear_scores_and_predictions(pipeline: Pipeline, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clf = pipeline.named_steps["clf"]

    if hasattr(clf, "predict_proba"):
        scores = pipeline.predict_proba(X)[:, 1]
    else:
        scores = pipeline.decision_function(X)

    preds = pipeline.predict(X)
    return scores, preds


def run_pairwise_model_benchmarks(
    partc_inputs: dict[str, pd.DataFrame | Path | float],
    task_table: pd.DataFrame,
    feature_spaces: tuple[str, ...] = ("main", "high", "representative"),
    model_names: tuple[str, ...] = ("logreg_l1", "linear_svm"),
    k_best: int = 20,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Repeated stratified CV benchmark across tasks / feature spaces / models.

    Returns:
    - fold_metrics_df
    - benchmark_summary_df
    - feature_stability_df
    """
    pairwise_tasks = task_table.loc[task_table["task_type"] == "pairwise"].copy()

    fold_rows = []
    coef_rows = []

    for _, task in pairwise_tasks.iterrows():
        task_name = task["task_name"]
        positive_class = task["positive_class"]
        negative_class = task["negative_class"]

        for feature_space in feature_spaces:
            X, y, feature_names, _ = prepare_pairwise_modeling_data(
                partc_inputs=partc_inputs,
                positive_class=positive_class,
                negative_class=negative_class,
                feature_space=feature_space,
            )

            cv = RepeatedStratifiedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )

            for model_name in model_names:
                for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    pipe = _build_linear_pipeline(
                        model_name=model_name,
                        n_features=X.shape[1],
                        k_best=k_best,
                        random_state=random_state,
                    )
                    pipe.fit(X_train, y_train)

                    scores, preds = _get_linear_scores_and_predictions(pipe, X_test)

                    fold_rows.append(
                        {
                            "task_name": task_name,
                            "positive_class": positive_class,
                            "negative_class": negative_class,
                            "feature_space": feature_space,
                            "model_name": model_name,
                            "fold_id": fold_id,
                            "auroc": _safe_auc(y_test, scores),
                            "balanced_accuracy": balanced_accuracy_score(y_test, preds),
                            "f1": f1_score(y_test, preds, zero_division=0),
                            "precision": precision_score(y_test, preds, zero_division=0),
                            "recall": recall_score(y_test, preds, zero_division=0),
                            "n_train": len(train_idx),
                            "n_test": len(test_idx),
                        }
                    )

                    selected_mask = pipe.named_steps["select"].get_support()
                    selected_features = np.array(feature_names)[selected_mask]

                    clf = pipe.named_steps["clf"]
                    if hasattr(clf, "coef_"):
                        selected_coefs = clf.coef_.ravel()
                    else:
                        selected_coefs = np.zeros(len(selected_features), dtype=float)

                    for feat, coef in zip(selected_features, selected_coefs):
                        coef_rows.append(
                            {
                                "task_name": task_name,
                                "feature_space": feature_space,
                                "model_name": model_name,
                                "fold_id": fold_id,
                                "feature": feat,
                                "coef": float(coef),
                                "coef_abs": float(abs(coef)),
                                "coef_positive": float(coef > 0),
                            }
                        )

    fold_metrics_df = pd.DataFrame(fold_rows)
    coef_df = pd.DataFrame(coef_rows)

    benchmark_summary_df = (
        fold_metrics_df.groupby(["task_name", "feature_space", "model_name"], as_index=False)
        .agg(
            mean_auroc=("auroc", "mean"),
            std_auroc=("auroc", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_f1=("f1", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            n_folds=("fold_id", "nunique"),
        )
        .sort_values(["task_name", "mean_auroc", "mean_balanced_accuracy"], ascending=[True, False, False])
        .reset_index(drop=True)
    )

    if coef_df.empty:
        feature_stability_df = pd.DataFrame(
            columns=[
                "task_name",
                "feature_space",
                "model_name",
                "feature",
                "n_selected",
                "n_folds",
                "selection_frequency",
                "mean_coef",
                "mean_abs_coef",
                "positive_coef_fraction",
                "sign_consistency",
            ]
        )
    else:
        fold_counts = (
            fold_metrics_df.groupby(["task_name", "feature_space", "model_name"])["fold_id"]
            .nunique()
            .rename("n_folds")
            .reset_index()
        )

        feature_stability_df = (
            coef_df.groupby(["task_name", "feature_space", "model_name", "feature"], as_index=False)
            .agg(
                n_selected=("feature", "size"),
                mean_coef=("coef", "mean"),
                mean_abs_coef=("coef_abs", "mean"),
                positive_coef_fraction=("coef_positive", "mean"),
            )
            .merge(
                fold_counts,
                on=["task_name", "feature_space", "model_name"],
                how="left",
                validate="many_to_one",
            )
        )

        feature_stability_df["selection_frequency"] = (
            feature_stability_df["n_selected"] / feature_stability_df["n_folds"]
        )
        feature_stability_df["sign_consistency"] = np.abs(
            2.0 * feature_stability_df["positive_coef_fraction"] - 1.0
        )

        feature_stability_df = feature_stability_df.sort_values(
            ["task_name", "feature_space", "model_name", "selection_frequency", "mean_abs_coef"],
            ascending=[True, True, True, False, False],
        ).reset_index(drop=True)

    return fold_metrics_df, benchmark_summary_df, feature_stability_df


def plot_model_benchmark_summary(
    benchmark_summary_df: pd.DataFrame,
    task_name: str,
) -> plt.Figure:
    """
    Compare AUROC and balanced accuracy across feature spaces and models for one task.
    """
    sub = benchmark_summary_df.loc[benchmark_summary_df["task_name"] == task_name].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    sns.barplot(
        data=sub,
        x="feature_space",
        y="mean_auroc",
        hue="model_name",
        ax=axes[0],
    )
    axes[0].set_title(f"{task_name} — mean AUROC")
    axes[0].set_xlabel("Feature space")
    axes[0].set_ylabel("Mean AUROC")
    axes[0].set_ylim(0.0, 1.02)

    sns.barplot(
        data=sub,
        x="feature_space",
        y="mean_balanced_accuracy",
        hue="model_name",
        ax=axes[1],
    )
    axes[1].set_title(f"{task_name} — mean balanced accuracy")
    axes[1].set_xlabel("Feature space")
    axes[1].set_ylabel("Mean balanced accuracy")
    axes[1].set_ylim(0.0, 1.02)

    for ax in axes:
        ax.grid(alpha=0.2)
        if ax.get_legend() is not None:
            ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")

    return fig


def plot_feature_stability(
    feature_stability_df: pd.DataFrame,
    task_name: str,
    feature_space: str = "representative",
    model_name: str = "logreg_l1",
    top_n: int = 12,
) -> plt.Figure:
    """
    Plot top stable features for one task / feature space / model.
    """
    sub = feature_stability_df.loc[
        (feature_stability_df["task_name"] == task_name)
        & (feature_stability_df["feature_space"] == feature_space)
        & (feature_stability_df["model_name"] == model_name)
    ].copy()

    sub = sub.sort_values(
        ["selection_frequency", "mean_abs_coef"],
        ascending=[False, False],
    ).head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)

    sns.barplot(
        data=sub,
        y="feature",
        x="selection_frequency",
        ax=axes[0],
    )
    axes[0].set_title(
        f"{task_name} — selection frequency\n({feature_space}, {model_name})"
    )
    axes[0].set_xlabel("Selection frequency")
    axes[0].set_ylabel("Feature")
    axes[0].set_xlim(0.0, 1.02)

    sns.barplot(
        data=sub,
        y="feature",
        x="mean_abs_coef",
        ax=axes[1],
    )
    axes[1].set_title(
        f"{task_name} — mean |coefficient|\n({feature_space}, {model_name})"
    )
    axes[1].set_xlabel("Mean |coefficient|")
    axes[1].set_ylabel("Feature")

    return fig


def build_biomarker_hierarchy(
    pairwise_tables_main: dict[str, pd.DataFrame],
    feature_stability_df: pd.DataFrame,
    feature_manifest_main: pd.DataFrame,
    model_name: str = "logreg_l1",
) -> pd.DataFrame:
    """
    Build a tiered biomarker shortlist.

    Tier 1:
    - representative feature
    - high-confidence retained in Part B
    - significant in at least one task
    - abs(AUC) >= 0.90 in at least one task
    - selection frequency >= 0.50 in at least one representative or main CV view
    - no carryover / blank veto warning

    Tier 2:
    - representative feature
    - weaker but still meaningful evidence

    Tier 3:
    - supportive family members / retained secondary signals
    """
    all_main = pd.concat(pairwise_tables_main.values(), ignore_index=True)

    task_agg = (
        all_main.groupby("feature", as_index=False)
        .agg(
            n_tasks_significant=("significant_fdr_05", "sum"),
            best_q_value=("q_value", "min"),
            best_auc_abs=("auc_abs", "max"),
            mean_auc_abs=("auc_abs", "mean"),
            max_abs_cliffs=("cliffs_delta", lambda s: np.nanmax(np.abs(s))),
            n_tasks_auc_ge_090=("auc_abs", lambda s: int(np.sum(s >= 0.90))),
        )
    )

    stab = feature_stability_df.loc[
        feature_stability_df["model_name"] == model_name
    ].copy()

    stab_agg = (
        stab.groupby("feature", as_index=False)
        .agg(
            max_selection_frequency=("selection_frequency", "max"),
            mean_selection_frequency=("selection_frequency", "mean"),
            max_sign_consistency=("sign_consistency", "max"),
            n_task_views_selection_ge_050=("selection_frequency", lambda s: int(np.sum(s >= 0.50))),
        )
    )

    out = feature_manifest_main.merge(task_agg, on="feature", how="left")
    out = out.merge(stab_agg, on="feature", how="left")

    for col in [
        "n_tasks_significant",
        "n_tasks_auc_ge_090",
        "max_selection_frequency",
        "mean_selection_frequency",
        "max_sign_consistency",
        "n_task_views_selection_ge_050",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna(0)

    if "redundancy_representative_flag" not in out.columns:
        out["redundancy_representative_flag"] = True

    if "retained_high_confidence" not in out.columns:
        out["retained_high_confidence"] = False

    if "blank_dominant_veto" not in out.columns:
        out["blank_dominant_veto"] = False

    if "potential_carryover_flag" not in out.columns:
        out["potential_carryover_flag"] = False

    is_representative = out["redundancy_representative_flag"].fillna(False)
    is_clean = (~out["blank_dominant_veto"].fillna(False)) & (~out["potential_carryover_flag"].fillna(False))

    tier1 = (
        is_representative
        & out["retained_high_confidence"].fillna(False)
        & is_clean
        & (out["n_tasks_significant"] >= 1)
        & (out["best_auc_abs"] >= 0.90)
        & (out["max_selection_frequency"] >= 0.50)
        & (out["max_sign_consistency"] >= 0.70)
    )

    tier2 = (
        is_representative
        & is_clean
        & (out["best_auc_abs"] >= 0.85)
        & (
            (out["n_tasks_significant"] >= 1)
            | (out["max_selection_frequency"] >= 0.25)
        )
    )

    out["target_tier"] = "Tier 3"
    out.loc[tier2, "target_tier"] = "Tier 2"
    out.loc[tier1, "target_tier"] = "Tier 1"

    strong_clusters = set(
        out.loc[out["target_tier"].isin(["Tier 1", "Tier 2"]), "redundancy_cluster_id"]
        .dropna()
        .tolist()
    )

    non_rep_support = (~is_representative) & out["redundancy_cluster_id"].isin(strong_clusters)
    out.loc[non_rep_support, "target_tier"] = "Tier 3_supportive_family_member"

    out = out.sort_values(
        [
            "target_tier",
            "best_q_value",
            "best_auc_abs",
            "max_selection_frequency",
        ],
        ascending=[True, True, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return out
def build_feature_clinical_profile_table(
    pairwise_tables_main: dict[str, pd.DataFrame],
    transformed_matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    min_auc: float = 0.85,
    max_q: float = 0.05,
    min_abs_log2_fc: float = 0.10,
    class_order: tuple[str, str, str] = ("French", "LMU", "Dunn"),
) -> pd.DataFrame:
    """
    Assign a compact clinical profile to each feature using pairwise evidence
    and class-level transformed medians.

    Profiles:
    - cancer-specific high / low
    - general disease-associated
    - benign-associated
    - healthy-associated
    - non-specific but discriminative
    """
    meta = sample_metadata[["sample", "class"]].drop_duplicates().copy()

    long_df = (
        transformed_matrix.merge(meta, on="sample", how="left")
        .melt(id_vars=["sample", "class"], var_name="feature", value_name="value")
    )

    medians = (
        long_df.groupby(["feature", "class"])["value"]
        .median()
        .unstack("class")
        .reset_index()
    )

    for cls in class_order:
        if cls not in medians.columns:
            medians[cls] = np.nan

    out = medians.copy()

    for task_name, table in pairwise_tables_main.items():
        keep_cols = [
            "feature",
            "q_value",
            "auc_abs",
            "raw_log2_fc",
            "effect_direction",
            "significant_fdr_05",
        ]
        renamed = table[keep_cols].rename(
            columns={
                "q_value": f"q_value__{task_name}",
                "auc_abs": f"auc_abs__{task_name}",
                "raw_log2_fc": f"raw_log2_fc__{task_name}",
                "effect_direction": f"effect_direction__{task_name}",
                "significant_fdr_05": f"significant_fdr_05__{task_name}",
            }
        )
        out = out.merge(renamed, on="feature", how="left", validate="one_to_one")

    def _is_strong(row: pd.Series, task_name: str) -> bool:
        q = row.get(f"q_value__{task_name}", np.nan)
        auc = row.get(f"auc_abs__{task_name}", np.nan)
        fc = row.get(f"raw_log2_fc__{task_name}", np.nan)
        return (
            pd.notna(q)
            and pd.notna(auc)
            and pd.notna(fc)
            and (q <= max_q)
            and (auc >= min_auc)
            and (abs(fc) >= min_abs_log2_fc)
        )

    profiles = []
    notes = []
    n_strong_tasks = []

    delta_small = 0.05

    for _, row in out.iterrows():
        f = row.get("French", np.nan)
        l = row.get("LMU", np.nan)
        d = row.get("Dunn", np.nan)

        s_fd = _is_strong(row, "French_vs_Dunn")
        s_fl = _is_strong(row, "French_vs_LMU")
        s_ld = _is_strong(row, "LMU_vs_Dunn")

        n_strong = int(s_fd) + int(s_fl) + int(s_ld)
        n_strong_tasks.append(n_strong)

        profile = "non-specific but discriminative"
        note = "Strong separation exists, but the pattern is not cleanly attributable to one clinical interpretation."

        if s_fd and s_fl and (f > l + delta_small) and (f > d + delta_small):
            profile = "cancer-specific high"
            if s_ld and (l > d + delta_small):
                note = "French is highest, with LMU intermediate and Dunn lowest."
            elif s_ld and (d > l + delta_small):
                note = "French is highest, while LMU stays below healthy."
            else:
                note = "French is clearly highest relative to both comparison groups."

        elif s_fd and s_fl and (f < l - delta_small) and (f < d - delta_small):
            profile = "cancer-specific low"
            if s_ld and (d > l + delta_small):
                note = "French is lowest, Dunn highest, and LMU intermediate."
            elif s_ld and (l > d + delta_small):
                note = "French is clearly lowest relative to both comparison groups."
            else:
                note = "French is clearly depleted relative to both comparison groups."

        elif s_fd and s_ld and (f > d + delta_small) and (l > d + delta_small) and not s_fl:
            profile = "general disease-associated"
            note = "Both disease groups are elevated relative to healthy, but French and LMU are not cleanly separated."

        elif s_fd and s_ld and (f < d - delta_small) and (l < d - delta_small) and not s_fl:
            profile = "healthy-associated"
            note = "Healthy samples are highest, while both disease groups are lower."

        elif s_ld and (l > f + delta_small) and (l > d + delta_small):
            profile = "benign-associated"
            note = "LMU is the dominant class for this feature."

        elif s_ld and (l < f - delta_small) and (l < d - delta_small):
            profile = "benign-depleted"
            note = "LMU is specifically low for this feature."

        elif (d > f + delta_small) and (d > l + delta_small):
            profile = "healthy-associated"
            note = "Dunn is highest overall."

        elif (d < f - delta_small) and (d < l - delta_small):
            profile = "healthy-depleted"
            note = "Dunn is lowest overall."

        profiles.append(profile)
        notes.append(note)

    out["n_strong_tasks"] = n_strong_tasks
    out["clinical_profile"] = profiles
    out["profile_note"] = notes

    return out


def plot_pairwise_decision_boundary(
    transformed_matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    cluster_aware_table: pd.DataFrame,
    positive_class: str,
    negative_class: str,
    title_prefix: str | None = None,
) -> plt.Figure:
    """
    Plot a simple 2D logistic decision boundary using the top two representative features
    from the cluster-aware table for a pairwise task.
    """
    top_features = (
        cluster_aware_table.sort_values(
            ["q_value", "best_q_value_in_cluster", "auc_abs"],
            ascending=[True, True, False],
            na_position="last",
        )["feature"]
        .drop_duplicates()
        .head(2)
        .tolist()
    )

    if len(top_features) < 2:
        raise ValueError("Need at least two representative features to draw a 2D boundary.")

    f1, f2 = top_features

    meta = sample_metadata[["sample", "class"]].drop_duplicates().copy()
    meta = meta.loc[meta["class"].isin([positive_class, negative_class])].copy()

    plot_df = (
        transformed_matrix[["sample", f1, f2]]
        .merge(meta, on="sample", how="inner", validate="one_to_one")
        .copy()
    )

    X = plot_df[[f1, f2]].to_numpy(dtype=float)
    y = (plot_df["class"] == positive_class).astype(int).to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    clf.fit(Xs, y)

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    dx = max((x1_max - x1_min) * 0.08, 0.05)
    dy = max((x2_max - x2_min) * 0.08, 0.05)

    xx, yy = np.meshgrid(
        np.linspace(x1_min - dx, x1_max + dx, 200),
        np.linspace(x2_min - dy, x2_max + dy, 200),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(scaler.transform(grid))[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)

    contour = ax.contourf(xx, yy, probs, levels=np.linspace(0, 1, 11), cmap="coolwarm", alpha=0.18)
    ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.8, linestyles="--")

    sns.scatterplot(
        data=plot_df,
        x=f1,
        y=f2,
        hue="class",
        palette=CLASS_PALETTE,
        s=85,
        ax=ax,
    )

    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    if title_prefix is None:
        title_prefix = f"{positive_class} vs {negative_class}"
    ax.set_title(f"{title_prefix} — 2D decision boundary ({f1}, {f2})")
    ax.grid(alpha=0.2)

    if ax.get_legend() is not None:
        ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.colorbar(contour, ax=ax, label=f"P({positive_class})")
    return fig


def build_warning_filtered_feature_space(
    partc_inputs: dict[str, pd.DataFrame | Path | float],
    base_feature_space: str = "main",
    exclude_carryover: bool = True,
    exclude_blank_dominant: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build a filtered feature space excluding warning-flagged features.
    """
    transformed_matrix, raw_matrix, feature_manifest = _get_feature_space_objects(
        partc_inputs, base_feature_space
    )

    manifest = feature_manifest.copy()
    keep_mask = pd.Series(True, index=manifest.index)

    if exclude_carryover and "potential_carryover_flag" in manifest.columns:
        keep_mask &= ~manifest["potential_carryover_flag"].fillna(False)

    if exclude_blank_dominant and "blank_dominant_veto" in manifest.columns:
        keep_mask &= ~manifest["blank_dominant_veto"].fillna(False)

    kept_manifest = manifest.loc[keep_mask].copy().reset_index(drop=True)
    kept_features = kept_manifest["feature"].drop_duplicates().tolist()

    transformed_subset = transformed_matrix[["sample", *kept_features]].copy()
    raw_subset = raw_matrix[["sample", *kept_features]].copy()

    return transformed_subset, raw_subset, kept_manifest


def _prepare_pairwise_modeling_data_from_matrix(
    transformed_matrix: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    positive_class: str,
    negative_class: str,
) -> tuple[np.ndarray, np.ndarray]:
    meta = sample_metadata[["sample", "class"]].drop_duplicates().copy()
    meta = meta.loc[meta["class"].isin([positive_class, negative_class])].copy()

    feature_cols = _feature_columns(transformed_matrix)
    X_df = transformed_matrix.set_index("sample").loc[meta["sample"], feature_cols].copy()
    y = (meta["class"] == positive_class).astype(int).to_numpy()

    return X_df.to_numpy(dtype=float), y


def run_pairwise_model_benchmarks_from_space_dict(
    matrix_space_dict: dict[str, pd.DataFrame],
    sample_metadata: pd.DataFrame,
    task_table: pd.DataFrame,
    model_names: tuple[str, ...] = ("logreg_l1", "linear_svm"),
    k_best: int = 20,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Flexible benchmark runner for a custom dictionary of feature spaces.
    """
    pairwise_tasks = task_table.loc[task_table["task_type"] == "pairwise"].copy()

    fold_rows = []

    for _, task in pairwise_tasks.iterrows():
        task_name = task["task_name"]
        positive_class = task["positive_class"]
        negative_class = task["negative_class"]

        for feature_space, transformed_matrix in matrix_space_dict.items():
            X, y = _prepare_pairwise_modeling_data_from_matrix(
                transformed_matrix=transformed_matrix,
                sample_metadata=sample_metadata,
                positive_class=positive_class,
                negative_class=negative_class,
            )

            cv = RepeatedStratifiedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )

            for model_name in model_names:
                for fold_id, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    pipe = _build_linear_pipeline(
                        model_name=model_name,
                        n_features=X.shape[1],
                        k_best=k_best,
                        random_state=random_state,
                    )
                    pipe.fit(X_train, y_train)

                    scores, preds = _get_linear_scores_and_predictions(pipe, X_test)

                    fold_rows.append(
                        {
                            "task_name": task_name,
                            "feature_space": feature_space,
                            "model_name": model_name,
                            "fold_id": fold_id,
                            "auroc": _safe_auc(y_test, scores),
                            "balanced_accuracy": balanced_accuracy_score(y_test, preds),
                            "f1": f1_score(y_test, preds, zero_division=0),
                            "precision": precision_score(y_test, preds, zero_division=0),
                            "recall": recall_score(y_test, preds, zero_division=0),
                        }
                    )

    fold_metrics_df = pd.DataFrame(fold_rows)

    benchmark_summary_df = (
        fold_metrics_df.groupby(["task_name", "feature_space", "model_name"], as_index=False)
        .agg(
            mean_auroc=("auroc", "mean"),
            std_auroc=("auroc", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
            mean_f1=("f1", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            n_folds=("fold_id", "nunique"),
        )
        .sort_values(["task_name", "mean_auroc", "mean_balanced_accuracy"], ascending=[True, False, False])
        .reset_index(drop=True)
    )

    return fold_metrics_df, benchmark_summary_df


def build_final_biomarker_decision_table(
    biomarker_hierarchy: pd.DataFrame,
    feature_profile_table: pd.DataFrame,
    pairwise_tables_main: dict[str, pd.DataFrame],
    feature_stability_df: pd.DataFrame,
    keep_tiers: tuple[str, ...] = ("Tier 1", "Tier 2"),
    model_name: str = "logreg_l1",
    representative_only: bool = True,
) -> pd.DataFrame:
    """
    Build the final Task 1 biomarker decision table.
    """
    base = biomarker_hierarchy.copy()

    if representative_only and "redundancy_representative_flag" in base.columns:
        base = base.loc[base["redundancy_representative_flag"].fillna(False)].copy()

    base = base.loc[base["target_tier"].isin(list(keep_tiers))].copy()

    pairwise_keep = []
    for task_name, table in pairwise_tables_main.items():
        tmp = table[
            [
                "feature",
                "task_name",
                "effect_direction",
                "raw_log2_fc",
                "q_value",
                "auc_abs",
                "positive_detection_rate_raw",
                "negative_detection_rate_raw",
            ]
        ].copy()
        pairwise_keep.append(tmp)

    pairwise_long = pd.concat(pairwise_keep, ignore_index=True)

    pairwise_wide = pairwise_long.pivot(index="feature", columns="task_name")
    pairwise_wide.columns = [f"{metric}__{task}" for metric, task in pairwise_wide.columns]
    pairwise_wide = pairwise_wide.reset_index()

    stab = feature_stability_df.loc[feature_stability_df["model_name"] == model_name].copy()
    stab_summary = (
        stab.groupby("feature", as_index=False)
        .agg(
            model_max_selection_frequency=("selection_frequency", "max"),
            model_mean_abs_coef=("mean_abs_coef", "mean"),
            model_max_sign_consistency=("sign_consistency", "max"),
        )
    )

    profile_keep_cols = ["feature", "clinical_profile", "profile_note", "French", "LMU", "Dunn", "n_strong_tasks"]
    final_table = (
        base.merge(feature_profile_table[profile_keep_cols], on="feature", how="left")
        .merge(pairwise_wide, on="feature", how="left")
        .merge(stab_summary, on="feature", how="left")
    )

    final_table["biological_interpretation_note"] = final_table["profile_note"]

    tier_order = {"Tier 1": 1, "Tier 2": 2}
    final_table["_tier_order"] = final_table["target_tier"].map(tier_order).fillna(99)

    final_table = final_table.sort_values(
        ["_tier_order", "best_q_value", "best_auc_abs", "model_max_selection_frequency"],
        ascending=[True, True, False, False],
        na_position="last",
    ).reset_index(drop=True)

    final_table["priority_rank"] = np.arange(1, len(final_table) + 1)
    final_table = final_table.drop(columns="_tier_order")

    return final_table
