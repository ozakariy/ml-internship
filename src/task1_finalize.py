from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from eda import get_feature_columns
from task1_processing import BIO_CLASSES


@dataclass
class _UnionFind:
    items: list[str]

    def __post_init__(self) -> None:
        self.parent = {x: x for x in self.items}

    def find(self, x: str) -> str:
        parent = self.parent[x]
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _safe_bool_sum(series: pd.Series) -> int:
    return int(series.fillna(False).astype(bool).sum())


def annotate_feature_redundancy(
    modeling_matrix: pd.DataFrame,
    retained_feature_manifest: pd.DataFrame,
    corr_reporting_threshold: float = 0.95,
    corr_cluster_threshold: float = 0.97,
    isomer_mz_tol: float = 0.02,
    isomer_rt_min: float = 5.0,
    adduct_rt_tol: float = 20.0,
    adduct_mz_min: float = 0.5,
    adduct_mz_max: float = 30.0,
    near_duplicate_mz_tol: float = 5.0,
    near_duplicate_rt_tol: float = 60.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Annotate structural redundancy among retained main features.

    We do NOT collapse features automatically here.
    We only:
    - label candidate isomer pairs
    - label candidate isotope/adduct pairs
    - label highly correlated near-duplicate pairs
    - build redundancy clusters
    - choose one representative feature per cluster
    """
    matrix = modeling_matrix.copy()
    if "sample" in matrix.columns:
        matrix = matrix.set_index("sample")

    feature_manifest = retained_feature_manifest.copy()
    features = [f for f in feature_manifest["feature"].tolist() if f in matrix.columns]
    matrix = matrix[features].astype(float)

    meta = feature_manifest.set_index("feature").copy()
    corr = matrix.corr(method="spearman")

    uf = _UnionFind(features)
    pair_rows: list[dict] = []

    for i, fa in enumerate(features):
        for fb in features[i + 1:]:
            rho = corr.loc[fa, fb]
            mz_diff = abs(float(meta.at[fa, "mz"]) - float(meta.at[fb, "mz"]))
            rt_diff = abs(float(meta.at[fa, "rt"]) - float(meta.at[fb, "rt"]))

            candidate_isomer = (mz_diff <= isomer_mz_tol) and (rt_diff >= isomer_rt_min)
            candidate_adduct_or_isotope = (
                (rt_diff <= adduct_rt_tol)
                and (adduct_mz_min <= mz_diff <= adduct_mz_max)
            )
            high_corr_near_duplicate = (
                pd.notna(rho)
                and (rho >= corr_cluster_threshold)
                and (mz_diff <= near_duplicate_mz_tol)
                and (rt_diff <= near_duplicate_rt_tol)
            )
            high_corr_other = pd.notna(rho) and (rho >= corr_reporting_threshold)

            keep_pair = (
                candidate_isomer
                or candidate_adduct_or_isotope
                or high_corr_near_duplicate
                or high_corr_other
            )
            if not keep_pair:
                continue

            if candidate_isomer and high_corr_other:
                relation_type = "candidate_isomer_correlated"
            elif candidate_isomer:
                relation_type = "candidate_isomer"
            elif candidate_adduct_or_isotope and high_corr_other:
                relation_type = "candidate_isotope_or_adduct_correlated"
            elif candidate_adduct_or_isotope:
                relation_type = "candidate_isotope_or_adduct"
            elif high_corr_near_duplicate:
                relation_type = "high_corr_near_duplicate"
            else:
                relation_type = "high_corr_other"

            pair_rows.append(
                {
                    "feature_a": fa,
                    "feature_b": fb,
                    "spearman_rho": rho,
                    "mz_a": float(meta.at[fa, "mz"]),
                    "mz_b": float(meta.at[fb, "mz"]),
                    "rt_a": float(meta.at[fa, "rt"]),
                    "rt_b": float(meta.at[fb, "rt"]),
                    "mz_diff": mz_diff,
                    "rt_diff": rt_diff,
                    "candidate_isomer": candidate_isomer,
                    "candidate_isotope_or_adduct": candidate_adduct_or_isotope,
                    "high_corr_near_duplicate": high_corr_near_duplicate,
                    "high_corr_other": high_corr_other,
                    "relation_type": relation_type,
                }
            )

            if candidate_isomer or candidate_adduct_or_isotope or high_corr_near_duplicate:
                uf.union(fa, fb)

    pairwise_table = pd.DataFrame(pair_rows)

    cluster_members: dict[str, list[str]] = defaultdict(list)
    for f in features:
        cluster_members[uf.find(f)].append(f)

    cluster_id_map = {}
    cluster_size_map = {}
    for idx, (_, members) in enumerate(cluster_members.items(), start=1):
        cluster_id = f"RG-{idx:03d}"
        for f in members:
            cluster_id_map[f] = cluster_id
            cluster_size_map[f] = len(members)

    annotated = feature_manifest.copy()
    annotated["redundancy_cluster_id"] = annotated["feature"].map(cluster_id_map)
    annotated["redundancy_cluster_size"] = annotated["feature"].map(cluster_size_map).fillna(1).astype(int)

    if pairwise_table.empty:
        annotated["candidate_isomer_neighbor_count"] = 0
        annotated["candidate_isotope_or_adduct_neighbor_count"] = 0
        annotated["high_corr_near_duplicate_neighbor_count"] = 0
        annotated["high_corr_other_neighbor_count"] = 0
    else:
        long_pairs_a = pairwise_table.rename(columns={"feature_a": "feature", "feature_b": "neighbor"})
        long_pairs_b = pairwise_table.rename(columns={"feature_b": "feature", "feature_a": "neighbor"})
        long_pairs = pd.concat([long_pairs_a, long_pairs_b], ignore_index=True)

        counts = (
            long_pairs.groupby("feature", dropna=False)
            .agg(
                candidate_isomer_neighbor_count=("candidate_isomer", _safe_bool_sum),
                candidate_isotope_or_adduct_neighbor_count=("candidate_isotope_or_adduct", _safe_bool_sum),
                high_corr_near_duplicate_neighbor_count=("high_corr_near_duplicate", _safe_bool_sum),
                high_corr_other_neighbor_count=("high_corr_other", _safe_bool_sum),
            )
            .reset_index()
        )
        annotated = annotated.merge(counts, on="feature", how="left", validate="one_to_one")
        for col in [
            "candidate_isomer_neighbor_count",
            "candidate_isotope_or_adduct_neighbor_count",
            "high_corr_near_duplicate_neighbor_count",
            "high_corr_other_neighbor_count",
        ]:
            annotated[col] = annotated[col].fillna(0).astype(int)

    annotated["redundancy_representative_flag"] = False

    for cluster_id, cluster_df in annotated.groupby("redundancy_cluster_id", dropna=False):
        cluster_df = cluster_df.copy()
        cluster_df = cluster_df.sort_values(
            [
                "retained_high_confidence",
                "qc_cv_pct",
                "d_ratio_log",
                "bio_detection_rate_max",
            ],
            ascending=[False, True, True, False],
            na_position="last",
        )
        rep_feature = cluster_df.iloc[0]["feature"]
        annotated.loc[annotated["feature"] == rep_feature, "redundancy_representative_flag"] = True

    annotated["redundancy_role"] = np.where(
        annotated["redundancy_cluster_size"] <= 1,
        "singleton",
        np.where(
            annotated["redundancy_representative_flag"],
            "representative",
            "member",
        ),
    )

    cluster_summary = (
        annotated.groupby("redundancy_cluster_id", dropna=False)
        .agg(
            cluster_size=("feature", "size"),
            representative_feature=("feature", lambda s: annotated.loc[
                (annotated["feature"].isin(s)) & (annotated["redundancy_representative_flag"]),
                "feature"
            ].iloc[0]),
        )
        .reset_index()
        .sort_values(["cluster_size", "redundancy_cluster_id"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return annotated, pairwise_table, cluster_summary


def plot_redundancy_summary(
    annotated_feature_manifest: pd.DataFrame,
) -> plt.Figure:
    df = annotated_feature_manifest.copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    plot_df = df.copy()
    plot_df["in_multifeature_cluster"] = plot_df["redundancy_cluster_size"] > 1

    sns.scatterplot(
        data=plot_df,
        x="rt",
        y="mz",
        hue="in_multifeature_cluster",
        style="redundancy_role",
        s=80,
        ax=axes[0],
    )
    axes[0].set_title("Retained feature landscape with redundancy annotation")
    axes[0].set_xlabel("Retention time")
    axes[0].set_ylabel("m/z")

    cluster_size_counts = (
        plot_df.loc[plot_df["redundancy_cluster_size"] > 1, "redundancy_cluster_size"]
        .value_counts()
        .sort_index()
        .rename_axis("cluster_size")
        .reset_index(name="n_clusters")
    )

    if cluster_size_counts.empty:
        axes[1].text(0.5, 0.5, "No multi-feature clusters", ha="center", va="center")
        axes[1].set_axis_off()
    else:
        sns.barplot(
            data=cluster_size_counts,
            x="cluster_size",
            y="n_clusters",
            ax=axes[1],
        )
        axes[1].set_title("Distribution of redundancy cluster sizes")
        axes[1].set_xlabel("Cluster size")
        axes[1].set_ylabel("Number of clusters")

    return fig


def compute_partb_threshold_ablation(
    feature_filter_table: pd.DataFrame,
    qc_cv_thresholds: tuple[float, ...] = (20.0, 30.0, 40.0),
    bio_detection_thresholds: tuple[float, ...] = (0.50, 0.70, 0.90),
    d_ratio_log_thresholds: tuple[float, ...] = (0.75, 1.00, 1.25),
    mz_threshold: float = 500.0,
    default_qc_cv_threshold: float = 30.0,
    default_bio_detection_threshold: float = 0.70,
) -> pd.DataFrame:
    df = feature_filter_table.copy()

    rows = []

    for thr in qc_cv_thresholds:
        n = (
            (df["mz"] > mz_threshold)
            & (df["qc_cv_pct"] < thr)
            & (df["bio_detection_rate_max"] >= default_bio_detection_threshold)
            & (~df["blank_dominant_veto"])
        ).sum()
        rows.append(
            {
                "criterion_type": "qc_cv_threshold",
                "threshold": thr,
                "n_features": int(n),
            }
        )

    for thr in bio_detection_thresholds:
        n = (
            (df["mz"] > mz_threshold)
            & (df["qc_cv_pct"] < default_qc_cv_threshold)
            & (df["bio_detection_rate_max"] >= thr)
            & (~df["blank_dominant_veto"])
        ).sum()
        rows.append(
            {
                "criterion_type": "bio_detection_threshold",
                "threshold": thr,
                "n_features": int(n),
            }
        )

    for thr in d_ratio_log_thresholds:
        n = (
            (df["mz"] > mz_threshold)
            & (df["qc_cv_pct"] < default_qc_cv_threshold)
            & (df["bio_detection_rate_max"] >= default_bio_detection_threshold)
            & (~df["blank_dominant_veto"])
            & df["d_ratio_log"].notna()
            & (df["d_ratio_log"] <= thr)
        ).sum()
        rows.append(
            {
                "criterion_type": "d_ratio_log_threshold",
                "threshold": thr,
                "n_features": int(n),
            }
        )

    return pd.DataFrame(rows)


def plot_partb_threshold_ablation(
    threshold_ablation_table: pd.DataFrame,
) -> plt.Figure:
    df = threshold_ablation_table.copy()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    order = [
        "qc_cv_threshold",
        "bio_detection_threshold",
        "d_ratio_log_threshold",
    ]
    titles = {
        "qc_cv_threshold": "Sensitivity to QC CV threshold",
        "bio_detection_threshold": "Sensitivity to detection-support threshold",
        "d_ratio_log_threshold": "Sensitivity to D-ratio(log) threshold",
    }

    for ax, crit in zip(axes, order):
        sub = df.loc[df["criterion_type"] == crit].copy()
        sns.lineplot(
            data=sub,
            x="threshold",
            y="n_features",
            marker="o",
            ax=ax,
        )
        ax.set_title(titles[crit])
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Retained features")

    return fig


def build_partc_handoff_tables(
    sample_review_table: pd.DataFrame,
    feature_decision_table: pd.DataFrame,
    selected_transform: str,
    detection_threshold: float,
) -> dict[str, pd.DataFrame]:
    sample_summary = (
        sample_review_table.groupby("class", dropna=False)
        .agg(
            n_total=("sample", "size"),
            n_retained=("recommend_exclude", lambda s: int((~s).sum())),
            n_excluded=("recommend_exclude", lambda s: int(s.sum())),
            n_minor_or_worse=("review_tier", lambda s: int(s.isin(["minor_review", "review", "strong_review"]).sum())),
            n_review_or_worse=("review_tier", lambda s: int(s.isin(["review", "strong_review"]).sum())),
        )
        .reset_index()
    )

    retained_main = feature_decision_table.loc[feature_decision_table["retained_main"]].copy()

    feature_summary = pd.DataFrame(
        {
            "metric": [
                "n_features_total",
                "n_features_main_retained",
                "n_features_high_confidence",
                "n_main_features_with_potential_carryover_flag",
                "n_main_features_in_multifeature_clusters",
                "n_main_representative_features",
            ],
            "value": [
                int(feature_decision_table["feature"].nunique()),
                int(feature_decision_table["retained_main"].sum()),
                int(feature_decision_table["retained_high_confidence"].sum()),
                int(retained_main["potential_carryover_flag"].fillna(False).sum()),
                int((retained_main["redundancy_cluster_size"].fillna(1) > 1).sum()),
                int(retained_main["redundancy_representative_flag"].fillna(False).sum()),
            ],
        }
    )

    config_summary = pd.DataFrame(
        {
            "setting": [
                "selected_transform",
                "operational_detection_threshold",
                "qc_cv_threshold",
                "bio_detection_threshold_any_class",
                "mz_threshold",
            ],
            "value": [
                selected_transform,
                detection_threshold,
                30.0,
                0.70,
                500.0,
            ],
        }
    )

    caveat_summary = pd.DataFrame(
        {
            "caveat": [
                "excluded_biological_samples",
                "main_features_with_carryover_warning",
                "main_features_in_multifeature_redundancy_clusters",
            ],
            "n": [
                int(sample_review_table["recommend_exclude"].sum()),
                int(retained_main["potential_carryover_flag"].fillna(False).sum()),
                int((retained_main["redundancy_cluster_size"].fillna(1) > 1).sum()),
            ],
        }
    )

    return {
        "handoff_sample_summary": sample_summary,
        "handoff_feature_summary": feature_summary,
        "handoff_config_summary": config_summary,
        "handoff_caveat_summary": caveat_summary,
    }
