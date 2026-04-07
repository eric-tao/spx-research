from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _ensure_output(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_probability_heatmaps(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> list[Path]:
    root = _ensure_output(output_dir)
    paths: list[Path] = []
    for width_label, table in tables.items():
        pivot = table.pivot(index="move_bucket", columns="checkpoint", values="p_le_1_0x")
        fig, ax = plt.subplots(figsize=(10, 5))
        image = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(item) for item in pivot.index])
        ax.set_title(f"P(remaining excursion <= 1.0 x width), width={width_label}")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        path = root / f"heatmap_width_{width_label}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def save_probability_lines(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> list[Path]:
    root = _ensure_output(output_dir)
    paths: list[Path] = []
    for width_label, table in tables.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        for move_bucket, group in table.groupby("move_bucket", sort=False):
            ax.plot(group["checkpoint"], group["p_le_1_0x"], marker="o", label=str(move_bucket))
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Probability by checkpoint, width={width_label}")
        ax.set_ylabel("P(remaining excursion <= 1.0 x width)")
        ax.set_xlabel("Checkpoint")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = root / f"line_width_{width_label}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def save_expected_value_lines(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> list[Path]:
    root = _ensure_output(output_dir)
    paths: list[Path] = []
    for width_label, table in tables.items():
        for (short_distance, credit_ratio), group in table.groupby(["short_distance_multiple", "credit_ratio"], sort=True):
            fig, ax = plt.subplots(figsize=(10, 5))
            for move_bucket, bucket_group in group.groupby("move_bucket", sort=False):
                ax.plot(
                    bucket_group["checkpoint"],
                    bucket_group["expected_value_points"],
                    marker="o",
                    label=str(move_bucket),
                )
            ax.axhline(0.0, color="black", linewidth=1, alpha=0.5)
            ax.set_title(
                f"Expected value by checkpoint, width={width_label}, "
                f"short={short_distance:.2f}x, credit={credit_ratio:.2f}w"
            )
            ax.set_ylabel("Expected value (points)")
            ax.set_xlabel("Checkpoint")
            ax.legend(loc="best", fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            short_label = str(short_distance).replace(".", "p")
            credit_label = str(credit_ratio).replace(".", "p")
            path = root / f"ev_width_{width_label}_short_{short_label}_credit_{credit_label}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
    return paths


def save_viability_heatmaps(
    breakeven_tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
    credit_ratio_thresholds: list[float],
    width_display_multiplier: float = 1.0,
) -> list[Path]:
    root = _ensure_output(output_dir)
    paths: list[Path] = []
    checkpoint_order: list[str] = []
    width_items: list[tuple[float, str]] = []
    for width_label, table in breakeven_tables.items():
        width_items.append((float(width_label), width_label))
        if not checkpoint_order and "checkpoint" in table.columns:
            checkpoint_order = list(dict.fromkeys(table["checkpoint"].tolist()))
    width_items.sort(key=lambda item: item[0])
    width_display_labels = {
        width_label: (
            str(int(width_value * width_display_multiplier))
            if float(width_value * width_display_multiplier).is_integer()
            else f"{width_value * width_display_multiplier:.3f}"
        )
        for width_value, width_label in width_items
    }

    sample_table = next(iter(breakeven_tables.values()), None)
    if sample_table is None or sample_table.empty:
        return paths
    regimes = [str(item) for item in sample_table["vol_regime"].dropna().drop_duplicates().tolist()]

    for threshold in credit_ratio_thresholds:
        for regime in regimes:
            matrix_rows: list[list[float]] = []
            row_labels: list[str] = []
            for _, width_label in width_items:
                table = breakeven_tables[width_label]
                subset = table[(table["short_distance_multiple"] == 0.5) & (table["vol_regime"] == regime)].copy()
                if subset.empty:
                    continue
                subset = subset.set_index("checkpoint").reindex(checkpoint_order)
                values = (subset["breakeven_credit_ratio"] <= threshold).astype(float).tolist()
                matrix_rows.append(values)
                row_labels.append(width_display_labels[width_label])
            if not matrix_rows:
                continue

            fig, ax = plt.subplots(figsize=(10, 4))
            image = ax.imshow(matrix_rows, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
            ax.set_xticks(range(len(checkpoint_order)))
            ax.set_xticklabels(checkpoint_order, rotation=45, ha="right")
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels)
            ax.set_xlabel("Checkpoint")
            ax.set_ylabel("Displayed Width")
            ax.set_title(f"Viable cells for credit >= {threshold:.2f}w, regime={regime}")
            fig.colorbar(image, ax=ax, ticks=[0.0, 1.0])
            fig.tight_layout()
            threshold_label = str(threshold).replace(".", "p")
            path = root / f"viability_regime_{regime}_credit_{threshold_label}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
    return paths
