import os
import numpy as np
import matplotlib.pyplot as plt

from .attention_plots import build_month_axis


def plot_future_variable_importance(feature_importances, plots_dir, forecast_start_date):
    """Stacked bar chart of future covariate importance per forecast horizon (with month labels)."""
    df_feature_future = feature_importances["Future variable importance over time"]

    # --- Version 1: numeric horizon labels (t+1, t+2, ...) ---
    fig, ax = plt.subplots(figsize=(20, 10))
    bottom = np.zeros(len(df_feature_future.index))

    for col in df_feature_future.columns:
        ax.bar(
            np.arange(1, len(df_feature_future) + 1),
            df_feature_future[col].values,
            0.6,
            label=col,
            bottom=bottom
        )
        bottom += df_feature_future[col]

    ax.set_xticks(np.arange(1, len(df_feature_future) + 1))
    ax.set_xticklabels([f"t+{i}" for i in range(1, len(df_feature_future) + 1)])
    ax.set_title("Future Variable Importance Over Time", fontsize=16)
    ax.set_ylabel("Importance")
    ax.set_xlabel("Forecast Horizon")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    output_path = os.path.join(plots_dir, "future_variable_importance_over_time.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Interpretability] Saved plot to: {output_path}")

    # --- Version 2: calendar month labels ---
    _, future_dates = build_month_axis(
        forecast_start_date,
        past_len=0,
        future_len=len(df_feature_future)
    )

    fig, ax = plt.subplots(figsize=(20, 10))
    bottom = np.zeros(len(df_feature_future))
    x = np.arange(len(future_dates))

    for col in df_feature_future.columns:
        ax.bar(x, df_feature_future[col].values, 0.6, bottom=bottom, label=col)
        bottom += df_feature_future[col].values

    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%b\n%Y") for d in future_dates], fontsize=10)
    ax.set_title("Future Variable Importance Over Time", fontsize=16)
    ax.set_xlabel("Month")
    ax.set_ylabel("Importance")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    output_path = os.path.join(plots_dir, "future_variable_importance_over_time_with_month.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Interpretability] Saved plot to: {output_path}")


def plot_past_variable_importance(feature_importances, plots_dir, forecast_start_date):
    """Stacked bar chart of past covariate importance (with month labels)."""
    df_feature_past = feature_importances["Past variable importance over time"]

    # --- Version 1: relative numeric time axis ---
    fig, ax = plt.subplots(figsize=(20, 10))
    bottom = np.zeros(len(df_feature_past.index))

    for col in df_feature_past.columns:
        ax.bar(np.arange(-len(df_feature_past), 0), df_feature_past[col].values, 0.6, label=col, bottom=bottom)
        bottom += df_feature_past[col]

    ax.set_title("Past Variable Importance Over Time")
    ax.set_ylabel("Importance")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    output_path = os.path.join(plots_dir, "past_variable_importance_over_time.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Interpretability] Saved plot to: {output_path}")

    # --- Version 2: calendar month labels ---
    past_dates, _ = build_month_axis(
        forecast_start_date,
        past_len=len(df_feature_past),
        future_len=0
    )

    x = np.arange(len(past_dates))
    fig, ax = plt.subplots(figsize=(20, 10))
    bottom = np.zeros(len(past_dates))

    for col in df_feature_past.columns:
        ax.bar(x, df_feature_past[col].values, 0.6, bottom=bottom, label=col)
        bottom += df_feature_past[col].values

    ax.set_xticks(x[::2])
    ax.set_xticklabels([d.strftime("%b\n%Y") for d in past_dates[::2]], fontsize=10)
    ax.set_title("Past Variable Importance Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Importance")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()

    output_path = os.path.join(plots_dir, "past_variable_importance_over_time_with_month.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Interpretability] Saved plot to: {output_path}")


def plot_variable_importance_means(feature_importances, plots_dir):
    """Horizontal bar charts of mean past and future variable importances."""

    # Past mean
    past_mean = (
        feature_importances["Past variable importance over time"]
        .mean()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    past_mean.plot(kind="barh", ax=ax)
    ax.set_title("Past Variable Importance (Mean)")
    ax.set_xlabel("Mean Importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "past_variable_importance_mean.png"), dpi=150)
    plt.close(fig)

    # Future mean
    future_mean = (
        feature_importances["Future variable importance over time"]
        .mean()
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    future_mean.plot(kind="barh", ax=ax)
    ax.set_title("Future Variable Importance (Mean)")
    ax.set_xlabel("Mean Importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "future_variable_importance_mean.png"), dpi=150)
    plt.close(fig)

    print("[Interpretability] Saved past/future variable importance mean plots.")
