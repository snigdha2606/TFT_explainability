import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_month_axis(
    forecast_start_date: str,
    past_len: int,
    future_len: int,
):
    """
    Returns:
        past_dates   : DatetimeIndex (length = past_len)
        future_dates : DatetimeIndex (length = future_len)
    """
    forecast_start = pd.Timestamp(forecast_start_date)

    past_dates = pd.date_range(
        end=forecast_start - pd.DateOffset(months=1),
        periods=past_len,
        freq="MS"
    )

    future_dates = pd.date_range(
        start=forecast_start,
        periods=future_len,
        freq="MS"
    )

    return past_dates, future_dates


def plot_attention(
    tft_model,
    plots_dir,
    plot: str = "time",
    output: str = "save",
    width: int = 800,
    height: int = 400,
    filename: str | None = None,
):
    attn = (
        tft_model
        .mean_on_batch(tft_model.interpretability_params["attn_wts"])
        .mean(dim=0)
        .cpu()
        .numpy()
    )

    input_size = tft_model.input_size
    h = tft_model.h
    time_axis = np.arange(-input_size, h)

    forecast_attn = attn[input_size:, :]

    fig, ax = plt.subplots(figsize=(width / 100, height / 100))

    if plot == "time":
        mean_attn = forecast_attn.mean(axis=0)
        ax.plot(time_axis, mean_attn, linewidth=2)
        ax.axvline(0, color="black", linestyle="--", linewidth=2, label="prediction start")
        ax.set_title("Mean Attention Over Time")
        ax.set_xlabel("Time (relative to forecast start)")
        ax.set_ylabel("Attention")
        ax.legend()
        default_name = "attention_mean_time.png"

    elif plot == "all":
        for i in range(h):
            ax.plot(time_axis, forecast_attn[i], label=f"horizon {i+1}", alpha=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=2, label="prediction start")
        ax.set_title("Attention per Forecast Horizon")
        ax.set_xlabel("Time (relative to forecast start)")
        ax.set_ylabel("Attention")
        ax.legend(title="Horizon", bbox_to_anchor=(1.02, 1), loc="upper left")
        default_name = "attention_all_horizons.png"

    elif plot == "heatmap":
        im = ax.imshow(
            attn,
            aspect="auto",
            cmap="viridis",
            extent=[-input_size, h, -input_size, h],
        )
        fig.colorbar(im, ax=ax)
        ax.set_title("Attention Heatmap")
        ax.set_xlabel("Attending to (time step)")
        ax.set_ylabel("Query time step")
        default_name = "attention_heatmap.png"

    elif isinstance(plot, int) and 1 <= plot <= h:
        i = plot - 1
        ax.plot(time_axis, forecast_attn[i], linewidth=2, label=f"horizon {plot}")
        ax.axvline(0, color="black", linestyle="--", linewidth=2, label="prediction start")
        ax.set_title(f"Attention at Horizon t+{plot}")
        ax.set_xlabel("Time (relative to forecast start)")
        ax.set_ylabel("Attention")
        ax.legend()
        default_name = f"attention_horizon_t{plot}.png"

    else:
        raise ValueError("Invalid plot argument")

    fig.subplots_adjust(left=0.08, right=0.85, top=0.9, bottom=0.15)
    os.makedirs(plots_dir, exist_ok=True)
    fname = filename if filename else default_name
    save_path = os.path.join(plots_dir, fname)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[Attention] Saved: {save_path}")

    if output in ("show", "both"):
        plt.show()
    plt.close(fig)


def plot_attention_with_dates(
    tft_model,
    plots_dir,
    forecast_start_date,
    width: int = 1000,
    height: int = 500,
    filename: str = "attention_mean_time_with_months.png",
):
    attn = (
        tft_model
        .mean_on_batch(tft_model.interpretability_params["attn_wts"])
        .mean(dim=0)
        .cpu()
        .numpy()
    )

    if attn.ndim == 3:
        attn = attn.mean(axis=0)

    input_size = tft_model.input_size
    h = tft_model.h

    forecast_attn = attn[input_size:, :]
    mean_attn = forecast_attn.mean(axis=0)

    forecast_start = pd.Timestamp(forecast_start_date)

    past_dates = pd.date_range(
        end=forecast_start - pd.DateOffset(months=1),
        periods=input_size,
        freq="MS"
    )

    future_dates = pd.date_range(
        start=forecast_start,
        periods=h,
        freq="MS"
    )

    all_dates = past_dates.append(future_dates)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100))

    ax.plot(all_dates, mean_attn, linewidth=2, color="steelblue")
    ax.axvline(
        forecast_start,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Forecast start"
    )

    if input_size >= 12:
        ax.axvline(
            forecast_start - pd.DateOffset(months=12),
            color="tomato",
            linestyle=":",
            linewidth=1.5,
            label="1 year ago"
        )

    ax.set_xlim(all_dates.min(), all_dates.max())
    ax.set_title("Mean Attention Over Time", fontsize=14)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Attention", fontsize=12)

    tick_dates = all_dates[::2]
    ax.set_xticks(tick_dates)
    ax.set_xticklabels(
        [d.strftime("%b\n%Y") for d in tick_dates],
        fontsize=8
    )

    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, filename)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[Attention] Saved: {save_path}")


def split_attention_selections(
    attention_df: pd.DataFrame,
    percentile: float = 90,
    std_k: float = 1.0,
    top_k: int = 3,
    peak_alpha: float = 0.6,
    mean_factor: float = 1.5,
):
    """
    Returns five DataFrames:
      1) Percentile-based selection
      2) Mean + k*Std selection
      3) Top-K selection
      4) Relative-to-peak selection
      5) Global-mean-threshold selection
    """
    attn = attention_df["mean_attention"].values

    percentile_threshold = np.percentile(attn, percentile)
    df_percentile = attention_df[attention_df["mean_attention"] >= percentile_threshold].copy()
    df_percentile["selection_method"] = "percentile"
    df_percentile["threshold"] = percentile_threshold

    # mu = attn.mean()
    # sigma = attn.std()
    # std_threshold = mu + std_k * sigma
    # df_std = attention_df[attention_df["mean_attention"] >= std_threshold].copy()
    # df_std["selection_method"] = "mean_plus_k_std"
    # df_std["threshold"] = std_threshold

    df_topk = (
        attention_df
        .sort_values("mean_attention", ascending=False)
        .head(top_k)
        .copy()
    )
    df_topk["selection_method"] = "top_k"
    df_topk["threshold"] = np.nan

    peak_threshold = peak_alpha * attn.max()
    df_peak = attention_df[attention_df["mean_attention"] >= peak_threshold].copy()
    df_peak["selection_method"] = "relative_to_peak"
    df_peak["threshold"] = peak_threshold

    # mean_threshold = mean_factor * mu
    # df_global_mean = attention_df[attention_df["mean_attention"] >= mean_threshold].copy()
    # df_global_mean["selection_method"] = "global_mean_scaled"
    # df_global_mean["threshold"] = mean_threshold

    #return df_percentile, df_std, df_topk, df_peak, df_global_mean
    return df_percentile, df_topk, df_peak


def plot_attention_with_dates_with_coords(
    tft_model,
    plots_dir,
    forecast_start_date,
    width: int = 1000,
    height: int = 500,
    filename: str = "attention_mean_time_dated_coords.png",
):
    attn = (
        tft_model
        .mean_on_batch(tft_model.interpretability_params["attn_wts"])
        .mean(dim=0)
        .cpu()
        .numpy()
    )

    if attn.ndim == 3:
        attn = attn.mean(axis=0)

    input_size = tft_model.input_size
    h = tft_model.h

    forecast_attn = attn[input_size:, :]
    mean_attn = forecast_attn.mean(axis=0)

    forecast_start = pd.Timestamp(forecast_start_date)

    past_dates = pd.date_range(
        end=forecast_start - pd.DateOffset(months=1),
        periods=input_size,
        freq="MS"
    )
    future_dates = pd.date_range(
        start=forecast_start,
        periods=h,
        freq="MS"
    )
    all_dates = past_dates.append(future_dates)

    attention_coordinates = pd.DataFrame({
        "relative_month": range(-input_size, h),
        "month": all_dates,
        "mean_attention": mean_attn
    })

    os.makedirs(plots_dir, exist_ok=True)
    coords_path = os.path.join(plots_dir, "attention_mean_time_with_dates_coordinates.csv")
    attention_coordinates.to_csv(coords_path, index=False)
    print(f"[Attention] Coordinates saved: {coords_path}")

    (
        df_percentile,
        #df_std,
        df_topk,
        df_peak,
        #df_global_mean,
    ) = split_attention_selections(
        attention_coordinates,
        percentile=90,
        std_k=1.0,
        top_k=3,
        peak_alpha=0.6,
        mean_factor=1.5
    )

    df_percentile.to_csv(os.path.join(plots_dir, "attention_percentile_selected.csv"), index=False)
    # df_std.to_csv(os.path.join(plots_dir, "attention_mean_plus_std_selected.csv"), index=False)
    df_topk.to_csv(os.path.join(plots_dir, "attention_topk_selected.csv"), index=False)
    df_peak.to_csv(os.path.join(plots_dir, "attention_relative_to_peak_selected.csv"), index=False)
    # df_global_mean.to_csv(os.path.join(plots_dir, "attention_global_mean_selected.csv"), index=False)

    print("[Attention] Saved attention selections: percentile, top-k, relative to peak")
