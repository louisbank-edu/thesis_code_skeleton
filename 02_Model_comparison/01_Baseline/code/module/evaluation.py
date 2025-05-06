import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_mad(actual, forecast):
    #return mean_absolute_error(actual, forecast)
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    errors = np.abs(actual - forecast)
    return float(errors.sum() / errors.shape[0])

def error_summary(y_true, forecasts_dict):
    summary = {}
    y_true = np.asarray(y_true)
    for name, y_pred in forecasts_dict.items():
        err = np.abs(y_true - np.asarray(y_pred))
        summary[name] = {
            "mean":   float(np.mean(err)),
            "median": float(np.median(err)),
            "std":    float(np.std(err))
        }
    return summary

def error_summary_excluding_fallbacks(y_true, forecasts_dict, flags_dict):
    y_true = np.asarray(y_true)
    total = y_true.shape[0]
    summary_nofb = {}
    fb_ratio     = {}

    for name, y_pred in forecasts_dict.items():
        y_pred = np.asarray(y_pred)
        flags = np.asarray(flags_dict[name], dtype=bool)
        if flags.ndim == 0:
            flags = np.full(total, flags, dtype=bool)
        fb_count = flags.sum()
        fb_ratio[name] = float(fb_count) / total

        if fb_count >= total:
            summary_nofb[name] = {"mean": np.nan, "median": np.nan, "std": np.nan}
        else:
            mask = ~flags
            err_nf = np.abs(y_true[mask] - y_pred[mask])
            summary_nofb[name] = {
                "mean":   float(np.mean(err_nf)),
                "median": float(np.median(err_nf)),
                "std":    float(np.std(err_nf))
            }

    return summary_nofb, fb_ratio

def format_summary_text(material_id, train_index, test_index,
                        summary_stats, best_mean, best_median,
                        nonfb_stats, fb_ratio):
    start_train = train_index[0].date()
    end_train   = train_index[-1].date()
    start_test  = test_index[0].date()
    end_test    = test_index[-1].date()

    lines = [
        f"Material: {material_id}",
        f"Train Period: {start_train} to {end_train}",
        f"Test Period:  {start_test} to {end_test}",
        "",
        "Forecast Error (MAD) Summary:"
    ]

    for method in sorted(summary_stats):
        o  = summary_stats[method]
        nf = nonfb_stats[method]
        r  = fb_ratio[method]

        lines.append(
            f"{method.capitalize():<15} "
            f"mean = {o['mean']:.4f}, "
            f"median = {o['median']:.4f}, "
            f"std = {o['std']:.4f}, "
            f"fb_ratio = {r:.4f}"
        )
        lines.append(
            f"{'':15} "
            f"(excl. fallback) mean = {nf['mean']:.4f}, "
            f"median = {nf['median']:.4f}, "
            f"std = {nf['std']:.4f}"
        )

    lines.extend([
        "",
        f"Best Method (by Mean):   {best_mean}",
        f"Best Method (by Median): {best_median}",
        ""
    ])

    return "\n".join(lines)

def format_overall_summary(
    overall_df,
    best_mean_counts,
    best_median_counts,
    method_names,
    include_nofb: bool = False
):
    """
    Generate an overall summary text list including:
      - Average MAD (mean, median, std)
      - avg_fb_ratio
      - (optional) non-fallback mean & median
      - Best‑Model counts across materials
    """
    lines = []
    for method in method_names:
        mean_overall   = overall_df[f"{method}_mean"].mean()
        median_overall = overall_df[f"{method}_median"].median()
        std_overall    = overall_df[f"{method}_std"].mean()
        fb_overall     = overall_df[f"{method}_fb_ratio"].mean()

        line = (
            f"{method.capitalize():<15} "
            f"Average MAD: mean = {mean_overall:.4f}, "
            f"median = {median_overall:.4f}, "
            f"std = {std_overall:.4f}, "
            f"avg_fb_ratio = {fb_overall:.4f}"
        )

        if include_nofb:
            mean_nofb   = overall_df[f"{method}_mean_nofb"].mean()
            median_nofb = overall_df[f"{method}_median_nofb"].median()
            line += (
                f", non‑fb mean = {mean_nofb:.4f}, "
                f"non‑fb median = {median_nofb:.4f}"
            )

        lines.append(line)

    lines.extend([
        "",
        "Best Model Counts across Materials (by Mean):"
    ])
    for method in method_names:
        lines.append(f"  {method}: {best_mean_counts.get(method, 0)}")

    lines.extend([
        "",
        "Best Model Counts across Materials (by Median):"
    ])
    for method in method_names:
        lines.append(f"  {method}: {best_median_counts.get(method, 0)}")

    return lines