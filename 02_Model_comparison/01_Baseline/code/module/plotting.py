import os
import matplotlib.pyplot as plt


def plot_forecast_comparison(material_id, train_df, test_df, forecast_dict, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_df.index, train_df["demand"], label="Training Demand", color="grey", alpha=0.7)
    plt.plot(test_df.index, test_df["demand"], label="Test Actual Demand", marker="o", color="black")
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    for (i, (method, fc)) in enumerate(forecast_dict.items()):
        plt.plot(test_df.index, fc, label=method, marker="x", linestyle="--", color=colors[i % len(colors)])
    plt.axvline(x=test_df.index[0], color="red", linestyle=":", label="Train/Test Split")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.title(f"Forecast Comparison for Material {material_id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_best_method_histogram(best_method_counts, title, save_path):
    methods = list(best_method_counts.keys())
    counts = [best_method_counts[m] for m in methods]

    plt.figure(figsize=(8, 6))
    plt.bar(methods, counts, color='skyblue')
    plt.xlabel("Method", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.title(title, fontsize=12)
    # Rotate x-tick labels by 45° and set font size
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
