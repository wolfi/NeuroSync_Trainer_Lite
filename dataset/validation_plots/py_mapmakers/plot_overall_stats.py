import os
import re
import matplotlib.pyplot as plt

def extract_overall_stats(file_path):
    """
    Extract overall statistics from a given stats file.
    
    Args:
        file_path (str): Path to the stats file.
    
    Returns:
        dict: A dictionary containing the overall stats.
    """
    stats = {}
    with open(file_path, 'r') as file:
        for line in file:
            if "Mean Absolute Error (MAE)" in line:
                stats["MAE"] = float(re.search(r"([\d.]+)$", line).group(1))
            elif "Mean Absolute Percentage Error (MAPE)" in line:
                stats["MAPE"] = float(re.search(r"([\d.]+)$", line).group(1))
            elif "Mean Squared Error (MSE)" in line:
                stats["MSE"] = float(re.search(r"([\d.]+)$", line).group(1))
            elif "Root Mean Squared Error (RMSE)" in line:
                stats["RMSE"] = float(re.search(r"([\d.]+)$", line).group(1))
            elif "Correlation Coefficient (r)" in line:
                stats["r"] = float(re.search(r"([\d.]+)$", line).group(1))
    return stats

def plot_stats(stats_over_epochs, output_path):
    """
    Plot overall statistics over epochs.
    
    Args:
        stats_over_epochs (list of dict): List of stats dictionaries, one per epoch.
        output_path (str): Path to save the output plot.
    """
    epochs = list(range(1, len(stats_over_epochs) + 1))
    mae = [stat["MAE"] for stat in stats_over_epochs]
    mape = [stat["MAPE"] for stat in stats_over_epochs]
    mse = [stat["MSE"] for stat in stats_over_epochs]
    rmse = [stat["RMSE"] for stat in stats_over_epochs]
    correlation = [stat["r"] for stat in stats_over_epochs]
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, mae, label="MAE", marker='o')
    plt.plot(epochs, mape, label="MAPE", marker='o')
    plt.title("Error Metrics Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, mse, label="MSE", marker='o')
    plt.plot(epochs, rmse, label="RMSE", marker='o')
    plt.plot(epochs, correlation, label="Correlation Coefficient (r)", marker='o')
    plt.title("MSE, RMSE, and Correlation Coefficient Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def main(stats_directory, output_plot_path):
    """
    Main function to read stats files and plot the overall statistics.
    
    Args:
        stats_directory (str): Path to the directory containing stats files.
        output_plot_path (str): Path to save the output plot.
    """
    stats_over_epochs = []
    stats_files = sorted(
        [os.path.join(stats_directory, f) for f in os.listdir(stats_directory) if f.endswith(".txt")]
    )
    
    for stats_file in stats_files:
        stats = extract_overall_stats(stats_file)
        stats_over_epochs.append(stats)
    
    plot_stats(stats_over_epochs, output_plot_path)

if __name__ == "__main__":
    stats_dir = "dataset/validation_plots/stats"
    output_plot = "dataset/validation_plots/py_mapmakers/plots/stats_overview.png"
    main(stats_dir, output_plot)
