import os
import re
import matplotlib.pyplot as plt

def extract_per_dimension_stats(file_path):
    """
    Extract per-dimension statistics from a given stats file.
    
    Args:
        file_path (str): Path to the stats file.
    
    Returns:
        dict: A dictionary with dimensions as keys and their stats as values.
    """
    dimension_stats = {}
    current_dimension = None
    with open(file_path, 'r') as file:
        for line in file:
            # Match dimension names (e.g., EyeBlinkLeft:)
            dimension_match = re.match(r"^(\w+):$", line.strip())
            if dimension_match:
                current_dimension = dimension_match.group(1)
                dimension_stats[current_dimension] = {}
            elif current_dimension:
                # Match metric lines (e.g., MAE: 0.0696)
                metric_match = re.match(r"^(\w+): ([\d.]+)$", line.strip())
                if metric_match:
                    metric_name = metric_match.group(1)
                    metric_value = float(metric_match.group(2))
                    dimension_stats[current_dimension][metric_name] = metric_value
    return dimension_stats

def plot_dimension_stats(stats_over_epochs, output_base_path):
    """
    Plot per-dimension statistics over epochs and save each dimension's plot.
    
    Args:
        stats_over_epochs (list of dict): List of per-dimension stats dictionaries, one per epoch.
        output_base_path (str): Base path to save the plots.
    """
    # Get all dimension names
    dimensions = stats_over_epochs[0].keys()

    # Create the base folder for the plots
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    for dimension in dimensions:
        # Gather metrics for the current dimension
        epochs = list(range(1, len(stats_over_epochs) + 1))
        mae = [epoch_stats[dimension].get("MAE", None) for epoch_stats in stats_over_epochs]
        mape = [epoch_stats[dimension].get("MAPE", None) for epoch_stats in stats_over_epochs]
        mse = [epoch_stats[dimension].get("MSE", None) for epoch_stats in stats_over_epochs]
        rmse = [epoch_stats[dimension].get("RMSE", None) for epoch_stats in stats_over_epochs]
        correlation = [epoch_stats[dimension].get("Correlation Coefficient", None) for epoch_stats in stats_over_epochs]

        # Create a subfolder for the current dimension
        dimension_folder = os.path.join(output_base_path, dimension)
        if not os.path.exists(dimension_folder):
            os.makedirs(dimension_folder)

        # Plotting
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(epochs, mae, label="MAE", marker='o')
        plt.plot(epochs, mape, label="MAPE", marker='o')
        plt.title(f"Error Metrics Over Epochs - {dimension}")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, mse, label="MSE", marker='o')
        plt.plot(epochs, rmse, label="RMSE", marker='o')
        plt.plot(epochs, correlation, label="Correlation Coefficient (r)", marker='o')
        plt.title(f"MSE, RMSE, and Correlation Coefficient Over Epochs - {dimension}")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.legend()

        plt.tight_layout()
        output_file_path = os.path.join(dimension_folder, f"{dimension}_stats_overview.png")
        plt.savefig(output_file_path)
        plt.close()

def main(stats_directory, output_base_path):
    """
    Main function to read stats files and plot per-dimension statistics.
    
    Args:
        stats_directory (str): Path to the directory containing stats files.
        output_base_path (str): Base path to save the output plots.
    """
    stats_over_epochs = []
    stats_files = sorted(
        [os.path.join(stats_directory, f) for f in os.listdir(stats_directory) if f.endswith(".txt")]
    )

    for stats_file in stats_files:
        stats = extract_per_dimension_stats(stats_file)
        stats_over_epochs.append(stats)

    plot_dimension_stats(stats_over_epochs, output_base_path)

if __name__ == "__main__":
    stats_dir = "dataset/validation_plots/stats"
    output_base = "dataset/validation_plots/py_mapmakers/dimensions"
    main(stats_dir, output_base)
