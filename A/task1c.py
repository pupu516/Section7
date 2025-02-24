import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

# Sample sizes for bootstrapping
sample_sizes = [5, 15, 40, 60, 90, 150, 210, 300, 400]
num_bootstraps = 100  # Number of bootstrap resamples

# Load datasets
datasets = {}
for i in range(1, 4):
    with open(f"dataset_{i}.json", "r") as file:
        datasets[f"Dataset {i}"] = np.array(json.load(file))  # Convert to NumPy array

# Function to perform bootstrapping
def bootstrap(data, sample_sizes, num_bootstraps):
    results = {}

    for size in sample_sizes:
        means = []
        variances = []

        for _ in range(num_bootstraps):
            resample = np.random.choice(data, size=size, replace=True)  # Resample with replacement
            means.append(np.mean(resample))
            variances.append(np.var(resample, ddof=1))  # Unbiased variance estimation

        results[size] = {"means": means, "variances": variances}

    return results

# Perform bootstrapping for each dataset
bootstrap_results = {name: bootstrap(data, sample_sizes, num_bootstraps) for name, data in datasets.items()}

# Plot (3x3) histograms for expectation values
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for dataset_idx, (dataset_name, results) in enumerate(bootstrap_results.items()):
    for i, size in enumerate(sample_sizes):
        row, col = dataset_idx, i % 3  # Position in subplot

        ax = axes[row, col]
        ax.hist(results[size]["means"], bins=15, alpha=0.7, color='b')
        ax.set_title(f"{dataset_name}, Size={size}")
        ax.set_xlabel("Expectation Value")
        ax.set_ylabel("Frequency")

plt.tight_layout()
plt.savefig('part1c.png')

# Displaying bootstrapped variances for comparison
variance_data = {dataset_name: {size: np.mean(results[size]["variances"]) for size in sample_sizes} for dataset_name, results in bootstrap_results.items()}
variance_df = pd.DataFrame(variance_data, index=sample_sizes)

print("\nBootstrapped Variances:\n")
print(variance_df)

