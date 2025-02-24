import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Load datasets
datasets = []
for i in range(1, 4):
    with open(f"dataset_{i}.json", "r") as file:
        datasets.append(json.load(file))  # Each dataset contains 500 coin flips

# Function to compute posterior distribution
def bayesian_inference(data):
    N = len(data)  # Total number of flips
    M = sum(data)  # Number of heads
    alpha, beta_param = M + 1, (N - M) + 1  # Beta distribution parameters

    # Probability range for visualization
    p_values = np.linspace(0, 1, 1000)
    posterior = beta.pdf(p_values, alpha, beta_param)  # Compute Beta distribution

    # Compute mean and variance of posterior
    mean_p = alpha / (alpha + beta_param)
    variance_p = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))

    return p_values, posterior, mean_p, variance_p

# Plot posterior distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, data in enumerate(datasets):
    p_values, posterior, mean_p, variance_p = bayesian_inference(data)
    
    axes[i].plot(p_values, posterior, label=f"Dataset {i+1}")
    axes[i].axvline(mean_p, color='r', linestyle='--', label=f"Mean: {mean_p:.3f}")
    axes[i].set_title(f"Posterior Distribution (Dataset {i+1})")
    axes[i].set_xlabel("p (Probability of Heads)")
    axes[i].set_ylabel("Density")
    axes[i].legend()

plt.tight_layout()
plt.savefig('part_a.png')

