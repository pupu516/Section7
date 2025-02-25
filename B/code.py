import numpy as np
import matplotlib.pyplot as plt

# Given parameters
a = 4
b = 4

# Target probability density function (PDF)
def target_pdf(t):
    return np.exp(-b * t) * (np.cos(a * t) ** 2)

# Rejection Sampling using Uniform Proposal Distribution
def rejection_sampling_uniform(n_samples, t_max):
    accepted_samples = []
    rejected_samples = 0

    while len(accepted_samples) < n_samples:
        t_proposed = np.random.uniform(0, t_max)  # Uniform proposal
        y_proposed = np.random.uniform(0, 1)  # Uniform in [0,1] for acceptance
        if y_proposed < target_pdf(t_proposed):  # Accept-Reject condition
            accepted_samples.append(t_proposed)
        else:
            rejected_samples += 1

    rejection_ratio = rejected_samples / (n_samples + rejected_samples)
    return np.array(accepted_samples), rejection_ratio

# Rejection Sampling using Exponential Proposal Distribution
def rejection_sampling_exponential(n_samples, lambda_exp=2):
    accepted_samples = []
    rejected_samples = 0

    while len(accepted_samples) < n_samples:
        t_proposed = np.random.exponential(1 / lambda_exp)  # Exponential proposal
        y_proposed = np.random.uniform(0, 1)  # Uniform in [0,1] for acceptance
        if y_proposed < (target_pdf(t_proposed) / (lambda_exp * np.exp(-lambda_exp * t_proposed))):
            accepted_samples.append(t_proposed)
        else:
            rejected_samples += 1

    rejection_ratio = rejected_samples / (n_samples + rejected_samples)
    return np.array(accepted_samples), rejection_ratio

# Generate samples using both methods
n_samples = 10000  # Number of samples to generate
t_max = 5  # Max time for uniform proposal

# Uniform proposal sampling
samples_uniform, rejection_ratio_uniform = rejection_sampling_uniform(n_samples, t_max)

# Exponential proposal sampling
samples_exponential, rejection_ratio_exponential = rejection_sampling_exponential(n_samples)

# Plot histograms of generated samples
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(samples_uniform, bins=50, density=True, alpha=0.6, color='b', label="Uniform Proposal")
axes[0].set_title(f"Rejection Sampling (Uniform Proposal)\nRejection Ratio: {rejection_ratio_uniform:.3f}")
axes[0].set_xlabel("t")
axes[0].set_ylabel("Density")

axes[1].hist(samples_exponential, bins=50, density=True, alpha=0.6, color='r', label="Exponential Proposal")
axes[1].set_title(f"Rejection Sampling (Exponential Proposal)\nRejection Ratio: {rejection_ratio_exponential:.3f}")
axes[1].set_xlabel("t")
axes[1].set_ylabel("Density")

plt.tight_layout()
plt.savefig('plot.png')

# Display rejection ratios for comparison
import pandas as pd
rejection_ratios_df = pd.DataFrame({
    "Method": ["Uniform Proposal", "Exponential Proposal"],
    "Rejection Ratio": [rejection_ratio_uniform, rejection_ratio_exponential]
})

print(rejection_ratios_df)

