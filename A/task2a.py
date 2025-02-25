import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit

# Function to model exponential decay
def exponential_pdf(x, lambda_):
    return (1 / lambda_) * np.exp(-x / lambda_)

# Load decay datasets
datasets = {}
dataset_names = ["Vacuum", "Cavity"]
for name in dataset_names:
    with open(f"{name}_decay_dataset.json", "r") as file:
        datasets[name] = np.array(json.load(file))

# Function to estimate lambda using Maximum Likelihood Estimation (MLE)
def estimate_lambda(data):
    return np.mean(data)  # MLE estimate for exponential decay parameter λ

# Estimate λ for both datasets
lambda_estimates = {name: estimate_lambda(data) for name, data in datasets.items()}

# Plot histograms and fitted exponential distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, (name, data) in enumerate(datasets.items()):
    lambda_ = lambda_estimates[name]
    
    # Plot histogram of decay distances
    axes[i].hist(data, bins=50, density=True, alpha=0.6, color='b', label="Observed Data")
    
    # Generate theoretical exponential distribution
    x_values = np.linspace(0, max(data), 100)
    y_values = exponential_pdf(x_values, lambda_)
    axes[i].plot(x_values, y_values, 'r-', label=f"Fitted Exp. Dist. (λ={lambda_:.3f})")
    
    # Labels and title
    axes[i].set_title(f"{name} Decay Distribution")
    axes[i].set_xlabel("Decay Distance")
    axes[i].set_ylabel("Probability Density")
    axes[i].legend()

plt.tight_layout()
plt.savefig('task2_a.png')

# Display estimated decay constants
import pandas as pd
lambda_df = pd.DataFrame.from_dict(lambda_estimates, orient='index', columns=["Estimated λ"])
import ace_tools as tools
tools.display_dataframe_to_user(name="Estimated Decay Constants", dataframe=lambda_df)

