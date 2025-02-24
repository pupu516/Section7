import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

# Define range for N
N_values = np.arange(1, 11)

# Compute exact factorial values using Gamma function
factorial_values = [sp.gamma(N + 1) for N in N_values]

# Compute Stirling’s approximation
stirling_values = [np.sqrt(2 * np.pi * N) * (N / np.e) ** N for N in N_values]

# --- Plot 1: Factorial vs Stirling’s Approximation ---
plt.figure(figsize=(10, 5))
plt.scatter(N_values, factorial_values, label="Factorial (Gamma Function)", color='blue', marker='o')
plt.plot(N_values, stirling_values, label="Stirling's Approximation", color='red', linestyle="--")
plt.xlabel("N")
plt.ylabel("Factorial Value")
plt.title("Factorial vs Stirling’s Approximation")
plt.legend()
plt.show()

# --- Plot 2: Error between Exact Factorial and Stirling’s Approximation ---
error = np.abs(np.array(factorial_values) - np.array(stirling_values))

plt.figure(figsize=(10, 5))
plt.plot(N_values, error, label="Error", color="green", marker='o', linestyle="--")
plt.xlabel("N")
plt.ylabel("Error")
plt.title("Error in Stirling’s Approximation")
plt.legend()
plt.savefig('part_b.png')

