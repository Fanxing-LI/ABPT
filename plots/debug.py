import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Sample data (noisy sine wave)
np.random.seed(0)
x = np.linspace(0, 10, 100)
data = np.sin(x) + np.random.normal(0, 0.5, x.shape)

# Define smoothing parameters
window_size = 5
alpha = 0.1
sigma = 2

# Apply Simple Moving Average
sma_smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Apply Gaussian Smoothing
gaussian_smoothed = gaussian_filter1d(data, sigma=sigma)

# Apply Exponential Moving Average
ema_smoothed = np.zeros_like(data)
ema_smoothed[0] = data[0]
for i in range(1, len(data)):
    ema_smoothed[i] = alpha * data[i] + (1 - alpha) * ema_smoothed[i - 1]

# Plot original and smoothed data for comparison
plt.figure(figsize=(12, 8))

# Original data
plt.plot(x, data, label='Original Data', color='black', alpha=0.5)

# Simple Moving Average
plt.plot(x[(window_size-1):], sma_smoothed, label='Simple Moving Average', color='blue')

# Gaussian Smoothing
plt.plot(x, gaussian_smoothed, label='Gaussian Filter', color='red')

# Exponential Moving Average
plt.plot(x, ema_smoothed, label='Exponential Moving Average', color='green')

# Labels and legend
plt.title('Comparison of Smoothing Techniques')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
