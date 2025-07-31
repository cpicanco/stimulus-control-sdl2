import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
np.random.seed(42)
x = np.arange(20)
y = np.random.randn(20)
binary = np.random.choice([False, True], 20)

# Create subplots with shared x-axis
fig, (ax1, ax2) = plt.subplots(
    2, 1, 
    figsize=(8, 6), 
    sharex=True, 
    gridspec_kw={'height_ratios': [3, 1]}
)

# Top plot: Scatter plot
ax1.scatter(x, y, c='blue', edgecolor='k', label='Data')
ax1.set_ylabel('Y Values')
ax1.grid(True)
ax1.legend()

# Bottom plot: Horizontal binary indicators
# Map True/False to colors and positions (e.g., 1 and 0)
colors = ['red' if not val else 'green' for val in binary]
ax2.scatter(x, np.ones_like(x), c=colors, marker='s', edgecolor='k', s=50)

# Customize the binary plot
ax2.set_yticks([1])  # Single tick to center labels (optional)
ax2.set_yticklabels(['State'])
ax2.set_ylim(0.5, 1.5)
ax2.grid(axis='x')
ax2.set_xlabel('X Values')

plt.tight_layout()
plt.show()