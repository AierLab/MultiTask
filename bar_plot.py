import matplotlib.pyplot as plt
import numpy as np

# Data from the table
methods = ["All-in-One", "Chen et al.", "TransWeather", "Ours"]
params_million = [44.00, 28.71, 38.05, 8.87]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Bar plot with academic color scheme
colors = ["#B7B7EB", "#9BBBE1", "#F09BA0", "#EAB883"]
bars = ax.bar(methods, params_million, color=colors, edgecolor="black")

# Labels and title
ax.set_xlabel("Methods", fontsize=14)
ax.set_ylabel("Parameters (Million)", fontsize=14)
ax.set_title("Parameter Count Comparison Among Methods", fontsize=16)

# Add values on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}", ha="center", va="bottom", fontsize=12)

# Show grid for better readability
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Save the plot as an image
plt.tight_layout()
plt.savefig("/home/4paradigm/WGWS-Net/parameter_comparison_chart.png", dpi=300)  # Save as PNG with 300 DPI

# Show plot
plt.show()
