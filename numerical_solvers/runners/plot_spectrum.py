
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from PIL import Image
import cv2
import pandas as pd
import os



# %%
spectrums_array = pd.read_csv('/home/computergraphics/Documents/jjmeixner/generative-inverse-heat-dissipation/numerical_solvers/runners/spectrums_array.csv')


spectrums_array = np.array(spectrums_array).T  # Transpose to have iterations on the x-axis

# Create the heatmap plot
plt.figure(figsize=(10, 10))
plt.title("Energy Spectrum Heatmap over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Wavenumber $k$")
plt.imshow(
    spectrums_array,
    aspect='auto',
    # extent=[0, 240, 0, 900],
    origin='lower',
    norm=LogNorm(),  # Use log scale for better visualization
    cmap='inferno'
)
plt.colorbar(label='Energy Spectrum $E(k)$')
plt.show()

print(spectrums_array.shape)


# %%

slice_k = 10
plt.figure(figsize=(10, 10))

plt.plot(spectrums_array[:,100])
plt.yscale("log")
plt.grid()
# Plot SSIM vs. Iterations
# ax.plot(iterations, ssim_values, 'g-', marker='o', label='SSIM')

# # Set y-axis limits for better visualization
# ax.set_ylim([0.96, 1])  # SSIM ranges from 0 to 1

# # Add grid, title, and labels
# ax.grid(True, which="both", ls="--")
# ax.set_title(self.title, fontsize=14)
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Structural Similarity Index (SSIM)')
# ax.legend(loc='best')

# %%

slice_k = 50
plt.figure(figsize=(10, 10))

plt.plot(spectrums_array[slice_k,:])