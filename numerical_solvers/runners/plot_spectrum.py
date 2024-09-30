
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from PIL import Image
import cv2



# %%
spectrums_array = np.read_from_csv(asdfbks)
target_dpi = 100

fig, ax = plt.subplots(figsize=(256, 2.56), dpi=target_dpi)

ax.imshow(
spectrums_array,
aspect='auto',
extent=[min(self.iterations), max(self.iterations), min(SpectrumHeatmapPlotter.k_values), max(SpectrumHeatmapPlotter.k_values)],
origin='lower',
norm=LogNorm(),
cmap='inferno'
)