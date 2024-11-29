import numpy as np
import matplotlib.pyplot as plt

def plot_matrix(matrix, title="Temperature Map (Matrix)", range=None):
  # Create a heatmap using matplotlib
  plt.imshow(
    matrix,
    cmap='hot',
    interpolation='nearest',
    vmin=range[0] if range else None,
    vmax=range[1] if range else None
    )
  plt.colorbar()
  plt.title(title)
  plt.xlabel("X-axis")
  plt.ylabel("Y-axis")
  plt.show()

def plot_matrices_in_grid(matrices, titles=None, columns=5, value_range=None):
    num_matrices = len(matrices)
    if titles is None: titles = [f"Matrix {i+1}" for i in range(num_matrices)]
    elif len(titles) != num_matrices: raise ValueError("Number of titles must match the number of matrices.")

    cols = columns
    rows = int(np.ceil(num_matrices / cols))

    if value_range is None:
        vmin = min(matrix.min() for matrix in matrices)
        vmax = max(matrix.max() for matrix in matrices)
    else:
        vmin, vmax = value_range

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).reshape(-1)

    for i, matrix in enumerate(matrices):
        ax = axes[i]
        im = ax.imshow(matrix, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(titles[i])
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        fig.colorbar(im, ax=ax)

    for ax in axes[num_matrices:]:ax.axis('off')

    plt.tight_layout()
    plt.show()
  
def plot_matrix_side_by_side(mat1, mat2, title="Matrices Side by Side", cmap='hot', labels=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im1 = axes[0].imshow(mat1, cmap=cmap, interpolation='nearest')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    if labels and len(labels) >= 1: axes[0].set_title(labels[0])
    else: axes[0].set_title("LBM")

    im2 = axes[1].imshow(mat2, cmap=cmap, interpolation='nearest')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    if labels and len(labels) >= 2:axes[1].set_title(labels[1])
    else:axes[1].set_title("DFT")
    fig.suptitle(title)

    plt.tight_layout()
    plt.show()
   
def display_schedules(Fo, Fo_realizable):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    plt.plot(Fo,  'rx', label=f'Fo_schedule')
    plt.plot(Fo_realizable,  'gx', label=f'Fo_realizable')

    ax.grid(True, which="both", ls="--")
    ax.set_xlabel(r"denoising steps")
    ax.set_ylabel(r"Fo")

    plt.legend()
