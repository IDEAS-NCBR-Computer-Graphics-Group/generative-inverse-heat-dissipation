import numpy as np
import matplotlib
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
    """
    Plot multiple matrices in a grid layout with consistent value range.

    Args:
        matrices (list of 2D arrays): List of 2D matrices to plot as heatmaps.
        titles (list of str, optional): List of titles for each plot. Defaults to None.
        columns (int, optional): Number of columns for the grid. Defaults to 5.
        value_range (tuple, optional): Tuple (vmin, vmax) for consistent color range. Defaults to None.
    """
    # Validate input sizes
    num_matrices = len(matrices)
    if titles is None:
        titles = [f"Matrix {i+1}" for i in range(num_matrices)]
    elif len(titles) != num_matrices:
        raise ValueError("Number of titles must match the number of matrices.")

    # Calculate grid size
    cols = columns
    rows = int(np.ceil(num_matrices / cols))

    # Determine value range
    if value_range is None:
        vmin = min(matrix.min() for matrix in matrices)
        vmax = max(matrix.max() for matrix in matrices)
    else:
        vmin, vmax = value_range

    # Create the subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).reshape(-1)  # Flatten axes for easy indexing

    for i, matrix in enumerate(matrices):
        ax = axes[i]
        im = ax.imshow(matrix, cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_title(titles[i])
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        fig.colorbar(im, ax=ax)

    # Hide unused subplots
    for ax in axes[num_matrices:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()

  
def plot_matrix_side_by_side(mat1, mat2, title="Matrices Side by Side", cmap='hot', labels=None):
    """
    Plots two matrices side-by-side using matplotlib.

    Args:
        mat1: The first matrix (NumPy array or list of lists).
        mat2: The second matrix (NumPy array or list of lists).
        title: The overall title for the plot.
        cmap: The colormap to use for the heatmaps (e.g., 'viridis', 'plasma', 'magma', 'inferno', 'hot', etc.)
        labels:  A list or tuple of strings to use for individual subplot titles.  Defaults to None

    Returns:
        None. Displays the plot.
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    # Plot the first matrix
    im1 = axes[0].imshow(mat1, cmap=cmap, interpolation='nearest')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)  # Consistent colorbar placement

    if labels and len(labels) >= 1: # Check if labels provided
        axes[0].set_title(labels[0])
    else:
        axes[0].set_title("LBM")


    # Plot the second matrix
    im2 = axes[1].imshow(mat2, cmap=cmap, interpolation='nearest')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    if labels and len(labels) >= 2: # Check if labels provided
        axes[1].set_title(labels[1])
    else:
         axes[1].set_title("DFT")

    fig.suptitle(title)  # Overall title


    plt.tight_layout()  # Adjusts subplot params so that subplots are nicely fit in the figure.
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

def calc_Fo(sigma, L):
    Fo = sigma / np.power(L, 2)
    return Fo

def get_timesteps_from_sigma(diffusivity, sigma):
    # sigma = np.sqrt(2 * diffusivity * tc)
    tc = np.power(sigma, 2)/(2*diffusivity)
    return int(tc)

def get_sigma_from_Fo(Fo, L):
    sigma = Fo * np.power(L, 2)
    return sigma

def get_timesteps_from_Fo_niu_L(Fo, diffusivity, L):
    sigma = get_sigma_from_Fo(Fo, L)
    tc = get_timesteps_from_sigma(diffusivity,sigma)
    return int(tc)

def get_Fo_from_tc_niu_L(tc, diffusivity, L):
    sigma = np.sqrt(2. * diffusivity * tc)
    Fo = calc_Fo(sigma, L)
    return Fo

def calculate_t_niu_array(Fo, niu_min, niu_max, L):
  """Calculates `t` and `niu` for each element in `Fo,
  knowing that Fo=np.sqrt(2*t*niu)/(L*L).

  Assumptions:
    `t` and `niu` are both monotonically increasing.
    `t` is a positive integer.
    `niu` is a float within range `niu_min` and `niu_max`.
    `L` is a positive float.
    `Fo` is a NumPy array.

  Args:
    Fo: The NumPy array of `Fo` values.
    niu_min: The minimum value of `niu`.
    niu_max: The maximum value of `niu`.

  Returns:
    Two NumPy arrays containing the values of `t` and `niu`
    corresponding to each element in `Fo`.
  """

  dt_values = []
  niu_values = []

  realizable_dFo = []
  niu_realizable_values= []

  for i in range(1, len(Fo)):
    dFo = Fo[i] - Fo[i-1]

    dt = 1
    while True:
      niu = ((L*L*dFo) ** 2)/ (2 * dt)
      if niu <= niu_max:
        dt_values.append(dt)
        niu_values.append(niu)

        if niu < niu_min:
          dFo_step_realizable = np.sqrt(2*dt*niu_min)/np.power(L, 2)
          niu_realizable_values.append(niu_min)
        else:
          dFo_step_realizable = get_Fo_from_tc_niu_L(dt, niu, L)
          niu_realizable_values.append(niu)

        realizable_dFo.append(dFo_step_realizable)
        break
      else:
        # print(f"niu out of range at i= {i} \t dt={dt} niu={niu[-1]:.2e} \t dFo={dFo:.2e}, Fo = {Fo[i]:.2e}")
        dt += 1

      if dt > 1000:
        break

  return np.array(dt_values, dtype=int), np.array(niu_values), np.array(niu_realizable_values), np.array(realizable_dFo)

def calculate_t_niu_array_from_0(Fo, niu_min, niu_max, L):
  """Calculates `t` and `niu` for each element in `Fo,
  knowing that Fo=np.sqrt(2*t*niu)/(L*L).

  Assumptions:
    `t` and `niu` are both monotonically increasing.
    `t` is a positive integer.
    `niu` is a float within range `niu_min` and `niu_max`.
    `L` is a positive float.
    `Fo` is a NumPy array.

  Args:
    Fo: The NumPy array of `Fo` values.
    niu_min: The minimum value of `niu`.
    niu_max: The maximum value of `niu`.

  Returns:
    Two NumPy arrays containing the values of `t` and `niu`
    corresponding to each element in `Fo`.
  """

  dt_values = []
  niu_values = []

  realizable_dFo = []
  niu_realizable_values= []

  for i in range(0, len(Fo)):
    dFo = Fo[i]

    dt = 1
    while True:
      niu = ((L*L*dFo) ** 2)/ (2 * dt)
      if niu <= niu_max:
        dt_values.append(dt)
        niu_values.append(niu)

        if niu < niu_min:
          dFo_step_realizable = np.sqrt(2*dt*niu_min)/np.power(L, 2)
          niu_realizable_values.append(niu_min)
        else:
          dFo_step_realizable = get_Fo_from_tc_niu_L(dt, niu, L)
          niu_realizable_values.append(niu)

        realizable_dFo.append(dFo_step_realizable)
        break
      else:
        # print(f"niu out of range at i= {i} \t dt={dt} niu={niu[-1]:.2e} \t dFo={dFo:.2e}, Fo = {Fo[i]:.2e}")
        dt += 1

      if dt > 100000000:
        break

  return np.array(dt_values, dtype=int), np.array(niu_values), np.array(niu_realizable_values), np.array(realizable_dFo)