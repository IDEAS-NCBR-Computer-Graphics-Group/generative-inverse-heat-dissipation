import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt

from visualization.KolmogorovSpectrumPlotter import compute_kolmogorov_spectrum


np.random.seed(123)




def compute_divergence(u, v, dx, dy):
    """
    Computes the divergence of a 2D vector field (u, v) using finite difference stencils.
    """
    div_u = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
    div_v = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)
    return div_u + div_v

def plot_velocity_components(u_random, v_random, u_rfm, v_rfm, u_spec, v_spec, time):
    """
    Plot the u and v velocity components for both RFM and Spectral methods in a single figure with four subplots.

    Args:
        u_rfm: 2D array of x-velocity fluctuations from the RFM method.
        v_rfm: 2D array of y-velocity fluctuations from the RFM method.
        u_spec: 2D array of x-velocity fluctuations from the Spectral method.
        v_spec: 2D array of y-velocity fluctuations from the Spectral method.
        time: Time at which the turbulence was computed.
    """
    plt.figure(figsize=(12, 15))

    # Plot random method u-component
    plt.subplot(3, 2, 1)
    plt.contourf(u_random, levels=50, cmap='jet')
    plt.colorbar(label='u-velocity')
    plt.title(f'Synthetic random Turbulence u-component at t={time:.2f}')

    # Plot random method v-component
    plt.subplot(3, 2, 2)
    plt.contourf(v_random, levels=50, cmap='jet')
    plt.colorbar(label='v-velocity')
    plt.title(f'Synthetic random Turbulence v-component at t={time:.2f}')

    # Plot RFM method u-component
    plt.subplot(3, 2, 3)
    plt.contourf(u_rfm, levels=50, cmap='jet')
    plt.colorbar(label='u-velocity')
    plt.title(f'Synthetic RFM Turbulence u-component at t={time:.2f}')

    # Plot RFM method v-component
    plt.subplot(3, 2, 4)
    plt.contourf(v_rfm, levels=50, cmap='jet')
    plt.colorbar(label='v-velocity')
    plt.title(f'Synthetic RFM Turbulence v-component at t={time:.2f}')

    # Plot Spectral method u-component
    plt.subplot(3, 2, 5)
    plt.contourf(u_spec, levels=50, cmap='jet')
    plt.colorbar(label='u-velocity')
    plt.title(f'Synthetic Spectral Turbulence u-component at t={time:.2f}')

    # Plot Spectral method v-component
    plt.subplot(3, 2, 6)
    plt.contourf(v_spec, levels=50, cmap='jet')
    plt.colorbar(label='v-velocity')
    plt.title(f'Synthetic Spectral Turbulence v-component at t={time:.2f}')

    plt.tight_layout()
    plt.show()
    



def plot_kolmogorov_spectrum(u_random, v_random, u_rfm, v_rfm, u_spec, v_spec, domain_size):
    """
    Plot the Kolmogorov spectrum for both the RFM and Spectral methods.

    Args:
        u_rfm: 2D array of x-velocity fluctuations from the RFM method.
        v_rfm: 2D array of y-velocity fluctuations from the RFM method.
        u_spec: 2D array of x-velocity fluctuations from the Spectral method.
        v_spec: 2D array of y-velocity fluctuations from the Spectral method.
        domain_size: tuple (Lx, Ly) representing the size of the domain.
    """
    Lx, Ly = domain_size

    # Compute the energy spectra for both methods
    
    k_random, energy_spectrum_random = compute_kolmogorov_spectrum(u_random, v_random, Lx, Ly)
    k_rfm, energy_spectrum_rfm = compute_kolmogorov_spectrum(u_rfm, v_rfm, Lx, Ly)
    k_spec, energy_spectrum_spec = compute_kolmogorov_spectrum(u_spec, v_spec, Lx, Ly)

    # Plot the energy spectra
    plt.figure(figsize=(10, 6))
    plt.loglog(k_random[1:len(k_random)//2], energy_spectrum_random[1:len(k_random)//2], '>', label='Random Method')
    plt.loglog(k_rfm[1:len(k_rfm)//2], energy_spectrum_rfm[1:len(k_rfm)//2], 'o', label='RFM Method')
    plt.loglog(k_spec[1:len(k_spec)//2], energy_spectrum_spec[1:len(k_spec)//2], 'x', label='Spectral Method')

    # Plot the -5/3 slope line for reference
    k_ref = k_rfm[1:len(k_rfm)//6]
    E_ref = k_ref**(-5.0/3.0)
    E_ref *= energy_spectrum_rfm[10] / E_ref[0]  # Adjust scaling to match the spectrum
    plt.loglog(k_ref, E_ref, 'r--', label='-5/3 slope')

    plt.xlabel(r"Wavenumber $k$")
    plt.ylabel(r"Energy Spectrum $E(k)$")


    # plt.ylim(1E-0, 1E4)
    # plt.ylim(1E-1, 1E5)
    # plt.xlim(2.0 * np.pi, 2.0 * np.pi / (1 / 20))


    plt.title("Kolmogorov Spectrum - RFM vs Spectral Method")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()
    