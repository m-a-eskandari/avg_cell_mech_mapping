import numpy as np
import matplotlib.pyplot as plt

def plot_fkv_fit(
    frequencies,
    storage_modulus,
    loss_modulus,
    params,
    masked_indices=None,
    n_points=1000,
    title=None,
    figsize=(6, 4.5)
):
    A, B, alpha, beta = params

    # 1. Reconstruct the boolean mask inside the function
    mask = (storage_modulus > 0.0) & (loss_modulus > 0.0)
    if masked_indices:
        mask &= ~np.isin(np.arange(len(frequencies)), masked_indices)

    # 2. Generate dense frequency array for smooth model curve
    freqs_dense = np.geomspace(frequencies.min(), frequencies.max(), n_points)

    # 3. Evaluate complex shear modulus model
    g_fit = A * (1j * 2 * np.pi * freqs_dense) ** alpha + B * (1j * 2 * np.pi * freqs_dense) ** beta
    storage_modulus_fit = g_fit.real
    loss_modulus_fit = g_fit.imag

    fig, ax = plt.subplots(figsize=figsize)

    # Plot continuous fit curves
    ax.loglog(freqs_dense, storage_modulus_fit, label="G' fit", c="tab:blue", linestyle="-")
    ax.loglog(freqs_dense, loss_modulus_fit, label='G" fit', c="tab:orange", linestyle="-")

    # Plot valid (unmasked) data points
    ax.scatter(frequencies[mask], storage_modulus[mask], c="tab:blue", marker="o")
    ax.scatter(frequencies[mask], loss_modulus[mask], c="tab:orange", marker="o")

    # Plot masked data points (marked with 'x')
    if not np.all(mask):
        ax.scatter(frequencies[~mask], storage_modulus[~mask], c="k", marker="x", s=60, label="G' masked")
        ax.scatter(frequencies[~mask], loss_modulus[~mask], c="k", marker="x", s=60, label='G" masked')

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Modulus [Pa]")
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)

    plt.tight_layout()
    plt.show()
    
    return fig, ax

def plot_passive_psds(
        passive_psds,
        title=""
):
    n_plots = len(passive_psds)

    # Auto-calculate figure size based on the number of items
    fig, axes = plt.subplots(
        nrows=n_plots,
        ncols=1,
        figsize=(7, 2.5 * n_plots),
        sharex=True
    )

    if n_plots == 1:
        axes = [axes]

    for i, (ax, data) in enumerate(zip(axes, passive_psds)):
        # Auto-detect whether data is binned or raw dictionary, or a LabVIEW list
        if isinstance(data, dict):
            freq_key = "freq" if "freq" in data else "freq"
            psd_key = "psd_y" if "psd_y" in data else "psd_y"
            freq, psd = data[freq_key], data[psd_key]

        ax.loglog(freq, psd, color="tab:orange", label=f"Run {i}")
        ax.set_ylabel("PSD y [um²/Hz]")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(loc="upper right")
    
    axes[0].set_title(title)
    axes[-1].set_xlabel("Frequency [Hz]")
    plt.tight_layout()
    plt.show()

    return fig, axes


def plot_effective_energy(
        frequencies: np.ndarray,
        e_eff: np.ndarray,
        fit_params: tuple | list | np.ndarray,
        masked_indices: list | np.ndarray | None = None
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the measured effective energy spectrum alongside the fitted power-law model.

    Parameters:
    -----------
    frequencies : np.ndarray
        Array of driving frequencies [Hz].
    e_eff : np.ndarray
        Measured effective energy in units of [k_B T].
    fit_params : tuple, list, or np.ndarray
        Fitted model parameters (A, alpha, C) where E_eff(f) = A * f^alpha + C.
    masked_indices : list, np.ndarray, or None
        Indices of frequencies excluded from the fit due to instrumental noise.
    ax : plt.Axes or None
        Matplotlib Axes object. If None, a new figure and axes are created.

    Returns:
    --------
    fig, ax : tuple[plt.Figure, plt.Axes]
        The Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(7, 5), dpi=100)


    frequencies = np.asarray(frequencies)
    e_eff = np.asarray(e_eff)

    # Separate valid and masked points for distinct visualization
    if masked_indices is not None and len(masked_indices) > 0:
        mask = np.ones(len(frequencies), dtype=bool)
        mask[masked_indices] = False

        ax.scatter(
            frequencies[mask],
            e_eff[mask],
            color="navy",
            s=40,
            zorder=3,
            label="Valid Data"
        )
        ax.scatter(
            frequencies[~mask],
            e_eff[~mask],
            color="crimson",
            marker="x",
            s=50,
            linewidths=1.5,
            zorder=3,
            label="Masked (Excluded)"
        )
    else:
        ax.scatter(
            frequencies,
            e_eff,
            color="navy",
            s=40,
            zorder=3,
            label="Data Points"
        )

    # Evaluate model: E_eff(f) = A * f^alpha + C
    A, alpha, C = fit_params
    f_smooth = np.logspace(np.log10(frequencies.min()), np.log10(frequencies.max()), 300)
    e_eff_fit = A * (f_smooth ** alpha) + C

    # Plot fitted model curve
    label_fit = rf"Fit: ${A:.2f} \cdot f^{{{alpha:.2f}}} + {C:.2f}$"
    ax.plot(f_smooth, e_eff_fit, "k--", linewidth=2, zorder=2, label=label_fit)

    # Thermal threshold reference line (E_eff = 1 k_B T)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, zorder=1, label=r"$1\ k_B T$ Thermal Limit")

    # Formatting
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency $f$ [Hz]", fontsize=11)
    ax.set_ylabel(r"Effective Energy $E_{\mathrm{eff}}$ [$k_B T$]", fontsize=11)
    ax.set_title("Effective Energy Spectrum", fontsize=12, pad=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", frameon=True, fontsize=9)

    fig.tight_layout()
    plt.show()
    return fig, ax