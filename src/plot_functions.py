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

def plot_autocorrelation_profiles(
    structure_autocorr_data, ensemble_autocorr_data, cfg=None
):
    """Plots spatial autocorrelation profiles and 1/e correlation lengths (l_c)

    for individual structures alongside the global ensemble average.
    """
    # Fallback formatting configuration if cfg is not provided
    if cfg is None:
        cfg = {
            "lines": {"width": 1.5},
            "axes": {
                "labelsize": 11,
                "labelweight": "bold",
                "labelcolor": "black",
                "linewidth": 1.0,
            },
            "ticks": {"width": 1.0, "length": 4, "labelsize": 10, "labelweight": "normal"},
            "legend": {"fontsize": 9, "frameon": True},
        }

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    threshold = 1 / np.e
    color_map = plt.cm.tab10

    # 1. Plot individual structure profiles
    for i, (name, data) in enumerate(structure_autocorr_data.items()):
        line_color = color_map(i % 10)
        dist_1e = data["dist_1e"]

        # Format label with chemical/structure formatting and l_c value
        if name.lower() in ["endoplasmic reticulum", "er"]:
            label_name = "ER"
        elif name.lower() in ["microtubules", "mt"]:
            label_name = "MT"
        else:
            label_name = name.capitalize()

        lc_str = f"{dist_1e:.2f}" if dist_1e is not None else "N/A"
        label_str = rf"{label_name} ($\mathbf{{l_c = }}$ {lc_str} µm)"

        # Mean curve & std envelope
        ax.plot(
            data["x"],
            data["mean"],
            linestyle="--",
            linewidth=cfg["lines"]["width"],
            color=line_color,
            alpha=0.85,
            label=label_str,
        )
        ax.fill_between(
            data["x"],
            data["mean"] - data["std"],
            data["mean"] + data["std"],
            color=line_color,
            alpha=0.1,
        )

    # 2. Plot global ensemble average profile
    global_x = ensemble_autocorr_data["x"]
    global_mean = ensemble_autocorr_data["mean"]
    global_lc = ensemble_autocorr_data["dist_1e"]

    global_lc_str = f"{global_lc:.2f}" if global_lc is not None else "N/A"
    global_label = rf"Average ($\mathbf{{l_c = }}$ {global_lc_str} µm)"

    ax.plot(
        global_x,
        global_mean,
        color="black",
        linewidth=cfg["lines"]["width"] * 2,
        label=global_label,
    )

    # 3. Mark 1/e threshold and global l_c vertical indicator
    ax.axhline(threshold, color="black", linestyle=":", alpha=0.7, label="1/e Threshold")
    if global_lc is not None:
        ax.axvline(global_lc, color="black", linestyle=":", linewidth=1.5, alpha=0.8)

    # 4. Axes, styling, and legend formatting
    ax.set_xlabel(
        r"Distance [$\mathbf{\mu m}$]",
        fontsize=cfg["axes"]["labelsize"],
        fontweight=cfg["axes"]["labelweight"],
        color=cfg["axes"]["labelcolor"],
    )
    ax.set_ylabel(
        "Autocorrelation Function",
        fontsize=cfg["axes"]["labelsize"],
        fontweight=cfg["axes"]["labelweight"],
        color=cfg["axes"]["labelcolor"],
    )

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        colors=cfg["axes"]["labelcolor"],
        width=cfg["ticks"]["width"],
        length=cfg["ticks"]["length"],
        labelsize=cfg["ticks"]["labelsize"],
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(cfg["axes"]["labelcolor"])
        spine.set_linewidth(cfg["axes"]["linewidth"])

    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2, linestyle="--")

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=cfg["legend"]["fontsize"],
        frameon=cfg["legend"]["frameon"],
    )

    plt.tight_layout()
    plt.show()
    return fig, ax