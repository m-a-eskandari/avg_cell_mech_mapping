import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

def show_multichannel(img_multichannel, title=""):
    """Display all channels of a 2D multichannel image (C, Y, X) side-by-side."""
    ch_count = img_multichannel.shape[0]
    fig, axes = plt.subplots(1, ch_count, figsize=(4 * ch_count, 4))

    # Handle single subplot case
    if ch_count == 1:
        axes = [axes]

    for c in range(ch_count):
        axes[c].imshow(img_multichannel[c], cmap='gray')
        axes[c].axis('off')

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
    print("")


def plot_nucleus_anchor_points(
        nucleus_sample_slice,
        nucleus_reference_image,
        nuc_anchor_x,
        nuc_anchor_y,
        nuc_anchor_u,
        nuc_anchor_v,
        nuc_thresh=100.0,
        figsize=(8, 8)
):
    """
    Plots the nucleus sample slice, template/sample borders, and computed nucleus anchor vectors.
    """
    # 1. Recompute borders for visualization overlay
    n_nucleus_ch = 255.0 * nucleus_sample_slice / np.max(nucleus_sample_slice)
    mask_sample = (n_nucleus_ch > nuc_thresh)
    mask_avg = (nucleus_reference_image > np.max(nucleus_reference_image) / 20.0)

    border_sample = mask_sample ^ ndi.binary_erosion(mask_sample)
    border_avg = mask_avg ^ ndi.binary_erosion(mask_avg)

    # 2. Setup figure
    fig, ax = plt.subplots(figsize=figsize)

    # Background image (sample intensity)
    ax.imshow(nucleus_sample_slice, cmap="gray", origin="upper")

    # Overlay sample border (cyan) and template border (magenta)
    by_smp, bx_smp = np.where(border_sample)
    by_avg, bx_avg = np.where(border_avg)

    # Individual Cell Nucleus Border -> RED
    ax.scatter(nuc_anchor_x + nuc_anchor_u, nuc_anchor_y + nuc_anchor_v, c="red", s=3, alpha=0.9, label="Individual Nucleus Border")

    # Average Cell Nucleus Border -> YELLOW
    ax.scatter(bx_avg, by_avg, c="lime", s=3, alpha=0.9, label="Average Nucleus Border")

    # Plot Anchor Vectors (yellow arrows)
    if len(nuc_anchor_x) > 0:
        ax.quiver(
            nuc_anchor_x, nuc_anchor_y,
            nuc_anchor_u, nuc_anchor_v,
            angles="xy", scale_units="xy", scale=1,
            color="yellow", width=0.001, headwidth=1.5, headlength=2.0,
            label=f"Anchor Vectors (n={len(nuc_anchor_x)})"
        )

    ax.set_title(f"Nucleus Anchor Points Alignment (N = {len(nuc_anchor_x)})", fontsize=14)
    ax.legend(loc="upper right")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_actin_anchor_points(
        actin_sample_slice,
        actin_reference_image,
        actin_anchor_x,
        actin_anchor_y,
        actin_anchor_u,
        actin_anchor_v,
        actin_thresh=150.0,
        figsize=(8, 8)
):
    """
    Plots the actin sample slice, eroded template/sample borders, and computed actin anchor vectors.
    """
    # 1. Recompute borders for visualization overlay
    n_actin_ch = 255.0 * actin_sample_slice / np.max(actin_sample_slice)
    mask_sample = (n_actin_ch > actin_thresh)
    mask_avg = ndi.binary_erosion((actin_reference_image > 0), iterations=5)

    border_sample = mask_sample ^ ndi.binary_erosion(mask_sample)
    border_avg = mask_avg ^ ndi.binary_erosion(mask_avg)

    # 2. Setup figure
    fig, ax = plt.subplots(figsize=figsize)

    # Background image (sample intensity)
    ax.imshow(actin_sample_slice, cmap="gray", origin="upper")

    # Overlay sample border (cyan) and eroded template border (green)
    by_smp, bx_smp = np.where(border_sample)
    by_avg, bx_avg = np.where(border_avg)

    # Individual Cell Actin Border -> GREEN
    ax.scatter(bx_smp, by_smp, c="red", s=3, alpha=0.9, label="Individual Actin Border")

    # Average Cell Actin Border -> YELLOW
    ax.scatter(bx_avg, by_avg, c="lime", s=3, alpha=0.9, label="Average Actin Border")

    # Plot Anchor Vectors (lime/white arrows)
    if len(actin_anchor_x) > 0:
        ax.quiver(
            actin_anchor_x, actin_anchor_y,
            actin_anchor_u, actin_anchor_v,
            angles="xy", scale_units="xy", scale=1,
            color="yellow", width=0.001, headwidth=1.5, headlength=2.0,
            label=f"Anchor Vectors (n={len(actin_anchor_x)})"
        )

    ax.set_title(f"Actin Anchor Points Alignment (N = {len(actin_anchor_x)})", fontsize=14)
    ax.legend(loc="upper right")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_transformation_results(
        segmented_sample_stack,
        nucleus_reference,
        actin_reference,
        nuc_anchors_tuple,
        act_anchors_tuple,
        Xnew,
        Ynew,
        sample_particle_xy,
        transformed_particle_xy,
        nuc_ch_ix=3,
        actin_ch_ix=1,
        ds_grid=8,
        figsize=(12, 6)
):
    """
    Plots a 2-panel transformation comparison:
    - Left: Sample cell with anchor vectors, undeformed grid, and original particle.
    - Right: Average cell reference with deformed grid and transformed particle.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    def norm01(img):
        return (img - np.min(img)) / (np.ptp(img) or 1)

    h, w = segmented_sample_stack.shape[1:]

    # =========================================================================
    # LEFT PANEL: Individual Cell Coordinates (Sample)
    # =========================================================================
    sample_rgb = np.zeros((h, w, 3))
    sample_rgb[..., 0] = norm01(segmented_sample_stack[nuc_ch_ix])  # Red: Nucleus
    sample_rgb[..., 1] = norm01(segmented_sample_stack[actin_ch_ix])  # Green: Actin

    ax1.imshow(sample_rgb, origin='upper')

    # Grid mask (Actin & not Nucleus)
    mask = (segmented_sample_stack[actin_ch_ix] > 150) & ~(segmented_sample_stack[nuc_ch_ix] > 100)
    mask_ds = mask[::ds_grid, ::ds_grid]

    X_orig, Y_orig = np.meshgrid(np.arange(0, w, ds_grid), np.arange(0, h, ds_grid))

    # Overlay undeformed grid
    for i in range(X_orig.shape[0]):
        valid = mask_ds[i, :]
        ax1.plot(np.where(valid, X_orig[i, :], np.nan), np.where(valid, Y_orig[i, :], np.nan), color='white', lw=0.5,
                 alpha=0.3)
    for j in range(X_orig.shape[1]):
        valid = mask_ds[:, j]
        ax1.plot(np.where(valid, X_orig[:, j], np.nan), np.where(valid, Y_orig[:, j], np.nan), color='white', lw=0.5,
                 alpha=0.3)

    # Original Particle Location
    ax1.scatter(sample_particle_xy[0], sample_particle_xy[1], color='crimson', s=60, edgecolors='white', linewidth=1.5,
                zorder=5, label='Sample Particle')
    ax1.set_title("Individual Cell Coordinates (Sample)", fontsize=13)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.axis('off')

    # =========================================================================
    # RIGHT PANEL: Transformed Coordinates (Average Cell Reference)
    # =========================================================================
    avg_rgb = np.zeros((*nucleus_reference.shape, 3))
    avg_rgb[..., 0] = norm01(nucleus_reference)  # Red: Average Nucleus
    avg_rgb[..., 1] = norm01(actin_reference)  # Green: Average Actin

    ax2.imshow(avg_rgb, origin='upper')

    # Overlay deformed grid
    for i in range(Xnew.shape[0]):
        valid = mask_ds[i, :]
        ax2.plot(np.where(valid, Xnew[i, :], np.nan), np.where(valid, Ynew[i, :], np.nan), color='white', lw=0.5,
                 alpha=0.4)
    for j in range(Xnew.shape[1]):
        valid = mask_ds[:, j]
        ax2.plot(np.where(valid, Xnew[:, j], np.nan), np.where(valid, Ynew[:, j], np.nan), color='white', lw=0.5,
                 alpha=0.4)

    # Mapped Particle Location
    ax2.scatter(transformed_particle_xy[0], transformed_particle_xy[1], color='limegreen', s=60, edgecolors='white',
                linewidth=1.5, zorder=5, label='Transformed Particle')
    ax2.set_title("Transformed Coordinates (Average Cell)", fontsize=13)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()