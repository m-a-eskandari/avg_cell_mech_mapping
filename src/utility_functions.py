import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def print_cell_particle_data(
    cell_txt_path,
    particle_tdms_path,
    cell_motor,
    part_motor,
    cell_piezo,
    part_piezo,
    col_widths=(28, 35, 35)
):
    """
    Prints a formatted summary comparing stage metadata between cell and particle files.
    """
    def fmt(arr):
        return f"[{', '.join(f'{v:.2f}' for v in arr)}]"

    # Dynamic headers with entity prefix and filenames
    cell_header = f"Cell ({cell_txt_path.name})"
    particle_header = f"Particle ({particle_tdms_path.name})"

    col_w0, col_w1, col_w2 = col_widths
    total_w = sum(col_widths) + 6

    border = "═" * total_w
    divider = "─" * total_w

    print(border)
    print(f"{'METADATA SUMMARY':^{total_w}}")
    print(border)
    print(f"  {'Parameter':<{col_w0}} │ {cell_header:^{col_w1}} │ {particle_header:^{col_w2}}")
    print(divider)
    print(f"  {'Motor (X, Y) [um]':<{col_w0}} │ {fmt(cell_motor):^{col_w1}} │ {fmt(part_motor):^{col_w2}}")
    print(f"  {'Piezo (X, Y) [um]':<{col_w0}} │ {fmt(cell_piezo):^{col_w1}} │ {fmt(part_piezo):^{col_w2}}")
    print(border)



def summarize_and_plot_particle_transform(
    particle_px_raw,
    p_full_rot_x,
    p_full_rot_y,
    raw_img_center,
    best_coords,
    rotation_angle,
    raw_img_shape,
    crop_dim=650,
    rel_x_um=None,
    rel_y_um=None,
):
    """Prints transformation metrics and displays particle positions, centers,

    and crop regions on a single plot set to the raw image bounds.
    """
    crop_center = crop_dim / 2.0
    cy_pat, cx_pat = best_coords  # Pattern center in (Y, X)

    # Calculate transformed particle coordinates in crop space
    dx_from_pattern = p_full_rot_x - cx_pat
    dy_from_pattern = p_full_rot_y - cy_pat
    abs_px_x = crop_center + dx_from_pattern
    abs_px_y = crop_center + dy_from_pattern

    # --- 1. Terminal Summary Table ---
    col_w = [30, 22]
    total_w = sum(col_w) + 3
    border = "═" * total_w
    divider = "─" * total_w

    print(border)
    print(f"{'PARTICLE TRANSFORM SUMMARY':^{total_w}}")
    print(border)
    print(f"  {'Metric':<{col_w[0]}} │ {'Value':^{col_w[1]}}")
    print(divider)
    print(
        f"  {'Raw Particle (X, Y)':<{col_w[0]}} │ {f'({particle_px_raw[0]:.2f}, {particle_px_raw[1]:.2f})':^{col_w[1]}}"
    )
    print(
        f"  {'Rotated Particle (X, Y)':<{col_w[0]}} │ {f'({p_full_rot_x:.2f}, {p_full_rot_y:.2f})':^{col_w[1]}}"
    )
    print(
        f"  {'Raw Center (X, Y)':<{col_w[0]}} │ {f'({raw_img_center[1]:.2f}, {raw_img_center[0]:.2f})':^{col_w[1]}}"
    )
    print(
        f"  {'Pattern Center (X, Y)':<{col_w[0]}} │ {f'({cx_pat:.2f}, {cy_pat:.2f})':^{col_w[1]}}"
    )
    print(
        f"  {'In-Crop Particle (X, Y)':<{col_w[0]}} │ {f'({abs_px_x:.2f}, {abs_px_y:.2f})':^{col_w[1]}}"
    )

    if rel_x_um is not None and rel_y_um is not None:
        print(
            f"  {'Relative Offset (X, Y) [um]':<{col_w[0]}} │ {f'({rel_x_um:.3f}, {rel_y_um:.3f})':^{col_w[1]}}"
        )
    print(border)

    # --- 2. Single Combined Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))

    raw_h, raw_w = raw_img_shape

    # Plot centers and particle locations
    ax.scatter(
        raw_img_center[1],
        raw_img_center[0],
        color="gray",
        marker="x",
        s=30,
        linewidths=2,
        label="Raw Image Center",
    )
    ax.scatter(
        particle_px_raw[0],
        particle_px_raw[1],
        color="crimson",
        s=30,
        zorder=4,
        label="Raw Particle",
    )
    ax.scatter(
        p_full_rot_x,
        p_full_rot_y,
        color="limegreen",
        s=30,
        zorder=4,
        label="Rotated Particle",
    )
    ax.scatter(
        cx_pat,
        cy_pat,
        color="blue",
        marker="+",
        s=30,
        linewidths=2,
        zorder=4,
        label="Pattern Center",
    )

    # --- Crop Squares ---
    half_crop = crop_dim / 2.0

    # 1. Unrotated Crop Square (Axis-Aligned around Pattern Center)
    unrot_rect = patches.Rectangle(
        (cx_pat - half_crop, cy_pat - half_crop),
        crop_dim,
        crop_dim,
        linewidth=1.5,
        edgecolor="darkorange",
        facecolor="none",
        linestyle="--",
        label="Unrotated Crop Boundary",
    )
    ax.add_patch(unrot_rect)

    # 2. Rotated Crop Square (Rotated by `rotation_angle` around Raw Image Center)
    # Define local square corners around pattern center
    corners_local = np.array(
        [
            [-half_crop, -half_crop],
            [half_crop, -half_crop],
            [half_crop, half_crop],
            [-half_crop, half_crop],
            [-half_crop, -half_crop],  # Close shape
        ]
    )

    # Translate to pattern position relative to image center
    corners_global = corners_local + np.array(
        [cx_pat - raw_img_center[1], cy_pat - raw_img_center[0]]
    )

    # Apply rotation around raw image center
    theta = np.radians(rotation_angle)
    c, s = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[c, -s], [s, c]])

    corners_rotated = np.dot(corners_global, rot_matrix.T) + np.array(
        [raw_img_center[1], raw_img_center[0]]
    )

    ax.plot(
        corners_rotated[:, 0],
        corners_rotated[:, 1],
        color="purple",
        linewidth=2,
        linestyle="-",
        label=f"Rotated Crop Boundary ({rotation_angle:.1f}°)",
    )

    # Axes setup
    ax.set_xlim(0, raw_w)
    ax.set_ylim(raw_h, 0)  # Inverted Y to match image space conventions
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_title(
        f"Particle Transformation & Crop Region ({raw_w}×{raw_h} px)",
        fontsize=12,
    )
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()