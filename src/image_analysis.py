import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from cupyx.scipy.ndimage import affine_transform
from cupyx.scipy.signal import fftconvolve
import scipy.fft as fft


def rotate_2d_cupy(image_2d, angle_deg=0.0):
    if image_2d.ndim != 2:
        raise ValueError(f"Expected 2D input image, got shape {image_2d.shape}")

    # Transfer CPU numpy array to GPU cupy array
    img_gpu = cp.asarray(image_2d) if isinstance(image_2d, np.ndarray) else image_2d

    angle_rad = cp.deg2rad(angle_deg)
    cos_a, sin_a = cp.cos(angle_rad), cp.sin(angle_rad)

    h, w = img_gpu.shape
    rot_matrix = cp.array(
        [[cos_a, -sin_a],
         [sin_a, cos_a]],
        dtype=cp.float32
    )
    center = cp.array([(w - 1) / 2, (h - 1) / 2])
    offset = center - rot_matrix @ center

    # Perform GPU rotation
    rotated_gpu = affine_transform(
        img_gpu,
        rot_matrix,
        offset=offset,
        order=1,
        mode='constant',
        cval=0.0,
        output_shape=(h, w)
    )

    if rotated_gpu.shape != (h, w):
        padded = cp.zeros((h, w), dtype=rotated_gpu.dtype)
        rh, rw = rotated_gpu.shape
        padded[:min(h, rh), :min(w, rw)] = rotated_gpu[:min(h, rh), :min(w, rw)]
        rotated_gpu = padded

    # Return as CPU NumPy array
    return cp.asnumpy(rotated_gpu)

def crop_and_pad_2d(
    img2d,
    center,
    out_shape
):
    """
    Crop a 2D image around the given center (y, x) with shape out_shape (y, x).
    If crop extends beyond the image, zero-padding is applied to ensure the output
    has exactly the requested shape.

    Args:
        img2d: Input 2D NumPy array of shape (Y, X).
        center: Center coordinates (y, x), all integers.
        out_shape: Desired output crop shape (Yc, Xc).

    Returns:
        dst_img: Cropped and padded 2D NumPy array with shape `out_shape`.
    """
    cy, cx = center
    oy, ox = out_shape
    half = np.array([oy // 2, ox // 2])

    # Source start/end in image coords
    src_start = np.array([cy, cx]) - half
    src_end = src_start + np.array([oy, ox])

    dst_img = np.zeros(out_shape, dtype=img2d.dtype)

    # Compute clamped ranges
    src_slices = []
    dst_slices = []
    for dim, (s0, s1, maxdim) in enumerate(zip(src_start, src_end, img2d.shape)):
        ss, se = max(s0, 0), min(s1, maxdim)
        ds0 = ss - s0
        ds1 = ds0 + (se - ss)
        src_slices.append(slice(ss, se))
        dst_slices.append(slice(ds0, ds1))

    dst_img[tuple(dst_slices)] = img2d[tuple(src_slices)]
    return dst_img

def find_pattern_centroid_ncc_2d(
    image2d,
    template2d,
    crop_size = None,
    show = False
):
    """
    Search for a 2D template in a 2D image and return the best match.

    Args:
        image2d: Input 2D image (Y, X) as NumPy array.
        template2d: 2D template to match (Ty, Tx) as NumPy array.
        crop_size: Optional output crop size (Yc, Xc). Returns zero-padded crop.
        show: If True, show the image with overlay and response map.

    Returns:
        best_coord: Tuple of integers (y, x) with best match.
        crop: Cropped 2D patch centered at best_coord or None.
        response: 2D NCC response map.
    """
    assert image2d.ndim == 2, f"Expected a 2D image, got shape {image2d.shape}"

    # Normalize image globally to reduce dtype issues
    img = image2d.astype(np.float64)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())

    # Calculate 2D NCC response map directly
    response = ncc_response_partial_overlap(img, template2d)

    # Find coordinates of maximum score
    idx = np.nanargmax(response)
    y, x = np.unravel_index(idx, response.shape)
    best_coord = (int(round(y)), int(round(x)))
    best_score = float(response[y, x])

    # Crop 2D region if requested using crop_and_pad_2d
    crop = None
    if crop_size is not None:
        crop = crop_and_pad_2d(image2d, best_coord, crop_size)

    if show:
        y, x = best_coord
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(image2d, cmap='gray')
        ax[0].scatter([x], [y], c='r', s=40)
        ax[0].set_title(f"Found pattern center: y={y}, x={x}")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(response, cmap='gray')
        ax[1].set_title("Response map")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        plt.tight_layout()
        plt.show()

    return best_coord, crop, response


def ncc_response_partial_overlap(
        img2d: np.ndarray,
        template2d: cp.ndarray,
        eps: float = 1e-9
) -> np.ndarray:
    """
    Returns normalized cross-correlation response (Pearson correlation coefficient) for every pixel
    in img2d accounting for partial overlap of template2d at image borders, using GPU.

    Args:
        img2d: 2D NumPy array
        template2d: 2D NumPy array
        eps: small number to avoid division by zero

    Returns:
        response: 2D NumPy array of NCC values in [-1, 1], -inf where undefined
    """

    # Move data to GPU
    I = cp.asarray(img2d, dtype=cp.float64)
    T = cp.asarray(template2d, dtype=cp.float64)

    # reversed template for convolution (so fftconvolve(I, T_rev) == correlation I * T)
    T_rev = T[::-1, ::-1]

    ones_T = cp.ones_like(T)
    ones_I = cp.ones_like(I)

    # local sums on image windows (size = template)
    sum_I = fftconvolve(I, ones_T, mode='same')
    sum_I2 = fftconvolve(I * I, ones_T, mode='same')

    # cross term: sum of I * T over overlap
    sum_IT = fftconvolve(I, T_rev, mode='same')

    # sum of template values within the overlap window (varies near borders)
    sum_T = fftconvolve(ones_I, T_rev, mode='same')  # sum of T values overlapping
    sum_T2 = fftconvolve(ones_I, (T * T)[::-1, ::-1], mode='same')  # sum of T^2 overlapping

    # number of overlapping pixels at each shift (<= T.size near edges)
    n = fftconvolve(ones_I, ones_T, mode='same')

    # compute numerator and denominator of NCC
    # numerator = sum_IT - sum_I * sum_T / n
    # denominator = sqrt( (sum_I2 - sum_I^2/n) * (sum_T2 - sum_T^2/n) )
    # avoid division by zero where n==0
    valid = n > 0
    response = np.full_like(sum_IT, fill_value=-np.inf, dtype=np.float64)

    num = cp.zeros_like(sum_IT)
    num[valid] = (sum_IT[valid] - (sum_I[valid] * sum_T[valid]) / n[valid])

    t_var = (sum_T2 - (sum_T * sum_T) / n)  # could be zero if template constant or n small
    i_var = (sum_I2 - (sum_I * sum_I) / n)

    denom = cp.sqrt(cp.maximum(i_var * t_var, 0.0))
    good = (denom > eps) & valid

    response[good] = num[good] / denom[good]

    response = cp.nan_to_num(response, nan=-cp.inf, posinf=-cp.inf, neginf=-cp.inf)

    # response is in [-1, 1] where defined; -inf where undefined (we can mask)
    return cp.asnumpy(response)

def get_radial_profile(data):
    """Compute the radial average of a 2D centered array."""
    y, x = np.indices(data.shape)
    center = np.array(data.shape) // 2
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    nr[nr == 0] = 1
    return tbin / nr


def analyze_structure_with_stats(stack):
    """Computes mean and std of autocorrelation for a stack of images."""
    n_images = stack.shape[0]
    all_profiles = []

    for i in range(n_images):
        img = stack[i].astype(float)
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)

        f = fft.fft2(img)
        acf = fft.ifft2(f * np.conj(f)).real
        acf = fft.fftshift(acf)

        profile = get_radial_profile(acf)
        profile /= (profile[0] + 1e-8)
        all_profiles.append(profile)

    all_profiles = np.array(all_profiles)
    return np.mean(all_profiles, axis=0), np.std(all_profiles, axis=0)