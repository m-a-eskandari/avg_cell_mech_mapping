from . import tdms_functions as tdms
import numpy as np
import pandas
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic

#Boltzman constant
k = 1.380649e-23

def bin_psd_log(freqs, psd, target_freqs, exclude_range=None):
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    target_freqs = np.asarray(target_freqs)

    mask = np.ones_like(freqs, dtype=bool)
    if exclude_range is not None:
        fmin, fmax = exclude_range
        mask &= ~((freqs >= fmin) & (freqs <= fmax))

    log_target = np.log(target_freqs)
    log_edges = np.zeros(len(target_freqs) + 1)
    log_edges[1:-1] = (log_target[:-1] + log_target[1:]) / 2
    log_edges[0] = log_target[0] - (log_edges[1] - log_target[0])
    log_edges[-1] = log_target[-1] + (log_target[-1] - log_edges[-2])

    edges = np.exp(log_edges)
    bin_indices = np.digitize(freqs, edges) - 1

    binned_psd = np.empty(len(target_freqs))
    bin_counts = np.zeros(len(target_freqs), dtype=int)

    for i in range(len(target_freqs)):
        in_bin = (bin_indices == i) & mask
        bin_counts[i] = np.sum(in_bin)
        if bin_counts[i] > 0:
            binned_psd[i] = np.mean(psd[in_bin])
        else:
            binned_psd[i] = np.nan

    binned_freqs = np.sqrt(edges[:-1] * edges[1:])
    return binned_freqs, binned_psd


def bin_data_at_frequencies(freq, values, bin_centers, statistic="mean"):
    # Log warning if bin_centers are not equally spaced in log-space
    if len(np.unique(np.round(np.diff(np.log(bin_centers)), 4))) > 1:
        logging.warning("Using this function does not guarantee matching the frequencies directly.")

    # Calculate log-spaced bin edges from centers
    log_centers = np.log(bin_centers)
    middle = (log_centers[1:] - log_centers[:-1]) / 2.0
    log_edges = np.concatenate([
        [log_centers[0] - middle[0]],
        log_centers[:-1] + middle,
        [log_centers[-1] + middle[-1]]
    ])

    bin_edges = np.exp(log_edges)
    freq_bin, mean_bin_values = bin_data_from_bins(freq, values, bin_edges, statistic=statistic)

    return freq_bin, mean_bin_values, bin_edges


def bin_data_from_bins(freq, values, bin_edges, statistic="mean"):
    mean_bin_values, _, _ = binned_statistic(freq, values, statistic=statistic, bins=bin_edges)

    # Calculate geometric mean of bin edges for representative frequencies
    freq_bin = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    # Filter out empty bins (NaN values)
    mask = ~np.isnan(mean_bin_values)
    return freq_bin[mask], mean_bin_values[mask]

def fit_fkv(f, g1, g2, require_causality=True, p0=(1e1, 1e0, 0.1, 0.9), masked_indices=None):
    # Construct mask for valid positive data and user-excluded indices
    mask = (g1 > 0.0) & (g2 > 0.0)
    if masked_indices:
        mask &= ~np.isin(np.arange(len(f)), masked_indices)

    # Log-space combined storage and loss moduli
    g = np.log(np.hstack((g1[mask], g2[mask])))

    # Configure bounds and fit model
    bounds = ([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, 1.0]) if require_causality else ([0.0, 0.0, 0.0, 0.0], np.inf)

    params, pcov, _, mesg, ier = curve_fit(
        fractional_kelvin_voigt_model_fit,
        f[mask],
        g,
        p0=p0,
        bounds=bounds,
        full_output=True,
    )

    if ier not in [1, 2, 3, 4]:
        raise ValueError(f"Fit failed with status {ier}: {mesg}")

    perr = np.sqrt(np.diag(pcov))

    # Swap parameters if alpha > beta to maintain ordering
    if params[2] > params[3]:
        params = [params[1], params[0], params[3], params[2]]
        perr = [perr[1], perr[0], perr[3], perr[2]]

    return np.array(params), np.array(perr), mask


def fractional_kelvin_voigt_model_fit(f, a, b, alpha, beta):
    g = fractional_kelvin_voigt_model(f, a, b, alpha, beta)
    return np.append(np.log(g.real), np.log(g.imag))


def fractional_kelvin_voigt_model(f, a, b, alpha, beta):
    g = np.array(
        a * np.power(2j * np.pi * f, alpha) + b * np.power(2j * np.pi * f, beta),
        dtype=np.complex128,
    )

    # Log warnings for invalid/non-positive values
    invalid_real = np.any(g.real <= 0)
    invalid_imag = np.any(g.imag <= 0)

    # Mask non-positive values as complex NaN
    g[(g.real < 0) | (g.imag < 0)] = np.nan + 1j * np.nan
    return g

def passive_analysis(xy, slope_position_xy, rate):
    if len(xy) != 4:
        raise ValueError(
            f"Invalid length of position signals: {len(xy)}. Should be 4 (x, y, sum x, sum y)."
        )
    if len(slope_position_xy) != 2:
        raise ValueError(
            f"Invalid length of slopes: {len(slope_position_xy)}. Should be 2 (x, y)."
        )

    x, y, sum_x, sum_y = np.asarray(xy)
    slope_x, slope_y = np.asarray(slope_position_xy)

    # Normalize voltage signals and convert to position in µm
    norm_x = x * slope_x / sum_x
    norm_y = y * slope_y / sum_y

    n = np.size(norm_x)

    # Positive frequency half
    freq = np.fft.fftfreq(n, 1 / rate)[: n // 2]
    fft_x = np.fft.fft(norm_x)[: n // 2]
    fft_y = np.fft.fft(norm_y)[: n // 2]

    # Calculate Power Spectral Density (PSD) and convert µm² to m²
    psd_x = np.power(np.abs(fft_x), 2) / n / rate * 1e-12
    psd_y = np.power(np.abs(fft_y), 2) / n / rate * 1e-12


    return {
        "freq": freq,
        "psd_x": psd_x,
        "psd_y": psd_y,
    }

def active_analysis(
    signals,
    slope_position,
    slope_force,
    bead_radius,
    acquisition_rate,
    oscillation_frequency
):
    if len(signals) != 3:
        raise ValueError(
            f"Invalid length of position signals: {len(signals)}. Should be 3 (position, position_sum, force)."
        )

    position, position_sum, force = np.asarray(signals)
    buffer_size = len(position)

    # Normalize position signal and convert voltages to physical units
    position *= slope_position / position_sum # in um
    force *= slope_force # in N
    # Fourier transforms and frequencies
    force_f = np.fft.fft(force)
    position_f = np.fft.fft(position)
    fft_freq = np.fft.fftfreq(buffer_size, 1.0 / acquisition_rate)

    # Ratio of FFT at oscillation frequency peak
    fft_peak_index = int(np.round(buffer_size / acquisition_rate * oscillation_frequency))
    ratio_d_f = position_f[fft_peak_index] / force_f[fft_peak_index]

    # Linear response function and complex shear modulus
    res_func = ratio_d_f * 1e-6 # To convert um/N to m/N
    bead_radius *= 1e-6 # to convert um to m
    shear_mod = 1.0 / (6.0 * np.pi * bead_radius * res_func)

    # Return positive-frequency half
    return {
        "fft_freq": fft_freq[: buffer_size // 2],
        "position_f": position_f[: buffer_size // 2],
        "force_f": force_f[: buffer_size // 2],
        "res_func": res_func,
        "shear_mod": shear_mod,
    }

def basic_analysis(
    tdms_file_path, kind = None
):
    """
    Perform basic analysis of data in a TDMS file.

    The active measurement DataFrame is always sorted for ascending
    frequencies.

    Args:
        tdms_file_path (str):
            Path to TDMS file of interest.
        kind (str | None):
            Explicit experiment kind ("active" or "passive"). If None (default)
            left to auto-detect by microrheology_analysis.tdms.load_tdms.

    Returns:
        tuple[
            TdmsFile,
            pandas.DataFrame,
            pandas.DataFrame | list[dict[str, NDArray[np.float64]]],
            pandas.DataFrame | list[pandas.DataFrame],
        ]:
            1. TDMS file object
            2. Raw data as a pandas.DataFrame
            3. Result 1:
                - Active: Response function and complex shear modulus
                - Passive: Dict of full resolution and binned PSDs in x and y.
            4. Result 2:
                - Active: FFT of signals (position, force)
                - Passive: list of pandas.DataFrame's of binned PSDs (one for each run).

    """
    tdms_file, data, kind = tdms.load_tdms(tdms_file_path, kind=kind)

    # define a few shorthands
    def active_single_freq(freq, frequencies, signal_ffts, results, oscillation_axis,
    ):
        """Get active analysis of single frequency."""
        idx = frequencies.get_loc(freq)
        if isinstance(idx, int):
            group_idx = len(frequencies) - 1 - idx
            # Check that the frequency matches the one from the file
            if freq != float(
                tdms_file.groups()[group_idx].properties["Frequency"].replace(",", ".")
            ):
                group_idx = idx
                # Check again with inverted sorting
                if freq != float(
                    tdms_file.groups()[group_idx]
                    .properties["Frequency"]
                    .replace(",", ".")
                ):
                    raise ValueError(
                        "Could not find correct frequency entry in TDMS file. "
                        "Contact developers of microrheology_analysis package to "
                        "find a solution for your file."
                    )
        else:
            raise ValueError(
                f'Got invalid index "{idx}" (type {type(idx)}) for frequency {freq}'
            )
        props = tdms_file.groups()[group_idx].properties

        analysis = active_analysis(
            *tdms.extract_active_args(data, props, freq, oscillation_axis)
        )
        signal_ffts[0].append(analysis["fft_freq"])
        signal_ffts[1].append(analysis["position_f"])
        signal_ffts[2].append(analysis["force_f"])
        results[0].append(analysis["res_func"])
        results[1].append(analysis["shear_mod"])

    def passive_single_run(
        run,
    ):
        """Get passive analysis of single run."""
        # TODO Check if this is the right order. Do we care about it?
        group_idx = len(tdms_file.groups()) - 1 - run
        props = tdms_file.groups()[group_idx].properties

        analysis = passive_analysis(
            *tdms.extract_passive_args(data, props, run)
        )
        return analysis

    # full analysis
    if kind == "active":
        frequencies = data.columns[::8].get_level_values("frequency")
        oscillation_axis = "xy"[
            int(tdms_file.groups()[0].properties.get("x(0)_or_y(1)"))
        ]
        signal_ffts: tuple[
            list[NDArray[np.float64]],
            list[NDArray[np.complex128]],
            list[NDArray[np.complex128]],
        ] = ([], [], [])
        results: tuple[
            list[np.complex128],
            list[np.complex128],
        ] = ([], [])
        for freq in frequencies:
            active_single_freq(freq, frequencies, signal_ffts, results, oscillation_axis)
        signal_ffts_df = pandas.DataFrame(
            signal_ffts, columns=frequencies, index=["fft_freq", "position_f", "force_f"]
        ).T
        results_df = pandas.DataFrame(
            results, columns=frequencies, index=["chi", "G*"]
        ).T
        results_df = results_df.sort_index(ascending=True)
        return (tdms_file, data, results_df, signal_ffts_df)
    elif kind == "passive":
        runs = data.columns[::4].get_level_values("Iteration")
        # list of dicts defined in live_analysis.passive_analysis
        analyses = [passive_single_run(run) for run in runs]
        return (tdms_file, data, analyses)
    else:
        raise ValueError(f"Invalid experiment kind {kind}.")


def effective_energy(
    frequencies,
    psd,
    res_func,
    temperature = 273.15 + 37.0,
):
    """
    Compute the effective energy.

    E_eff = [C(f) * pi * f] / [Im(chi)(f) * k_B * T]
        f            Frequencies
        C(f)         Power spectral density of the passive recording
        chi(f)       Response function obtained from an active measurement
        Im(chi)(f)   Imaginary part of chi(f)
        k_B          Boltzmann constant
        T            Absolute temperature in Kelvin

    In practice Im(chi) can have a sign, so we take the absolute value of it.

    Args:
        frequencies (NDArray[np.float64]):
            Frequencies of PSD and response function
        psd (NDArray[np.float64]):
            Power spectral density
        res_func (NDArray[np.complex128]):
            Complex response function. The effective enery only needs the
            negative imaginary part.
        temperature (float):
            Absolute temperature in Kelvin. Defaults to 37 °C = 310.15 K.

    Returns:
        NDArray[np.float64]:
            Effective energy. Can be negative if np.imag(res_func) contains
            positive values.

    """

    return (psd * np.pi * frequencies) / (-np.imag(res_func) * k * temperature)  # type: ignore



def fit_e_eff(
    f,
    e_eff,
    p0= (1e0, -1.0),
    masked_indices = None,
    force_one_kbt = True,
):
    """
    Fit offset powerlaw to effective energy.

    Uses analysis.offset_powerlaw for length 3 p0 and False force_one_kbt,
    otherwise analysis.constrained_offset_powerlaw.

    Args:
        f (NDArray[np.float64]):
            Applied oscillation frequencies in units of 1 Hz.
        e_eff (NDArray[np.float64]):
            Effective energy in units of k_B T.
        p0 (tuple[float, float, float] | tuple[float, float]):
            Initial guess for parameters in the order (E_0, gamma, offset).
            If only two parameters are passed, use them as (E_0, gamma).
            In that case not offset_powerlaw, but constrained_offset_powerlaw
            is used. Defaults to (1e0, -1.0).
        masked_indices (list[int] | None):
            Indices that should be ignored during fitting. Defaults to None in
            which all indices are valid.
        force_one_kbt (bool):
            If offset_powerlaw should be constrained to an offset of 1 k_B T.
            This results in constrained_offset_powerlaw being used.  Defaults
            to True.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool]]:
            1. The fit paramaters (A, B, alpha, beta)
            2. The estimated parameter errors
            3. The boolean array mask for the input arrays.

    """
    if e_eff.shape != f.shape:
        raise ValueError("Shape mismatch for e_eff and f.")
    # construct boolean array mask for invalid values
    mask: NDArray[np.bool]
    if masked_indices is None or len(masked_indices) == 0:
        mask = e_eff > 0.0
    else:
        mask = np.array([i not in masked_indices for i in range(len(f))])
    if not np.all(mask):
        print(f"Masked frequencies: {np.array(f[~mask])}")

    if force_one_kbt or len(p0) == 2:
        model = constrained_offset_powerlaw
        if len(p0) == 3:
            p0 = p0[:-1]
        bounds = ([0.0, -np.inf], [np.inf, np.inf])
    else:
        model = offset_powerlaw
        bounds = ([0.0, -np.inf, 0.0], [np.inf, np.inf, np.inf])

    # fit fractional viscoelastic model, A, B are positive, alpha is smaller than beta
    params, pcov, infodict, mesg, ier = curve_fit(
        model,
        f[mask],
        e_eff[mask],
        p0=p0,
        bounds=bounds,
        full_output=True,
    )
    # check that fit succeeded
    if ier in [1, 2, 3, 4]:
        print(f"Fit succeeded with status {ier}: {mesg}")
    else:
        raise ValueError(f"Fit failed with status {ier}: {mesg}")

    # error estimation of fit parameters
    perr = np.sqrt(np.diag(pcov))

    return np.array(params), np.array(perr), mask


def offset_powerlaw(
    f,
    e_0,
    gamma,
    offset,
):
    """
    Offset powerlaw, used to fit the effective energy.

    E_eff(f) = E_0 * (f)^gamma + offset
    Typically offset = 1 and gamma is close to -1.

    Args:
        f (NDArray[np.float64]):
            Frequency
        e_0 (np.float64):
            Energy prefactor
        gamma (np.float64):
            Powerlaw exponent
        offset (float):
            Offset

    Returns:
        NDArray[np.float64]:
            Offset powerlaw.

    """
    return e_0 * np.power(f, gamma) + offset


def constrained_offset_powerlaw(
    f,
    e_0,
    gamma,
):
    """
    Constrained offset powerlaw, used to fit the effective energy.

    Constraint: offset = 1 k_B T

    E_eff(f) = E_0 * (f)^gamma + 1
    Typically gamma is close to -1.

    Args:
        f (NDArray[np.float64]):
            Frequency
        e_0 (np.float64):
            Energy prefactor
        gamma (np.float64):
            Powerlaw exponent

    Returns:
        NDArray[np.float64]:
            Offset powerlaw.

    """
    return offset_powerlaw(f, e_0, gamma, offset=1.0)