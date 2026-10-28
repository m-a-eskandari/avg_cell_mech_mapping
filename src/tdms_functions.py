import numpy as np
import pandas
from nptdms import TdmsFile


def extract_active_args(data, props, freq, axis):
    # Extract signals and filter out NaNs
    cols = (freq, (f"Position {axis}", f"Position sum {axis}", f"Force {axis}"))
    signals = data.loc[:, cols].to_numpy()
    signals = signals[~np.isnan(signals).any(axis=1), :].T

    # Helper function to parse floating point properties handling comma decimals
    def get_float(key, error_msg):
        val = props.get(key)
        if val is None:
            raise ValueError(error_msg)
        return float(str(val).replace(",", "."))

    # Extract required properties
    slope_position = get_float("Slope_position_y", "Invalid or missing position slope encountered.")
    slope_force = get_float(f"Slope_force_{axis}", "Invalid or missing force slope encountered.")
    radius = get_float("radius(um)", "Invalid or missing bead radius encountered.")

    acq_rate = props.get("Acq_rate")
    if acq_rate is None:
        raise ValueError("Invalid or missing acquisition rate encountered.")
    acquisition_rate = int(acq_rate)

    oscillation_frequency = get_float("Frequency", "Invalid or missing TDMS frequency encountered.")

    # Apply calibration multiplier for x-axis
    if axis == "x":
        slope_position *= 1.6875

    return (
        signals,
        slope_position,
        slope_force,
        radius,
        acquisition_rate,
        oscillation_frequency,
    )


def extract_passive_args(data, props, iteration):
    xy = data[iteration].to_numpy().T

    # Extract and parse position slope
    slope_y = props.get("Slope_position_y")
    if slope_y is None:
        raise ValueError("Invalid slope encountered.")

    slope_val = float(str(slope_y).replace(",", "."))
    slope_position_xy = (slope_val * 1.6875, slope_val)

    # Extract acquisition rate
    acq_rate = props.get("Acq_rate")
    if acq_rate is None:
        raise ValueError("Invalid acquisition rate encountered.")

    rate = float(acq_rate)

    return (xy, slope_position_xy, rate)

def load_tdms(
    file_path: str, kind: str | None = None
) -> tuple[TdmsFile, pandas.DataFrame, str]:
    """
    Load data from a TDMS file.

    Args:
        file_path (str):
            Path to the TDMS file
        kind (str | None):
            Kind of experiment ("passive" or "active"). Defaults to None in
            which case an attempt is made to detect it automatically.

    Returns:
        tuple[TdmsFile, pandas.DataFrame, str]:
            - TdmsFile containing all data and metadata for further analysis.
                See e.g. file.properties, file.groups()[0].properties or
                file.groups()[0].channels()[0].properties
            - data from channels as pandas.DataFrame
            - kind of experiment. Exported in case it was detected
                automatically.

    """
    tdms_file = TdmsFile.read(file_path)
    if kind is None:
        if "passive" in file_path.lower():
            kind = "passive"
        elif "active" in file_path.lower():
            kind = "active"
        else:
            raise ValueError(
                f"kind is None and could not be extracted from file path {file_path}"
            )
    elif kind not in ["active", "passive"]:
        raise ValueError(f"kind {kind} not implemented (yet).")
    # convert to pandas.DataFrame
    data = tdms_file.as_dataframe()
    # make column names nicer: MultiIndex from frequencies/iterations and data kind
    if kind == "passive":
        # use run number as index
        group_numbers = [
            int(c.name.replace("Iteration_", "")) for c in tdms_file.groups()
        ]
        channel_names = [c.name for c in tdms_file.groups()[0].channels()]
        data.columns = pandas.MultiIndex.from_product(
            [group_numbers, channel_names], names=["Iteration", "Channel"]
        )
    elif kind == "active":
        # use frequencies as index
        data.columns = pandas.Index([c.replace("'", "") for c in data.columns])
        cols = [
            c[len("/frequency_00/") :]
            for c in data.columns
            if c.startswith("/frequency_00/")
        ]
        freq_nums = [
            int(
                c.replace("/", "")
                .replace("Position x", "")
                .replace("Position y", "")
                .replace("frequency_", "")
            )
            for c in data.columns[:: len(cols)]
        ]
        freqs = [
            float(tdms_file.groups()[fn].properties["Frequency"].replace(",", "."))
            for fn in freq_nums
        ]
        data.columns = pandas.MultiIndex.from_product(
            [freqs, cols], names=["frequency", "kind"]
        )
    return tdms_file, data, kind