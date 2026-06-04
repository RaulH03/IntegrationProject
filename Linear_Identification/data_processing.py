import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt

if __name__ == "__main__":
    data = np.loadtxt("experiments/chirp_05_10_4_amp02.csv", delimiter=",", skiprows=1)
    dt = data[0, 1] - data[0, 0]
    datas = data

    t_eval = datas[0, :]
    u_data = datas[1, :]
    x_meas_pos = datas[2:4, :]

    cutoff = 0.25  # Cutoff frequency in Hz
    order = 4  # Order of the filter (higher = steeper roll-off)

    b, a = butter(order, cutoff, btype="highpass", fs=1 / dt)

    x_meas_pos[0, :] = filtfilt(b, a, savgol_filter(np.unwrap(x_meas_pos[0, :]), 7, 3))
    x_meas_pos[1, :] = filtfilt(
        b, a, (savgol_filter(np.unwrap(x_meas_pos[1, :]), 7, 3) + x_meas_pos[0, :])
    )

    x_meas = np.vstack(
        (
            x_meas_pos,
            np.gradient(x_meas_pos[0, :], dt),
            np.gradient(x_meas_pos[1, :], dt),
        )
    )

    np.savez(
        "experiments/chirp_05_10_4_amp02_processed.npz",
        t_eval=t_eval,
        u_data=u_data,
        x_meas=x_meas,
    )
