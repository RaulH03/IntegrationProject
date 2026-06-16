import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import differential_evolution
from pendulum_core import get_lumped_bounds, simulate_n_step


def objective_global(p, u_data, x_meas, dt, Kt):
    x_sim = simulate_n_step(p, u_data, x_meas, dt, Kt, n_steps=200, delay_steps=1)
    if np.isinf(x_sim[0, 0]):
        return 1e6

    mse_th1 = np.mean((x_sim[0, :] - x_meas[0, :]) ** 2)
    mse_th2 = np.mean((x_sim[1, :] - x_meas[1, :]) ** 2)
    mse_dq1 = np.mean((x_sim[2, :] - x_meas[2, :]) ** 2)
    mse_dq2 = np.mean((x_sim[3, :] - x_meas[3, :]) ** 2)

    # Weight positions heavily, keep velocities to constrain explosions
    return (mse_th1 * 1.0) + (mse_th2 * 1.0) + (mse_dq1 * 0.01) + (mse_dq2 * 0.01)


if __name__ == "__main__":
    KNOWN_G, KNOWN_KT = 9.81, 2.73

    # 1. Tight constraints to force physical compliance
    phys_min = {
        "m1": 0.05,
        "m2": 0.01,
        "I1": 0.0001,
        "I2": 0.0001,
        "l1": 0.1,
        "l2": 0.1,
        "b1": 0.0,
        "b2": 0.0,
        "dry1": 0.0,
        "dry2": 0.0,
        "g": KNOWN_G,
    }
    phys_max = {
        "m1": 0.5,
        "m2": 0.2,
        "I1": 0.1,
        "I2": 0.1,
        "l1": 0.1,
        "l2": 0.1,
        "b1": 5.0,
        "b2": 0.5,
        "dry1": 2.0,
        "dry2": 2.0,
        "g": KNOWN_G,
    }
    phys_guess = {
        "m1": 0.11,
        "m2": 0.03,
        "I1": 0.004,
        "I2": 0.001,
        "l1": 0.1,
        "l2": 0.1,
        "b1": 0.85,
        "b2": 0.001,
        "dry1": 0.05,
        "dry2": 0.05,
        "g": KNOWN_G,
    }

    # 2. Extract bounds
    p0, (lower_bounds, upper_bounds) = get_lumped_bounds(phys_guess, phys_min, phys_max)
    de_bounds = list(zip(lower_bounds, upper_bounds))

    # 3. Load & Process Data
    # data = np.loadtxt(
    #     "experiments/NEW_chirp44_amp015_dt001.csv", delimiter=",", skiprows=1
    # )
    # dt = data[0, 1] - data[0, 0]
    # datas = data  # [:, :int(5.0 / dt)]

    # u_data = datas[1, :]
    # x_meas_pos = datas[2:4, :]
    # # x_meas_pos[0, :] = savgol_filter(np.unwrap(x_meas_pos[0, :] + 3.799) - 3.799, 7, 3)
    # # x_meas_pos[1, :] = savgol_filter(np.unwrap(x_meas_pos[1, :] + 1.21) - 1.21, 7, 3)
    # # # skip offset compensation
    # # x_meas_pos[0, :] = savgol_filter(np.unwrap(x_meas_pos[0, :]) - 3.799, 7, 3)
    # # x_meas_pos[1, :] = savgol_filter(np.unwrap(x_meas_pos[1, :]) - 1.21, 7, 3) + x_meas_pos[0, :]
    # # x_meas = np.vstack((x_meas_pos, np.gradient(x_meas_pos[0, :], dt), np.gradient(x_meas_pos[1, :], dt)))

    # x_meas_pos[0, :] = savgol_filter(np.unwrap(x_meas_pos[0, :]), 7, 3)
    # x_meas_pos[1, :] = (
    #     savgol_filter(np.unwrap(x_meas_pos[1, :]), 7, 3) + x_meas_pos[0, :]
    # )
    # x_meas = np.vstack(
    #     (
    #         x_meas_pos,
    #         np.gradient(x_meas_pos[0, :], dt),
    #         np.gradient(x_meas_pos[1, :], dt),
    #     )
    # )

    data = np.load("experiments/chirp_05_10_4_amp015_processed_unfiltered.npz")

    x_meas = data["x_meas"]
    u_data = data["u_data"]
    t_eval = data["t_eval"]
    dt = t_eval[1] - t_eval[0]

    # 4. Run Optimizer
    print("\nRunning Global N-Step Horizon Optimization (This will take a minute)...")
    res = differential_evolution(
        objective_global,
        bounds=de_bounds,
        args=(u_data, x_meas, dt, KNOWN_KT),
        strategy="best1bin",
        popsize=15,
        disp=True,
        tol=1e-2,
        workers=-1,
        updating="deferred",
    )

    print("\n--- COPY THIS LIST INTO validate.py ---")
    print(list(float(res) for res in np.round(res.x, 10)))
