import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, detrend
from pendulum_core import simulate_open_loop, augment_input

if __name__ == "__main__":
    print("Loading data...")
    # data = np.loadtxt('experiments/idinput_dt005_amp02.csv', delimiter=',', skiprows=1) 
    data = np.loadtxt('expirement_data_freq_sweep_UTF8_dot.csv', delimiter=';', skiprows=1).T
    dt = data[0, 1] - data[0, 0]
    datas = data[:, :int(12.0 / dt)]
    
    t_eval = datas[0, :]
    u_data = datas[1, :]
    x_meas_pos = datas[2:4, :]
    
    # Must detrend here as well to match what the optimizer saw!
    x_meas_pos[0, :] = savgol_filter(np.unwrap(x_meas_pos[0, :] + 3.799) - 3.799, 7, 3)
    x_meas_pos[1, :] = savgol_filter(np.unwrap(x_meas_pos[1, :] + 1.21) - 1.21, 7, 3)
    x_meas = np.vstack((x_meas_pos, np.gradient(x_meas_pos[0, :], dt), np.gradient(x_meas_pos[1, :], dt)))
    x_meas[2:4, 0] = 0
    KNOWN_KT = 2.73
    
    # ====================================================
    # PASTE THE LIST OUTPUT FROM optimize.py HERE!
    # ====================================================
    p_opt = [0.00267, 0.00143, 0.00119, 0.10703, 0.09362, 0.51714, 8e-05, 0.01546, 0.05117]
    
    print("Running Open-Loop Validation...")
    x_sim = simulate_open_loop(p_opt, u_data, x_meas, dt, KNOWN_KT, delay_steps=1)
    u_aug = augment_input(u_data, x_meas[2, :], p_opt, KNOWN_KT, delay_steps=1)
    
    rmse_th1 = np.sqrt(np.mean((x_sim[0, :] - x_meas[0, :])**2))
    rmse_th2 = np.sqrt(np.mean((x_sim[1, :] - x_meas[1, :])**2))
    print(f"\n--- VALIDATION RESULTS ---")
    print(f"Theta 1 (Base) RMSE:   {rmse_th1:.4f} rad")
    print(f"Theta 2 (Distal) RMSE: {rmse_th2:.4f} rad")

    # --- Plotting ---
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    plt.plot(t_eval, x_meas[0, :], 'k.', alpha=0.3, label='Measured $\\theta_1$')
    plt.plot(t_eval, x_sim[0, :], 'r-', linewidth=2, label=f'Linear ZOH (RMSE: {rmse_th1:.3f})')
    plt.title('Base Joint Angle ($\\theta_1$)')
    plt.ylabel('Angle (rad)'); plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(t_eval, x_meas[1, :], 'k.', alpha=0.3, label='Measured $\\theta_2$')
    plt.plot(t_eval, x_sim[1, :], 'b-', linewidth=2, label=f'Linear ZOH (RMSE: {rmse_th2:.3f})')
    plt.title('Distal Joint Angle ($\\theta_2$)')
    plt.ylabel('Angle (rad)'); plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(t_eval, u_data, 'k-', alpha=0.5, label='Raw Input ($u$)')
    plt.plot(t_eval, u_aug, 'orange', alpha=0.8, label='Augmented Input ($u_{aug}$)')
    plt.title('Control Signals')
    plt.ylabel('Input Signal'); plt.xlabel('Time (s)'); plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(t_eval, x_meas[2, :], 'k.', alpha=0.2, label='Measured $\\dot{\\theta}_1$')
    plt.plot(t_eval, x_sim[2, :], 'r-', linewidth=1.5, label='Simulated $\\dot{\\theta}_1$')
    plt.plot(t_eval, x_sim[3, :], 'b-', linewidth=1.5, label='Simulated $\\dot{\\theta}_2$')
    plt.title('Joint Velocities')
    plt.ylabel('Velocity (rad/s)'); plt.xlabel('Time (s)'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.show()