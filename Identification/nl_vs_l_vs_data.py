import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, cont2discrete
import warnings

# Silence overflow and Scipy warnings
np.seterr(all='ignore') 
warnings.filterwarnings('ignore') 

# Import generated physics functions
from pendulum_function_gen import derive_and_lambdify, fast_dynamics, linearize_system, get_equilibrium_models

# ==========================================
# SOLVER & SIMULATION FUNCTIONS
# ==========================================
def rk4_step(dynamics, t, x, dt, args, sub_steps=5):
    """A single step of 4th Order Runge-Kutta, with sub-stepping for stability."""
    h = dt / sub_steps
    x_curr = x.copy()
    t_curr = t
    
    for _ in range(sub_steps):
        k1 = np.array(dynamics(t_curr, x_curr, *args))
        k2 = np.array(dynamics(t_curr + h/2.0, x_curr + (h/2.0) * k1, *args))
        k3 = np.array(dynamics(t_curr + h/2.0, x_curr + (h/2.0) * k2, *args))
        k4 = np.array(dynamics(t_curr + h, x_curr + h * k3, *args))
        
        x_curr = x_curr + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        t_curr += h
        
    return x_curr

def get_augmented_input(u_raw, dq1, params, k_smooth=100.0):
    """
    Augments the control input by canceling out the base joint's dry friction.
    u_aug = u_raw - (friction / Kt)
    """
    b1, c1, Kt = params['b1'], params['c1'], params['Kt']
    
    # Calculate smooth asymmetric dry friction based on current velocity
    s_th1 = np.tanh(k_smooth * dq1)
    fric_tau1 = (b1 * ((1 + s_th1) / 2.0) + c1 * ((1 - s_th1) / 2.0)) * s_th1
    
    return u_raw - (fric_tau1 / Kt)

# ==========================================
# DATA PROCESSING
# ==========================================
def load_and_preprocess_data(filepath, cut_off_time=10.0, window=7, poly=3):
    """Loads experimental data, slices it, unwraps angles, and smooths derivatives."""
    print("Loading and pre-processing data...")
    data = np.loadtxt(filepath, delimiter=',', skiprows=1) 
    
    # Determine dt and cutoff index
    dt = data[0, 1] - data[0, 0]
    idx_end = int(cut_off_time / dt)
    datas = data[:, :idx_end].copy()
    
    t_eval = datas[0, :]
    u_data = datas[1, :]
    
    # Extract and unwrap measured state
    x_measured = datas[2:, :]
    x_measured[0, :] = np.unwrap(x_measured[0, :] + 3.799) - 3.799
    x_measured[1, :] = np.unwrap(x_measured[1, :] + 1.21) - 1.21
    x_measured_pos = x_measured[0:2, :]
    
    # Filter and differentiate
    th1_smooth = savgol_filter(x_measured_pos[0, :], window_length=window, polyorder=poly)
    th2_smooth = savgol_filter(x_measured_pos[1, :], window_length=window, polyorder=poly)
    
    dq1_est = np.gradient(th1_smooth, dt)
    dq2_est = np.gradient(th2_smooth, dt)
        
    x_measured_full = np.vstack((th1_smooth, th2_smooth, dq1_est, dq2_est))
    
    return t_eval, u_data, x_measured_full, dt


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # 1. Load Data
    DATA_PATH = 'experiments/sin_amp02.csv'
    t_eval, u_data, x_measured_full, dt = load_and_preprocess_data(DATA_PATH, cut_off_time=10.0)
    t0_global = t_eval[0]
    num_eval_steps = len(t_eval)

    # 2. Define Parameters
    KNOWN_PARAMS = {'l1': 0.1, 'l2': 0.1, 'g': 9.81, 'Kt': 2.73}
    optimized_params = {
        'm1': 0.1081, 'm2': 0.0338, 
        'I1': 0.004030, 'I2': 0.00095, 
        'b1': 0.8500, 'b2': 0.0003, 
        'c1': 0.6830, 
        **KNOWN_PARAMS
    }

    # 3. Initialize Math Engine and Linearize
    print("Initializing math engine...")
    M_clean, f_clean = derive_and_lambdify(lamdify=False)
    
    E_func, A_func, B_func = linearize_system(M_clean, f_clean) 

    lin_params = optimized_params.copy()
    lin_params['b1'] = 0.0  
    lin_params['c1'] = 0.0  
    lin_params['b2'] = 0.0  
    
    models = get_equilibrium_models(lin_params, E_func, A_func, B_func)

    A_cont = models['Down-Down']['A']
    B_cont = models['Down-Down']['B']
    
    print("\nDown-Down Equilibrium A Matrix:\n", np.round(A_cont, 3))
    print("Down-Down Equilibrium B Matrix:\n", np.round(B_cont, 3))
    
    # 4. Discretize Linear System (ZOH)
    # C and D matrices are required for cont2discrete but we only care about A and B
    C_dummy = np.eye(4)
    D_dummy = np.zeros((4, 1))
    sys_d = cont2discrete((A_cont, B_cont, C_dummy, D_dummy), dt, method='zoh')
    Ad, Bd = sys_d[0], sys_d[1]

    # 5. Initialize Simulation Arrays
    sol_opt_y = np.zeros((4, num_eval_steps))  # Non-linear RK4
    sol_lin_y = np.zeros((4, num_eval_steps))  # Linear ZOH
    u_augmented_history = np.zeros(num_eval_steps)
    
    # Set initial conditions directly from data
    sol_opt_y[:, 0] = x_measured_full[:, 0]
    sol_lin_y[:, 0] = x_measured_full[:, 0]
    u_augmented_history[0] = u_data[0]

    M_fast, f_fast = derive_and_lambdify()
    
    print("\nSimulating full dataset (Non-linear RK4 & Linear ZOH with Friction Augmentation)...")
    ode_args_final = (u_data, t0_global, dt, optimized_params, M_fast, f_fast)
    
    # Tuning parameter for hardware/filter delay (e.g., 1 to 3 timesteps)
    delay_steps = 3
    
    for k in range(num_eval_steps - 1):
        # --- A. Non-Linear Simulation (RK4) ---
        sol_opt_y[:, k+1] = rk4_step(
            fast_dynamics, t_eval[k], sol_opt_y[:, k], dt, ode_args_final, sub_steps=20
        )
        
        # --- B. Linear Simulation (ZOH w/ Augmented Input) ---
        current_lin_state = sol_lin_y[:, k]
        
        # --- FIX 2: Use MEASURED velocity (with delay) to prevent positive feedback loops ---
        # Shift the index back by delay_steps to account for the physical phase delay
        idx_vel = max(0, k - delay_steps)
        measured_dq1 = x_measured_full[2, idx_vel]
        
        u_aug = get_augmented_input(u_data[k], measured_dq1, optimized_params)
        u_augmented_history[k+1] = u_aug
        
        # Step the discrete linear state-space model
        sol_lin_y[:, k+1] = Ad @ current_lin_state + Bd.flatten() * u_aug

    # 7. Calculate Integration Tracking Error (RMSE)
    rmse_nl_th1 = np.sqrt(np.mean((sol_opt_y[0, :] - x_measured_full[0, :])**2))
    rmse_nl_th2 = np.sqrt(np.mean((sol_opt_y[1, :] - x_measured_full[1, :])**2))
    rmse_lin_th1 = np.sqrt(np.mean((sol_lin_y[0, :] - x_measured_full[0, :])**2))
    rmse_lin_th2 = np.sqrt(np.mean((sol_lin_y[1, :] - x_measured_full[1, :])**2))
    
    print(f"\n--- INTEGRATION TRACKING ERROR ---")
    print(f"Non-Linear Model | Theta 1 RMSE: {rmse_nl_th1:.4f} rad, Theta 2 RMSE: {rmse_nl_th2:.4f} rad")
    print(f"Linearized Model | Theta 1 RMSE: {rmse_lin_th1:.4f} rad, Theta 2 RMSE: {rmse_lin_th2:.4f} rad")

    # ==========================================
    # PLOTTING
    # ==========================================
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: Theta 1
    plt.subplot(2, 2, 1)
    plt.plot(t_eval, x_measured_full[0, :], 'k.', label=r'Measured $\theta_1$', alpha=0.3)
    plt.plot(t_eval, sol_opt_y[0, :], 'r-', label=f'NL RK4 (RMSE: {rmse_nl_th1:.2f})', linewidth=2)
    plt.plot(t_eval, sol_lin_y[0, :], 'g--', label=f'Linear ZOH (RMSE: {rmse_lin_th1:.2f})', linewidth=2)
    plt.ylabel('Angle (rad)')
    plt.title('Base Joint Angle ($\theta_1$)')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Theta 2
    plt.subplot(2, 2, 2)
    plt.plot(t_eval, x_measured_full[1, :], 'k.', label=r'Measured $\theta_2$', alpha=0.3)
    plt.plot(t_eval, sol_opt_y[1, :], 'b-', label=f'NL RK4 (RMSE: {rmse_nl_th2:.2f})', linewidth=2)
    plt.plot(t_eval, sol_lin_y[1, :], 'm--', label=f'Linear ZOH (RMSE: {rmse_lin_th2:.2f})', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title('Distal Joint Angle ($\theta_2$)')
    plt.legend()
    plt.grid(True)

    # Subplot 3: Inputs (Raw vs Augmented)
    plt.subplot(2, 2, 3)
    plt.plot(t_eval, u_data, 'k-', label='Raw Input Data ($u$)', alpha=0.5)
    plt.plot(t_eval, u_augmented_history, 'orange', label=r'Augmented Input ($u - \tau_{fric}/K_t$)', alpha=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Input Signal')
    plt.title('Control Signals')
    plt.legend()
    plt.grid(True)

    # Subplot 4: Velocities
    plt.subplot(2, 2, 4)
    plt.plot(t_eval, x_measured_full[2, :], 'k.', label=r'Measured $\dot{\theta}_1$', alpha=0.2)
    plt.plot(t_eval, sol_opt_y[2, :], 'r-', label=r'NL Fit $\dot{\theta}_1$', linewidth=1.5)
    plt.plot(t_eval, sol_lin_y[2, :], 'g--', label=r'Lin Fit $\dot{\theta}_1$', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.title('Base Joint Velocity ($\dot{\theta}_1$)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()