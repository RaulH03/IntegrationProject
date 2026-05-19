import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter 
import warnings

# Silence overflow and Scipy warnings
np.seterr(all='ignore') 
warnings.filterwarnings('ignore') 

# Import your generated physics
from pendulum_function_gen import derive_and_lambdify, fast_dynamics, linearize_system, get_equilibrium_models

# ==========================================
# CUSTOM FIXED-STEP SOLVER (RK4)
# ==========================================
def rk4_step(dynamics, t, x, dt, args, sub_steps=50):
    """A single step of 4th Order Runge-Kutta, with sub-stepping for stability."""
    h = dt / sub_steps
    x_curr = x.copy()
    t_curr = t
    
    for _ in range(sub_steps):
        # We pass the full args (which includes the u_data array) 
        # because your fast_dynamics handles the time-indexing internally.
        k1 = np.array(dynamics(t_curr, x_curr, *args))
        k2 = np.array(dynamics(t_curr + h/2.0, x_curr + (h/2.0) * k1, *args))
        k3 = np.array(dynamics(t_curr + h/2.0, x_curr + (h/2.0) * k2, *args))
        k4 = np.array(dynamics(t_curr + h, x_curr + h * k3, *args))
        
        x_curr = x_curr + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        t_curr += h
        
    return x_curr

# ==========================================
# MAIN SIMULATION SCRIPT
# ==========================================
if __name__ == "__main__":
    print("Initializing math engine...")
    M_fast, f_fast = derive_and_lambdify()

    print("Loading data...")
    # UPDATE THIS PATH IF YOUR DATA IS SOMEWHERE ELSE
    # data_full = np.loadtxt('experiments/sin_amp02.csv', delimiter=',', skiprows=1) 
    # num_samples = data_full.shape[1]
    # data = data_full[:, :int(num_samples/2)].copy()

    data = np.loadtxt('experiments/idinput_dt005_amp02.csv', delimiter=',', skiprows=1) 
    # data = np.loadtxt('experiments/sin_amp02.csv', delimiter=',', skiprows=1) 
    # data = np.loadtxt('experiments/idinput_dt005_amp01.csv', delimiter=',', skiprows=1) 
    # data = np.loadtxt('experiments/idinput_amp02.csv', delimiter=',', skiprows=1) 
    # data = np.loadtxt('experiments/IDinput_amp04.csv', delimiter=',', skiprows=1) 
   



    # Calculate dt directly from the first two time steps
    dt = data[0, 1] - data[0, 0]
    
    # Convert 10 seconds into an array index
    cut_off_time = 10.0 
    idx_end = int(cut_off_time / dt)
    
    # Slice the data to only include up to that index!
    datas = data[:, :idx_end].copy()
    
    t_eval = datas[0, :]
    u_data = datas[1, :]
    
    x_measured = datas[2:, :]
    x_measured[0, :] = np.unwrap(x_measured[0, :] + 3.799) - 3.799
    x_measured[1, :] = np.unwrap(x_measured[1, :] + 1.21) - 1.21
    x_measured_pos = x_measured[0:2, :]

    # t_eval = data[0, :]
    # u_data = data[1, :]
    # dt = t_eval[1] - t_eval[0]
    t0_global = t_eval[0]
    
    # x_measured = data[2:, :]
    # x_measured[0, :] = np.unwrap(x_measured[0, :] + 3.799) - 3.799
    # x_measured[1, :] = np.unwrap(x_measured[1, :] + 1.21) - 1.21
    # x_measured_pos = x_measured[0:2, :] 

    print("Pre-processing and filtering data...")
    window = 7 
    poly = 3
    
    th1_smooth = savgol_filter(x_measured_pos[0, :], window_length=window, polyorder=poly)
    th2_smooth = savgol_filter(x_measured_pos[1, :], window_length=window, polyorder=poly)
    x_target_smooth = np.vstack((th1_smooth, th2_smooth))

    dq1_est = np.gradient(th1_smooth, dt)
    dq2_est = np.gradient(th2_smooth, dt)
        
    x_measured_full = np.vstack((x_target_smooth, dq1_est, dq2_est))

    # ==========================================
    # SET YOUR OPTIMIZED PARAMETERS HERE
    # ==========================================
    KNOWN_PARAMS = {'l1': 0.1, 'l2': 0.1, 'g': 9.81, 'Kt': 2.73}
    
    # Plug in the final values you got from your optimization script
    # optimized_params = {
    #     'm1': 0.2499, 'm2': 0.0698, 
    #     'I1': 0.001948, 'I2': 0.000133, 
    #     'b1': 1.2560, 'b2': 0.0003, 
    #     'c1': 0.8492, 
    #     **KNOWN_PARAMS
    # }

    optimized_params = {
        'm1': 0.1081, 'm2': 0.0338, 
        'I1': 0.004030, 'I2': 0.00095, 
        'b1': 0.8500, 'b2': 0.0003, 
        'c1': 0.6830, 
        **KNOWN_PARAMS
    }

       # After derive_and_lambdify()
    M_clean, f_clean = derive_and_lambdify(lamdify=False)
    
    # 1. Generate the symbolic Jacobian functions
    E_func, A_func, B_func = linearize_system(M_clean, f_clean) 
    
    # 2. Extract the A and B matrices at each equilibrium
    models = get_equilibrium_models(optimized_params, E_func, A_func, B_func)

    A = models['Down-Down']['A']
    B = models['Down-Down']['B']
    
    # Print the Up-Up unstable equilibrium matrices
    print("Down-Down Equilibrium A Matrix:\n", np.round(A, 3))
    print("Down-Down Equilibrium B Matrix:\n", np.round(A, 3))
    
    print("\nSimulating full dataset validation using RK4 (Open-Loop)...")
    
    # Bundle args exactly as fast_dynamics expects them
    ode_args_final = (u_data, t0_global, dt, optimized_params, M_fast, f_fast)
    
    # Initialize array to hold final simulation data
    num_eval_steps = len(t_eval)
    sol_opt_y = np.zeros((4, num_eval_steps))
    
    # Start the simulation at the EXACT initial state of the real data
    sol_opt_y[:, 0] = x_measured_full[:, 0]
    
    # Run the full validation with RK4
    for k in range(num_eval_steps - 1):
        sol_opt_y[:, k+1] = rk4_step(fast_dynamics, t_eval[k], sol_opt_y[:, k], dt, ode_args_final, sub_steps=5)

    # 3. Calculate Root Mean Square Error (RMSE) against actual data
    error_th1 = sol_opt_y[0, :] - x_measured_full[0, :]
    error_th2 = sol_opt_y[1, :] - x_measured_full[1, :]
    
    rmse_th1 = np.sqrt(np.mean(error_th1**2))
    rmse_th2 = np.sqrt(np.mean(error_th2**2))
    
    print(f"\n--- INTEGRATION TRACKING ERROR ---")
    print(f"Theta 1 (Base) RMSE:   {rmse_th1:.4f} rad")
    print(f"Theta 2 (Distal) RMSE: {rmse_th2:.4f} rad")

    # ==========================================
    # PLOTTING
    # ==========================================
    plt.figure(figsize=(10, 8))
    
    # Note: Using raw strings (r'...') prevents Python \d escape sequence warnings
    plt.subplot(2, 2, 1)
    plt.plot(t_eval, x_measured_full[0, :], 'k.', label=r'Measured $\theta_1$', alpha=0.3)
    plt.plot(t_eval, sol_opt_y[0, :], 'r-', label=f'RK4 Fit (RMSE: {rmse_th1:.2f})', linewidth=2)
    plt.ylabel('Angle (rad)')
    plt.title('System Identification Results: Model vs Real Data')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(t_eval, x_measured_full[1, :], 'k.', label=r'Measured $\theta_2$', alpha=0.3)
    plt.plot(t_eval, sol_opt_y[1, :], 'b-', label=f'RK4 Fit (RMSE: {rmse_th2:.2f})', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(t_eval, u_data, 'k.', label='Input Data', alpha=0.3)
    plt.ylabel('Input')
    plt.title('Input Data')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(t_eval, x_measured_full[2, :], 'k.', label=r'Measured $\dot{\theta}_1$', alpha=0.3)
    plt.plot(t_eval, x_measured_full[3, :], 'k.', label=r'Measured $\dot{\theta}_2$', alpha=0.3)
    plt.plot(t_eval, sol_opt_y[2, :], 'b-', label=r'Optimized Fit $\dot{\theta}_1$', linewidth=2)
    plt.plot(t_eval, sol_opt_y[3, :], 'b-', label=r'Optimized Fit $\dot{\theta}_2$', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()