import numpy as np
import matplotlib.pyplot as plt
import signal
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter # Add this to your imports

from pendulum_function_gen import derive_and_lambdify, fast_dynamics

KNOWN_PARAMS = {'l1': 0.1, 'l2': 0.1, 'g': 9.81}


print("Initializing math engine on this CPU core...")
M_fast, f_fast = derive_and_lambdify()


stop_optimization = False

def handle_ctrl_c(sig, frame):
    global stop_optimization
    print("\n[!] Ctrl+C detected! Finishing the current generation and stopping...")
    stop_optimization = True

signal.signal(signal.SIGINT, handle_ctrl_c)

def early_stop_callback(xk, convergence):
    global stop_optimization
    if stop_optimization:
        return True 

def cost_function_global(guess_array, t_data, x_target_smooth, full_state_est, u_data):
    m1_g, m2_g, I1_g, I2_g, b1_g, b2_g, c1_g, Kt_g = guess_array


    if abs(c1_g) >= b1_g:
        return 1e6
    
    current_params = {
        'm1': m1_g, 'm2': m2_g, 
        'I1': I1_g, 'I2': I2_g, 
        'b1': b1_g, 'b2': b2_g, 
        'c1': c1_g, 'Kt': Kt_g,
        **KNOWN_PARAMS
    }



    u_func = interp1d(t_data, u_data, bounds_error=False, fill_value="extrapolate")
    
    chunk_size = 100  # 100 points
    num_chunks = len(t_data) // chunk_size
    total_mse = 0.0
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(t_data))
        
        t_chunk = t_data[start_idx:end_idx]
        
        # Use the pre-computed smoothed targets
        x_target_chunk = x_target_smooth[:, start_idx:end_idx]
        x0_chunk = full_state_est[:, start_idx]
        
        sol = solve_ivp(
            fast_dynamics, 
            [t_chunk[0], t_chunk[-1]], 
            x0_chunk, 
            args=(u_func, current_params, M_fast, f_fast), 
            t_eval=t_chunk,
            method='Radau' # <--- VITAL: Use Radau for the tanh() stiffness
        )
        
        if not sol.success:
            return 1e6 
            
        # Calculate raw error against the smoothed target data

        error_th1 = sol.y[0, :] - x_target_chunk[0, :]
        error_th2 = sol.y[1, :] - x_target_chunk[1, :]
        
        # Divide by the global variance of each signal so they have equal weight
        var_th1 = np.var(full_state_est[0, :])
        var_th2 = np.var(full_state_est[1, :])
        
        # Add them together
        total_mse += np.mean((error_th1**2) / var_th1) + np.mean((error_th2**2) / var_th2)
        
    return total_mse / num_chunks
        

if __name__ == "__main__":
    print("Loading data...")
    data = np.loadtxt('expirement_data_freq_sweep_UTF8_dot.csv', delimiter=';', skiprows=1) 
    t_eval = data[:, 0]
    u_data = -data[:, 1]
    
    x_measured = data[:, 2:].T
    x_measured[0, :] = np.unwrap(x_measured[0, :] + 3.779) - 3.779
    x_measured[1, :] = np.unwrap(x_measured[1, :] + 1.21) - 1.21
    x_measured_pos = x_measured[0:2, :] 

    print("Pre-processing and filtering data once...")
    window = 21 
    poly = 3
    dt = t_eval[1] - t_eval[0]
    
    # 1. PRE-COMPUTE smoothed states ONE TIME
    th1_smooth = savgol_filter(x_measured_pos[0, :], window_length=window, polyorder=poly)
    th2_smooth = savgol_filter(x_measured_pos[1, :], window_length=window, polyorder=poly)
    x_target_smooth = np.vstack((th1_smooth, th2_smooth))

    # 2. PRE-COMPUTE velocities ONE TIME
    dq1_est = np.gradient(th1_smooth, dt)
    dq2_est = np.gradient(th2_smooth, dt)
        
    full_state_est = np.vstack((x_target_smooth, dq1_est, dq2_est))
    x_measured_full = full_state_est

    print("\nStarting Parallel System Identification...")
    print("Press Ctrl+C at any time to stop early and view the best parameters.")
    
    # Allow c1 to be negative so it can properly identify asymmetric friction direction
    bounds = [
        (0.01, 0.5),   # m1
        (0.01, 0.1),   # m2
        (0.01, 1.0),   # I1
        (0.01, 1.0),   # I2
        (0.00, 0.5),   # b1 (Viscous)
        (0.00, 0.5),   # b2 (Viscous)
        (-0.5, 0.5),   # c1 (Asymmetry Modifier)
        (0.01, 5.0)    # Kt
    ]

    # Run the evolutionary algorithm
    result = differential_evolution(
        cost_function_global, 
        bounds=bounds,
        # Pass the pre-computed arrays to the args!
        args=(t_eval, x_target_smooth, full_state_est, u_data),
        strategy='best1bin', 
        popsize=15,          
        maxiter=150,  # 150 is usually plenty to see convergence        
        disp=True,           
        tol=1e-4,
        callback=early_stop_callback,
        workers=-1,
        updating='deferred'
    )

    m1_opt, m2_opt, I1_opt, I2_opt, b1_opt, b2_opt, c1_opt, Kt_opt= result.x
    print("\n--- GLOBAL IDENTIFICATION RESULTS ---")
    if stop_optimization:
        print("(Note: Optimization was stopped early by user)")
    print(f"Masses:  m1 = {m1_opt:.4f} kg | m2 = {m2_opt:.4f} kg")
    print(f"Inertia: I1 = {I1_opt:.4f}    | I2 = {I2_opt:.4f}")
    print(f"Viscous: b1 = {b1_opt:.4f}    | b2 = {b2_opt:.4f}")
    print(f"Coulomb: c1 = {c1_opt:.4f}")
    print(f"Motor:   Kt = {Kt_opt:.4f}")
    print(f"Final Mean Squared Error: {result.fun:.6f}")

    # 6. Validation Plot
    optimized_params = {'m1': m1_opt, 'm2': m2_opt, 
                        'I1': I1_opt, 'I2': I2_opt, 
                        'b1': b1_opt, 'b2': b2_opt, 
                        'c1': c1_opt, 'Kt': Kt_opt, 
                        **KNOWN_PARAMS}
    u_func_eval = interp1d(t_eval, u_data, bounds_error=False, fill_value="extrapolate")
    
    sol_opt = solve_ivp(
        fast_dynamics, 
        [t_eval[0], t_eval[-1]], 
        x_measured_full[:, 0], 
        args=(u_func_eval, optimized_params, M_fast, f_fast), 
        t_eval=t_eval
    )

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(t_eval, x_measured_full[0, :], 'k.', label='Measured $\\theta_1$', alpha=0.3)
    plt.plot(t_eval, sol_opt.y[0, :], 'r-', label='Optimized Fit $\\theta_1$', linewidth=2)
    plt.ylabel('Angle (rad)')
    plt.title('System Identification Results: Model vs Real Data')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(t_eval, x_measured_full[1, :], 'k.', label='Measured $\\theta_2$', alpha=0.3)
    plt.plot(t_eval, sol_opt.y[1, :], 'b-', label='Optimized Fit $\\theta_2$', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(t_eval, u_data, 'k.', label='Measured $\\theta_1$', alpha=0.3)
    # plt.plot(t_eval, sol_opt.y[0, :], 'r-', label='Optimized Fit $\\theta_1$', linewidth=2)
    plt.ylabel('Input')
    plt.title('Input data')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(t_eval, x_measured_full[2, :], 'k.', label='Measured $\\dot{\\theta}_1$', alpha=0.3)
    plt.plot(t_eval, x_measured_full[3, :], 'k.', label='Measured $\\dot{\\theta}_2$', alpha=0.3)
    plt.plot(t_eval, sol_opt.y[2, :], 'b-', label='Optimized Fit $\\dot{\\theta}_1$', linewidth=2)
    plt.plot(t_eval, sol_opt.y[3, :], 'b-', label='Optimized Fit $\\dot{\\theta}_2$', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (radss^-1)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()