import numpy as np
import matplotlib.pyplot as plt
import signal
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter # Add this to your imports

from pendulum_function_gen import derive_and_lambdify, fast_dynamics

KNOWN_PARAMS = {'l1': 0.2, 'l2': 0.1, 'g': 9.81}


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

def cost_function_global(guess_array, t_data, x_data_pos, u_data):
    m1_g, m2_g, b1_g, b2_g = guess_array
    current_params = {'m1': m1_g, 'm2': m2_g, 'b1': b1_g, 'b2': b2_g, **KNOWN_PARAMS}
    
    u_func = interp1d(t_data, u_data, bounds_error=False, fill_value="extrapolate")
    
    # 1. Estimate velocities for the ENTIRE dataset upfront
    window = 21 
    poly = 3
    dt = t_data[1] - t_data[0]
    th1_smooth = savgol_filter(x_data_pos[0, :], window_length=window, polyorder=poly)
    th2_smooth = savgol_filter(x_data_pos[1, :], window_length=window, polyorder=poly)

    # 2. TAKE THE GRADIENT OF THE SMOOTHED DATA
    dq1_est = np.gradient(th1_smooth, dt)
    dq2_est = np.gradient(th2_smooth, dt)
        
    # Combine into a full 4xN state array [th1, th2, th1_dot, th2_dot]
    full_state_est = np.vstack((x_data_pos, dq1_est, dq2_est))
    
    # 2. Setup Multiple Shooting Chunks
    chunk_size = 100  # 50 points = 0.5 seconds (assuming 100Hz data)
    num_chunks = len(t_data) // chunk_size
    total_mse = 0.0
    
    # 3. Loop through each chunk
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(t_data))
        
        # Grab the data for this specific time window
        t_chunk = t_data[start_idx:end_idx]
        x_target_chunk = x_data_pos[:, start_idx:end_idx]
        
        # RESET the initial condition to the TRUE measured state at the start of this chunk
        x0_chunk = full_state_est[:, start_idx]
        
        # Simulate just this short window
        sol = solve_ivp(
            fast_dynamics, 
            [t_chunk[0], t_chunk[-1]], 
            x0_chunk, 
            args=(u_func, current_params, M_fast, f_fast), 
            t_eval=t_chunk,
            method='RK45'
        )
        
        # Penalize unstable parameters heavily
        if not sol.success:
            return 1e6 
            
        # Calculate raw error for this chunk (NO WRAPPING)
        chunk_error = sol.y[0:2, :] - x_target_chunk
        total_mse += np.mean(chunk_error**2)
        
    # Return the average error across all chunks
    return total_mse / num_chunks

# def cost_function_global(guess_array, t_data, x_data_pos, u_data):
#     m1_g, m2_g, b1_g, b2_g = guess_array
#     current_params = {'m1': m1_g, 'm2': m2_g, 'b1': b1_g, 'b2': b2_g, **KNOWN_PARAMS}
    
#     u_func = interp1d(t_data, u_data, bounds_error=False, fill_value="extrapolate")
    
#     dt = t_data[1] - t_data[0]
#     dth1_0 = (x_data_pos[0, 1] - x_data_pos[0, 0]) / dt
#     dth2_0 = (x_data_pos[1, 1] - x_data_pos[1, 0]) / dt
#     x0_guess = [x_data_pos[0, 0], x_data_pos[1, 0], dth1_0, dth2_0]
    
#     sol = solve_ivp(
#         fast_dynamics, 
#         [t_data[0], t_data[-1]], 
#         x0_guess, 
#         args=(u_func, current_params, M_fast, f_fast), # Uses globals
#         t_eval=t_data,
#         method='RK45'
#     )
    
#     if not sol.success:
#         return 1e6 
        
#     raw_error = sol.y[0:2, :] - x_data_pos
#     wrapped_error = (raw_error + np.pi) % (2 * np.pi) - np.pi
#     mse = np.mean(wrapped_error**2)
    
#     return mse

if __name__ == "__main__":
    print("Loading data...")
    data = np.load('pendulum_data.npz')
    t_eval = data['t']
    u_data = data['u']
    
    x_measured_full = data['x']
    x_measured_pos = x_measured_full[0:2, :] 

    print("\nStarting Parallel System Identification...")
    print("Press Ctrl+C at any time to stop early and view the best parameters.")
    
    bounds = [(0.01, 5.0), (0.01, 1.0), (0.00, 0.9), (0.00, 0.5)]

    # Run the evolutionary algorithm
    result = differential_evolution(
        cost_function_global, 
        bounds=bounds,
        args=(t_eval, x_measured_pos, u_data),
        strategy='best1bin', 
        popsize=15,          
        maxiter=250,         
        disp=True,           
        tol=1e-4,
        callback=early_stop_callback,
        workers=-1,
        updating='deferred'
    )

    m1_opt, m2_opt, b1_opt, b2_opt = result.x
    print("\n--- GLOBAL IDENTIFICATION RESULTS ---")
    if stop_optimization:
        print("(Note: Optimization was stopped early by user)")
    print(f"Optimized m1: {m1_opt:.4f} kg")
    print(f"Optimized m2: {m2_opt:.4f} kg")
    print(f"Optimized b1: {b1_opt:.4f} Nms/rad")
    print(f"Optimized b2: {b2_opt:.4f} Nms/rad")
    print(f"Final Mean Squared Error: {result.fun:.6f}")

    # 6. Validation Plot
    optimized_params = {'m1': m1_opt, 'm2': m2_opt, 'b1': b1_opt, 'b2': b2_opt, **KNOWN_PARAMS}
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