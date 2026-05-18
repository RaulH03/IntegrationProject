import numpy as np
import matplotlib.pyplot as plt
import signal
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter 

from pendulum_function_gen import derive_and_lambdify, fast_dynamics

# Added Kt to KNOWN_PARAMS based on your previous code snippet
KNOWN_PARAMS = {'l1': 0.1, 'l2': 0.1, 'g': 9.81, 'Kt': 2.73}

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

# ==========================================
# PHASE 1 COST FUNCTION (Optimizes Link 2)
# ==========================================
def cost_function_link2(guess_array, t_data, x_target_smooth, full_state_est, u_data, fixed_link1):
    m2_g, I2_g, b2_g = guess_array
    
    current_params = {
        'm1': fixed_link1['m1'], 'I1': fixed_link1['I1'], 
        'b1': fixed_link1['b1'], 'c1': fixed_link1['c1'],
        'm2': m2_g, 'I2': I2_g, 'b2': b2_g,
        **KNOWN_PARAMS
    }

    u_func = interp1d(t_data, u_data, bounds_error=False, fill_value="extrapolate")
    chunk_size = 50  
    num_chunks = len(t_data) // chunk_size
    total_mse = 0.0
    var_th2 = np.var(full_state_est[1, :])
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(t_data))
        t_chunk = t_data[start_idx:end_idx]
        
        x_target_chunk = x_target_smooth[:, start_idx:end_idx]
        x0_chunk = full_state_est[:, start_idx]
        
        sol = solve_ivp(
            fast_dynamics, [t_chunk[0], t_chunk[-1]], x0_chunk, 
            args=(u_func, current_params, M_fast, f_fast), 
            t_eval=t_chunk, method='Radau' 
        )
        
        if not sol.success:
            return 1e6 
            
        error_th2 = sol.y[1, :] - x_target_chunk[1, :]
        total_mse += np.mean((error_th2**2) / var_th2)
        
    return total_mse / num_chunks


# ==========================================
# PHASE 2 COST FUNCTION (Optimizes Link 1)
# ==========================================
def cost_function_link1(guess_array, t_data, x_target_smooth, full_state_est, u_data, opt_link2):
    m1_g, I1_g, b1_g, c1_g = guess_array

    # Safety Net: Prevent infinite energy generation
    if abs(c1_g) >= b1_g:
        return 1e6
    
    current_params = {
        'm1': m1_g, 'I1': I1_g, 'b1': b1_g, 'c1': c1_g,
        'm2': opt_link2['m2'], 'I2': opt_link2['I2'], 'b2': opt_link2['b2'],
        **KNOWN_PARAMS
    }

    u_func = interp1d(t_data, u_data, bounds_error=False, fill_value="extrapolate")
    chunk_size = 50  
    num_chunks = len(t_data) // chunk_size
    total_mse = 0.0
    var_th1 = np.var(full_state_est[0, :])
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(t_data))
        t_chunk = t_data[start_idx:end_idx]
        
        x_target_chunk = x_target_smooth[:, start_idx:end_idx]
        x0_chunk = full_state_est[:, start_idx]
        
        sol = solve_ivp(
            fast_dynamics, [t_chunk[0], t_chunk[-1]], x0_chunk, 
            args=(u_func, current_params, M_fast, f_fast), 
            t_eval=t_chunk, method='Radau' 
        )
        
        if not sol.success:
            return 1e6 
            
        error_th1 = sol.y[0, :] - x_target_chunk[0, :]
        total_mse += np.mean((error_th1**2) / var_th1)
        
    return total_mse / num_chunks


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Loading data...")
    # Make sure this matches your exact CSV structure
    data = np.loadtxt('experiments/IDinput_amp01.csv', delimiter=',', skiprows=1) 
    t_eval = data[0, :]
    u_data = data[1, :]
    
    x_measured = data[2:, :]
    x_measured[0, :] = np.unwrap(x_measured[0, :] + 3.799) - 3.799
    x_measured[1, :] = np.unwrap(x_measured[1, :] + 1.21) - 1.21
    x_measured_pos = x_measured[0:2, :] 

    print("Pre-processing and filtering data once...")
    window = 21 
    poly = 3
    dt = t_eval[1] - t_eval[0]
    
    th1_smooth = savgol_filter(x_measured_pos[0, :], window_length=window, polyorder=poly)
    th2_smooth = savgol_filter(x_measured_pos[1, :], window_length=window, polyorder=poly)
    x_target_smooth = np.vstack((th1_smooth, th2_smooth))

    dq1_est = np.gradient(th1_smooth, dt)
    dq2_est = np.gradient(th2_smooth, dt)
        
    full_state_est = np.vstack((x_target_smooth, dq1_est, dq2_est))
    x_measured_full = full_state_est

    print("\nStarting Parallel System Identification (Sequential Mode)...")
    print("Press Ctrl+C at any time to stop early and view the best parameters.")
    
    # ------------------------------------------------
    # PHASE 1: Optimize Link 2 (3 Parameters)
    # ------------------------------------------------
    print("\n--- PHASE 1: Isolating Link 2 Dynamics ---")
    
    # Reasonable dummy values for Link 1
    fixed_link1 = {'m1': 0.1, 'I1': 0.001, 'b1': 0.01, 'c1': 0.0}
    
    bounds_link2 = [
        (0.001, 0.05),        # m2
        (0.000001, 0.0005),   # I2
        (0.00, 0.5),          # b2
    ]

    result_link2 = differential_evolution(
        cost_function_link2, 
        bounds=bounds_link2,
        args=(t_eval, x_target_smooth, full_state_est, u_data, fixed_link1),
        strategy='best1bin', popsize=15, maxiter=100, disp=True, 
        tol=1e-4, callback=early_stop_callback, workers=-1, updating='deferred'
    )
    
    m2_opt, I2_opt, b2_opt = result_link2.x
    opt_link2 = {'m2': m2_opt, 'I2': I2_opt, 'b2': b2_opt}
    print(f"-> Link 2 Locked: m2={m2_opt:.4f}, I2={I2_opt:.6f}, b2={b2_opt:.4f}")

    # ------------------------------------------------
    # PHASE 2: Optimize Link 1 (4 Parameters)
    # ------------------------------------------------
    if not stop_optimization:
        print("\n--- PHASE 2: Isolating Link 1 Dynamics ---")
        
        bounds_link1 = [
            (0.01, 0.3),          # m1
            (0.00001, 0.005),     # I1
            (0.00, 0.5),          # b1
            (-0.5, 0.5),          # c1
        ]
    
        result_link1 = differential_evolution(
            cost_function_link1, 
            bounds=bounds_link1,
            args=(t_eval, x_target_smooth, full_state_est, u_data, opt_link2),
            strategy='best1bin', popsize=15, maxiter=100, disp=True, 
            tol=1e-4, callback=early_stop_callback, workers=-1, updating='deferred'
        )
        
        m1_opt, I1_opt, b1_opt, c1_opt = result_link1.x
        print(f"-> Link 1 Locked: m1={m1_opt:.4f}, I1={I1_opt:.6f}, b1={b1_opt:.4f}, c1={c1_opt:.4f}")

    # ------------------------------------------------
    # FINAL OUTPUT & VALIDATION
    # ------------------------------------------------
    print("\n--- SEQUENTIAL IDENTIFICATION COMPLETE ---")
    print(f"Masses:  m1 = {m1_opt:.4f} kg | m2 = {m2_opt:.4f} kg")
    print(f"Inertia: I1 = {I1_opt:.6f}    | I2 = {I2_opt:.6f}")
    print(f"Viscous: b1 = {b1_opt:.4f}    | b2 = {b2_opt:.4f}")
    print(f"Coulomb: c1 = {c1_opt:.4f}")

    optimized_params = {
        'm1': m1_opt, 'm2': m2_opt, 
        'I1': I1_opt, 'I2': I2_opt, 
        'b1': b1_opt, 'b2': b2_opt, 
        'c1': c1_opt, 
        **KNOWN_PARAMS
    }
    
    u_func_eval = interp1d(t_eval, u_data, bounds_error=False, fill_value="extrapolate")
    
    print("\nSimulating 15-second validation open-loop...")
    sol_opt = solve_ivp(
        fast_dynamics, 
        [t_eval[0], t_eval[-1]], 
        x_measured_full[:, 0], 
        args=(u_func_eval, optimized_params, M_fast, f_fast), 
        t_eval=t_eval, method='Radau'
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
    plt.plot(t_eval, u_data, 'k.', label='Input Data', alpha=0.3)
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
    plt.ylabel('Velocity (rad/s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()