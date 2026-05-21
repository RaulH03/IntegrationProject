import numpy as np
import matplotlib.pyplot as plt
import signal
from scipy.optimize import differential_evolution, minimize
from scipy.signal import savgol_filter 
import warnings
np.seterr(all='ignore') # <--- ADD THIS to silence overflow/NaN warnings
warnings.filterwarnings('ignore') # <--- ADD THIS to silence Scipy warnings

from pendulum_function_gen import derive_and_lambdify, fast_dynamics

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
# CUSTOM FIXED-STEP SOLVER
# ==========================================
def rk4_step(dynamics, t, x, dt, args, sub_steps=5):
    """A single data step of RK4, broken into sub-steps for mathematical stability."""
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

# ==========================================
# PHASE 1 COST FUNCTION (Optimizes Link 1)
# ==========================================
def cost_function_link1(guess_array, t_data, x_target_smooth, full_state_est, u_data, fixed_link2):
    m1_g, I1_g, b1_g, c1_g = guess_array

    # Safety Net: Prevent infinite energy generation
    if abs(c1_g) >= b1_g:
        return 1e6
    
    current_params = {
        'm1': m1_g, 'I1': I1_g, 'b1': b1_g, 'c1': c1_g,
        'm2': fixed_link2['m2'], 'I2': fixed_link2['I2'], 'b2': fixed_link2['b2'],
        **KNOWN_PARAMS
    }

    dt = t_data[1] - t_data[0]
    t0_global = t_data[0]
    
    chunk_size = 50  
    num_chunks = len(t_data) // chunk_size
    total_mse = 0.0
    var_th1 = np.var(full_state_est[0, :])
    
    # Bundle args exactly as fast_dynamics expects them
    ode_args = (u_data, t0_global, dt, current_params, M_fast, f_fast)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(t_data))
        t_chunk = t_data[start_idx:end_idx]
        
        x_target_chunk = x_target_smooth[:, start_idx:end_idx]
        x0_chunk = full_state_est[:, start_idx]
        
        num_steps = len(t_chunk)
        x_sim = np.zeros((4, num_steps))
        x_sim[:, 0] = x0_chunk
        
        # RK4 Integration Loop
        # RK4 Integration Loop with Safety Trap
        for k in range(num_steps - 1):
            next_state = rk4_step(fast_dynamics, t_chunk[k], x_sim[:, k], dt, ode_args, sub_steps=5)
            
            # --- NUMERICAL EXPLOSION TRAP ---
            # If the optimizer guessed parameters that make the physics blow up to Infinity or NaN,
            # abort immediately and penalize this parameter set.
            if not np.all(np.isfinite(next_state)):
                return 1e6 
                
            x_sim[:, k+1] = next_state
        
        # Calculate this specific chunk's error
        error_th1 = x_sim[0, :] - x_target_chunk[0, :]
        chunk_mse = np.mean((error_th1**2) / var_th1)
        
        # --- THE EARLY EXIT TRAP ---
        if chunk_mse > 2.0:
            return 1e6 
            
        total_mse += chunk_mse
        
    return total_mse / num_chunks


# ==========================================
# PHASE 2 COST FUNCTION (Optimizes Link 2)
# ==========================================
def cost_function_link2(guess_array, t_data, x_target_smooth, full_state_est, u_data, opt_link1):
    m2_g, I2_g, b2_g = guess_array
    
    current_params = {
        'm1': opt_link1['m1'], 'I1': opt_link1['I1'], 
        'b1': opt_link1['b1'], 'c1': opt_link1['c1'],
        'm2': m2_g, 'I2': I2_g, 'b2': b2_g,
        **KNOWN_PARAMS
    }

    dt = t_data[1] - t_data[0]
    t0_global = t_data[0]
    
    chunk_size = 50  
    num_chunks = len(t_data) // chunk_size
    total_mse = 0.0
    var_th2 = np.var(full_state_est[1, :])
    
    # Bundle args exactly as fast_dynamics expects them
    ode_args = (u_data, t0_global, dt, current_params, M_fast, f_fast)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(t_data))
        t_chunk = t_data[start_idx:end_idx]
        
        x_target_chunk = x_target_smooth[:, start_idx:end_idx]
        x0_chunk = full_state_est[:, start_idx]
        
        num_steps = len(t_chunk)
        x_sim = np.zeros((4, num_steps))
        x_sim[:, 0] = x0_chunk
        
        # RK4 Integration Loop with Safety Trap
        for k in range(num_steps - 1):
            next_state = rk4_step(fast_dynamics, t_chunk[k], x_sim[:, k], dt, ode_args, sub_steps=5)
            
            # --- NUMERICAL EXPLOSION TRAP ---
            # If the optimizer guessed parameters that make the physics blow up to Infinity or NaN,
            # abort immediately and penalize this parameter set.
            if not np.all(np.isfinite(next_state)):
                return 1e6 
                
            x_sim[:, k+1] = next_state

        # Calculate this specific chunk's error
        error_th2 = x_sim[1, :] - x_target_chunk[1, :]
        chunk_mse = np.mean((error_th2**2) / var_th2)
        
        # --- THE EARLY EXIT TRAP ---
        if chunk_mse > 2.0:
            return 1e6 
            
        total_mse += chunk_mse
        
    return total_mse / num_chunks


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Loading data...")
    
    data = np.loadtxt('experiments/sin_amp02.csv', delimiter=',', skiprows=1)
    # data_full = np.loadtxt('experiments/idinput_dt005_amp02.csv', delimiter=',', skiprows=1) 
    # num_samples = data_full.shape[1]
    # data = data_full[:, :int(num_samples/2)].copy()
    t_eval = data[0, :]
    u_data = data[1, :]
    
    x_measured = data[2:, :]
    x_measured[0, :] = np.unwrap(x_measured[0, :] + 3.799) - 3.799
    x_measured[1, :] = np.unwrap(x_measured[1, :] + 1.21) - 1.21
    x_measured_pos = x_measured[0:2, :] 

    print("Pre-processing and filtering data once...")
    window = 40 
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
    # PHASE 1: Optimize Link 1 (The Proximal Base)
    # ------------------------------------------------
    print("\n--- PHASE 1: Isolating Link 1 Dynamics ---")
    
    # Reasonable dummy values for Link 2 (minor disturbance)
    fixed_link2 = {'m2': 0.0510, 'I2': 0.000100, 'b2': 0.0}
    
    bounds_link1 = [
        (0.1, 0.5),          # m1
        (0.00001, 0.05),     # I1
        (0.0, 2.0),          # b1
        (0.0, 1.0),          # c1
    ]

    previous_best_link1 = [0.2499, 0.001948, 1.2560, 0.8492]

    result_link1 = differential_evolution(
        cost_function_link1, 
        bounds=bounds_link1,
        args=(t_eval, x_target_smooth, full_state_est, u_data, fixed_link2),
        strategy='best1bin', 
        popsize=8, # Optimized for speed given the tight bounds
        mutation=(0.5, 1.5),  # Default is (0.5, 1.0). Widen it to force bold exploration.
        recombination=0.5,    # Default is 0.7. Lowering it forces more parameter mixing.
        maxiter=50, disp=True, 
        tol=1e-3, callback=early_stop_callback, workers=-1, updating='deferred',
        polish=False,
        x0=previous_best_link1
    )

    # m1_opt, I1_opt, b1_opt, c1_opt = result_link1.x

    polished_result1 = minimize(
        cost_function_link1, 
        x0=result_link1.x, 
        args=(t_eval, x_target_smooth, full_state_est, u_data, fixed_link2),
        method='Nelder-Mead',
        bounds=bounds_link1, # Note: Nelder-Mead in newer Scipy versions supports bounds!
        options={'xatol': 1e-4, 'fatol': 1e-4, 'disp': True, 'maxiter': 50}
    )

    m1_opt, I1_opt, b1_opt, c1_opt = polished_result1.x
    

    opt_link1 = {'m1': m1_opt, 'I1': I1_opt, 'b1': b1_opt, 'c1': c1_opt}
    print(f"-> Link 1 Locked: m1={m1_opt:.4f}, I1={I1_opt:.6f}, b1={b1_opt:.4f}, c1={c1_opt:.4f}")

    # ------------------------------------------------
    # PHASE 2: Optimize Link 2 (The Distal End)
    # ------------------------------------------------
    if not stop_optimization:
        print("\n--- PHASE 2: Isolating Link 2 Dynamics ---")
        
        bounds_link2 = [
            (0.001, 0.1),        # m2
            (0.000001, 0.0005),   # I2
            (0.00, 0.5),          # b2
        ]

        previous_best_link2 = [0.0698, 0.000133, 0.0002]
    
        result_link2 = differential_evolution(
            cost_function_link2, 
            bounds=bounds_link2,
            args=(t_eval, x_target_smooth, full_state_est, u_data, opt_link1),
            mutation=(0.5, 1.5),  
            recombination=0.5,    
            strategy='best1bin', popsize=8, maxiter=50, disp=True, 
            tol=1e-3, callback=early_stop_callback, workers=-1, updating='deferred', 
            polish=False,
            x0=previous_best_link2
        )
        
        # m2_opt, I2_opt, b2_opt = result_link2.x

        polished_result2 = minimize(
            cost_function_link2, 
            x0=result_link2.x, 
            args=(t_eval, x_target_smooth, full_state_est, u_data, opt_link1),
            method='Nelder-Mead',
            bounds=bounds_link2, # Note: Nelder-Mead in newer Scipy versions supports bounds!
            options={'xatol': 1e-4, 'fatol': 1e-4, 'disp': True, 'maxiter': 50},
        )

        m2_opt, I2_opt, b2_opt = polished_result2.x

        print(f"-> Link 2 Locked: m2={m2_opt:.4f}, I2={I2_opt:.6f}, b2={b2_opt:.4f}")
    else:
        print('link 2 not optimized')
        m2_opt, I2_opt, b2_opt = 0.05, 0.0001, 0.005

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
    
    print("\nSimulating 15-second validation open-loop...")
    
    # Bundle args for final RK4 run
    ode_args_final = (u_data, t_eval[0], dt, optimized_params, M_fast, f_fast)
    
    # Initialize array to hold final simulation data
    num_eval_steps = len(t_eval)
    sol_opt_y = np.zeros((4, num_eval_steps))
    sol_opt_y[:, 0] = x_measured_full[:, 0]
    
    # Run the full validation with RK4
    for k in range(num_eval_steps - 1):
        sol_opt_y[:, k+1] = rk4_step(fast_dynamics, t_eval[k], sol_opt_y[:, k], dt, ode_args_final)


    # ------------------------------------------------
    # PLOTTING
    # ------------------------------------------------
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(t_eval, x_measured_full[0, :], 'k.', label='Measured $\\theta_1$', alpha=0.3)
    plt.plot(t_eval, sol_opt_y[0, :], 'r-', label='Optimized Fit $\\theta_1$', linewidth=2)
    plt.ylabel('Angle (rad)')
    plt.title('System Identification Results: Model vs Real Data')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(t_eval, x_measured_full[1, :], 'k.', label='Measured $\\theta_2$', alpha=0.3)
    plt.plot(t_eval, sol_opt_y[1, :], 'b-', label='Optimized Fit $\\theta_2$', linewidth=2)
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
    plt.plot(t_eval, sol_opt_y[2, :], 'b-', label='Optimized Fit $\\dot{\\theta}_1$', linewidth=2)
    plt.plot(t_eval, sol_opt_y[3, :], 'b-', label='Optimized Fit $\\dot{\\theta}_2$', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()