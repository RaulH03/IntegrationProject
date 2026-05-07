import numpy as np
import matplotlib.pyplot as plt
import signal
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

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
    
    dt = t_data[1] - t_data[0]
    dth1_0 = (x_data_pos[0, 1] - x_data_pos[0, 0]) / dt
    dth2_0 = (x_data_pos[1, 1] - x_data_pos[1, 0]) / dt
    x0_guess = [x_data_pos[0, 0], x_data_pos[1, 0], dth1_0, dth2_0]
    
    sol = solve_ivp(
        fast_dynamics, 
        [t_data[0], t_data[-1]], 
        x0_guess, 
        args=(u_func, current_params, M_fast, f_fast), # Uses globals
        t_eval=t_data,
        method='RK45'
    )
    
    if not sol.success:
        return 1e6 
        
    raw_error = sol.y[0:2, :] - x_data_pos
    wrapped_error = (raw_error + np.pi) % (2 * np.pi) - np.pi
    mse = np.mean(wrapped_error**2)
    
    return mse

if __name__ == "__main__":
    print("Loading data...")
    data = np.load('pendulum_data.npz')
    t_eval = data['t']
    u_data = data['u']
    
    x_measured_full = data['x']
    x_measured_pos = x_measured_full[0:2, :] 

    print("\nStarting Parallel System Identification...")
    print("Press Ctrl+C at any time to stop early and view the best parameters.")
    
    bounds = [(0.01, 1.0), (0.01, 1.0), (0.00, 0.5), (0.00, 0.5)]

    # Run the evolutionary algorithm
    result = differential_evolution(
        cost_function_global, 
        bounds=bounds,
        args=(t_eval, x_measured_pos, u_data),
        strategy='best1bin', 
        popsize=15,          
        maxiter=100,         
        disp=True,           
        tol=1e-3,
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
    plt.subplot(2, 1, 1)
    plt.plot(t_eval, x_measured_full[0, :], 'k.', label='Measured $\\theta_1$', alpha=0.3)
    plt.plot(t_eval, sol_opt.y[0, :], 'r-', label='Optimized Fit $\\theta_1$', linewidth=2)
    plt.ylabel('Angle (rad)')
    plt.title('System Identification Results: Model vs Real Data')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t_eval, x_measured_full[1, :], 'k.', label='Measured $\\theta_2$', alpha=0.3)
    plt.plot(t_eval, sol_opt.y[1, :], 'b-', label='Optimized Fit $\\theta_2$', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()