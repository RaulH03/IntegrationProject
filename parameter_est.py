import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

# Import the math engine and simulation function from File 1
from pendulum_function_gen import derive_and_lambdify, fast_dynamics

# Known system constants
KNOWN_PARAMS = {'l1': 0.2, 'l2': 0.1, 'g': 9.81}

def cost_function(guess_array, t_data, x_data_pos, u_data, M_func, f_func):
    """
    Simulates the system with guessed params and returns error on POSITIONS ONLY.
    x_data_pos: A 2xN array containing only [theta1, theta2] measurements.
    """
    m1_g, m2_g, b1_g, b2_g = guess_array
    current_params = {
        'm1': m1_g, 'm2': m2_g, 'b1': b1_g, 'b2': b2_g, 
        **KNOWN_PARAMS
    }
    
    u_func = interp1d(t_data, u_data, bounds_error=False, fill_value="extrapolate")
    
    # --- ESTIMATE INITIAL CONDITIONS ---
    # We have starting positions, but need starting velocities for the solver.
    th1_0, th2_0 = x_data_pos[0, 0], x_data_pos[1, 0]
    
    # Estimate initial velocity (finite difference of first two points)
    dt = t_data[1] - t_data[0]
    dth1_0 = (x_data_pos[0, 1] - x_data_pos[0, 0]) / dt
    dth2_0 = (x_data_pos[1, 1] - x_data_pos[1, 0]) / dt
    
    x0_guess = [th1_0, th2_0, dth1_0, dth2_0]
    
    # --- SIMULATE ---
    sol = solve_ivp(
        fast_dynamics, 
        [t_data[0], t_data[-1]], 
        x0_guess, 
        args=(u_func, current_params, M_func, f_func), 
        t_eval=t_data,
        method='RK45'
    )
    
    if not sol.success:
        return np.ones_like(x_data_pos.flatten()) * 1e6
        
    # --- CALCULATE ERROR ON POSITIONS ONLY ---
    # sol.y is 4xN (pos & vel). x_data_pos is 2xN (pos only).
    # We slice sol.y[0:2, :] to only compare the angles.
    error = sol.y[0:2, :] - x_data_pos
    
    return error.flatten()


if __name__ == "__main__":
    # 1. Load Data
    print("Loading data...")
    # Replace this with your actual hardware data loading logic
    data = np.load('pendulum_data.npz')
    t_eval = data['t']
    u_data = data['u']
    
    # Simulate loading ONLY the position rows from your dataset
    x_measured_full = data['x']
    x_measured_pos = x_measured_full[0:2, :] # Slice out velocities
    
    M_fast, f_fast = derive_and_lambdify()

    print("\nStarting System Identification (Position Only)...")
    initial_guess = [0.05, 0.1, 0.1, 0.1] 
    bounds = ([0.01, 0.01, 0.00, 0.00], [1.00, 1.00, 0.50, 0.50])

    # Notice we pass x_measured_pos instead of the full state array
    result = least_squares(
        cost_function, 
        initial_guess, 
        bounds=bounds,
        args=(t_eval, x_measured_pos, u_data, M_fast, f_fast),
        verbose=2,     
        loss='soft_l1' 
    )

    # 5. Results
    m1_opt, m2_opt, b1_opt, b2_opt = result.x
    print("\n--- IDENTIFICATION RESULTS ---")
    print(f"Optimized m1: {m1_opt:.4f} kg")
    print(f"Optimized m2: {m2_opt:.4f} kg")
    print(f"Optimized b1: {b1_opt:.4f} Nms/rad")
    print(f"Optimized b2: {b2_opt:.4f} Nms/rad")

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