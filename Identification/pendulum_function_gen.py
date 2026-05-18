import numpy as np
import numpy.linalg as lin
import sympy as sm
import sympy.physics.mechanics as me
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Define SymPy symbols globally so they can be referenced
t = me.dynamicsymbols._t
l1, l2 = sm.symbols('l1 l2')
I1, I2 = sm.symbols('I1 I2')
m1, m2 = sm.symbols('m1 m2')
b1, b2 = sm.symbols('b1 b2')
c1 = sm.symbols('c1')          
Kt = sm.symbols('Kt')
# bias1 = sm.symbols('bias1')
g, u = sm.symbols('g u')
th1, th2 = me.dynamicsymbols('theta1 theta2')
th1_dot, th2_dot = th1.diff(t), th2.diff(t)

def derive_and_lambdify():
    """
    Derives the non-linear dynamics using SymPy and returns fast, 
    numerical NumPy functions for the Mass matrix (M) and forcing vector (f).
    """
    print("Deriving SymPy equations (this takes a few seconds)...")
    N = me.ReferenceFrame('N')
    A1 = me.ReferenceFrame('A1')
    A2 = me.ReferenceFrame('A2')

    A1.orient_axis(N, th1, N.z)
    A2.orient_axis(A1, th2, A1.z)

    O = me.Point('0')
    O.set_vel(N, 0)

    r_O_m1 = -l1*A1.y
    r_O_m2 = r_O_m1 - l2*A2.y

    Pm1 = me.Point('Pm1')
    Pm1.set_pos(O, r_O_m1)
    Pm2 = me.Point('Pm2')
    Pm2.set_pos(O, r_O_m2)
    
    Pm1.set_vel(N, r_O_m1.dt(N))
    Pm2.set_vel(N, r_O_m2.dt(N))

    r_O_o1 = -l1*A1.y/2
    r_O_o2 = r_O_m1 - l2*A2.y/2

    Po1 = me.Point('Po1')
    Po1.set_pos(O, r_O_o1)
    Po2 = me.Point('Po2')
    Po2.set_pos(O, r_O_o2)

    Po1.set_vel(N, r_O_o1.dt(N))
    Po2.set_vel(N, r_O_o2.dt(N))

    Inertia_1 = (me.inertia(A1, 0, 0, I1), Po1)
    Inertia_2 = (me.inertia(A2, 0, 0, I2), Po2)

    B1 = me.RigidBody('B1', Po1, A1, m1, Inertia_1)
    B2 = me.RigidBody('B2', Po2, A2, m2, Inertia_2)

    B1.potential_energy = m1 * g * r_O_o1.dot(N.y)
    B2.potential_energy = m2 * g * r_O_o2.dot(N.y)

    # P1 = me.Particle('P1', Pm1, m1)
    # P2 = me.Particle('P2', Pm2, m2)

    # P1.potential_energy = m1 * g * r_O_m1.dot(N.y)
    # P2.potential_energy = m2 * g * r_O_m2.dot(N.y)

    L = me.Lagrangian(N, B1, B2)

    # Corrected friction and reaction forces
    k_smooth = 5.0
    fric_tau1 = b1 * th1_dot + c1 * th1_dot * sm.tanh(k_smooth * th1_dot)
    fric_tau2 = b2 * th2_dot

    forces = [
        (A1, (Kt*u - fric_tau1 + fric_tau2) * N.z),
        (A2, -fric_tau2 * N.z)
    ]

    LM = me.LagrangesMethod(L, [th1, th2], forcelist=forces, frame=N)
    LM.form_lagranges_equations()

    M = LM.mass_matrix
    f = LM.forcing

    # 1. Create plain, static symbols
    q1, q2, dq1, dq2 = sm.symbols('q1 q2 dq1 dq2')
    
    # 2. Swap the complex time-derivatives for the static symbols
    subs_dict = {th1: q1, th2: q2, th1_dot: dq1, th2_dot: dq2}
    M_clean = M.subs(subs_dict)
    f_clean = f.subs(subs_dict)

    # 3. Lambdify using the clean static symbols (cse=True will now work perfectly!)
    M_func = sm.lambdify((q1, q2, m1, m2, I1, I2, l1, l2, b1, b2, c1, Kt, g), M_clean, "numpy", cse=True)
    f_func = sm.lambdify((q1, q2, dq1, dq2, u, m1, m2, I1, I2, l1, l2, b1, b2, c1, Kt, g), f_clean, "numpy", cse=True)
    
    print("Derivation complete.")
    return M_func, f_func


def fast_dynamics(t, state, u_func, p, M_func, f_func):
    """
    The numerical function used by SciPy's solve_ivp to integrate the system.
    """
    q1, q2, dq1, dq2 = state
    u_val = u_func(t)
    
    # Evaluate M and f using the lambdified functions
    M_val = M_func(q1, q2, p['m1'], p['m2'], p['I1'], p['I2'], p['l1'], p['l2'], p['b1'], p['b2'], p['c1'], p['Kt'], p['g'])
    f_val = f_func(q1, q2, dq1, dq2, u_val, p['m1'], p['m2'], p['I1'], p['I2'], p['l1'], p['l2'], p['b1'], p['b2'], p['c1'], p['Kt'], p['g'])

    # Solve M * q_ddot = f for q_ddot
    q_ddot = lin.solve(M_val, f_val).flatten()
    
    return [dq1, dq2, q_ddot[0], q_ddot[1]]


if __name__ == "__main__":
    # ---------------------------------------------------------
    # DATA GENERATION SCRIPT (Run this file directly to generate data)
    # ---------------------------------------------------------
    M_fast, f_fast = derive_and_lambdify()
    
    # True parameters of the "Hardware"
    true_params = {
        'm1': 0.5, 'm2': 0.1, 
        'I1': 0.01, 'I2': 0.005,  # NEW: Added Inertias
        'b1': 0.7, 'b2': 0.015, 
        'c1': 0.05,               # NEW: Added Asymmetric friction
        'Kt': 1.0,                # NEW: Added Motor Torque constant
        'l1': 0.15, 'l2': 0.15, 'g': 9.81
    }

    # Create a rich excitation signal (chirp-like behavior)
    t_eval = np.linspace(0, 5, 1000) # 5 seconds at 100Hz
    # u_data = 1.2 * np.sin(2 * np.pi * 0.5 * t_eval) + 0.8 * np.sin(2 * np.pi * 2.3 * t_eval)
    u_data = 2 * np.sin(t_eval**2) 
    u_func_real = interp1d(t_eval, u_data, bounds_error=False, fill_value="extrapolate")
    
    print("Simulating real hardware to collect data...")
    x0 = [0.0, 0.0, 0.0, 0.0]
    sol = solve_ivp(
        fast_dynamics, 
        [t_eval[0], t_eval[-1]], 
        x0, 
        args=(u_func_real, true_params, M_fast, f_fast), 
        t_eval=t_eval,
        method='RK45'
    )
    
    # Add realistic sensor noise (e.g., encoder quantization/noise)
    noise_std = 0.05 
    x_measured = sol.y + np.random.normal(0, noise_std, sol.y.shape)
    
    # Save the input/output data to a file for File 2 to load
    filename = 'pendulum_data.npz'
    np.savez(filename, t=t_eval, u=u_data, x=x_measured)
    print(f"Data successfully saved to {filename}")