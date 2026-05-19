import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from scipy.signal import cont2discrete

# ==========================================
# GLOBAL AUTO-DERIVATION CACHE
# ==========================================
_A_func = None
_B_func = None
_P_funcs = None

def _auto_derive_math():
    """Uses SymPy to automatically derive lumped mappings and state-space matrices."""
    global _A_func, _B_func, _P_funcs
    print("Auto-deriving physics and state-space matrices via SymPy (runs once)...")
    
    t = me.dynamicsymbols._t
    l1, l2, I1, I2, m1, m2, b1, b2, g, u, Kt_sym = sm.symbols('l1 l2 I1 I2 m1 m2 b1 b2 g u Kt', real=True, positive=True)
    th1, th2 = me.dynamicsymbols('theta1 theta2')
    th1_d, th2_d = th1.diff(t), th2.diff(t)

    N, A1, A2 = me.ReferenceFrame('N'), me.ReferenceFrame('A1'), me.ReferenceFrame('A2')
    A1.orient_axis(N, th1, N.z); A2.orient_axis(A1, th2, A1.z)
    O = me.Point('0'); O.set_vel(N, 0)

    r_m1 = -l1 * A1.y
    r_o1, r_o2 = r_m1 / 2, r_m1 - (l2 * A2.y) / 2
    Po1, Po2 = me.Point('Po1'), me.Point('Po2')
    Po1.set_pos(O, r_o1); Po2.set_pos(O, r_o2)
    Po1.set_vel(N, r_o1.dt(N)); Po2.set_vel(N, r_o2.dt(N))

    B1 = me.RigidBody('B1', Po1, A1, m1, (me.inertia(A1, 0, 0, I1), Po1))
    B2 = me.RigidBody('B2', Po2, A2, m2, (me.inertia(A2, 0, 0, I2), Po2))
    B1.potential_energy = m1 * g * r_o1.dot(N.y)
    B2.potential_energy = m2 * g * r_o2.dot(N.y)

    L = me.Lagrangian(N, B1, B2)
    
    # Your correct physics: Positive u creates negative torque
    forces = [(A1, (-Kt_sym*u - b1*th1_d + b2*th2_d) * N.z), (A2, -b2*th2_d * N.z)]

    LM = me.LagrangesMethod(L, [th1, th2], forcelist=forces, frame=N)
    LM.form_lagranges_equations()

    f_sym = LM.forcing
    K_sym = -f_sym.jacobian(sm.Matrix([th1, th2]))
    D_sym = -f_sym.jacobian(sm.Matrix([th1_d, th2_d]))
    
    # --- NEW: Automatically extract the true B vector from the physics ---
    B_sym_full = f_sym.jacobian(sm.Matrix([u]))

    # Evaluate at Down-Down equilibrium
    eq_dict = {th1: 0, th2: 0, th1_d: 0, th2_d: 0, u: 0}
    M_eq = sm.simplify(LM.mass_matrix.subs(eq_dict))
    K_eq = sm.simplify(K_sym.subs(eq_dict))
    D_eq = sm.simplify(D_sym.subs(eq_dict))
    B_eq = sm.simplify(B_sym_full.subs(eq_dict)) # This perfectly evaluates to [[-Kt_sym], [0]]

    # Extract lumped groupings
    P_exprs = [M_eq[0, 0], M_eq[0, 1], M_eq[1, 1], K_eq[0, 0], K_eq[0, 1], D_eq[0, 0], D_eq[1, 1]]
    phys_syms = (m1, m2, I1, I2, l1, l2, b1, b2, g)
    _P_funcs = [sm.lambdify(phys_syms, expr, 'numpy') for expr in P_exprs]

    # Convert to explicit A and B matrices using Adjugate
    P1, P2, P3, P4, P5, P6, P7 = sm.symbols('P1 P2 P3 P4 P5 P6 P7', real=True)
    M_lump = sm.Matrix([[P1, P2], [P2, P3]])
    K_lump = sm.Matrix([[P4, P5], [P5, P5]])
    D_lump = sm.Matrix([[P6, 0],  [0, P7]])

    det_sym = P1 * P3 - P2**2
    M_inv = M_lump.adjugate() / det_sym
    I_mat, Z_mat = sm.eye(2), sm.zeros(2, 2)

    A_sym = sm.Matrix.vstack(sm.Matrix.hstack(Z_mat, I_mat), sm.Matrix.hstack(-M_inv*K_lump, -M_inv*D_lump))
    
    # --- NEW: Use the true SymPy derivation for B ---
    B_sym = sm.Matrix.vstack(Z_mat[:, 0:1], M_inv * B_eq) 

    _A_func = sm.lambdify((P1, P2, P3, P4, P5, P6, P7), A_sym, 'numpy')
    _B_func = sm.lambdify((P1, P2, P3, Kt_sym), B_sym, 'numpy')

# ==========================================
# TRANSLATION & MATRICES
# ==========================================
def get_lumped_bounds(p_guess, p_min, p_max):
    if _P_funcs is None: _auto_derive_math()
    
    keys = ['m1', 'm2', 'I1', 'I2', 'l1', 'l2', 'b1', 'b2', 'g']
    P_guess = [float(f(*[p_guess[k] for k in keys])) for f in _P_funcs] + [p_guess['dry1'], p_guess['dry2']]
    P_lower = [float(f(*[p_min[k] for k in keys])) for f in _P_funcs] + [p_min['dry1'], p_min['dry2']]
    P_upper = [float(f(*[p_max[k] for k in keys])) for f in _P_funcs] + [p_max['dry1'], p_max['dry2']]
    return np.array(P_guess), (P_lower, P_upper)

def build_matrices(p, Kt, eq='Down-Down'):
    if _A_func is None: _auto_derive_math()
    
    P1, P2, P3, P4, P5, P6, P7 = p[:7]
    if eq == 'Up-Up':
        P4, P5 = -P4, -P5  # Gravity flips for inverted equilibrium

    det = P1 * P3 - P2**2
    if det < 1e-6: return None, None
    return _A_func(P1, P2, P3, P4, P5, P6, P7), _B_func(P1, P2, P3, Kt)

# ==========================================
# INPUT AUGMENTATION & SIMULATION
# ==========================================
def augment_input(u_raw, dq1_meas, p, Kt, delay_steps=1):
    dq1_delayed = np.roll(dq1_meas, delay_steps)
    dq1_delayed[:delay_steps] = 0.0 
    vel_sign = np.tanh(10.0 * dq1_delayed)
    fric_val = (p[7] * ((1 + vel_sign) / 2.0) + p[8] * ((1 - vel_sign) / 2.0)) * vel_sign
    return u_raw - (fric_val / Kt)

def simulate_n_step(p, u_data, x_meas, dt, Kt, n_steps=100, delay_steps=1):
    A, B = build_matrices(p, Kt) 
    if A is None: return np.inf * np.ones((4, len(u_data)))
        
    Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(4), np.zeros((4,1))), dt, method='zoh')
    u_aug = augment_input(u_data, x_meas[2, :], p, Kt, delay_steps)
    
    x_sim = np.zeros_like(x_meas)
    x_sim[:, 0] = x_meas[:, 0]
    Bd_flat = Bd.flatten()
    
    for k in range(len(u_data) - 1):
        current_state = x_meas[:, k] if k % n_steps == 0 else x_sim[:, k]
        x_sim[:, k+1] = Ad @ current_state + Bd_flat * u_aug[k]
    return x_sim

def simulate_open_loop(p, u_data, x_meas, dt, Kt, delay_steps=1):
    A, B = build_matrices(p, Kt) 
    if A is None: return np.inf * np.ones((4, len(u_data)))
        
    Ad, Bd, _, _, _ = cont2discrete((A, B, np.eye(4), np.zeros((4,1))), dt, method='zoh')
    u_aug = augment_input(u_data, x_meas[2, :], p, Kt, delay_steps)
    
    x_sim = np.zeros((4, len(u_data)))
    x_sim[:, 0] = x_meas[:, 0]
    Bd_flat = Bd.flatten()
    
    for k in range(len(u_data) - 1):
        x_sim[:, k+1] = Ad @ x_sim[:, k] + Bd_flat * u_aug[k]
    return x_sim