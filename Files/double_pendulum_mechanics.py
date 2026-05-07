import numpy as np
import numpy.linalg as lin
import sympy as sm
import sympy.physics.mechanics as me
import scipy.signal as sig

t = me.dynamicsymbols._t

l1, l2 = sm.symbols('l1:3')
m1, m2 = sm.symbols('m1:3')
g, u = sm.symbols('g, u')

N, A1, A2 = sm.symbols('N, A1, A2', cls=me.ReferenceFrame)

th1, th2 = me.dynamicsymbols('theta1, theta2')
th1_dot, th2_dot = th1.diff(t), th2.diff(t)

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

P1 = me.Particle('P1', Pm1, m1)
P2 = me.Particle('P2', Pm2, m2)

P1.potential_energy = m1 * g * r_O_m1.dot(N.y)
P2.potential_energy = m2 * g * r_O_m2.dot(N.y)

L = me.Lagrangian(N, P1, P2)

forces = [(A1, u * N.z)]

LM = me.LagrangesMethod(L, [th1, th2], forcelist=forces, frame=N)
LM.form_lagranges_equations()

M = sm.simplify(LM.mass_matrix)
f = sm.simplify(LM.forcing)

thddot = M.inv() * f 

X = sm.Matrix([th1, th2, th1_dot, th2_dot])
U = sm.Matrix([u])

F = sm.Matrix([
    th1_dot,
    th2_dot,
    thddot[0],
    thddot[1]
])

Ac_sym = F.jacobian(X)
Bc_sym = F.jacobian(U)

# q1=0 (down), q2=pi (up relative to q1), velocities=0, torque=0
eq_dict = {
    th1: 0,
    th2: sm.pi,
    th1_dot: 0,
    th2_dot: 0,
    u: 0
}

# Substitute the equilibrium values and simplify to get the final matrices
Ac = sm.simplify(Ac_sym.subs(eq_dict))
Bc = sm.simplify(Bc_sym.subs(eq_dict))

m1_val, m2_val = 0.1, 0.3
l1_val, l2_val = 0.2, 0.1

params = {
    m1: m1_val,
    m2: m2_val,
    l1: l1_val,
    l2: l2_val,
    g: 9.81
}

# Substitute the numbers into the symbolic continuous matrices
Ac_num = Ac.subs(params)
Bc_num = Bc.subs(params)

# Convert SymPy matrices to standard NumPy float arrays for SciPy
A = np.array(Ac_num).astype(np.float64)
B = np.array(Bc_num).astype(np.float64)

n = A.shape[0]

K = B 

# Loop to stack AB, A^2B, A^3B horizontally
for i in range(1, n):
    term = lin.matrix_power(A, i) @ B
    K = np.hstack((K, term))


eigenvalues = lin.eigvals(A)

# Find the fastest dynamics (maximum distance from the origin in the s-plane)
max_omega = np.max(np.abs(eigenvalues)) # in rad/s
max_freq_hz = max_omega / (2 * np.pi)   # convert to Hz

# Rule of thumb: Sample ~20 times faster than the highest system frequency
f_sample = 20 * max_freq_hz
Ts = 1.0 / f_sample

num_states = A.shape[0]
num_inputs = B.shape[1]

# Full state output and no direct feedthrough
C_dummy = np.eye(num_states)
D_dummy = np.zeros((num_states, num_inputs))

sys_continuous = (A, B, C_dummy, D_dummy)

# ZOH discretization
sys_discrete = sig.cont2discrete(sys_continuous, Ts, method='zoh')

# Extract the discrete matrices
Ad = sys_discrete[0]
Bd = sys_discrete[1]

F_num = F.subs(params)

non_linear_dynamics = sm.lambdify((*X, u), F_num, "numpy")


if __name__ == "__main__":
    print('Numerical A\n', np.round(A, 4))
    print('Numerical B\n', np.round(B, 4))

    print('Controlability matrix\n', K)
    print('Rank controlability matrix\n', lin.matrix_rank(K))

    print('Discrete A\n', np.round(Ad, 4))
    print('Discrete B\n', np.round(Bd, 4))

    sm.pprint(f)
    sm.pprint(M)




