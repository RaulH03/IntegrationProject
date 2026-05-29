[A, B, C, D] = Dynamics(a = 1);
sysc = ss(A, B, C, D);
sysd = c2d(ss(A,B,C,D),h);

n_states = 4;
n_inputs = 1;
N = 20;

Q = diag([5e2, 1e1, 1e1, 1e0]);
R = 5e2;

[P, ~, K] = idare(Ad, Bd, Q, R);

u_max = 2.0;
theta1_max = deg2rad(90);
theta2_dev_max = deg2rad(90);

alpha = compute_terminal_alpha_double_pendulum(P, K, theta1_max, theta2_dev_max, u_max);
fprintf('Calculated terminal alpha: %.4f\n', alpha);

function alpha_min = compute_terminal_alpha_double_pendulum(P, K, theta1_max, theta2_dev_max, u_max)
    P_inv = inv(P);
    c1 = [1; 0; 0; 0];
    c2 = [0; 1; 0; 0]; 
    c3 = -K';                  

    b1 = theta1_max;
    b2 = theta2_dev_max;
    b3 = u_max;

    alpha1 = (b1^2) / (c1' * P_inv * c1);
    alpha2 = (b2^2) / (c2' * P_inv * c2);
    alpha3 = (b3^2) / (c3' * P_inv * c3);

    alpha_min = min([alpha1, alpha2, alpha3]);
end