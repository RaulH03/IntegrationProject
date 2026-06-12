clear; clc;
h = 0.02;
[A, B, C, D] = Dynamics(1); 
sysc = ss(A, B, C, D);
sysd = c2d(sysc, h);
Ad = sysd.A;
Bd = sysd.B;
n_states = 4;
n_inputs = 1;
N = 35;
Q = diag([1/(deg2rad(45)^2), 1/(deg2rad(5)^2), 1/(0.2^2), 1/(0.2^2)]);
R = 1;

% Kalman weights 
R1 = diag([1e-4,1e-4,5e-1,5e-1]);
R2 = diag([0.018039541207762, 0.017742484041180]);

[K, P, CLP] = dlqr(Ad, Bd, Q, R);


u_max = 1.0;
theta1_max = deg2rad(90);
theta2_max = deg2rad(90);


[H_f, h_f] = compute_terminal_polyhedral_double_pendulum(Ad, Bd, K, theta1_max, theta2_max, u_max);


Q_bar = blkdiag(kron(eye(N-1), Q), P);
R_bar = kron(eye(N), R);

Sx = zeros(n_states * N, n_states);
Su = zeros(n_states * N, n_inputs * N);

for i = 1:N
    Sx((i-1)*n_states+1 : i*n_states, :) = Ad^i;
    for j = 1:i
        Su((i-1)*n_states+1 : i*n_states, (j-1)*n_inputs+1 : j*n_inputs) = (Ad^(i-j)) * Bd;
    end
end

H_qp = 2*(Su' * Q_bar * Su + R_bar);
% H_qp = (H_qp + H_qp') / 2; % Ensure perfect symmetry to prevent solver warnings
F_qp = 2 * Su' * Q_bar * Sx; % f will be calculated in real-time as: f = F_qp * x_curr

A_stage = [ 1  0  0  0; 
           -1  0  0  0; 
            0  1  0  0; 
            0 -1  0  0];
b_stage = [theta1_max; theta1_max; theta2_max; theta2_max];

A_block = blkdiag(kron(eye(N-1), A_stage), H_f);
b_block = [repmat(b_stage, N-1, 1); h_f];

% Substitute X = Sx*x_curr + Su*U into constraints: A_block * (Sx*x + Su*U) <= b_block
% Rearranged for U: (A_block * Su) * U <= b_block - (A_block * Sx) * x_curr
A_ineq = A_block * Su;
b_ineq_base = b_block;
S_ineq = A_block * Sx; % We subtract (S_ineq * x_curr) from b_ineq_base in real-time

lb = -u_max * ones(N, 1);
ub =  u_max * ones(N, 1);
x0 = zeros(N, 1);


function [H_set, h_set] = compute_terminal_polyhedral_double_pendulum(Ad, Bd, K, theta1_max, theta2_max, u_max)
    A_cl = Ad - Bd*K;
    tol = 1e-6; 
    
    F = [1  0  0  0
        -1  0  0  0
         0  1  0  0
         0 -1  0  0
         0  0  1  0
         0  0 -1  0
         0  0  0  1
         0  0  0 -1];

    F = [F
         K
        -K];

    theta1_dot_max = 100;
    theta2_dot_max = 100;


    g = [theta1_max
         theta1_max
         theta2_max
         theta2_max
         theta1_dot_max
         theta1_dot_max
         theta2_dot_max
         theta2_dot_max
         u_max
         u_max];
         
    H_set = F;
    h_set = g;
    
    for i = 1:100
        H_new = F*(A_cl^i);
        added_any = false;

        for rowid = 1:size(H_new, 1)
            c = H_new(rowid, :)';
            b_val = g(rowid);
            f = -c;
            [~, fval, exitflag, ~] = linprog(f, H_set, h_set);
            
            if exitflag == 1 
                max_val = -fval;
                if max_val > b_val + tol
                    H_set = [H_set; c'];
                    h_set = [h_set; b_val];
                    added_any = true;
                end
            else
                error('LP failed while checking redundancy at iteration %d, row %d', i, rowid); 
            end
        end
        
        if ~added_any
            fprintf('Polyhedral terminal set found after %d iterations with %d inequalities.\n', i, size(H_set, 1));
            break;
        end
    end
end