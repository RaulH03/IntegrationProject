function u_opt = double_pendulum_mpc_qp(x_curr, H_qp, F_qp, A_ineq, b_ineq_base, S_ineq, lb, ub, x0)
    
    f = F_qp * x_curr;

    b_ineq = b_ineq_base - S_ineq * x_curr;
    
    options = optimoptions('quadprog', 'Display', 'none', 'Algorithm', 'active-set');
    
    % quadprog signature: x = quadprog(H, f, A, b, Aeq, beq, lb, ub, x0, options)
    [U_opt, ~, exitflag, ~] = quadprog(H_qp, f, A_ineq, b_ineq, [], [], lb, ub, x0, options);

    if exitflag <= 0
        u_opt = 0; % Fallback if solver fails (e.g., infeasible)
    else
        u_opt = U_opt(1); % Apply only the first optimal control action
%         x0 = [U_opt(2:end); 0];
    end
end