% Load the data
dataStruct = load('MPC_up_up_equili_dist.mat');
data = dataStruct.ans';
data = data(150:350, :);

% Extract columns
time         = data(:, 1);
u_input      = data(:, 2);
reference    = data(:, 3);
meas_1       = data(:, 4);
meas_2_abs   = data(:, 5);
est_1        = data(:, 6);
est_2        = data(:, 7);
est_speed_1  = data(:, 8);
est_speed_2  = data(:, 9);

% Create figure
figure('Name', 'Double Pendulum MPC', 'Position', [100, 100, 850, 850]);

% 1. Control Input (Bounded & Saturated)
subplot(4, 1, 1);
hold on;
plot(time, u_input, 'Color', [0.4940 0.1840 0.5560], 'LineWidth', 1.5);
% Add saturation boundary lines
yline(1, 'r--', 'Max (+1)', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
yline(-1, 'r--', 'Min (-1)', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');
hold off;
ylim([-1.2, 1.2]); % Force bounds slightly wider than saturation
ylabel('Input', 'FontWeight', 'bold');
legend('Control Effort', 'Location', 'northeast');
grid on; set(gca, 'XTickLabel', []);

% 2. Link 1 Angle
subplot(4, 1, 2);
hold on;
plot(time, reference, 'k--', 'LineWidth', 1.5);
plot(time, meas_1, 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);
plot(time, est_1, '-.', 'LineWidth', 1.5, 'Color', [0.8500 0.3250 0.0980]);
hold off;
ylim([-1, 1]);
ylabel('Link 1 Angle (rad)', 'FontWeight', 'bold');
legend('Reference', 'Measured', 'Estimated', 'Location', 'northeast');
grid on; set(gca, 'XTickLabel', []);

% 3. Link 2 Angle
subplot(4, 1, 3);
hold on;
plot(time, meas_2_abs, 'LineWidth', 1.5, 'Color', [0.4660 0.6740 0.1880]);
plot(time, est_2, '-.', 'LineWidth', 1.5, 'Color', [0.6350 0.0780 0.1840]);
hold off;
ylim([-0.1, 0.1]);
ylabel('Link 2 Angle (rad)', 'FontWeight', 'bold');
legend('Measured (abs)', 'Estimated', 'Location', 'northeast');
grid on; set(gca, 'XTickLabel', []);

% 4. Angular Velocities
subplot(4, 1, 4);
hold on;
plot(time, est_speed_1, 'LineWidth', 1.5);
plot(time, est_speed_2, 'LineWidth', 1.5);
hold off;
ylim([-2, 2]);
xlabel('Time', 'FontWeight', 'bold');
ylabel('Angular Vel (rad/s)', 'FontWeight', 'bold');
legend('Est Velocity 1', 'Est Velocity 2', 'Location', 'northeast');
grid on;

% Add overall title
sgtitle('Double Pendulum MPC: Up-Up Equilibrium disturbed', 'FontSize', 15, 'FontWeight', 'bold');