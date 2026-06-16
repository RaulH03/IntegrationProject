% Load the data
dataStruct = load('MPC_up_up_equili.mat');
data = dataStruct.ans';

% Extract columns based on labels
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
figure('Name', 'MPC Performance', 'Position', [100, 100, 800, 800]);

% 1. Control Input
subplot(4, 1, 1);
plot(time, u_input, 'Color', [0.4940 0.1840 0.5560], 'LineWidth', 1.5);
ylabel('Control Input', 'FontWeight', 'bold');
legend('Input', 'Location', 'best');
grid on; set(gca, 'XTickLabel', []);

% 2. State 1: Reference vs Measured vs Estimated
subplot(4, 1, 2);
hold on;
plot(time, reference, 'k--', 'LineWidth', 1.5);
plot(time, meas_1, 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);
plot(time, est_1, '-.', 'LineWidth', 1.5, 'Color', [0.8500 0.3250 0.0980]);
hold off;
ylabel('State 1', 'FontWeight', 'bold');
legend('Reference', 'Measured 1', 'Estimated 1', 'Location', 'best');
grid on; set(gca, 'XTickLabel', []);

% 3. State 2: Measured vs Estimated
subplot(4, 1, 3);
hold on;
plot(time, meas_2_abs, 'LineWidth', 1.5, 'Color', [0.4660 0.6740 0.1880]);
plot(time, est_2, '-.', 'LineWidth', 1.5, 'Color', [0.6350 0.0780 0.1840]);
hold off;
ylabel('State 2', 'FontWeight', 'bold');
legend('Measured 2 (abs)', 'Estimated 2', 'Location', 'best');
grid on; set(gca, 'XTickLabel', []);

% 4. Speeds
subplot(4, 1, 4);
hold on;
plot(time, est_speed_1, 'LineWidth', 1.5);
plot(time, est_speed_2, 'LineWidth', 1.5);
hold off;
xlabel('Time', 'FontWeight', 'bold');
ylabel('Est Speeds', 'FontWeight', 'bold');
legend('Est Speed 1', 'Est Speed 2', 'Location', 'best');
grid on;

% Add overall title (R2018b or later)
sgtitle('MPC Performance & Estimation Results', 'FontSize', 14, 'FontWeight', 'bold');