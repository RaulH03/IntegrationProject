%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DSCS FPGA interface board: init and I/O conversions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% gains and offsets
daoutoffs = [0.00];                   % output offset
daoutgain = 1*[-6];                   % output gain

% Sensor calibration:
adinoffs = -[0 0];
adingain = [1 1];

adinoffs = [adinoffs 0 0 0 0 0];    % input offset
adingain = [adingain 1 1 1 1 1];     % input gain (to radians)

