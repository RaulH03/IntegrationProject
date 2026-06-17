For LQR control: matlab file to run is "controllers.m". Simulink file to run is "rotpen_control". to change between reference tracking and holding the equilibrium change variable "y_input" in file "controllers.m"

For MPC control: matlab file is "MPC_poly.m". Simulink file is "rotpen_MPC_poly_2021"

The coordinate system and the equilibrium point is transformed by the function in file "rescale"

The function for calling up the dynamics for a certain equilibrium point can be found in matlab file "Dynamics.m"

The parameter identification is done in the folder "linear_Identification"