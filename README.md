# Sym_robot_arm
lagrangian_dynamics_env: Custom 2D reacher env. Unlike fixed target in openai Gym Reacher, the target is moving in certain trajectory. Lagrange euler method used for dynamics.
trajectory_generator: Create target trajectory in cartesian coordinate. Code mostly from https://github.com/rparak/2-Link_Manipulator by MIT.
                      'coorinates.csv' file is the output.
demonstration_generator: Calculate needed joint angle, joint veloctiy, joint torque to reach target trajectory. This data is used to create demonstration data for DDPG.
                         'stateaction.csv' file is the output.
twolink_DDPG_trial: DDPG with demonstration. PER (Prioritized Experience Replay) not used, just ERB (Experience Replay Buffer).

