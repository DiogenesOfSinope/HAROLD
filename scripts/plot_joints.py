import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    # 1. Load the logged data
    file_path = "/home/c/Documents/harold/sim/logs/rsl_rl/leg/2026-04-04_17-32-03/joint_data_log.npz"
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Make sure you are in the correct directory.")
        return

    joint_pos = data['joint_pos']  # Shape: (num_steps, num_envs, num_joints)
    joint_vel = data['joint_vel']  # Shape: (num_steps, num_envs, num_joints)

    # 2. Select a single environment to plot (Environment 0)
    env_idx = 0
    pos_env0 = joint_pos[:, env_idx, :]
    vel_env0 = joint_vel[:, env_idx, :]

    num_steps = pos_env0.shape[0]
    num_joints = pos_env0.shape[1]

    # Create an array for the x-axis (time steps)
    time_steps = np.arange(num_steps)

    # 3. Setup the figure and subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Plot Joint Positions ---
    for j in range(num_joints):
        axs[0].plot(time_steps, pos_env0[:, j], label=f'Joint {j}')
    
    axs[0].set_title('Robot Joint Positions over Time (Env 0)')
    axs[0].set_ylabel('Position (rad)')
    axs[0].grid(True, linestyle='--', alpha=0.7)
    # Put legend outside the plot
    axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # --- Plot Joint Velocities ---
    for j in range(num_joints):
        axs[1].plot(time_steps, vel_env0[:, j], label=f'Joint {j}')
        
    axs[1].set_title('Robot Joint Velocities over Time (Env 0)')
    axs[1].set_xlabel('Simulation Steps')
    axs[1].set_ylabel('Velocity (rad/s)')
    axs[1].grid(True, linestyle='--', alpha=0.7)
    # Put legend outside the plot
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Adjust layout to prevent clipping of the external legend
    plt.tight_layout()
    
    # 4. Display the plot
    plt.savefig("joint_trajectories.png", dpi=300, bbox_inches='tight') # <-- ADD THIS
    print("Plot saved successfully to 'joint_trajectories.png'")

if __name__ == "__main__":
    main()