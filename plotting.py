import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

def plot_simulation(map_matrix, state_history, control_history, goal_pos, plot_path, resolution=0.1, arrow_step=10):
    """
    Plots the 2D environment, goal, system's trajectory, and state/control inputs.
    Includes directional arrows to indicate heading.
    """
    # Convert JAX arrays to standard numpy for plotting
    map_matrix = np.array(map_matrix)
    state_history = np.array(state_history)
    control_history = np.array(control_history)
    
    # Extract states
    path_x = state_history[:, 0]
    path_y = state_history[:, 1]
    theta = state_history[:, 2]   # --- NEW: Extract heading (theta) ---
    velocity = state_history[:, 3]
    
    # Extract controls (if the simulation ran for T steps, we have T controls)
    if len(control_history) > 0:
        acceleration = control_history[:, 0]
        angular_vel = control_history[:, 1]
    else:
        # Fallback if simulation exits immediately
        acceleration = np.zeros(len(state_history)-1)
        angular_vel = np.zeros(len(state_history)-1)
        
    # Create the figure and GridSpec layout
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.2, 1]) 
    
    # --- 1. Map Plot (Left Column, spans all 3 rows) ---
    ax_map = fig.add_subplot(gs[:, 0])
    
    max_y_m = map_matrix.shape[0] * resolution
    max_x_m = map_matrix.shape[1] * resolution
    
    ax_map.imshow(map_matrix, cmap='Greys', origin='lower', 
                  extent=[0, max_x_m, 0, max_y_m], alpha=0.5) 
    
    # Boundary 
    bounds_x = [0, max_x_m, max_x_m, 0, 0]
    bounds_y = [0, 0, max_y_m, max_y_m, 0]
    ax_map.plot(bounds_x, bounds_y, color='black', linewidth=3)
    
    # Poses
    ax_map.scatter(goal_pos[0], goal_pos[1], c='red', marker='x', s=100, label='Goal')
    ax_map.scatter(path_x[0], path_y[0], c='green', marker='o', s=100, label='Start')
    ax_map.plot(path_x, path_y, c='blue', linewidth=2, label='Trajectory')
    
    # --- NEW: Heading Arrows (Quiver Plot) ---
    # Calculate directional components (u, v) from theta
    u = np.cos(theta)
    v_dir = np.sin(theta) 
    
    # Plot arrows every 'arrow_step' to prevent visual clutter
    ax_map.quiver(path_x[::arrow_step], path_y[::arrow_step], 
                  u[::arrow_step], v_dir[::arrow_step], 
                  color='red', scale=25, width=0.005, headwidth=4, alpha=0.7, label='Heading')
    
    ax_map.set_title("2D MPPI Autonomous Navigation")
    ax_map.set_xlabel("X Position (m)")
    ax_map.set_ylabel("Y Position (m)")
    ax_map.legend(loc='upper left')
    ax_map.grid(True, linestyle='--', alpha=0.6)
    ax_map.axis('equal')
    
    # --- 2. Velocity Plot (Top Right) ---
    ax_vel = fig.add_subplot(gs[0, 1])
    ax_vel.plot(velocity, c='purple', linewidth=2)
    ax_vel.set_title("Velocity over Time")
    ax_vel.set_ylabel("v (m/s)")
    ax_vel.grid(True, linestyle='--', alpha=0.6)
    
    # --- 3. Acceleration Plot (Middle Right) ---
    ax_acc = fig.add_subplot(gs[1, 1])
    ax_acc.plot(acceleration, c='orange', linewidth=2)
    ax_acc.set_title("Acceleration Input (a)")
    ax_acc.set_ylabel("a (m/s²)")
    ax_acc.grid(True, linestyle='--', alpha=0.6)
    
    # --- 4. Angular Velocity Plot (Bottom Right) ---
    ax_omega = fig.add_subplot(gs[2, 1])
    ax_omega.plot(angular_vel, c='teal', linewidth=2)
    ax_omega.set_title("Angular Velocity Input (ω)")
    ax_omega.set_ylabel("ω (rad/s)")
    ax_omega.set_xlabel("Time Step")
    ax_omega.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(plot_path)
    # plt.show() 


def animate_simulation(ground_truth_map, state_history, control_history, goal_pos, resolution=0.1, save_path="simulation_video.mp4", fps=20):

    ground_truth_map = np.array(ground_truth_map)
    state_history = np.array(state_history)
    control_history = np.array(control_history)
    
    path_x = state_history[:, 0]
    path_y = state_history[:, 1]
    theta = state_history[:, 2]
    velocity = state_history[:, 3]
    
    if len(control_history) > 0:
        acceleration = control_history[:, 0]
        angular_vel = control_history[:, 1]
    else:
        acceleration = np.zeros(len(state_history) - 1)
        angular_vel = np.zeros(len(state_history) - 1)
        
    num_frames = len(state_history)
    
 
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.2, 1]) 
    

    ax_map = fig.add_subplot(gs[:, 0])
    max_y_m = ground_truth_map.shape[0] * resolution
    max_x_m = ground_truth_map.shape[1] * resolution
    

    ax_map.imshow(ground_truth_map, cmap='Greys', origin='lower', 
                  extent=[0, max_x_m, 0, max_y_m], alpha=0.5) 
    
    bounds_x = [0, max_x_m, max_x_m, 0, 0]
    bounds_y = [0, 0, max_y_m, max_y_m, 0]
    ax_map.plot(bounds_x, bounds_y, color='black', linewidth=3)
    ax_map.scatter(goal_pos[0], goal_pos[1], c='red', marker='x', s=100, label='Goal')
    

    start_dot = ax_map.scatter([], [], c='green', marker='o', s=100, label='Start')
    traj_line, = ax_map.plot([], [], c='blue', linewidth=2, label='Trajectory')
    
    ax_map.set_title("2D MPPI Autonomous Navigation (Animation)")
    ax_map.set_xlabel("X Position (m)")
    ax_map.set_ylabel("Y Position (m)")
    ax_map.legend(loc='upper left')
    ax_map.grid(True, linestyle='--', alpha=0.6)
    ax_map.axis('equal')
    

    ax_vel = fig.add_subplot(gs[0, 1])
    vel_line, = ax_vel.plot([], [], c='purple', linewidth=2)
    ax_vel.set_title("Velocity over Time")
    ax_vel.set_ylabel("v (m/s)")
    ax_vel.set_xlim(0, num_frames)
    ax_vel.set_ylim(float(np.min(velocity)) - 0.2, float(np.max(velocity)) + 0.2)
    ax_vel.grid(True, linestyle='--', alpha=0.6)
    
    ax_acc = fig.add_subplot(gs[1, 1])
    acc_line, = ax_acc.plot([], [], c='orange', linewidth=2)
    ax_acc.set_title("Acceleration Input (a)")
    ax_acc.set_ylabel("a (m/s²)")
    ax_acc.set_xlim(0, num_frames)
    if len(acceleration) > 0:
        ax_acc.set_ylim(float(np.min(acceleration)) - 1.0, float(np.max(acceleration)) + 1.0)
    ax_acc.grid(True, linestyle='--', alpha=0.6)
    
    ax_omega = fig.add_subplot(gs[2, 1])
    omega_line, = ax_omega.plot([], [], c='teal', linewidth=2)
    ax_omega.set_title("Angular Velocity Input (ω)")
    ax_omega.set_ylabel("ω (rad/s)")
    ax_omega.set_xlabel("Time Step")
    ax_omega.set_xlim(0, num_frames)
    if len(angular_vel) > 0:
        ax_omega.set_ylim(float(np.min(angular_vel)) - 0.5, float(np.max(angular_vel)) + 0.5)
    ax_omega.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    

    quiver_container = [None]
    
    def init():
        start_dot.set_offsets(np.c_[path_x[0], path_y[0]])
        traj_line.set_data([], [])
        vel_line.set_data([], [])
        acc_line.set_data([], [])
        omega_line.set_data([], [])
        return start_dot, traj_line, vel_line, acc_line, omega_line

    def update(frame):

        traj_line.set_data(path_x[:frame+1], path_y[:frame+1])
        

        if quiver_container[0] is not None:
            quiver_container[0].remove()
            quiver_container[0] = None
            
        if frame > 0:
            arrow_idx = np.arange(0, frame + 1, 10) 
            if len(arrow_idx) > 0:
                u = np.cos(theta[arrow_idx])
                v_dir = np.sin(theta[arrow_idx])
                quiver_container[0] = ax_map.quiver(
                    path_x[arrow_idx], path_y[arrow_idx], u, v_dir,
                    color='red', scale=25, width=0.005, headwidth=4, alpha=0.7
                )
        

        vel_line.set_data(np.arange(frame + 1), velocity[:frame + 1])
        if frame > 0 and len(acceleration) >= frame:
            acc_line.set_data(np.arange(frame), acceleration[:frame])
            omega_line.set_data(np.arange(frame), angular_vel[:frame])
            
        return start_dot, traj_line, vel_line, acc_line, omega_line

    print("Compiling simulation animation frames...")
    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False)
    

    try:
        print(f"Saving video to {save_path} using FFMpegWriter...")
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        ani.save(save_path, writer=writer)
        print("Video rendering complete!")
    except Exception as e:
        print(f"FFMpegWriter unavailable ({e}). Defaulting to Pillow (.gif) animation wrapper...")
        gif_path = save_path.replace(".mp4", ".gif")
        ani.save(gif_path, writer='pillow', fps=fps)
        print(f"Animation saved successfully as: {gif_path}")
        
    plt.close(fig)