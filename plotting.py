import matplotlib.pyplot as plt
import numpy as np

def plot_simulation(map_matrix, state_history, goal_pos, resolution=0.1):
    """
    plots the 2D environment, the goal, and the drone's trajectory 
    convert JAX arrays to numpy arrays for matplotlib
    """
    # convert JAX arrays to standard numpy for plotting
    map_matrix = np.array(map_matrix)
    state_history = np.array(state_history)
    
    # extract X and Y coordinates from the state history
    # state_history shape is (T, 4) so [px, py, theta, v]
    path_x = state_history[:, 0]
    path_y = state_history[:, 1]
    
    # create the figure
    plt.figure(figsize=(8, 10))
    
    # plot the map using imshow. 
    # the extent argument scales the matrix indices back to real-world meters.
    max_y_m = map_matrix.shape[0] * resolution
    max_x_m = map_matrix.shape[1] * resolution
    
    plt.imshow(map_matrix, cmap='Greys', origin='lower', 
               extent=[0, max_x_m, 0, max_y_m], alpha=0.5) 
    
    # boundary 
    bounds_x = [0, max_x_m, max_x_m, 0, 0]
    bounds_y = [0, 0, max_y_m, max_y_m, 0]
    plt.plot(bounds_x, bounds_y, color='black', linewidth=3)
    
    # goal
    plt.scatter(goal_pos[0], goal_pos[1], c='red', marker='x', s=100, label='Goal')
    
    # start position
    plt.scatter(path_x[0], path_y[0], c='green', marker='o', s=100, label='Start')
    
    # drone trajectory
    plt.plot(path_x, path_y, c='blue', linewidth=2, label='Trajectory')
    
    # adjustments
    plt.title("2D MPPI Drone Navigation")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.tight_layout()
    
    # Show the plot
    plt.show()