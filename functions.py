import jax 
import jax.numpy as jnp 

def initialize_maps(width_m : float = 3.0, length_m : float = 10.0, resolution : float = 0.1): 
    """
    creates the 2d environment based on the input parameters
    """ 
    # calculating the grid dimensions based on the resolution 
    rows = int(length_m / resolution) 
    cols = int(width_m / resolution) 

    # ground truth (real) map 
    # using jnp.int32 because states are discrete (-1,0,1) 
    # 0 : free space, -1 : unkown, 1 : obstacle 
    # we initialize the map all with zeros (free) and then add obstacles 
    ground_truth_map = jnp.zeros((rows, cols), dtype=jnp.int32) 

    # adding an obstacle in the middle just to test our setup 
    # using .at[...].set(...) syntax to return a NEW array with the updates values 
    # vertical wall in the middle 
    ground_truth_map = ground_truth_map.at[47:52, 0:20].set(1)
    ground_truth_map = ground_truth_map.at[20:25, 10:30].set(1) 
    ground_truth_map = ground_truth_map.at[68:72, 0:12].set(1) 
    ground_truth_map = ground_truth_map.at[68:72, 15:30].set(1) 
    ground_truth_map = ground_truth_map.at[35:38, 5:25].set(1)
    ground_truth_map = ground_truth_map.at[58:61, 5:25].set(1)

    # belief map 
    # initialize all with -1 (unkown) 
    belief_map = jnp.full((rows, cols), -1, dtype=jnp.int32) 

    return ground_truth_map, belief_map 

# The @jax.jit decorator compiles this function down to optimized machine code (XLA).
# It makes this function incredibly fast, which we need for MPPI. 
@jax.jit 
def update_belief_map(belief_map, ground_truth_map, drone_row, drone_col, radius_cells): 
    """
    simulates the drone sensor updating its internal map 
    using vector operations instead of for loops
    """ 
    # getting shape of our belief map 
    rows, cols = belief_map.shape 

    # creating 1d arrays of the row and column indices 
    r_idx = jnp.arange(rows) 
    c_idx = jnp.arange(cols) 

    # creating 2d coordinate grids 
    # R[i, j] will be i, and C[i, j] will be j for every cell in the map (to be used later for broadcasting) 
    R, C = jnp.meshgrid(r_idx, c_idx, indexing='ij') 

    # calculating the squared distance from drone to all cells 
    # square distance to save computation time 
    dist_sq = (R - drone_row)**2 + (C - drone_col)**2 

    # boolean checking 
    in_sensor_range = dist_sq <= (radius_cells**2) 

    # using jnp.where(condition, x, y) 
    # where in_sensor_range is true, pick from ground_truth_map 
    # otherwise keep existing 
    new_belief_map = jnp.where(in_sensor_range, ground_truth_map, belief_map)  

    return new_belief_map  

@jax.jit 
def update_belief_map_limitedfov(belief_map, ground_truth_map, drone_row, drone_col, drone_theta, radius_cells, fov_rad): 
    """
    simulates the drone sensor updating its internal map based on the fov and its heading
    using vector operations instead of for loops
    """ 
    # getting shape of our belief map 
    rows, cols = belief_map.shape 

    # creating 1d arrays of the row and column indices 
    r_idx = jnp.arange(rows) 
    c_idx = jnp.arange(cols) 

    # creating 2d coordinate grids 
    # R[i, j] will be i, and C[i, j] will be j for every cell in the map (to be used later for broadcasting) 
    R, C = jnp.meshgrid(r_idx, c_idx, indexing='ij') 

    # calculating the squared distance from drone to all cells 
    # square distance to save computation time 
    dist_sq = (R - drone_row)**2 + (C - drone_col)**2 

    # boolean checking 
    in_sensor_range = dist_sq <= (radius_cells**2)  

    # fov limited update 
    # calculating the dx and dy from the drone to all the cell in the grid 
    # C corresponds to x (columns) and R corresponds to y (rows) 
    dx = C - drone_col 
    dy = R - drone_row 

    # calculating the absolute angle to each cell 
    cell_angles = jnp.arctan2(dy, dx) 

    # difference between the angles 
    angle_diff = cell_angles - drone_theta 

    # normalization between [-pi, pi] 
    angle_diff = jnp.arctan2(jnp.sin(angle_diff), jnp.cos(angle_diff)) 

    # checking fov (half the limit)
    in_fov = jnp.abs(angle_diff) <= (fov_rad / 2.0) 

    # current cell is always visible (arctan(0,0) edge case) 
    is_drone_cell = dist_sq == 0 

    # combine distance, FOV, and drone cell conditions
    is_visible = (in_sensor_range & in_fov) | is_drone_cell 

    # using jnp.where(condition, x, y) 
    # where in_sensor_range is true, pick from ground_truth_map 
    # otherwise keep existing 
    new_belief_map = jnp.where(is_visible, ground_truth_map, belief_map) 

    return new_belief_map 

@jax.jit 
def dynamics_step(state, control, dt, max_speed) : 
    """
    update the drone state using unicycle kinematics via euler integration 
    can handle a single state or a batch of mppi states
    inputs : array of shape (..., 4) that has [px, py, theta, v] 
    control : array of shape (..., 2) that has [a, omega] 
    dt : time step in seconds
    returns the next state with the same shape 
    """

    # unpack the state 
    px = state[..., 0] 
    py = state[..., 1] 
    theta = state[..., 2] 
    v = state[..., 3]  

    # unpacking the controls 
    a = control[..., 0] 
    omega = control[..., 1] 

    # hw limits 
    max_a = 15
    max_omega = jnp.pi
    a = jnp.clip(a, -max_a, max_a)
    omega = jnp.clip(omega, -max_omega, max_omega)  

    # euler integration (simple) 
    # using jnp.sin and jnp.cos for jit compatibility 
    next_px = px + v * jnp.cos(theta) * dt 
    next_py = py + v * jnp.sin(theta) * dt 

    next_theta = theta + omega * dt 
    next_v = v + a * dt 

    # clipping the max drone velocity 
    next_v = jnp.clip(next_v, -max_speed, max_speed) 

    # repacking the state 
    next_state = jnp.stack([next_px, next_py, next_theta, next_v], axis=-1) 

    return next_state 

def single_trajectory_rollout(state_init, control_sequence, dt, max_speed): 
    """
    simulate a single trajectory over the control horizon steps
    """
    def step_fn(current_state, current_control): 
        # update the state 
        next_state = dynamics_step(current_state, current_control, dt, max_speed) 
        # jax.lax.scan requires returning the 'carry' (state to pass to next step) 
        # and the 'output' (what we want to save in the final array) 
        return next_state, next_state 
    
    # jax.lax.scan loops over the control_sequence instead of a for loop 
    # it returns the very final state, and an array of all intermediate states (the path) 
    _, path = jax.lax.scan(step_fn, state_init, control_sequence) 

    return path  

@jax.jit
def check_line_of_sight(start_pos, goal_pos, belief_map, resolution, num_ray_points=50):
    """
    check if there is a clear, known path from the current position to the goal
    true if LOS is clear, false if blocked by obstacle (1) or unknown (-1)
    """
    max_rows, max_cols = belief_map.shape
    
    # sampling points along a single line from start to goal
    t = jnp.linspace(0, 1, num_ray_points)
    vec = goal_pos - start_pos
    ray_points = start_pos + t[:, None] * vec  # shape: (num_ray_points, 2)
    
    # conversion to grid indices and clip to bounds
    ray_cols = jnp.clip(jnp.floor(ray_points[:, 0] / resolution).astype(jnp.int32), 0, max_cols - 1)
    ray_rows = jnp.clip(jnp.floor(ray_points[:, 1] / resolution).astype(jnp.int32), 0, max_rows - 1)
    
    # getting the values of the belief map along the ray
    ray_voxels = belief_map[ray_rows, ray_cols]
    
    # LOS is clear if all voxels along the ray are 0 (free space) 
    is_blocked = jnp.any(ray_voxels != 0)
    return jnp.logical_not(is_blocked) 

@jax.jit
def calculate_perception_cost(endpoints, goal_pose, belief_map, resolution, num_ray_points=20):
    """
    ray tracing (vector form) from N trajectory endpoints to goal
    endpoints: shape (N, 2)
    """
    N = endpoints.shape[0]
    max_rows, max_cols = belief_map.shape

    # interpolation steps (0 to 1)
    t = jnp.linspace(0, 1, num_ray_points) # shape: (M,)
    
    # geting vectors from each endpoint to the goal
    vecs = goal_pose - endpoints # shape: (N, 2)
    
    # broadcasting (helpful) to create M points along N rays simultaneously
    # endpoints: (N, 1, 2) | t: (1, M, 1) | vecs: (N, 1, 2)
    ray_points = endpoints[:, None, :] + t[None, :, None] * vecs[:, None, :] # Shape: (N, M, 2)

    # convert physical coordinates to map indices
    ray_cols = jnp.floor(ray_points[..., 0] / resolution).astype(jnp.int32)
    ray_rows = jnp.floor(ray_points[..., 1] / resolution).astype(jnp.int32)

    # clip to map bounds (safety)
    ray_cols = jnp.clip(ray_cols, 0, max_cols - 1)
    ray_rows = jnp.clip(ray_rows, 0, max_rows - 1)

    # check voxel values: Shape (N, M)
    ray_voxels = belief_map[ray_rows, ray_cols] 

    # finding the first collision along each ray
    # boolean mask of not free space (either 1 or -1)
    is_not_free = (ray_voxels != 0)
    
    # jnp.argmax returns the index of the first True value
    first_col_idx = jnp.argmax(is_not_free, axis=1) # Shape: (N,)
    has_collision = jnp.any(is_not_free, axis=1)    # Shape: (N,)

    # actual voxel value (-1 or 1) at the first collision
    first_voxels = ray_voxels[jnp.arange(N), first_col_idx]

    # assigning perception costs 
    C_occupied = 2.0
    C_unknown = 500.0 

    p_costs = jnp.zeros(N)
    
    # penalty if the ray hits a wall first
    p_costs = jnp.where(has_collision & (first_voxels == 1), C_occupied, p_costs)
    
    # reward (negative cost) if the ray hits an unknown frontier first
    p_costs = jnp.where(has_collision & (first_voxels == -1), -C_unknown, p_costs)

    return p_costs

# magic of vmap 
# no need to map the initial state (all trajectories start from the same initial state) 
# but map the first axis of the control array (ofcourse) 
# so we can roll N parallel trajectories simultaenously 
batch_rollout = jax.vmap(single_trajectory_rollout, in_axes=(None, 0, None, None)) 

# @jax.jit(static_argnums=(5, 6))
def mppi_step(state, nominal_controls, belief_map, goal_pose, prng_key, N=1000, H=15, dt=0.1, lam=0.02, resolution=0.1, max_speed=2.0): 
    """
    performs one complete MPPI optimization step
    """
    # mppi noise sampling 
    key, subkey = jax.random.split(prng_key) 
    noise_std = jnp.array([8, 8]) 
    noise = jax.random.normal(subkey, shape=(N, H, 2)) * noise_std 
    perturbed_controls = nominal_controls + noise  

    # clipping limits (hw) 
    # limits: [max_acceleration, max_angular_velocity] 
    max_controls = jnp.array([15, jnp.pi]) 
    min_controls = jnp.array([-15, -jnp.pi]) 
    perturbed_controls = jnp.clip(perturbed_controls, min_controls, max_controls) 

    # rollout 
    paths = batch_rollout(state, perturbed_controls, dt, max_speed) 
    # extracting the states
    px = paths[..., 0] 
    py = paths[..., 1] 
    theta = paths[..., 2]
    v_seq = paths[..., 3] 
    all_positions = paths[..., :2] 

    # costs  

    # trajectory and final pos costs

    # trajectory cost wrt the goal (penaliing the trajectory for wandering)
    # squared euclidean distance between drone and the goal at all the time steps
    step_goal_costs = jnp.sum((all_positions - goal_pose)**2, axis=-1)  # shape (N,H)
    trajectory_goal_cost = jnp.sum(step_goal_costs, axis=-1) * 0.05 # sums along the last axis, shape (N,) 

    # obstacle and collisions costs  

    # real distance to voxels
    col_indices = jnp.floor(px / resolution).astype(jnp.int32) 
    row_indices = jnp.floor(py / resolution).astype(jnp.int32) 

    # shape of the belief map (not constant) 
    max_rows, max_cols = belief_map.shape  
    # (|) logical or operator, returns true if TRUE if the conditions are met
    out_of_bounds = (col_indices < 0) | (col_indices >= max_cols) | \
                    (row_indices < 0) | (row_indices >= max_rows)
    
    # assuming not out of bounds ! (safety)
    col_indices = jnp.clip(col_indices, 0, max_cols - 1) 
    row_indices = jnp.clip(row_indices, 0, max_rows - 1) 
    # getting all the map values !
    map_values = belief_map[row_indices, col_indices] 

    # fatal if hitting walls (1) or going out of the map (bounds)
    is_fatal = (map_values == 1) | out_of_bounds
    obstacle_cost = jnp.sum(is_fatal, axis=-1) * 10000.0 

    # perception cost
    
    # check LOS to goal from the current position
    has_los = check_line_of_sight(state[:2], goal_pose, belief_map, resolution)

    # 2. calculate ray tracing costs from the ENDPOINTS (last) of the sampled trajectories
    # final step of the horizon: all_positions[:, -1, :]
    endpoints = all_positions[:, -1, :] 
    perception_costs = calculate_perception_cost(endpoints, goal_pose, belief_map, resolution)

    # only apply perception costs if no LOS
    # has_los is True, active_perception_costs becomes 0.0 (exploitation)
    # has_los is False, we use the calculated costs (exploration)
    active_perception_costs = jnp.where(has_los, 0.0, perception_costs)

    # total cost
    total_cost = trajectory_goal_cost + obstacle_cost + 5000*active_perception_costs

    # weight update and control extraction
    beta = jnp.min(total_cost) 
    weights = jnp.exp(- (total_cost - beta) / lam) 
    weights = weights / jnp.sum(weights) 

    optimal_control_sequence = nominal_controls + jnp.sum(weights.reshape(N, 1, 1) * noise, axis=0)  
    optimal_control_sequence = jnp.clip(optimal_control_sequence, min_controls, max_controls)

    return optimal_control_sequence, key

mppi_step = jax.jit(mppi_step, static_argnums=(5, 6))