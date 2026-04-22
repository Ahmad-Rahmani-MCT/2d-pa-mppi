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
    ground_truth_map = ground_truth_map.at[50, 10:20].set(1) 

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

