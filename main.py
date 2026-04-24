import jax 
import jax.numpy as jnp 
from functions import initialize_maps, mppi_step, dynamics_step 
from plotting import plot_simulation

def main(): 
    # setup parameters 
    dt = 0.1 
    max_steps = 150 
    resolution = 0.1 
    max_speed = 2.0 

    # mppi parameters 
    H = 50  
    N = 1000 
    lam = 0.05 # mppi temperature 

    # initializing the environments 
    ground_truth_map, _ = initialize_maps(width_m=3.0, length_m=10.0, resolution=resolution) 

    # assume perfect map knowledge 
    belief_map = ground_truth_map 

    # initial conditions 
    # starting at bottom middle facing upwards 
    start_state = jnp.array([1.5, 1.0, jnp.pi/2, 0.0]) 
    goal_pos = jnp.array([1.5, 9.0]) 

    # intialize nominal controls (to zero) 
    nominal_controls = jnp.zeros((H, 2)) 

    # initialize JAX random key 
    prng_key = jax.random.PRNGKey(42) 

    # simulation loop 
    current_state = start_state
    state_history = [current_state] 

    print("starting the simulation") 

    for step in range(max_steps): 
        # calculate distance to goal for completion 
        dist_to_goal = jnp.linalg.norm(current_state[:2]-goal_pos) 
        if dist_to_goal <= 0.1 : 
            print(f"goal reached at step {step}")
            break 

        # mppi optimization 
        optimal_controls, prng_key = mppi_step(current_state, nominal_controls, belief_map,
                                               goal_pos, prng_key, N, H, dt, lam, resolution, max_speed) 
        
        # receeding horizon 
        # shift the control sequence forward by 1 step 
        # the last step is filled with zeros
        best_action = optimal_controls[0]  
        shifted_controls = jnp.roll(optimal_controls, shift=-1, axis=0) 
        nominal_controls = shifted_controls.at[-1].set(jnp.array([0.0, 0.0]))

        # updating true physics 
        current_state = dynamics_step(current_state, best_action, dt, max_speed) 
        state_history.append(current_state) 

        if step %10 == 0 : 
            print(f"Step {step}: Distance to goal: {dist_to_goal:.2f}m") 
        
    
    print("simulation completed") 

    plot_simulation(ground_truth_map, state_history, goal_pos, resolution) 

if __name__ == "__main__" : 
    main()