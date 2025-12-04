import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization for parameter fitting.
    """
    def __init__(self, 
                 n_particles: int = 30,
                 n_iterations: int = 100,
                 w: float = 0.7,  # Inertia weight
                 c1: float = 1.5,  # Cognitive parameter
                 c2: float = 1.5,  # Social parameter
                 bounds: dict[str, tuple[float, float]] = None,
                 initial_params: dict[str, float] = None,  # NEW
                 objective_function: callable([[dict[str, float]], float]) = None,
                 verbose: bool = True):
        """
        Initialize PSO optimizer.
        
        Args:
            n_particles: Number of particles in swarm
            n_iterations: Number of optimization iterations
            w: Inertia weight (controls exploration vs exploitation)
            c1: Cognitive parameter (particle's own best)
            c2: Social parameter (swarm's best)
            bounds: Dictionary of parameter bounds {'param_name': (min, max)}
            initial_params: Dictionary of initial parameter values to seed first particle
            verbose: Print progress during optimization
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds
        self.initial_params = initial_params  # NEW
        self.verbose = verbose
        
        # To be initialized
        self.param_names = None
        self.n_params = None
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = None
        self.history = {'best_scores': [], 'mean_scores': []}
        self.objective_function = objective_function
        
    def initialize_swarm(self):
        """Initialize particle positions and velocities within bounds."""
        self.param_names = list(self.bounds.keys())
        self.n_params = len(self.param_names)
        
        # Initialize positions randomly within bounds
        self.positions = np.zeros((self.n_particles, self.n_params))
        for i, param_name in enumerate(self.param_names):
            lower, upper = self.bounds[param_name]
            self.positions[:, i] = np.random.uniform(lower, upper, self.n_particles)
        
        # NEW: Set first particle to initial parameters if provided
        if self.initial_params is not None:
            if self.verbose:
                print("Seeding first particle with initial parameters:")
            for i, param_name in enumerate(self.param_names):
                if param_name in self.initial_params:
                    self.positions[0, i] = self.initial_params[param_name]
                    if self.verbose:
                        print(f"  {param_name:15s} = {self.initial_params[param_name]:.6e}")
            if self.verbose:
                print()
        
        # Initialize velocities (small random values)
        velocity_range = np.array([self.bounds[p][1] - self.bounds[p][0] 
                                   for p in self.param_names])
        self.velocities = np.random.uniform(-0.1, 0.1, (self.n_particles, self.n_params)) * velocity_range
        
        # Initialize personal bests
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.n_particles, np.inf)
        
        # Initialize global best
        initial_array = np.array([self.initial_params[name] for name in self.param_names])
        self.global_best_position = initial_array.copy() # Store as array
        self.global_best_score = self.objective_function(self.initial_params)        
        print(f"\nMean Squared Error (MSE) between model and data: {self.global_best_score:.6e}")

    def params_to_dict(self, params_array: np.ndarray) -> dict[str, float]:
        """Convert parameter array to dictionary."""
        return {name: params_array[i] for i, name in enumerate(self.param_names)}

    def params_to_dict_jax(self, params_array: np.ndarray) -> dict[str, list]:
        """Convert parameter array to dictionary."""
        return {name: jnp.array(params_array[:, i]) for i, name in enumerate(self.param_names)}

    def clip_to_bounds(self):
        """Ensure all particle positions are within bounds."""
        for i, param_name in enumerate(self.param_names):
            lower, upper = self.bounds[param_name]
            self.positions[:, i] = np.clip(self.positions[:, i], lower, upper)
    
    def optimize(self, objective_function: callable([[dict[str, float]], float])) -> tuple[dict[str, float], float, dict[str, float]]:
        """
        Run PSO optimization.
        
        Args:
            objective_function: Function that takes parameter dict and returns cost (MSE)
            
        Returns:
            best_params: Dictionary of optimized parameters
            best_score: Best objective function value achieved
        """
        self.initialize_swarm()
        # # You need to specify which axis to map over for EACH dictionary entry:
        # in_axes_dict = {key: 0 for key in self.param_names}
        #
        # jax_obj_fn = jax.vmap(objective_function, in_axes=(in_axes_dict,))
        for iteration in range(self.n_iterations):
            # Evaluate all particles
            # scores = np.zeros(self.n_particles)
            params_dict = []
            for p in range(self.n_particles):
                params_dict.append(self.params_to_dict(self.positions[p]))
            # params_dict = self.params_to_dict_jax(self.positions)
            scores = objective_function(params_dict)
            
            # Update personal bests
            improved = scores < self.personal_best_scores
            self.personal_best_scores[improved] = scores[improved]
            self.personal_best_positions[improved] = self.positions[improved]
            
            # Update global best
            best_particle_idx = np.argmin(scores)
            if scores[best_particle_idx] < self.global_best_score:
                self.global_best_score = float(scores[best_particle_idx])
                self.global_best_position = self.positions[best_particle_idx].copy()
            
            # Store history
            self.history['best_scores'].append(self.global_best_score)
            self.history['mean_scores'].append(np.mean(scores))
            
            # Print progress
            if self.verbose: #and (iteration % 10 == 0 or iteration == self.n_iterations - 1):
                print(f"Iteration {iteration+1}/{self.n_iterations}: "
                      f"Best MSE = {self.global_best_score:.6e}, "
                      f"Mean MSE = {np.mean(scores):.6e}")
            
            # Update velocities and positions
            r1 = np.random.random((self.n_particles, self.n_params))
            r2 = np.random.random((self.n_particles, self.n_params))
            
            cognitive = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social = self.c2 * r2 * (self.global_best_position - self.positions)
            
            self.velocities = self.w * self.velocities + cognitive + social
            self.positions += self.velocities
            
            # Ensure positions stay within bounds
            self.clip_to_bounds()
        
        # Return best parameters found
        best_params = self.params_to_dict(self.global_best_position)
        all_particle_params = [self.params_to_dict(self.positions[p]) for p in range(self.n_particles)]
        return best_params, self.global_best_score, all_particle_params
    
    def plot_convergence(self):
        """Plot optimization convergence history."""
        plt.figure(figsize=(10, 6))
        plt.semilogy(self.history['best_scores'], label='Best MSE', linewidth=2)
        plt.semilogy(self.history['mean_scores'], label='Mean MSE', linewidth=2, alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.title('PSO Convergence History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

