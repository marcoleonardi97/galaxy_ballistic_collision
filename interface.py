from amuse.units import units, constants, nbody_system
from amuse.lab import Particles
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from galaxy import MilkyWay_galaxy

class GalaxyPenetrationSimulation:
    def __init__(self, milky_way=None, cluster_mass=1e7|units.MSun, cluster_radius=0.5|units.kpc,
                 impact_parameter=5|units.kpc, initial_distance=30|units.kpc, 
                 velocity=300|units.km/units.s):
        """
        Initialize a simulation of a cluster penetrating the Milky Way.
        
        Parameters:
        -----------
        milky_way : MilkyWay_galaxy object
            Pre-initialized Milky Way galaxy model
        cluster_mass : mass unit
            Total mass of the penetrating cluster
        cluster_radius : length unit
            Radius of the penetrating cluster
        impact_parameter : length unit
            Distance from galaxy center at closest approach
        initial_distance : length unit
            Initial distance of the cluster from the galaxy center
        velocity : velocity unit
            Initial velocity of the cluster
        """
        # Create or use existing Milky Way model
        if milky_way is None:
            self.milky_way = MilkyWay_galaxy(ndisk=2000, nhalo=1000, gas_mass=5e4|units.MSun)
            self.milky_way.setup_initial_conditions()
        else:
            self.milky_way = milky_way
        
        # Cluster parameters
        self.cluster_mass = cluster_mass
        self.cluster_radius = cluster_radius
        self.impact_parameter = impact_parameter
        self.initial_distance = initial_distance
        self.velocity = velocity
        
        # Create the penetrating cluster
        self.create_cluster()
        
        # Tracking data
        self.times = [] | units.Myr
        self.hole_densities = []  # Normalized density in hole region
        self.disk_densities = []  # Average disk density for comparison
        self.hole_center_positions = []  # Track the center of the hole
        
        # Output directory for data and visualizations
        self.output_dir = "galaxy_penetration_results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def create_cluster(self, n_particles=200):
        """Create a cluster of stars and gas that will penetrate the Milky Way"""
        cluster = Particles(n_particles)
        
        # Set cluster on a trajectory toward the galaxy
        # The cluster starts at (-initial_distance, impact_parameter, 0)
        # and moves along the x-axis with velocity (velocity, 0, 0)
        
        # Distribute particles in a Plummer sphere
        r = self.cluster_radius * np.random.power(2.5, n_particles)
        theta = np.arccos(np.random.uniform(-1, 1, n_particles))
        phi = np.random.uniform(0, 2*np.pi, n_particles)
        
        # Convert to Cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        # Position the cluster
        cluster.x = x - self.initial_distance
        cluster.y = y + self.impact_parameter
        cluster.z = z
        
        # Set masses (80% stars, 20% gas)
        n_stars = int(0.8 * n_particles)
        star_masses = np.random.power(1.5, n_stars) * (20 | units.MSun)
        gas_masses = np.ones(n_particles - n_stars) * (self.cluster_mass * 0.2 / (n_particles - n_stars))
        
        cluster[:n_stars].mass = star_masses
        cluster[n_stars:].mass = gas_masses
        
        # Adjust masses to reach the desired total mass
        current_mass = cluster.mass.sum()
        scaling_factor = self.cluster_mass / current_mass
        cluster.mass *= scaling_factor
        
        # Set velocities
        sigma = (constants.G * self.cluster_mass / (5 * self.cluster_radius)).sqrt()
        vx = np.random.normal(0, sigma.value_in(units.km/units.s), n_particles) | units.km/units.s
        vy = np.random.normal(0, sigma.value_in(units.km/units.s), n_particles) | units.km/units.s
        vz = np.random.normal(0, sigma.value_in(units.km/units.s), n_particles) | units.km/units.s
        
        # Add bulk velocity toward the galaxy
        cluster.vx = vx + self.velocity
        cluster.vy = vy
        cluster.vz = vz
        
        # Set internal energy for gas particles
        cluster[n_stars:].u = (10**4 | units.K) * constants.kB / (0.6 * constants.proton_mass)
        
        # Add cluster to Milky Way
        self.cluster_stars = cluster[:n_stars].copy()
        self.cluster_gas = cluster[n_stars:].copy()
        
        # Add cluster particles to the Milky Way model
        self.milky_way.stars.add_particles(self.cluster_stars)
        self.milky_way.gas_particles.add_particles(self.cluster_gas)
        self.milky_way.all_particles.add_particles(cluster)
        
        print(f"Created cluster with {n_particles} particles (total mass: {self.cluster_mass})")
        print(f"Initial position: ({-self.initial_distance}, {self.impact_parameter}, 0)")
        print(f"Initial velocity: ({self.velocity}, 0, 0)")
    
    def analyze_density_distribution(self, bin_size=1.0):
        """
        Analyze the density distribution of the galaxy, focusing on the hole region.
        Returns the normalized density in the hole region and the average disk density.
        
        Parameters:
        -----------
        bin_size : float
            Size of bins for density calculation in kpc
        """
        # Get current time
        current_time = self.milky_way.system_time
        
        # Find current position of the cluster (approximated by the center of mass)
        cluster_particles = Particles()
        cluster_particles.add_particles(self.cluster_stars)
        cluster_particles.add_particles(self.cluster_gas)
        
        cluster_com = cluster_particles.center_of_mass()
        
        # Define the hole region (cylindrical region around the cluster COM)
        hole_radius = 2 * self.cluster_radius  # Hole radius is larger than cluster radius
        
        # Define a function to calculate distance from cluster center in xy plane
        def xy_distance(x, y):
            return ((x - cluster_com[0])**2 + (y - cluster_com[1])**2).sqrt()
        
        # Count particles in the hole region (stars and gas from the Milky Way, not from the cluster)
        original_mw_stars = self.milky_way.stars[:-(len(self.cluster_stars))]
        original_mw_gas = self.milky_way.gas_particles[:-(len(self.cluster_gas))]
        
        # Find particles in the hole region
        stars_in_hole = original_mw_stars[xy_distance(original_mw_stars.x, original_mw_stars.y) < hole_radius]
        gas_in_hole = original_mw_gas[xy_distance(original_mw_gas.x, original_mw_gas.y) < hole_radius]
        
        # Calculate hole area
        hole_area = np.pi * hole_radius**2
        
        # Calculate densities (particles per kpc^2)
        hole_star_density = len(stars_in_hole) / hole_area.value_in(units.kpc**2)
        hole_gas_density = len(gas_in_hole) / hole_area.value_in(units.kpc**2)
        hole_total_density = hole_star_density + hole_gas_density
        
        # Calculate average disk density (within 20 kpc for comparison)
        disk_radius = 20 | units.kpc
        disk_area = np.pi * disk_radius**2
        
        stars_in_disk = original_mw_stars[
            (original_mw_stars.x**2 + original_mw_stars.y**2).sqrt() < disk_radius
        ]
        gas_in_disk = original_mw_gas[
            (original_mw_gas.x**2 + original_mw_gas.y**2).sqrt() < disk_radius
        ]
        
        disk_star_density = len(stars_in_disk) / disk_area.value_in(units.kpc**2)
        disk_gas_density = len(gas_in_disk) / disk_area.value_in(units.kpc**2)
        disk_total_density = disk_star_density + disk_gas_density
        
        # Normalized hole density (ratio to average disk density)
        normalized_hole_density = hole_total_density / disk_total_density
        
        # Store results
        self.times.append(current_time)
        self.hole_densities.append(normalized_hole_density)
        self.disk_densities.append(disk_total_density)
        self.hole_center_positions.append((cluster_com[0], cluster_com[1], cluster_com[2]))
        
        print(f"Time: {current_time.in_(units.Myr)}")
        print(f"Hole center: ({cluster_com[0].in_(units.kpc)}, {cluster_com[1].in_(units.kpc)}, {cluster_com[2].in_(units.kpc)})")
        print(f"Normalized hole density: {normalized_hole_density:.3f}")
        
        return normalized_hole_density, disk_total_density
    
    def create_density_map(self, bin_size=0.5, save=True, filename=None):
        """
        Create a 2D density map of the galaxy, showing the hole.
        
        Parameters:
        -----------
        bin_size : float
            Size of bins for density calculation in kpc
        save : bool
            Whether to save the plot
        filename : str
            Filename for the plot (if None, a default name will be used)
        """
        # Get current time
        current_time = self.milky_way.system_time
        
        # Create bins for the histogram
        x_range = (-25, 25)  # kpc
        y_range = (-25, 25)  # kpc
        
        x_bins = np.arange(x_range[0], x_range[1] + bin_size, bin_size)
        y_bins = np.arange(y_range[0], y_range[1] + bin_size, bin_size)
        
        # Get positions of all Milky Way particles (excluding the cluster)
        original_mw_stars = self.milky_way.stars[:-(len(self.cluster_stars))]
        original_mw_gas = self.milky_way.gas_particles[:-(len(self.cluster_gas))]
        
        stars_x = original_mw_stars.x.value_in(units.kpc)
        stars_y = original_mw_stars.y.value_in(units.kpc)
        gas_x = original_mw_gas.x.value_in(units.kpc)
        gas_y = original_mw_gas.y.value_in(units.kpc)
        
        # Create histograms
        star_hist, _, _ = np.histogram2d(stars_x, stars_y, bins=[x_bins, y_bins])
        gas_hist, _, _ = np.histogram2d(gas_x, gas_y, bins=[x_bins, y_bins])
        
        # Combined density
        total_hist = star_hist + gas_hist
        
        # Get positions of the cluster
        cluster_x = self.cluster_stars.x.value_in(units.kpc)
        cluster_y = self.cluster_stars.y.value_in(units.kpc)
        cluster_gas_x = self.cluster_gas.x.value_in(units.kpc)
        cluster_gas_y = self.cluster_gas.y.value_in(units.kpc)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot density map
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
        plt.imshow(total_hist.T, origin='lower', extent=extent, 
                   aspect='equal', cmap='inferno', norm=plt.cm.colors.LogNorm())
        plt.colorbar(label='Particle Count')
        
        # Plot the cluster
        plt.scatter(cluster_x, cluster_y, s=5, c='cyan', alpha=0.7, label='Cluster Stars')
        plt.scatter(cluster_gas_x, cluster_gas_y, s=5, c='blue', alpha=0.7, label='Cluster Gas')
        
        # Add labels and title
        plt.xlabel('X [kpc]')
        plt.ylabel('Y [kpc]')
        plt.title(f'Galaxy Density Map at T = {current_time.in_(units.Myr).number:.1f} Myr')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        if save:
            if filename is None:
                filename = f"{self.output_dir}/density_map_T{current_time.in_(units.Myr).number:.1f}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def run_simulation(self, total_time=200|units.Myr, dt=1|units.Myr, analyze_every=5):
        """
        Run the simulation for the specified time, analyzing the density distribution periodically.
        
        Parameters:
        -----------
        total_time : time unit
            Total simulation time
        dt : time unit
            Time step for the simulation
        analyze_every : int
            Number of time steps between analyses
        """
        start_time = time.time()
        
        # Number of time steps
        n_steps = int(total_time / dt)
        
        # Run the initial analysis
        print("Starting simulation...")
        self.analyze_density_distribution()
        self.create_density_map()
        self.milky_way.overview(show=False, save=True, 
                               filename=f"{self.output_dir}/galaxy_T{self.milky_way.system_time.in_(units.Myr).number:.1f}.png")
        
        # Run the simulation
        hole_refilled = False
        refill_time = None
        density_threshold = 0.9  # Consider the hole refilled when density reaches 90% of disk average
        
        for step in range(1, n_steps + 1):
            # Evolve the system
            _, _, _ = self.milky_way.evolve_system(self.milky_way.system_time + dt, dt=dt/10)
            
            # Analyze periodically
            if step % analyze_every == 0 or step == n_steps:
                # Analyze density
                hole_density, disk_density = self.analyze_density_distribution()
                
                # Create density map
                self.create_density_map()
                
                # Create galaxy overview
                self.milky_way.overview(show=False, save=True, 
                                       filename=f"{self.output_dir}/galaxy_T{self.milky_way.system_time.in_(units.Myr).number:.1f}.png")
                
                # Check if hole has refilled
                if not hole_refilled and hole_density >= density_threshold:
                    hole_refilled = True
                    refill_time = self.milky_way.system_time
                    print(f"Hole has refilled at T = {refill_time.in_(units.Myr)}!")
        
        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.2f} seconds")
        
        if hole_refilled:
            print(f"The hole refilled after {refill_time.in_(units.Myr)}")
        else:
            print(f"The hole did not refill within the simulation time ({total_time.in_(units.Myr)})")
        
        # Plot the hole density evolution
        self.plot_density_evolution()
        
        # Plot the hole trajectory
        self.plot_hole_trajectory()
        
        return self.times, self.hole_densities
    
    def plot_density_evolution(self):
        """Plot the evolution of the hole density over time"""
        plt.figure(figsize=(10, 6))
        
        times_myr = [t.value_in(units.Myr) for t in self.times]
        
        plt.plot(times_myr, self.hole_densities, 'b-', label='Normalized Hole Density')
        plt.axhline(y=0.9, color='r', linestyle='--', label='Refill Threshold (90%)')
        
        plt.xlabel('Time [Myr]')
        plt.ylabel('Normalized Density (Hole/Disk Average)')
        plt.title('Evolution of Hole Density')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Find refill time (if it occurred)
        refill_time = None
        for i in range(len(self.hole_densities)):
            if self.hole_densities[i] >= 0.9:
                refill_time = self.times[i]
                break
        
        if refill_time is not None:
            plt.axvline(x=refill_time.value_in(units.Myr), color='g', linestyle=':', 
                       label=f'Refill at {refill_time.in_(units.Myr)}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/hole_density_evolution.png", dpi=300)
        plt.close()
    
    def plot_hole_trajectory(self):
        """Plot the trajectory of the hole center"""
        plt.figure(figsize=(10, 8))
        
        # Extract trajectory points
        x_positions = [pos[0].value_in(units.kpc) for pos in self.hole_center_positions]
        y_positions = [pos[1].value_in(units.kpc) for pos in self.hole_center_positions]
        
        # Color points by time
        times_myr = [t.value_in(units.Myr) for t in self.times]
        
        plt.scatter(x_positions, y_positions, c=times_myr, cmap='viridis', 
                   s=50, alpha=0.8, edgecolors='k', linewidths=0.5)
        
        # Connect the points
        plt.plot(x_positions, y_positions, 'k--', alpha=0.3)
        
        # Mark start and end points
        plt.plot(x_positions[0], y_positions[0], 'go', markersize=10, label='Start')
        plt.plot(x_positions[-1], y_positions[-1], 'ro', markersize=10, label='End')
        
        # Add colorbar for time
        cbar = plt.colorbar()
        cbar.set_label('Time [Myr]')
        
        # Add labels and title
        plt.xlabel('X [kpc]')
        plt.ylabel('Y [kpc]')
        plt.title('Trajectory of the Cluster/Hole')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add galaxy center for reference
        plt.plot(0, 0, 'y*', markersize=15, label='Galaxy Center')
        plt.legend()
        
        # Set equal aspect ratio
        plt.axis('equal')
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/hole_trajectory.png", dpi=300)
        plt.close()
