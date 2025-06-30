#!/usr/bin/env python3
"""
Galaxy Collision Simulation using AMUSE Framework
Simulates a small projectile galaxy colliding with a Milky Way-type galaxy
Monitors hole formation and refilling in the target galaxy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from amuse.units import units, constants
from amuse.datamodel import Particles
from amuse.community.ph4.interface import Ph4
from amuse.community.fi.interface import Fi
from amuse.ic.kingmodel import new_king_model
from amuse.ic.plummer import new_plummer_model
from amuse.couple.bridge import Bridge
from amuse.lab import nbody_system
import time
from mpl_toolkits.mplot3d import Axes3D

class GalaxyCollisionSimulator:
    def __init__(self):
        self.stellar_code = Ph4(convert_nbody=nbody_system.nbody_to_si(1e10 | units.MSun, 10 | units.kpc))
        self.gas_code = Fi(convert_nbody=nbody_system.nbody_to_si(1e10 | units.MSun, 10 | units.kpc))
        self.bridge_code = None
        self.all_particles = None
        self.target_stars = None
        self.projectile_stars = None
        self.gas_particles = None
        self.time_data = []
        self.hole_data = []
        self.animation_data = []

    def converter(self, m, r):
        return nbody_system.nbody_to_si(m, r)
        
    def setup_codes(self):
        """Initialize and configure the physics codes"""
        # Configure stellar dynamics (Ph4)
        self.stellar_code.parameters.epsilon_squared = 0.01**2 | units.kpc**2
        self.stellar_code.parameters.timestep_parameter = 0.1 
        
        # Configure gas dynamics (Fi)
        self.gas_code.parameters.epsilon_squared = 0.01**2 | units.kpc**2
        self.gas_code.parameters.timestep = 0.1 | units.Myr
        self.gas_code.parameters.self_gravity_flag = True
        
    def create_milky_way_galaxy(self, n_stars=10000, n_gas=1000):
        """Create a Milky Way-type galaxy with stars and gas"""
        print("Creating Milky Way-type galaxy...")
        
        # Create stellar component using King model
        target_stars = new_king_model(
            n_stars, 
            W0=3.0,  # King model parameter
            Rtidal=50.0 | units.kpc,
            Mcluster=1.0e11 | units.MSun,
            convert_nbody=self.converter(1e10 | units.MSun, 10 | units.kpc)
        )
        
        # Add rotation to make it more realistic
        self.add_rotation(target_stars, v_rot=200 | units.km/units.s)
        
        # Create gas component with exponential disk profile
        target_gas = self.create_gas_disk(n_gas, 
                                        disk_radius=15.0 | units.kpc,
                                        scale_height=1.0 | units.kpc,
                                        total_mass=2.0e10 | units.MSun)
        
        return target_stars, target_gas
    
    def create_projectile_galaxy(self, n_stars=5000, n_gas=1000):
        """Create a smaller projectile galaxy"""
        print("Creating projectile galaxy...")
        
        # Smaller galaxy with King model
        projectile_stars = new_king_model(
            n_stars,
            W0=2.0,
            Rtidal=1.0 | units.kpc,
            Mcluster=1.0e10 | units.MSun,
            convert_nbody =self.converter(1e10 | units.MSun, 10 | units.kpc)
        )
        
        # Position projectile galaxy
        projectile_stars.x += 0.0 | units.kpc
        projectile_stars.y += 0.0 | units.kpc
        projectile_stars.z += 50.0 | units.kpc
        
        # Give it initial velocity toward target
        projectile_stars.vx = -100.0 | units.km/units.s
        projectile_stars.vy = -100.0 | units.km/units.s
        projectile_stars.vz = -500.0 | units.km/units.s
        
        # Add some gas to projectile
        projectile_gas = self.create_gas_disk(n_gas,
                                            disk_radius=5.0 | units.kpc,
                                            scale_height=0.5 | units.kpc,
                                            total_mass=5.0e9 | units.MSun)
        
        # Position and velocity for gas
        projectile_gas.x += 0.0 | units.kpc
        projectile_gas.y += 0.0 | units.kpc
        projectile_gas.z += 50.0 | units.kpc
        projectile_gas.vx = -100.0 | units.km/units.s
        projectile_gas.vy = -100.0 | units.km/units.s
        projectile_gas.vz = -500.0 | units.km/units.s
        
        return projectile_stars, projectile_gas
    
    def add_rotation(self, particles, v_rot):
        """Add rotational velocity to particles"""
        for p in particles:
            r = (p.x**2 + p.y**2)**0.5
            if r > 0 | units.kpc:
                # Circular velocity
                v_circ = v_rot * (r / (10.0 | units.kpc))**0.5
                p.vx = -v_circ * p.y / r
                p.vy = v_circ * p.x / r
    
    def create_gas_disk(self, n_particles, disk_radius, scale_height, total_mass):
        """Create gas particles in an exponential disk"""
        gas = Particles(n_particles)
        
        # Exponential disk profile
        r = np.random.exponential(disk_radius.value_in(units.kpc), n_particles) | units.kpc
        theta = np.random.uniform(0, 2*np.pi, n_particles)
        z = np.random.normal(0, scale_height.value_in(units.kpc), n_particles) | units.kpc
        
        gas.x = r * np.cos(theta)
        gas.y = r * np.sin(theta)
        gas.z = z
        
        # Initial velocities (rotation)
        gas.vx = 0.0 | units.km/units.s
        gas.vy = 0.0 | units.km/units.s
        gas.vz = 0.0 | units.km/units.s
        
        # Add rotation
        self.add_rotation(gas, 150.0 | units.km/units.s)
        
        # Assign masses
        gas.mass = total_mass / n_particles
        
        # Gas properties
        gas.u = 1.0e4 | (units.km/units.s)**2  # Specific internal energy
        gas.h_smooth = 0.1 | units.kpc  # Smoothing length
        
        return gas
    
    def setup_bridge(self):
        """Setup the bridge to couple stellar and gas dynamics"""
        print("Setting up bridge...")
        
        # Add particles to respective codes
        self.stellar_code.particles.add_particles(self.target_stars)
        self.stellar_code.particles.add_particles(self.projectile_stars)
        self.gas_code.particles.add_particles(self.gas_particles)
        
        # Create bridge
        self.bridge_code = Bridge()
        self.bridge_code.add_system(self.stellar_code, (self.gas_code,))
        self.bridge_code.add_system(self.gas_code, (self.stellar_code,))
        self.bridge_code.timestep = 1.0 | units.Myr
        
    def measure_hole_size(self, time):
        """Measure the size of the hole in the target galaxy"""
        # Get current positions of target stars
        target_positions = self.stellar_code.particles[:len(self.target_stars)]
        
        # Define collision region (where projectile passed through)
        collision_center_x = 5.0 | units.kpc
        collision_center_y = 3.0 | units.kpc
        collision_radius = 3.0 | units.kpc
        
        # Count stars in collision region
        distances = ((target_positions.x - collision_center_x)**2 + 
                    (target_positions.y - collision_center_y)**2)**0.5
        stars_in_region = (distances < collision_radius).sum()
        
        # Calculate hole size (inverse of density)
        initial_density = len(self.target_stars) / (np.pi * (20.0)**2)  # Rough estimate
        current_density = stars_in_region / (np.pi * collision_radius.value_in(units.kpc)**2)
        hole_fraction = 1.0 - (current_density / initial_density)
        
        return max(0.0, hole_fraction)
    
    def run_simulation(self, end_time=100.0 | units.Myr, dt=1.0 | units.Myr):
        """Run the collision simulation"""
        print("Starting simulation...")
        
        # Setup
        self.setup_codes()
        self.target_stars, target_gas = self.create_milky_way_galaxy()
        self.projectile_stars, projectile_gas = self.create_projectile_galaxy()
        
        # Combine gas particles
        self.gas_particles = Particles()
        self.gas_particles.add_particles(target_gas)
        self.gas_particles.add_particles(projectile_gas)
        
        self.setup_bridge()
        
        # Simulation loop
        current_time = 0.0 | units.Myr
        step = 0
        
        while current_time < end_time:
            print(f"Time: {current_time.value_in(units.Myr):.1f} Myr")
            
            # Evolve system
            self.bridge_code.evolve_model(current_time + dt)
            current_time += dt
            
            # Measure hole size
            hole_size = self.measure_hole_size(current_time)
            self.time_data.append(current_time.value_in(units.Myr))
            self.hole_data.append(hole_size)
            
            # Store animation data every few steps
            if step % 5 == 0:
                stellar_data = {
                    'target_x': self.stellar_code.particles[:len(self.target_stars)].x.value_in(units.kpc),
                    'target_y': self.stellar_code.particles[:len(self.target_stars)].y.value_in(units.kpc),
                    'target_z': self.stellar_code.particles[:len(self.target_stars)].z.value_in(units.kpc),
                    'projectile_x': self.stellar_code.particles[len(self.target_stars):].x.value_in(units.kpc),
                    'projectile_y': self.stellar_code.particles[len(self.target_stars):].y.value_in(units.kpc),
                    'projectile_z': self.stellar_code.particles[len(self.target_stars):].z.value_in(units.kpc),
                    'gas_x': self.gas_code.particles.x.value_in(units.kpc),
                    'gas_y': self.gas_code.particles.y.value_in(units.kpc),
                    'gas_z': self.gas_code.particles.y.value_in(units.kpc),
                    'time': current_time.value_in(units.Myr)
                }
                self.animation_data.append(stellar_data)
            
            step += 1
        
        print("Simulation completed!")
        self.cleanup()
    
    def cleanup(self):
        """Clean up the simulation codes"""
        if self.bridge_code:
            self.bridge_code.stop()
        self.stellar_code.stop()
        self.gas_code.stop()
    
    def plot_hole_evolution(self):
        """Plot the evolution of the hole size over time"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_data, self.hole_data, 'b-', linewidth=2)
        plt.xlabel('Time (Myr)')
        plt.ylabel('Hole Fraction')
        plt.title('Evolution of Hole in Target Galaxy')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, max(self.time_data))
        plt.ylim(0, max(self.hole_data) * 1.1 if self.hole_data else 1)
        
        # Mark refilling time
        if self.hole_data:
            max_hole = max(self.hole_data)
            max_time = self.time_data[self.hole_data.index(max_hole)]
            
            # Find when hole is 50% refilled
            refill_threshold = max_hole * 0.5
            refill_time = None
            for i, hole_size in enumerate(self.hole_data):
                if i > self.hole_data.index(max_hole) and hole_size <= refill_threshold:
                    refill_time = self.time_data[i]
                    break
            
            plt.axvline(max_time, color='r', linestyle='--', alpha=0.7, label=f'Max hole at {max_time:.1f} Myr')
            if refill_time:
                plt.axvline(refill_time, color='g', linestyle='--', alpha=0.7, 
                           label=f'50% refill at {refill_time:.1f} Myr')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def animate_collision(self, save_filename=None):
        """Create animation of the collision"""
        if not self.animation_data:
            print("No animation data available!")
            return
        
        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        #ax2 = fig.add_subplot(112, project="3d")
        
        def animate(frame):
            ax1.clear()
            #ax2.clear()
            
            data = self.animation_data[frame]
            
            # Plot stellar components
            ax1.scatter(data['target_x'], data['target_y'], data['target_z'],
                       s=0.5, c='blue', alpha=0.6, label='Target stars')
            ax1.scatter(data['projectile_x'], data['projectile_y'], data["projectile_z"], 
                       s=0.5, c='red', alpha=0.8, label='Projectile stars')
            
            ax1.set_xlim(-100, 100)
            ax1.set_ylim(-100, 100)
            ax1.set_zlim(-100, 100)
            ax1.set_xlabel('X (kpc)')
            ax1.set_ylabel('Y (kpc)')
            ax1.set_zlabel('Z (kpc)')
            ax1.set_title(f'Stellar Components - Time: {data["time"]:.1f} Myr')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot gas
            """
            ax2.scatter(data['gas_x'], data['gas_y'], data["gas_z"],
                       s=1.0, c='green', alpha=0.4, label='Gas')
            
            ax2.set_xlim(-30, 40)
            ax2.set_ylim(-30, 30)
            ax2.set_xlabel('X (kpc)')
            ax2.set_ylabel('Y (kpc)')
            ax2.set_zlabel('Z (kpc)')
            ax2.set_title(f'Gas Component - Time: {data["time"]:.1f} Myr')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            """
        anim = FuncAnimation(fig, animate, frames=len(self.animation_data), 
                           interval=200, repeat=True)
        
        if save_filename:
            print(f"Saving animation to {save_filename}...")
            anim.save(save_filename, writer='pillow', fps=5)
        
        plt.tight_layout()
        plt.show()
        
        return anim

