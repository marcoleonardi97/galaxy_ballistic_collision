from amuse.units import units, constants, nbody_system
from amuse.lab import Particles
from amuse.community.ph4.interface import Ph4
from amuse.community.fi.interface import Fi
from amuse.couple.bridge import Bridge
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from amuse.ic.kingmodel import new_king_model


class GalaxyCollision:
    def __init__(self, n_target_stars=8000, n_target_gas=2000, n_intruder_stars=4000, n_intruder_gas=1000,
                offset = [0, 0, 30] | units.kpc):
        """
        Galaxy collision simulator using Ph4 + Fi + Bridge
        """
        self.n_target_stars = n_target_stars
        self.n_target_gas = n_target_gas
        self.n_intruder_stars = n_intruder_stars
        self.n_intruder_gas = n_intruder_gas
        self.offset = offset
        
        self.time = 0 | units.Myr
        self.converter = nbody_system.nbody_to_si(1e11 | units.MSun, 20 | units.kpc)
        
        # Particle collections
        self.all_stars = Particles()
        self.all_gas = Particles()
        self.history = []
        
        self.setup_galaxies()
        
    def setup_galaxies(self):
        """Create target MW-like galaxy and smaller intruder with proper gravitational binding"""
        
        # TARGET GALAXY - Use King model properly
        target_stars = new_king_model(
            self.n_target_stars, 
            W0=7.0,  # Higher W0 for more concentrated, stable galaxy
            Rtidal=15.0 | units.kpc,
            Mcluster=8.0e10 | units.MSun,  # Reduced mass for stability
            convert_nbody=self.converter
        )
        
        # Add central supermassive black hole for target galaxy
        central_bh = Particles(1)
        central_bh.mass = 4.0e6 | units.MSun  # Milky Way-like SMBH
        central_bh.x = 0 | units.kpc
        central_bh.y = 0 | units.kpc  
        central_bh.z = 0 | units.kpc
        central_bh.vx = 0 | units.km/units.s
        central_bh.vy = 0 | units.km/units.s
        central_bh.vz = 0 | units.km/units.s
        target_stars.add_particle(central_bh)
        
        # Scale target galaxy to disk-like structure but keep King model velocities
        # Only modify z-coordinates to flatten the disk
        target_stars.z *= 0.2  # Flatten to disk
        target_stars.vz *= 0.5  # Reduce z-velocities for disk
        
        # TARGET GALAXY GAS - distributed like a thin disk
        target_gas = Particles(self.n_target_gas)
        gas_radius = np.random.exponential(6.0, self.n_target_gas) | units.kpc
        gas_theta = np.random.uniform(0, 2*np.pi, self.n_target_gas)
        
        target_gas.x = gas_radius * np.cos(gas_theta)
        target_gas.y = gas_radius * np.sin(gas_theta)
        target_gas.z = np.random.normal(0, 0.3, self.n_target_gas) | units.kpc
        target_gas.mass = (1e8 | units.MSun) * np.ones(self.n_target_gas)
        
        # Circular velocities for gas (more realistic)
        total_enclosed_mass = 8.0e10 | units.MSun  # Approximate
        v_gas = np.sqrt(constants.G * total_enclosed_mass / gas_radius)
        target_gas.vx = -v_gas * np.sin(gas_theta)
        target_gas.vy = v_gas * np.cos(gas_theta) 
        target_gas.vz = np.random.normal(0, 8, self.n_target_gas) | units.km/units.s
        
        # Gas internal energy (temperature)
        target_gas.u = (1e4 | units.K) * constants.kB / (0.6 * constants.proton_mass)
        
        # INTRUDER GALAXY - Smaller, denser galaxy (elliptical-like)
        intruder_stars = new_king_model(
            self.n_intruder_stars,
            W0=2.0,  # Well-bound
            Rtidal=1 | units.kpc,
            Mcluster=2.0e9 | units.MSun,
            convert_nbody=nbody_system.nbody_to_si(1e9 | units.Msun, 1 |units.kpc)
        )
        
        approach_vel = [-i.number * 10 for i in self.offset] | units.km/units.s
        
        # Add central black hole to intruder
        intruder_bh = Particles(1)
        intruder_bh.mass = 1.0e6 | units.MSun
        intruder_bh.x = self.offset[0]
        intruder_bh.y = self.offset[1]
        intruder_bh.z = self.offset[2]
        intruder_bh.vx = approach_vel[0]   # Approach velocity
        intruder_bh.vy = approach_vel[1] 
        intruder_bh.vz = approach_vel[2] 
        
        # Position intruder at offset
        intruder_stars.x += self.offset[0]
        intruder_stars.y += self.offset[1]
        intruder_stars.z += self.offset[2]
        
        # Give intruder approach velocity (preserve King model internal velocities)
        
        intruder_stars.vx += approach_vel[0] 
        intruder_stars.vy += approach_vel[1]
        intruder_stars.vz += approach_vel[2]
        
        intruder_stars.add_particle(intruder_bh)
        
        # INTRUDER GALAXY GAS - Compact distribution
        intruder_gas = Particles(self.n_intruder_gas)
        r_gas_int = np.random.exponential(2.0, self.n_intruder_gas) | units.kpc
        theta_gas_int = np.random.uniform(0, np.pi, self.n_intruder_gas)
        phi_gas_int = np.random.uniform(0, 2*np.pi, self.n_intruder_gas)
        
        intruder_gas.x = r_gas_int * np.sin(theta_gas_int) * np.cos(phi_gas_int) + self.offset[0]
        intruder_gas.y = r_gas_int * np.sin(theta_gas_int) * np.sin(phi_gas_int) + self.offset[1]
        intruder_gas.z = r_gas_int * np.cos(theta_gas_int) + self.offset[2]
        intruder_gas.mass = (5e6 | units.MSun) * np.ones(self.n_intruder_gas)
        
        # Intruder gas velocities - include both internal motion and approach
        v_internal = np.sqrt(constants.G * (2e10 | units.MSun) / r_gas_int) * 0.5  # Internal rotation
        intruder_gas.vx = approach_vel[0] 
        intruder_gas.vy = approach_vel[1]
        intruder_gas.vz = approach_vel[2] 
        intruder_gas.u = (1e4 | units.K) * constants.kB / (0.6 * constants.proton_mass)
        
        # Combine particles
        self.all_stars.add_particles(target_stars)
        self.all_stars.add_particles(intruder_stars)
        self.all_gas.add_particles(target_gas)
        self.all_gas.add_particles(intruder_gas)
        
        # Keep track of which is which for plotting (account for black holes)
        self.target_star_indices = list(range(len(target_stars)))
        self.intruder_star_indices = list(range(len(target_stars), len(self.all_stars)))
        self.target_gas_indices = list(range(self.n_target_gas))
        self.intruder_gas_indices = list(range(self.n_target_gas, len(self.all_gas)))
        
        print(f"Target galaxy: {len(target_stars)} stars (including central BH)")
        print(f"Intruder galaxy: {len(intruder_stars)} stars (including central BH)")
        print(f"Total gas particles: {len(self.all_gas)}")
        
    def run_simulation(self, t_end=100|units.Myr, dt=0.5|units.Myr):
        """Run the collision simulation"""
        print(f"Starting simulation for {t_end} with timestep {dt}...")
        
        # Setup Ph4 for stars (including black holes)
        stars_gravity = Ph4(self.converter)
        stars_gravity.parameters.epsilon_squared = (0.05 | units.kpc)**2  # Smaller softening
        stars_gravity.particles.add_particles(self.all_stars)
        
        # Setup Fi for gas
        gas_hydro = Fi(self.converter)
        gas_hydro.parameters.use_hydro_flag = True
        gas_hydro.parameters.gamma = 5.0/3.0
        gas_hydro.parameters.epsilon_squared = (0.1 | units.kpc)**2
        gas_hydro.parameters.timestep = dt / 5  # Smaller hydro timestep
        gas_hydro.gas_particles.add_particles(self.all_gas)
        
        # Setup Bridge with smaller timestep
        bridge = Bridge(use_threading=False)
        bridge.add_system(stars_gravity, (gas_hydro,))
        bridge.add_system(gas_hydro, (stars_gravity,))
        bridge.timestep = dt
        
        # Store initial state
        self.store_snapshot(self.time)
        
        # Evolution loop
        current_time = self.time
        step_count = 0
        while current_time < t_end:
            bridge.evolve_model(current_time + dt)
            current_time += dt
            step_count += 1
            
            # Update particle data from codes
            stars_gravity.particles.copy_values_of_attributes_to(['x', 'y', 'z', 'vx', 'vy', 'vz'], 
                                                                self.all_stars)
            gas_hydro.gas_particles.copy_values_of_attributes_to(['x', 'y', 'z', 'vx', 'vy', 'vz'], 
                                                                self.all_gas)
            
            # Store snapshot every few steps
            if step_count % 5 == 0:
                self.store_snapshot(current_time)
                
            if step_count % 20 == 0:
                print(f"Time: {current_time}, Step: {step_count}")
                
                # Check for runaway particles
                max_r_stars = max(np.sqrt(self.all_stars.x**2 + self.all_stars.y**2 + self.all_stars.z**2))
                max_r_gas = max(np.sqrt(self.all_gas.x**2 + self.all_gas.y**2 + self.all_gas.z**2))
                print(f"  Max stellar radius: {max_r_stars.in_(units.kpc).number:.1f}")
                print(f"  Max gas radius: {max_r_gas.in_(units.kpc).number:.1f}")
                
                if max_r_stars > 200 | units.kpc:
                    print("WARNING: Stars escaping to large distances!")
        
        # Store final state
        self.store_snapshot(current_time)
        
        # Cleanup
        stars_gravity.stop()
        gas_hydro.stop()
        
        self.time = t_end
        print("Simulation complete!")
        
    def store_snapshot(self, time):
        """Store current state for animation"""
        snapshot = {
            'time': time.value_in(units.Myr),
            'target_stars_x': self.all_stars[self.target_star_indices].x.value_in(units.kpc),
            'target_stars_y': self.all_stars[self.target_star_indices].y.value_in(units.kpc),
            'target_stars_z': self.all_stars[self.target_star_indices].z.value_in(units.kpc),
            'intruder_stars_x': self.all_stars[self.intruder_star_indices].x.value_in(units.kpc),
            'intruder_stars_y': self.all_stars[self.intruder_star_indices].y.value_in(units.kpc),
            'intruder_stars_z': self.all_stars[self.intruder_star_indices].z.value_in(units.kpc),
            'target_gas_x': self.all_gas[self.target_gas_indices].x.value_in(units.kpc),
            'target_gas_y': self.all_gas[self.target_gas_indices].y.value_in(units.kpc),
            'target_gas_z': self.all_gas[self.target_gas_indices].z.value_in(units.kpc),
            'intruder_gas_x': self.all_gas[self.intruder_gas_indices].x.value_in(units.kpc),
            'intruder_gas_y': self.all_gas[self.intruder_gas_indices].y.value_in(units.kpc),
            'intruder_gas_z': self.all_gas[self.intruder_gas_indices].z.value_in(units.kpc)
        }
        self.history.append(snapshot)

    def plot_density(self, ax, data, galaxy_type='target'):
        """Create 2D density plot"""
        if galaxy_type == 'target':
            x_data = np.concatenate([data['target_stars_x'], data['target_gas_x']])
            y_data = np.concatenate([data['target_stars_y'], data['target_gas_y']])
            title = f'Target Galaxy Density (t = {data["time"]:.1f} Myr)'
        else:
            x_data = np.concatenate([data['intruder_stars_x'], data['intruder_gas_x']])
            y_data = np.concatenate([data['intruder_stars_y'], data['intruder_gas_y']])
            title = f'Intruder Galaxy Density (t = {data["time"]:.1f} Myr)'
        
        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(x_data, y_data, bins=50, range=[[-30, 30], [-30, 30]])
        
        im = ax.imshow(H.T, origin='lower', extent=[-30, 30, -30, 30], 
                      cmap='viridis', aspect='equal')
        ax.set_xlabel('X [kpc]')
        ax.set_ylabel('Y [kpc]')
        ax.set_title(title)
    
    def create_animation(self, filename="galaxy_collision.mp4", fps=10):
        """Create animation of the collision"""
        if not self.history:
            print("No simulation data! Run simulation first.")
            return
            
        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(121, projection='3d')  
        ax2 = fig.add_subplot(122)  
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            data = self.history[frame]
            
            # 3D view
            ax1.scatter(data['target_stars_x'], data['target_stars_y'], data["target_stars_z"],
                       c='blue', s=0.5, alpha=0.6, label='Target Stars')
            ax1.scatter(data['intruder_stars_x'], data['intruder_stars_y'], data['intruder_stars_z'], 
                       c='red', s=0.5, alpha=0.8, label='Intruder Stars')
            ax1.scatter(data['target_gas_x'], data['target_gas_y'], data['target_gas_z'],
                       c='cyan', s=1, alpha=0.4, label='Target Gas')
            ax1.scatter(data['intruder_gas_x'], data['intruder_gas_y'], data['intruder_gas_z'],
                       c='orange', s=1, alpha=0.6, label='Intruder Gas')
            
            ax1.set_xlim(-60, 60)
            ax1.set_ylim(-60, 60)
            ax1.set_zlim(-60, 60)
            ax1.set_xlabel('X [kpc]')
            ax1.set_ylabel('Y [kpc]')
            ax1.set_zlabel('Z [kpc]')
            ax1.set_title(f'Galaxy Collision 3D (t = {data["time"]:.1f} Myr)')
            ax1.legend(loc='upper right')

            self.plot_density(ax2, data, 'target')

            
        anim = FuncAnimation(fig, animate, frames=len(self.history), interval=100)
        
        # Save animation
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='AMUSE'), bitrate=1800)
            anim.save(filename, writer=writer)
            print(f"Animation saved as {filename}")
        except:
            print("Could not save animation - ffmpeg may not be available")
            plt.show()
        
    def plot_3d(self, save_as=None):
        """Plot current 3D state"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(self.all_stars[self.target_star_indices].x.value_in(units.kpc), 
                  self.all_stars[self.target_star_indices].y.value_in(units.kpc), 
                  self.all_stars[self.target_star_indices].z.value_in(units.kpc),
                  color="blue", alpha=0.6, s=0.5, label="Target Galaxy")
        ax.scatter(self.all_stars[self.intruder_star_indices].x.value_in(units.kpc), 
                  self.all_stars[self.intruder_star_indices].y.value_in(units.kpc), 
                  self.all_stars[self.intruder_star_indices].z.value_in(units.kpc),
                  color="red", alpha=0.8, s=0.5, label="Intruder Galaxy")
        ax.scatter(self.all_gas[self.target_gas_indices].x.value_in(units.kpc), 
                  self.all_gas[self.target_gas_indices].y.value_in(units.kpc), 
                  self.all_gas[self.target_gas_indices].z.value_in(units.kpc),
                  color="cyan", alpha=0.4, s=1)
        ax.scatter(self.all_gas[self.intruder_gas_indices].x.value_in(units.kpc), 
                  self.all_gas[self.intruder_gas_indices].y.value_in(units.kpc), 
                  self.all_gas[self.intruder_gas_indices].z.value_in(units.kpc),
                  color="orange", alpha=0.6, s=1)

        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_zlim(-50, 50)
        ax.set_xlabel('X [kpc]')
        ax.set_ylabel('Y [kpc]')
        ax.set_zlabel('Z [kpc]')
        ax.set_title('Galaxy Collision - 3D View')
        
        if save_as:
            plt.savefig(save_as)

        plt.legend()
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Create galaxy collision with fewer particles for testing
    collision = GalaxyCollision(
        n_target_stars=8000, 
        n_target_gas=1000, 
        n_intruder_stars=100, 
        n_intruder_gas=5000,
        offset=[10, 10, 50] | units.kpc
    )
    
    # Plot initial state
    print("Plotting initial state...")
    collision.plot_3d()
    
    # Run simulation
    collision.run_simulation(t_end=400|units.Myr, dt=2|units.Myr)
    
    # Create animation
    collision.create_animation("galaxy_collision.mp4")
