from amuse.units import units, constants, nbody_system
from amuse.lab import Particles
from amuse.community.fi.interface import Fi
from amuse.community.ph4.interface import ph4
from amuse.couple.bridge import Bridge
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
import time

class MilkyWay_galaxy(object):
    def __init__(self, r=5 | units.pc, ndisk=1000, nhalo=500, gas_mass=10e3 | units.MSun):
        self.r = r
        self.ndisk = ndisk
        self.nhalo = nhalo
        self.gas_mass = gas_mass
        self.stars = Particles()
        self.gas_particles = Particles()
        self.all_particles = Particles()
        self.system_time = 0 | units.yr
        self.converter = nbody_system.nbody_to_si(self.ndisk | units.MSun, self.r)
        
    def get_gravity_at_point(self, eps, x, y, z):
        phi_0 = self.get_potential_at_point(eps, x, y, z)
        dpos = 0.001*(x**2+y**2+z**2).sqrt()
        phi_dx = self.get_potential_at_point(0, x+dpos, y, z) - phi_0
        phi_dy = self.get_potential_at_point(0, x, y+dpos, z) - phi_0
        phi_dz = self.get_potential_at_point(0, x, y, z+dpos) - phi_0
        return phi_dx/dpos, phi_dy/dpos, phi_dz/dpos
    
    def disk_and_bulge_potentials(self, x, y, z, a, b, mass):
        r = (x**2+y**2).sqrt()
        return constants.G * mass / (r**2 + (a + (z**2 + b**2).sqrt())**2).sqrt()
    
    def halo_potential(self, x, y, z, Mc=5.0E+10 | units.MSun, Rc=1.0 | units.kpc**2):
        r = (x**2+y**2+z**2).sqrt()
        rr = (r/Rc)
        return (-constants.G * (Mc/Rc) * (0.5*np.log(1 + rr**2) + np.arctan(rr)/rr))
    
    def get_potential_at_point(self, eps, x, y, z):
        pot_disk = self.disk_and_bulge_potentials(
                x, y, z,
                0.0 | units.kpc, 0.277 | units.kpc, 1.12E+10 | units.MSun)
        pot_bulge = self.disk_and_bulge_potentials(
                x, y, z,
                3.7 | units.kpc, 0.20 | units.kpc, 8.07E+10 | units.MSun)
        pot_halo = self.halo_potential(
                x, y, z,
                Mc=5.0E+10 | units.MSun, Rc=6.0 | units.kpc)
        return pot_disk + pot_bulge + pot_halo
    
    def setup_initial_conditions(self):
        """Set up initial conditions for stars and gas"""
        # Create disk stars
        disk_stars = Particles(self.ndisk)
        
        # Distribute stars in a disk
        radius = np.random.power(2, self.ndisk) * (10 | units.kpc)
        theta = np.random.uniform(0, 2*np.pi, self.ndisk)
        z_height = np.random.normal(0, 0.3, self.ndisk) | units.kpc
        
        disk_stars.x = radius * np.cos(theta)
        disk_stars.y = radius * np.sin(theta)
        disk_stars.z = z_height
        
        # Set masses
        disk_stars.mass = np.random.power(1.8, self.ndisk) * (10 | units.MSun)
        
        # Set velocities (circular orbits + dispersion)
        v_circ = (constants.G * ((1.12E+10 + 8.07E+10) | units.MSun) / radius).sqrt()
        disk_stars.vx = -v_circ * np.sin(theta)
        disk_stars.vy = v_circ * np.cos(theta)
        disk_stars.vz = np.random.normal(0, 10, self.ndisk) | units.km/units.s
        
        # Create halo stars
        halo_stars = Particles(self.nhalo)
        
        # Distribute halo stars spherically
        r_halo = np.random.power(2.5, self.nhalo) * (20 | units.kpc)
        theta_halo = np.random.uniform(0, np.pi, self.nhalo)
        phi_halo = np.random.uniform(0, 2*np.pi, self.nhalo)
        
        halo_stars.x = r_halo * np.sin(theta_halo) * np.cos(phi_halo)
        halo_stars.y = r_halo * np.sin(theta_halo) * np.sin(phi_halo)
        halo_stars.z = r_halo * np.cos(theta_halo)
        
        # Set masses
        halo_stars.mass = np.random.power(1.5, self.nhalo) * (5 | units.MSun)
        
        # Set velocities (random directions, magnitude based on virial theorem)
        v_disp = (constants.G * (5.0E+10 | units.MSun) / r_halo).sqrt() * 0.7
        v_phi = np.random.uniform(0, 2*np.pi, self.nhalo)
        v_theta = np.random.uniform(0, np.pi, self.nhalo)
        
        halo_stars.vx = v_disp * np.sin(v_theta) * np.cos(v_phi)
        halo_stars.vy = v_disp * np.sin(v_theta) * np.sin(v_phi)
        halo_stars.vz = v_disp * np.cos(v_theta)
        
        # Create gas particles
        gas = Particles(self.ndisk // 2)
        
        # Distribute gas in disk
        gas_radius = np.random.power(1.8, self.ndisk // 2) * (12 | units.kpc)
        gas_theta = np.random.uniform(0, 2*np.pi, self.ndisk // 2)
        gas_z = np.random.normal(0, 0.2, self.ndisk // 2) | units.kpc
        
        gas.x = gas_radius * np.cos(gas_theta)
        gas.y = gas_radius * np.sin(gas_theta)
        gas.z = gas_z
        
        # Set gas masses
        gas.mass = self.gas_mass / (self.ndisk // 2)
        
        # Set gas velocities
        gas_v_circ = (constants.G * ((1.12E+10 + 8.07E+10) | units.MSun) / gas_radius).sqrt()
        gas.vx = -gas_v_circ * np.sin(gas_theta)
        gas.vy = gas_v_circ * np.cos(gas_theta)
        gas.vz = np.random.normal(0, 5, self.ndisk // 2) | units.km/units.s
        
        # Set internal energy for gas (temperature ~ 10^4 K)
        gas.u = (10**4 | units.K) * constants.kB / (0.6 * constants.proton_mass)
        
        # Add all particles
        self.stars.add_particles(disk_stars)
        self.stars.add_particles(halo_stars)
        self.gas_particles.add_particles(gas)
        self.all_particles.add_particles(self.stars)
        self.all_particles.add_particles(self.gas_particles)
    
    def evolve_system(self, t_end, dt=0.1|units.Myr):
        """Evolve the system using Fi for gas and Ph4 for stars"""
        
        # Make sure we have particles to evolve
        if len(self.stars) == 0 or len(self.gas_particles) == 0:
            self.setup_initial_conditions()
        
        print(f"Evolving system from {self.system_time} to {t_end}...")
        start_time = time.time()
        
        # Setup gravity codes
        stars_gravity = ph4(self.converter)
        stars_gravity.parameters.epsilon_squared = (0.01 | units.pc)**2
        stars_gravity.particles.add_particles(self.stars)
        
        # Setup hydro code
        hydro = Fi(self.converter)
        hydro.parameters.use_hydro_flag = True
        hydro.parameters.isothermal_flag = False
        hydro.parameters.gamma = 5.0/3.0
        hydro.parameters.epsilon_squared = (0.1 | units.pc)**2
        hydro.parameters.timestep = dt / 2
        hydro.gas_particles.add_particles(self.gas_particles)
        
        # Create a gravity field from the external potential
        class GalaxyPotential:
            def __init__(self, galaxy_model):
                self.galaxy_model = galaxy_model
                
            def get_gravity_at_point(self, eps, x, y, z):
                return self.galaxy_model.get_gravity_at_point(eps, x, y, z)
            
            def get_potential_at_point(self, eps, x, y, z):
                return self.galaxy_model.get_potential_at_point(eps, x, y, z)
        
        # Create bridge to couple everything
        bridge = Bridge(use_threading=False)
        galaxy_potential = GalaxyPotential(self)
        
        # Add everything to the bridge
        bridge.add_system(stars_gravity, (hydro, galaxy_potential))
        bridge.add_system(hydro, (stars_gravity, galaxy_potential))
        
        # Set timestep
        bridge.timestep = dt
        
        # Evolve to desired time
        times = []
        potential_energies = []
        kinetic_energies = []
        
        # Add perturbation to break symmetry and ensure dynamics
        """
        commented out for now because p.vx is not a thing here and i should just
        change the whole vector
        for p in self.stars:
            p.vx += np.random.normal(0, 5) | units.km/units.s
            p.vy += np.random.normal(0, 5) | units.km/units.s
        
        for p in self.gas_particles:
            p.vx += np.random.normal(0, 7) | units.km/units.s
            p.vy += np.random.normal(0, 7) | units.km/units.s
            p.u *= (1.0 + 0.2 * np.random.random())  # Add energy perturbation
        """
        
        # Update the particle sets in the codes after perturbation
        channel_to_stars = stars_gravity.particles.new_channel_to(self.stars)
        channel_from_stars = self.stars.new_channel_to(stars_gravity.particles)
        channel_from_stars.copy()
        
        channel_to_gas = hydro.gas_particles.new_channel_to(self.gas_particles)
        channel_from_gas = self.gas_particles.new_channel_to(hydro.gas_particles)
        channel_from_gas.copy()
        
        current_time = self.system_time
        while current_time < t_end:
            target_time = current_time + dt
            bridge.evolve_model(target_time)
            current_time = target_time
            
            # Copy back the particles using channels for efficiency
            channel_to_stars.copy()
            channel_to_gas.copy()
            
            # Record energies
            potential_energy = stars_gravity.potential_energy + hydro.potential_energy
            kinetic_energy = stars_gravity.kinetic_energy + hydro.kinetic_energy
            
            times.append(current_time)
            potential_energies.append(potential_energy)
            kinetic_energies.append(kinetic_energy)
            
            print(f"Time: {current_time.in_(units.Myr)}, PE: {potential_energy}, KE: {kinetic_energy}")
        
        # Update system time
        self.system_time = t_end
        
        # Clean up
        stars_gravity.stop()
        hydro.stop()
        
        end_time = time.time()
        print(f"Evolution completed in {end_time - start_time:.2f} seconds")
        
        return times, potential_energies, kinetic_energies
    
    def overview(self, show=True, save=False, filename="galaxy_snapshot.png"):
        """Plot the current state of the system"""
        rcParams.update({'font.size': 12})
        fig = plt.figure(figsize=(16, 10))
        
        # Face-on view
        ax1 = fig.add_subplot(221)
        ax1.plot(self.stars.x.value_in(units.kpc), 
                 self.stars.y.value_in(units.kpc), 'b.', alpha=0.3, ms=2, label='Stars')
        ax1.plot(self.gas_particles.x.value_in(units.kpc), 
                 self.gas_particles.y.value_in(units.kpc), 'r.', alpha=0.8, ms=3, label='Gas')
        ax1.set_xlim(-25, 25)
        ax1.set_ylim(-25, 25)
        ax1.set_xlabel('X [kpc]')
        ax1.set_ylabel('Y [kpc]')
        ax1.set_title('Face-on view')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Edge-on view
        ax2 = fig.add_subplot(222)
        ax2.plot(self.stars.x.value_in(units.kpc), 
                 self.stars.z.value_in(units.kpc), 'b.', alpha=0.3, ms=2)
        ax2.plot(self.gas_particles.x.value_in(units.kpc), 
                 self.gas_particles.z.value_in(units.kpc), 'r.', alpha=0.8, ms=3)
        ax2.set_xlim(-25, 25)
        ax2.set_ylim(-10, 10)
        ax2.set_xlabel('X [kpc]')
        ax2.set_ylabel('Z [kpc]')
        ax2.set_title('Edge-on view')
        ax2.grid(True, alpha=0.3)
        
        # 3D view
        ax3 = fig.add_subplot(223, projection='3d')
        ax3.scatter(self.stars.x.value_in(units.kpc), 
                    self.stars.y.value_in(units.kpc),
                    self.stars.z.value_in(units.kpc),
                    c='blue', alpha=0.2, s=1)
        ax3.scatter(self.gas_particles.x.value_in(units.kpc), 
                    self.gas_particles.y.value_in(units.kpc),
                    self.gas_particles.z.value_in(units.kpc),
                    c='red', alpha=0.6, s=2)
        ax3.set_xlim(-20, 20)
        ax3.set_ylim(-20, 20)
        ax3.set_zlim(-10, 10)
        ax3.set_xlabel('X [kpc]')
        ax3.set_ylabel('Y [kpc]')
        ax3.set_zlabel('Z [kpc]')
        ax3.set_title('3D view')
        
        # Radial distribution
        ax4 = fig.add_subplot(224)
        r_stars = np.sqrt(self.stars.x.value_in(units.kpc)**2 + 
                          self.stars.y.value_in(units.kpc)**2)
        r_gas = np.sqrt(self.gas_particles.x.value_in(units.kpc)**2 + 
                        self.gas_particles.y.value_in(units.kpc)**2)
        
        bins = np.linspace(0, 25, 50)
        ax4.hist(r_stars, bins=bins, alpha=0.5, label='Stars', density=True)
        ax4.hist(r_gas, bins=bins, alpha=0.5, label='Gas', density=True)
        ax4.set_xlabel('Radial Distance [kpc]')
        ax4.set_ylabel('Normalized Density')
        ax4.set_title('Radial Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(f'Galaxy Model at T = {self.system_time.in_(units.Myr).number:.1f} Myr', 
                     fontsize=16, y=0.99)
        
        if save:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_animation(self, t_end, frames=20, filename="galaxy_evolution.mp4"):
        """Create an animation of the galaxy evolution"""
        from matplotlib.animation import FuncAnimation
        import matplotlib.animation as animation

        dt = (t_end.number / frames) t_end.unit
        
        # Make sure we have particles to evolve
        if len(self.stars) == 0 or len(self.gas_particles) == 0:
            self.setup_initial_conditions()
        
        # Create figure for animation
        fig = plt.figure(figsize=(12, 10))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223, projection='3d')
        ax4 = fig.add_subplot(224)
        
        # Setup axes
        ax1.set_xlim(-25, 25)
        ax1.set_ylim(-25, 25)
        ax1.set_xlabel('X [kpc]')
        ax1.set_ylabel('Y [kpc]')
        ax1.set_title('Face-on view')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlim(-25, 25)
        ax2.set_ylim(-10, 10)
        ax2.set_xlabel('X [kpc]')
        ax2.set_ylabel('Z [kpc]')
        ax2.set_title('Edge-on view')
        ax2.grid(True, alpha=0.3)
        
        ax3.set_xlim(-20, 20)
        ax3.set_ylim(-20, 20)
        ax3.set_zlim(-10, 10)
        ax3.set_xlabel('X [kpc]')
        ax3.set_ylabel('Y [kpc]')
        ax3.set_zlabel('Z [kpc]')
        ax3.set_title('3D view')
        
        ax4.set_xlabel('Time [Myr]')
        ax4.set_ylabel('Energy')
        ax4.set_title('Energy Evolution')
        ax4.grid(True, alpha=0.3)
        
        title = fig.suptitle('', fontsize=16, y=0.99)
        plt.tight_layout()
        
        # Reset system to initial state and add perturbation
        original_time = self.system_time
        self.system_time = 0 | units.yr
        
        # Store initial positions for comparison
        initial_stars_x = self.stars.x.copy()
        initial_stars_y = self.stars.y.copy()
        initial_gas_x = self.gas_particles.x.copy()
        initial_gas_y = self.gas_particles.y.copy()
        
        # Set up codes outside the animation loop
        stars_gravity = ph4(self.converter)
        stars_gravity.parameters.epsilon_squared = (0.01 | units.pc)**2
        stars_gravity.particles.add_particles(self.stars)
        
        hydro = Fi(self.converter)
        hydro.parameters.use_hydro_flag = True
        hydro.parameters.isothermal_flag = False
        hydro.parameters.gamma = 5.0/3.0
        hydro.parameters.epsilon_squared = (0.1 | units.pc)**2
        hydro.parameters.timestep = dt / 2
        hydro.gas_particles.add_particles(self.gas_particles)
        """
        # Add perturbation to break symmetry
        for p in stars_gravity.particles:
            p.vx += np.random.normal(0, 10) | units.km/units.s
            p.vy += np.random.normal(0, 10) | units.km/units.s
        
        for p in hydro.gas_particles:
            p.vx += np.random.normal(0, 15) | units.km/units.s
            p.vy += np.random.normal(0, 15) | units.km/units.s
            p.u *= (1.0 + 0.3 * np.random.random())
            """
        
        # Create channels for efficient data transfer
        channel_to_stars = stars_gravity.particles.new_channel_to(self.stars)
        channel_to_gas = hydro.gas_particles.new_channel_to(self.gas_particles)
        
        # Create galaxy potential
        class GalaxyPotential:
            def __init__(self, galaxy_model):
                self.galaxy_model = galaxy_model
                
            def get_gravity_at_point(self, eps, x, y, z):
                return self.galaxy_model.get_gravity_at_point(eps, x, y, z)
            
            def get_potential_at_point(self, eps, x, y, z):
                return self.galaxy_model.get_potential_at_point(eps, x, y, z)
        
        # Create bridge
        bridge = Bridge(use_threading=False)
        galaxy_potential = GalaxyPotential(self)
        
        bridge.add_system(stars_gravity, (hydro, galaxy_potential))
        bridge.add_system(hydro, (stars_gravity, galaxy_potential))
        bridge.timestep = dt
        
        # Store time and energy arrays for plotting
        sim_times = []
        ke_values = []
        pe_values = []
        
        # Calculate time step for each frame
        time_per_frame = t_end / frames
        
        def animate(i):
            # Target time for this frame
            target_time = (i + 1) * time_per_frame
            
            # Run evolution until we reach the target time
            while self.system_time < target_time:
                step_dt = min(dt, target_time - self.system_time)
                bridge.evolve_model(self.system_time + step_dt)
                self.system_time += step_dt
                
                # Update particle data
                channel_to_stars.copy()
                channel_to_gas.copy()
                
                # Record energy data
                ke = stars_gravity.kinetic_energy + hydro.kinetic_energy
                pe = stars_gravity.potential_energy + hydro.potential_energy
                sim_times.append(self.system_time)
                ke_values.append(ke)
                pe_values.append(pe)
            
            print(f"Frame {i+1}/{frames}, Time: {self.system_time.in_(units.Myr)}")
            
            # Clear all subplots for updating
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            
            # Face-on view
            ax1.plot(self.stars.x.value_in(units.kpc), 
                     self.stars.y.value_in(units.kpc), 'b.', alpha=0.3, ms=2)
            ax1.plot(self.gas_particles.x.value_in(units.kpc), 
                     self.gas_particles.y.value_in(units.kpc), 'r.', alpha=0.8, ms=3)
            # Plot initial positions with lighter color for comparison
            ax1.plot(initial_stars_x.value_in(units.kpc), 
                     initial_stars_y.value_in(units.kpc), 'c.', alpha=0.1, ms=1)
            ax1.plot(initial_gas_x.value_in(units.kpc), 
                     initial_gas_y.value_in(units.kpc), 'm.', alpha=0.1, ms=1)
            ax1.set_xlim(-25, 25)
            ax1.set_ylim(-25, 25)
            ax1.set_xlabel('X [kpc]')
            ax1.set_ylabel('Y [kpc]')
            ax1.set_title('Face-on view')
            ax1.grid(True, alpha=0.3)
            
            # Edge-on view
            ax2.plot(self.stars.x.value_in(units.kpc), 
                     self.stars.z.value_in(units.kpc), 'b.', alpha=0.3, ms=2)
            ax2.plot(self.gas_particles.x.value_in(units.kpc), 
                     self.gas_particles.z.value_in(units.kpc), 'r.', alpha=0.8, ms=3)
            ax2.set_xlim(-25, 25)
            ax2.set_ylim(-10, 10)
            ax2.set_xlabel('X [kpc]')
            ax2.set_ylabel('Z [kpc]')
            ax2.set_title('Edge-on view')
            ax2.grid(True, alpha=0.3)
            
            # 3D view
            ax3.scatter(self.stars.x.value_in(units.kpc), 
                        self.stars.y.value_in(units.kpc),
                        self.stars.z.value_in(units.kpc),
                        c='blue', alpha=0.2, s=1)
            ax3.scatter(self.gas_particles.x.value_in(units.kpc), 
                        self.gas_particles.y.value_in(units.kpc),
                        self.gas_particles.z.value_in(units.kpc),
                        c='red', alpha=0.6, s=2)
            ax3.set_xlim(-20, 20)
            ax3.set_ylim(-20, 20)
            ax3.set_zlim(-10, 10)
            ax3.set_xlabel('X [kpc]')
            ax3.set_ylabel('Y [kpc]')
            ax3.set_zlabel('Z [kpc]')
            
            # Energy plot
            if sim_times:
                times_myr = [t.value_in(units.Myr) for t in sim_times]
                ke_plot = [e.value_in(units.J) for e in ke_values]
                pe_plot = [e.value_in(units.J) for e in pe_values]
                total_e = [k + p for k, p in zip(ke_plot, pe_plot)]
                
                ax4.plot(times_myr, ke_plot, 'r-', label='Kinetic')
                ax4.plot(times_myr, pe_plot, 'b-', label='Potential')
                ax4.plot(times_myr, total_e, 'g-', label='Total')
                ax4.set_xlabel('Time [Myr]')
                ax4.set_ylabel('Energy [J]')
                ax4.set_title('Energy Evolution')
                ax4.legend(loc='best')
                ax4.grid(True, alpha=0.3)
            
            title.set_text(f'Galaxy Model at T = {self.system_time.in_(units.Myr).number:.1f} Myr')
            
            # Keep layout clean
            plt.tight_layout()
            fig.subplots_adjust(top=0.9)
            
            return ax1, ax2, ax3, ax4, title
        
        anim = FuncAnimation(fig, animate, frames=frames, blit=False, interval=200)
        
        # Save animation
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='AMUSE'), bitrate=1800)
        anim.save(filename, writer=writer)
        
        # Clean up
        stars_gravity.stop()
        hydro.stop()
        plt.close()
        
        # Restore original time
        self.system_time = original_time
        
        print(f"Animation saved to {filename}")
        
    def plot_energy_evolution(self, times, ke, pe, show=True, save=False, filename="energy_evolution.png"):
        """Plot the energy evolution of the system"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        times_myr = [t.value_in(units.Myr) for t in times]
        ke_values = [e.value_in(units.J) for e in ke]
        pe_values = [e.value_in(units.J) for e in pe]
        total_energy = [k + p for k, p in zip(ke_values, pe_values)]
        
        ax.plot(times_myr, ke_values, 'r-', label='Kinetic Energy')
        ax.plot(times_myr, pe_values, 'b-', label='Potential Energy')
        ax.plot(times_myr, total_energy, 'g-', label='Total Energy')
        
        ax.set_xlabel('Time [Myr]')
        ax.set_ylabel('Energy [J]')
        ax.set_title('Energy Evolution of Galaxy Model')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save:
            plt.savefig(filename, dpi=300)
        
        if show:
            plt.show()
        else:
            plt.close()
