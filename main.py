sim = GalaxyPenetrationSimulation(
    cluster_mass=5e6|units.MSun,           
    cluster_radius=0.5|units.kpc,          
    impact_parameter=3|units.kpc,          # Distance from galaxy center at closest approach
    initial_distance=30|units.kpc,         
    velocity=400|units.km/units.s         
)

sim.run_simulation(
    total_time=12|units.Myr,    
    dt=1|units.Myr,              
    analyze_every=5              
)
