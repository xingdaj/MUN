import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy.interpolate import BSpline
from sem_waveform import SEMSimulation
from sem_waveform.mesh import create_global_mesh
from sem_waveform.velocity import build_velocity_on_sem_nodes

# Start timing
start_time = time.time()

np.random.seed(42)

# Define configuration
config = {
    'domain': {
        'xmin': -1000, 'xmax': 1000,
        'zmin': -1000, 'zmax': 1000,
        'nelem_x': 50, 'nelem_z': 50
    },
    'time': {
        'total_time': 0.40,
        'dt': 0.00008
    },
    'source': {
        'position': (0.0, 0.0),
        'frequency': 20.0,
        'amplitude': 1.0
    },
    'receivers': {
        'num_receivers': 40,
        'radius': 400.0
    },
    'method': {
        'polynomial_order': 5,
        'pml_thickness': 300.0
    },
    'velocity': {
        'inside_velocity': 2000.0,
        'outside_velocity': 3000.0,
        'control_points': np.array([[-100, -100], [100, -100], [150, -50], 
                                   [200, 0], [0, 200], [-200, 100]]),
        'perturbations': np.random.normal(loc=0.0, scale=50.0, size=(6, 2)),
        'tau': 10.0,
        'spline_samples': 800 
    },
    'output': {
        'save_wavefield': True, # False, save wavefield
        'save_seismograms': True, # False, save receiver waveform
        'visualize': True, # False, show dynamic wavefield propogation
        'output_dir': 'sem_output'
    }
}

# Run simulation
print("Starting SEM simulation...")
#sim_start_time = time.time()
sim = SEMSimulation(config)
results = sim.run()
#sim_end_time = time.time()
#print(f"SEM simulation completed in {sim_end_time - sim_start_time:.2f} seconds")

# Access results
waveforms = results['receiver_data']
dt = results['dt']
nt = results['nt']
receiver_positions = results['receiver_coords']

# Use output_dir 
output_dir = config['output']['output_dir']

# Save results in the same format 
os.makedirs(output_dir, exist_ok=True)
np.savez(os.path.join(output_dir, 'sem_forward_results.npz'),
         receiver_data=waveforms,
         dt=dt,  # Save dt as scalar
         nt=nt,  # Save nt as scalar
         receiver_coords=receiver_positions)  # Save receiver coordinates
print(f"Saved SEM results to {os.path.join(output_dir, 'sem_forward_results.npz')}")

# Plot receiver seismograms
print("Plotting receiver seismograms...")
plot_start_time = time.time()
plt.figure(figsize=(18, 12))
for i in range(len(receiver_positions)):
    plt.subplot(5, 8, i+1)
    plt.plot(np.arange(nt) * dt, waveforms[:, i], 'b-', alpha=0.7, label = f'R{i+1}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-8e-8, 8e-8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sem_seismograms.png'), dpi=150)
plt.close()
plot_end_time = time.time()
print(f"Seismograms plotted in {plot_end_time - plot_start_time:.2f} seconds")
print(f"SEM forward simulation completed. Results saved to {output_dir}/")

#========= Draw velocity_model_true.png ==============
print("Creating velocity model visualization...")
velocity_plot_start_time = time.time()

_global_coords, _global_connectivity, _node_map = create_global_mesh(
    xmin=config['domain']['xmin'],
    xmax=config['domain']['xmax'],
    zmin=config['domain']['zmin'],
    zmax=config['domain']['zmax'],
    nelem_x=config['domain']['nelem_x'],
    nelem_z=config['domain']['nelem_z'],
    npol=config['method']['polynomial_order'],
)

# Absolute control points = base control_points + perturbations (true offsets)
_ctrl6_true = np.asarray(config['velocity']['control_points'], dtype=float) + \
              np.asarray(config['velocity']['perturbations'], dtype=float)

# Evaluate velocity on SEM nodes with the spline-defined interface
_velocity_model, _signed_dist, _extras = build_velocity_on_sem_nodes(
    nodes_xy=_global_coords,
    ctrl6_xy=_ctrl6_true,
    v_inside=float(config['velocity']['inside_velocity']),
    v_outside=float(config['velocity']['outside_velocity']),
    tau = float(config['velocity']['tau']),
    samples = int(config['velocity']['spline_samples']),
    newton_steps=7,
)

# Create B-spline curve for boundary visualization 
_k = 3  # cubic B-spline degree
_ctrl_pts = np.vstack([_ctrl6_true, np.tile(_ctrl6_true[0], (_k, 1))])
_n = len(_ctrl_pts) - 1
_total_knots = _n + _k + 2
_knots = np.zeros(_total_knots)
_knots[:_k+1] = 0
_knots[-_k-1:] = 1
_inner_knots = np.linspace(0, 1, _n - _k + 2)[1:-1]
_knots[_k+1:-_k-1] = _inner_knots

_spline = BSpline(_knots, _ctrl_pts, _k, extrapolate=False)
_t_curve = np.linspace(_knots[_k], _knots[-(_k+1)], 500)
_curve_points = _spline(_t_curve)

# Save figure with improved styling 
os.makedirs(config['output']['output_dir'], exist_ok=True)
    
plt.figure(figsize=(10, 8))
_x_coords = _global_coords[:, 0]
_z_coords = _global_coords[:, 1]
_sc = plt.scatter(_x_coords, _z_coords, c=_velocity_model, cmap='seismic', s=1)
plt.colorbar(_sc, label='Velocity (m/s)')
    
# Draw B-spline boundary curve
plt.plot(_curve_points[:, 0], _curve_points[:, 1], 'r-', lw=2, label='Boundary')

# Draw receivers (circle) if present in config
if 'receivers' in config and 'num_receivers' in config['receivers']:
    _num = int(config['receivers']['num_receivers'])
    _rad = float(config['receivers']['radius'])
    _ang = np.linspace(0, 2*np.pi, _num, endpoint=False)
    _rx = _rad * np.cos(_ang)
    _rz = _rad * np.sin(_ang)
    plt.plot(_rx, _rz, 'g^', markersize=8, label='Receivers',
                markeredgecolor='black', markeredgewidth=0.5)

# Draw source
if 'source' in config and 'position' in config['source']:
    _sx, _sz = config['source']['position']
    plt.plot(_sx, _sz, 'r*', markersize=15, label='Source',
                markeredgecolor='black', markeredgewidth=1.0)

# Draw B-spline control polygon (closed) with improved styling
try:
    _ctrl_pts_true_closed = np.vstack([_ctrl6_true, _ctrl6_true[0]])
    plt.plot(_ctrl_pts_true_closed[:, 0], _ctrl_pts_true_closed[:, 1], 'ro--',
                markersize=8, label='B-spline Nodes', markeredgecolor='black', markeredgewidth=0.5)
except Exception:
    pass

# Set limits and labels
plt.xlim(config['domain']['xmin'], config['domain']['xmax'])
plt.ylim(config['domain']['zmin'], config['domain']['zmax'])
plt.xlabel('X (m)')
plt.ylabel('Z (m)')
plt.title('Velocity Model with True Boundary')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(config['output']['output_dir'], 'velocity_model_true.png'), dpi=1200)
plt.close()
velocity_plot_end_time = time.time()
print(f"Velocity model plotted in {velocity_plot_end_time - velocity_plot_start_time:.2f} seconds")
print("Saved velocity figure to", os.path.join(config['output']['output_dir'], 'velocity_model_true.png'))


#### Calculate and print total execution time
end_time = time.time()
total_time = end_time - start_time

print("\n" + "="*50)
print("EXECUTION TIME SUMMARY")
print("="*50)
print(f"Total execution time: {total_time:.2f} seconds")
print("="*50)