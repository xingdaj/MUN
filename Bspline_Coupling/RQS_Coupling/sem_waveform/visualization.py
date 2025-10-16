import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

def plot_sem_velocity_model(velocity_model, global_coords, ctrl_pts_original, true_offset, 
                           receiver_coords, src_x, src_z, vmin, vmax, title="SEM Velocity Model", out_dir="sem_output"):
    """Plot SEM velocity model with control points and boundary"""
    
    os.makedirs(out_dir, exist_ok=True)

    ck = 3  # cubic B-spline degree
    
    plt.figure(figsize=(8, 8))
    sc = plt.scatter(global_coords[:, 0], global_coords[:, 1], 
                    c=velocity_model, s=10, cmap='jet', alpha=0.8)
    plt.colorbar(sc, label='Velocity (m/s)')
    
    # Plot control points and B-spline boundary
    if true_offset is not None and ctrl_pts_original is not None:
        ctrl_pts_plot = ctrl_pts_original + true_offset
        plt.plot(np.append(ctrl_pts_plot[:, 0], ctrl_pts_plot[0, 0]), 
                 np.append(ctrl_pts_plot[:, 1], ctrl_pts_plot[0, 1]), 'ro--', 
                 markersize=8, label='Control Points')
        
        # Create and plot B-spline boundary
        ctrl_closed_plot = np.vstack([ctrl_pts_plot, np.tile(ctrl_pts_plot[0], (ck, 1))])
        n = len(ctrl_closed_plot) - 1
        total_knots = n + ck + 2
        knots = np.zeros(total_knots)
        knots[:ck+1] = 0
        knots[-ck-1:] = 1
        if n > ck:
            inner_knots = np.linspace(0, 1, n - ck + 2)[1:-1]
            knots[ck+1:-ck-1] = inner_knots
            
        spline_plot = BSpline(knots, ctrl_closed_plot, ck)
        t_curve = np.linspace(knots[ck], knots[-(ck+1)], 500)
        curve_points_plot = spline_plot(t_curve)
        plt.plot(curve_points_plot[:, 0], curve_points_plot[:, 1], 'r-', lw=2, label='B-spline Boundary')
    
    # Plot receivers and source
    plt.scatter([p[0] for p in receiver_coords], [p[1] for p in receiver_coords], 
                c='green', s=50, marker='^', edgecolor='black', label='Receivers')
    plt.scatter([src_x], [src_z], c='red', s=200, marker='*', edgecolor='black', label='Source')
    # Set the research range
    #plt.axis("tight")
    #plt.xlim(-1000, 1000)
    #plt.ylim(-1000, 1000)
    plt.title(title)
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    #plt.axis('equal')
    #plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'sem_velocity_model.png'), dpi=150)
    #plt.savefig('sem_output/sem_velocity_model.png', dpi=150)
    plt.close()

def save_animation_gif(frames, out_dir="sem_output", name="wave_propagation.gif", fps=12):
    import imageio, os
    os.makedirs(out_dir, exist_ok=True)
    if frames:
        imageio.mimsave(os.path.join(out_dir, name), frames, fps=fps)

def save_wavefield_npz(snapshots, roi_mask=None, meta=None, out_dir="sem_output", name="wavefield_snapshots.npz"):
    import numpy as np, os
    os.makedirs(out_dir, exist_ok=True)
    stack = np.stack(snapshots, axis=0) if snapshots else np.empty((0,))
    meta = meta or {}
    np.savez(os.path.join(out_dir, name), snapshots=stack, roi_mask=roi_mask, **meta)

