import time
import os
import warnings
import numpy as np
import scipy.sparse as sp
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, griddata, interp1d, CubicSpline
from scipy.spatial import cKDTree
import imageio

from .mesh import create_global_mesh, find_closest_node
from .operators import (
    precompute_sem_operators,
    assemble_mass_matrix_lumped_acoustic,
    assemble_stiffness_matrix_acoustic,
    stiffness_matrix_vector_product,
    stiffness_matrix_vector_product_torch,
    accumulate_stiffness_sensitivity_wrt_velocity,
    assemble_mass_matrix_lumped_torch,
    assemble_stiffness_matrix_torch,
)
from .sources import create_source_wavelet, ricker_wavelet
from .receivers import setup_receivers, interpolate_to_receivers_sem, interpolate_to_receivers_torch
from .boundary import (
    setup_pml,
    apply_precomputed_pml_all_layers,
    apply_pml_torch,
    apply_pml_torch_optimized,
)
from .velocity import build_velocity_on_sem_nodes
from .utils import check_cfl_condition, adjust_time_step_for_stability
from .visualization import plot_sem_velocity_model


warnings.filterwarnings(
    "ignore",
    message="Sparse CSR tensor support is in beta state.*"
)

# ========================================================
# Torch helper functions and tensorized sensitivity kernel
# ========================================================
def _to_torch64(x):
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float64, device='cpu')
    return torch.from_numpy(np.asarray(x)).to(dtype=torch.float64, device='cpu')

@torch.no_grad()
def accumulate_sensitivity_dJdc_torch(
    u_all, lam_all, connectivity_t,
    Gx_t, Gz_t, S_interp_t, w2D_t,
    c_all, npol, npoints
):
    """
    Compute incremental gradient dJ/dc for one time step dJ/dc:
      1) c^2 at nodes -> quadrature points:c2_q = S_interp @ (c_e^2)
      2) weight:w_c = w2D * c2_q
      3) g_q = w_c * (∇u·∇λ)
      4) back-project to nodes:dJ/d(c^2)_local = S_interp^T @ g_q
      5) chain rule:dJ/dc = 2*c * dJ/d(c^2)
       torch.float64.
    """
    device = u_all.device
    nelem, ng = connectivity_t.shape

    gather_idx = connectivity_t.reshape(-1)
    u_loc  = u_all.index_select(0, gather_idx).reshape(nelem, ng)
    l_loc  = lam_all.index_select(0, gather_idx).reshape(nelem, ng)
    c_loc  = c_all.index_select(0, gather_idx).reshape(nelem, ng)

    ux = torch.einsum('ab,ib->ia', Gx_t, u_loc)
    uz = torch.einsum('ab,ib->ia', Gz_t, u_loc)
    lx = torch.einsum('ab,ib->ia', Gx_t, l_loc)
    lz = torch.einsum('ab,ib->ia', Gz_t, l_loc)

    c2_q = torch.einsum('ab,ib->ia', S_interp_t, c_loc * c_loc)
    g_q  = w2D_t * (ux * lx + uz * lz)

    dJ_dc2_local = torch.einsum('ba,ib->ia', S_interp_t, g_q)
    dJ_dc2 = torch.zeros(npoints, dtype=torch.float64, device=device)
    dJ_dc2.scatter_add_(0, gather_idx, dJ_dc2_local.reshape(-1))

    dJ_dc = 2.0 * c_all * dJ_dc2
    return dJ_dc


class SEMSimulation:
    def __init__(self, config):
        """
        Initialize SEM simulation with configuration parameters.
        """
        self.config = config
        self._validate_config()
        self._initialize_components()

        # Visualization parameters
        self.visualize = self.config['output'].get('visualize', False)
        self.frames = []

        # === Adjoint/History precision switch ===
        # 'float64 (default) or float32 (to save memory)
        self.adj_history_dtype = (
            self.config.get('method', {}).get('adj_history_dtype', 'float64')
        )
        if self.adj_history_dtype not in ('float32', 'float64'):
            self.adj_history_dtype = 'float64'

    def _align_observation_data(self, y_obs, obs_dt, obs_nt):
        """Resample observed data y_obs (obs_dt/obs_nt) onto the current SEM grid (self.dt/self.nt)."""
        src_time = np.arange(obs_nt, dtype=np.float64) * float(obs_dt)
        dst_time = np.arange(self.nt,  dtype=np.float64) * float(self.dt)

        y_obs = np.asarray(y_obs, dtype=np.float64)
        if y_obs.shape[0] > obs_nt:
            y_obs = y_obs[:obs_nt, :]

        if (y_obs.shape[0] == self.nt and
            np.allclose(src_time[:self.nt], dst_time, atol=1e-12, rtol=0.0)):
            return y_obs

        nrec = y_obs.shape[1]
        y_aligned = np.zeros((self.nt, nrec), dtype=np.float64)
        for i in range(nrec):
            f = interp1d(src_time, y_obs[:, i], kind="linear",
                         bounds_error=False, fill_value=0.0, assume_sorted=True)
            y_aligned[:, i] = f(dst_time)
        return y_aligned

    def _validate_config(self):
        """Validate configuration parameters"""
        required_keys = ['domain', 'time', 'source', 'receivers', 'method', 'velocity', 'output']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def _ensure_sem_ops_torch(self):
        """Ensure SEM operators are available as PyTorch tensors for adjoint computation"""
        if not hasattr(self, 'Gx_global_t') or not hasattr(self, 'connectivity_t'):
            self.Gx_global_t = torch.from_numpy(self.Gx_global).double()
            self.Gz_global_t = torch.from_numpy(self.Gz_global).double()
            self.w2D_global_t = torch.from_numpy(self.w2D_global).double()
            self.S_interp_global_t = torch.from_numpy(self.S_interp_global).double()

            _conn = np.asarray(self.global_connectivity, dtype=np.int64)
            self.connectivity_t = torch.from_numpy(_conn)

            if not hasattr(self, 'velocity_model_torch'):
                self.velocity_model_torch = torch.from_numpy(self.velocity_model).double()

    def _initialize_components(self):
        """Initialize all SEM components (CFL-adjusted at init; rebuild source & PML after dt update)."""
        # -------- 1) read config --------
        domain          = self.config['domain']
        time_params     = self.config['time']
        source_params   = self.config['source']
        receiver_params = self.config['receivers']
        method_params   = self.config['method']
        velocity_params = self.config['velocity']

        # -------- 2) basic scalars from config --------
        # domain / mesh
        self.xmin, self.xmax = float(domain['xmin']), float(domain['xmax'])
        self.zmin, self.zmax = float(domain['zmin']), float(domain['zmax'])
        self.nelem_x, self.nelem_z = int(domain['nelem_x']), int(domain['nelem_z'])
        self.npol = int(method_params['polynomial_order'])
        self.ngll = self.npol + 1

        # ===== Debug print: check domain parameters =====
        #print("==== Domain / Mesh Parameters ====")
        #print(f"xmin = {self.xmin}, xmax = {self.xmax}")
        #print(f"zmin = {self.zmin}, zmax = {self.zmax}")
        #print(f"nelem_x = {self.nelem_x}, nelem_z = {self.nelem_z}")
        #print(f"npol = {self.npol}, ngll = {self.ngll}")

        # time (raw, to be CFL-adjusted below)
        self.total_time = float(time_params['total_time'])
        # dt comes from config, but only as a candidate; it will be adjusted by CFL below
        self.dt = float(time_params['dt'])
        self.nt = int(np.ceil(self.total_time / self.dt))
        self.dt2 = self.dt * self.dt
        # ===== Debug print: check time parameters =====
        #print("==== Time Parameters ====")
        #print(f"total_time = {self.total_time}")
        #print(f"initial dt = {self.dt}, nt = {self.nt}")

        # source
        self.src_x, self.src_z = [float(v) for v in source_params['position']]
        self.src_f0 = float(source_params['frequency'])
        self.src_amp = float(source_params['amplitude'])

        # ===== Debug print: check source parameters =====
        #print("==== Source Parameters ====")
        #print(f"src_x = {self.src_x}, src_z = {self.src_z}")
        #print(f"src_f0 = {self.src_f0}, src_amp = {self.src_amp}")

        # receivers
        self.num_receivers   = int(receiver_params['num_receivers'])
        self.receiver_radius = float(receiver_params['radius'])

        #print("==== Receiver Parameters ====")
        #print(f"num_receivers = {self.num_receivers}, receiver_radius = {self.receiver_radius}")

        # velocity (inside/outside, ctrl points, optional “true” offset)
        self.vmin = float(velocity_params['inside_velocity'])
        self.vmax = float(velocity_params['outside_velocity'])
        self.ctrl_pts_original = np.asarray(velocity_params.get('control_points', None), dtype=np.float64)
        self.true_offset       = None if velocity_params.get('perturbations', None) is None else \
                                 np.asarray(velocity_params['perturbations'], dtype=np.float64)

        #print("==== Velocity Parameters ====")
        #print(f"vmin = {self.vmin}, vmax = {self.vmax}")
        #if self.ctrl_pts_original is not None:
        #    print(f"Control points shape: {self.ctrl_pts_original.shape}")
        #if self.true_offset is not None:
        #    print(f"True offset shape: {self.true_offset.shape}")

        # output
        self.output_dir       = self.config['output'].get('output_dir', 'sem_output')
        self.save_wavefield   = bool(self.config['output'].get('save_wavefield', False))
        self.save_seismograms = bool(self.config['output'].get('save_seismograms', True))
        self.visualize        = bool(self.config['output'].get('visualize', False))

        # adjoint/history precision (default float64)
        self.adj_history_dtype = torch.float64

        # -------- 3) build mesh --------
        self.global_coords, self.global_connectivity, self.node_map = create_global_mesh(
            self.xmin, self.xmax, self.zmin, self.zmax, self.nelem_x, self.nelem_z, self.npol
        )
        self.global_connectivity = np.asarray(self.global_connectivity, dtype=np.int64)
        self.npoints = int(len(self.global_coords))

        # element size
        self.Lx = self.xmax - self.xmin
        self.Lz = self.zmax - self.zmin
        self.dx_elem = self.Lx / self.nelem_x
        self.dz_elem = self.Lz / self.nelem_z

        # -------- 4) initial velocity model (for CFL check at init) --------
        # Use "base control points + (if provided) true offset" to build an initial velocity
        # for CFL; this aligns better with the velocity used later in the real forward pass.
        if self.ctrl_pts_original is not None:
            if self.true_offset is not None:
                ctrl6 = (self.ctrl_pts_original + self.true_offset).reshape(6, 2)
            else:
                ctrl6 = self.ctrl_pts_original.reshape(6, 2)

            tau_cfg = float(self.config['velocity']['tau'])
            spline_samples = int(self.config['velocity']['spline_samples'])

            # ===== Debug print: check velocity spline parameters =====
            #print("==== Velocity Spline Parameters ====")
            #print(f"tau_cfg = {tau_cfg}")
            #print(f"spline_samples = {spline_samples}")

            velocity_model_init, signed_dist_init, _extras = build_velocity_on_sem_nodes(
                nodes_xy=self.global_coords,
                ctrl6_xy=ctrl6,
                v_inside=self.vmin,
                v_outside=self.vmax,
                tau=tau_cfg,
                samples=spline_samples,
                newton_steps=7,
            )
        else:
            # If no spline control points are given, fall back to a constant-velocity model
            # (choose vmax); for CFL we only need the upper bound.
            velocity_model_init = np.full((self.npoints,), self.vmax, dtype=np.float64)

        self.velocity_model = velocity_model_init  # Will be overwritten/updated later in the real forward

        # -------- 5) CFL: adjust dt/nt BEFORE building time array & source --------
        self.dt, self.nt = adjust_time_step_for_stability(
            self.velocity_model, self.dt, self.total_time,
            self.npol, self.dx_elem, self.dz_elem, self.vmax
        )
        self.dt2 = self.dt * self.dt

        # -------- 6) time array & source (built with the *final* dt/nt) --------
        self.time_array = np.arange(self.nt, dtype=np.float64) * self.dt
        # Ricker wavelet with DC removal & gating
        self.src_wavelet = create_source_wavelet(
            nt=self.nt, dt=self.dt, f0=self.src_f0, t0=0.1, gate_time=0.3, amp=self.src_amp
        )
        self.src_wavelet_torch = torch.from_numpy(self.src_wavelet).to(dtype=torch.float64)

        # -------- 7) locate source node --------
        self.src_node = find_closest_node(self.src_x, self.src_z, self.global_coords)
        self.src_pos  = self.global_coords[self.src_node]

        # -------- 8) receivers & interpolation operator --------
        self.receiver_coords_sem, self.receiver_interp_info_sem, self.R_sparse, self.R_torch = setup_receivers(
            self.src_x, self.src_z, self.receiver_radius, self.num_receivers,
            self.xmin, self.zmin, self.dx_elem, self.dz_elem,
            self.nelem_x, self.nelem_z,
            self.global_connectivity, self.global_coords, self.npol
        )

        # -------- 9) precompute SEM operators --------
        self.Gx_global, self.Gz_global, self.S_interp_global, self.w2D_global = precompute_sem_operators(
            self.dx_elem, self.dz_elem, self.npol
        )

        # -------- 10) PML with the *final* dt (IMPORTANT!) --------
        pml_thickness = self.config['method']['pml_thickness']
        # ===== Debug print: check PML parameters =====
        #print("==== PML Parameters ====")
        #print(f"pml_thickness = {pml_thickness}")

        # NOTE: Keep consistent with core0.py: do not pass c_max here.
        self.pml_indices_sem, self.pml_mask_sem, self.decay_sem_torch = setup_pml(
            self.xmin, self.xmax, self.zmin, self.zmax,
            pml_thickness, self.global_coords, self.dt, c_max=self.vmax
        )
        self.decay_sem = self.decay_sem_torch.numpy()

        # -------- 11) ROI mask (for viz/debug) --------
        self.roi_xmin = self.config['output'].get('roi_xmin', self.xmin + 100.0)
        self.roi_xmax = self.config['output'].get('roi_xmax', self.xmax - 100.0)
        self.roi_zmin = self.config['output'].get('roi_zmin', self.zmin + 100.0)
        self.roi_zmax = self.config['output'].get('roi_zmax', self.zmax - 100.0)
        x, z = self.global_coords[:, 0], self.global_coords[:, 1]
        self.roi_mask = ((x >= self.roi_xmin) & (x <= self.roi_xmax) &
                         (z >= self.roi_zmin) & (z <= self.roi_zmax))

    def _create_roi_mask(self):
        x, z = self.global_coords[:, 0], self.global_coords[:, 1]
        roi_mask = ((x >= self.roi_xmin) & (x <= self.roi_xmax) &
                    (z >= self.roi_zmin) & (z <= self.roi_zmax))
        #print(f"ROI mask: {roi_mask.sum()}/{len(roi_mask)} nodes in ROI")
        return roi_mask

    def run(self, ctrl_offsets_flat=None, use_torch=True):
        """
        Run SEM forward simulation.
        """
        #print(f"\n=== Starting SEM Forward Simulation ({'PyTorch' if use_torch else 'NumPy'}) ===")
        start_time = time.perf_counter()

        # Build velocity model FIRST (needed for CFL check) 
        if ctrl_offsets_flat is None and self.true_offset is not None:
            ctrl_offsets_flat = torch.from_numpy(self.true_offset.flatten()).double()

        # offset(12,) -> absolute control points(6,2)
        if isinstance(ctrl_offsets_flat, torch.Tensor):
            perts_np = ctrl_offsets_flat.detach().cpu().numpy().reshape(6, 2)
        else:
            perts_np = np.asarray(ctrl_offsets_flat, dtype=np.float64).reshape(6, 2)
        ctrl6_abs = self.ctrl_pts_original + perts_np  # absolute control points

        # Unify geometry to 'true spline + Newton nearest point' — call velocity API
        vel_cfg = self.config['velocity']
        #tau     = float(vel_cfg.get('tau', 10.0))
        tau     = float(vel_cfg['tau'])
        #samples = int(vel_cfg.get('spline_samples', 1200))
        samples = int(vel_cfg['spline_samples'])

        # ===== Debug print: check velocity spline parameters in run() =====
        #print("==== Velocity Spline Parameters (run) ====")
        #print(f"tau = {tau}")
        #print(f"spline_samples = {samples}")

        self.velocity_model, self.signed_dist, self.spline_extras = build_velocity_on_sem_nodes(
            nodes_xy   = self.global_coords,     # (N,2)
            ctrl6_xy   = ctrl6_abs,              # (6,2)
            v_inside   = float(self.vmin),
            v_outside  = float(self.vmax),
            tau        = tau,
            samples    = samples,
            newton_steps = 7
        )

        # Adjust dt/nt for stability (CFL) and recompute PML decay with current dt
        self.dt, self.nt = adjust_time_step_for_stability(
            self.velocity_model, self.dt, self.total_time,
            self.npol, self.dx_elem, self.dz_elem, self.vmax
        )
        self.dt2 = self.dt * self.dt

        #pml_thickness = self.config['method'].get('pml_thickness', 300.0)
        pml_thickness = self.config['method']['pml_thickness']
        #print(f"pml_thickness = {pml_thickness}")

        self.pml_indices_sem, self.pml_mask_sem, self.decay_sem_torch = setup_pml(
            self.xmin, self.xmax, self.zmin, self.zmax,
            pml_thickness, self.global_coords, self.dt, c_max=self.vmax 
        )
        self.decay_sem = self.decay_sem_torch.numpy()

        # Time axis and source
        self.time_array = np.arange(self.nt, dtype=np.float64) * self.dt
        self.src_wavelet = create_source_wavelet(
            self.nt, self.dt, f0=self.src_f0, t0=0.1, gate_time=0.3, amp=self.src_amp
        )
        if use_torch:
            self.src_wavelet_torch = torch.from_numpy(self.src_wavelet).double()

        # Snapshot interval
        self.snapshot_interval = self.config['output'].get('snapshot_interval',
                                                          max(200, self.nt // 100))

        if use_torch:
            # Torch tensors
            self.velocity_model_torch = torch.from_numpy(self.velocity_model).double()
            self.global_coords_torch = torch.from_numpy(self.global_coords).double()
            self.dt_torch = torch.tensor(self.dt, dtype=torch.float64)
            self.dt2_torch = torch.tensor(self.dt2, dtype=torch.float64)

            #print("Assembling SEM matrices (PyTorch)...")
            self.M_diag_torch = assemble_mass_matrix_lumped_torch(
                self.global_connectivity, self.dx_elem, self.dz_elem, self.npol, self.npoints
            )
            self.Minv_diag_torch = 1.0 / self.M_diag_torch

            self.u_prev_torch = torch.zeros(self.npoints, dtype=torch.float64)
            self.u_curr_torch = torch.zeros(self.npoints, dtype=torch.float64)
            self.u_next_torch = torch.zeros(self.npoints, dtype=torch.float64)

            self.K_u_torch = torch.zeros(self.npoints, dtype=torch.float64)
            self.rhs_torch = torch.zeros(self.npoints, dtype=torch.float64)

            self.pml_mask_sem_torch = torch.from_numpy(self.pml_mask_sem)
            self.decay_sem_torch = torch.from_numpy(self.decay_sem).double()
            self.roi_mask_torch = torch.from_numpy(self.roi_mask)

            self._ensure_sem_ops_torch()
            results = self._run_torch_version()
        else:
            results = self._run_numpy_version()

        end_time = time.perf_counter()
        comp_time = end_time - start_time
        #print(f"SEM simulation completed in {comp_time:.2f} seconds")
        return results

    def _run_numpy_version(self):
        """Run SEM simulation using NumPy"""
        print("Assembling SEM matrices (NumPy)...")
        self.M = assemble_mass_matrix_lumped_acoustic(
            self.velocity_model, self.global_connectivity,
            self.dx_elem, self.dz_elem, self.npol, self.npoints
        )
        self.M_diag = self.M.diagonal().copy().astype(np.float64)
        self.Minv_diag = (1.0 / self.M_diag).astype(np.float64)
        self.Minv_dt2 = (self.dt2 * self.Minv_diag).astype(np.float64)

        self.u_prev = np.zeros(self.npoints, dtype=np.float64)
        self.u_curr = np.zeros(self.npoints, dtype=np.float64)
        self.u_next = np.zeros(self.npoints, dtype=np.float64)

        self.K_u = np.zeros(self.npoints, dtype=np.float64)
        self.rhs = np.zeros(self.npoints, dtype=np.float64)
        self.f = np.zeros(self.npoints, dtype=np.float64)

        if self.save_seismograms:
            self.receiver_data = np.zeros((self.nt, len(self.receiver_coords_sem)), dtype=np.float64)

        if self.save_wavefield:
            self.wavefield_snapshots = []
            self.snapshot_dtype = np.float32

        if self.visualize:
            self._setup_visualization()

        self._time_stepping_loop_numpy()

        results = {
            'receiver_data': self.receiver_data if self.save_seismograms else None,
            'wavefield_snapshots': self.wavefield_snapshots if self.save_wavefield else None,
            'computation_time': None,
            'velocity_model': self.velocity_model,
            'global_coords': self.global_coords,
            'dt': self.dt,
            'nt': self.nt,
            'receiver_coords': self.receiver_coords_sem
        }
        
        self._save_wavefield_snapshots()
        self._save_visualization()
        return results

    def _run_torch_version(self):
        """
        Forward SEM (PyTorch, CPU) — **delegates to `_time_stepping_loop_torch()`** so that
        wavefield snapshots and live visualization are handled uniformly.
        """
        # Note: time stepping uses `apply_pml_torch_optimized` internally

        torch.set_default_dtype(torch.float64)
        device = torch.device("cpu")

        nt   = int(self.nt)
        npts = int(self.npoints)

        # Assemble global stiffness matrix once (COO -> CSR)
        K_coo = assemble_stiffness_matrix_torch(
            torch.from_numpy(self.velocity_model).to(device=device, dtype=torch.float64),
            self.global_connectivity,
            self.Gx_global, self.Gz_global,
            self.S_interp_global, self.w2D_global,
            self.npol, self.npoints
        )
        self.K_csr_torch = K_coo.coalesce().to_sparse_csr()

        # Receiver operator (ensure CSR) + provide alias used by the loop
        R = getattr(self, "R_torch", None)
        nrec = 0
        if R is not None:
            if not R.is_sparse:
                R = R.to_sparse_csr()
            self.R_torch = R
            # compatibility: _time_stepping_loop_torch() expects `self.R_sparse_torch`
            self.R_sparse_torch = R
            nrec = int(R.shape[0])

        # Mass inverse times dt^2 (vector)
        # `self.Minv_diag_torch` is prepared in run(); keep it as-is, just scale by dt^2
        self.Minv_dt2_torch = (self.dt ** 2) * self.Minv_diag_torch

        # Allocate receiver buffers if needed
        if nrec > 0:
            # Torch loop writes into `self.receiver_data`
            self.receiver_data = torch.zeros((nt, nrec), dtype=torch.float64, device=device)

        # Initialize wavefield snapshot container for the Torch loop
        if self.save_wavefield:
            self.wavefield_snapshots = []
            # store as float32 to keep file size reasonable (matches NumPy path)
            self.snapshot_dtype = torch.float32

        # Ensure ROI mask in torch tensor
        if not hasattr(self, "roi_mask_torch"):
            self.roi_mask_torch = torch.from_numpy(self.roi_mask)

        # Call the unified Torch stepping loop (handles snapshots + visualization)
        self._time_stepping_loop_torch()

        # Pack outputs
        out = {"dt": float(self.dt), "nt": int(nt)}
        if nrec > 0:
            out["receiver_data"]  = self.receiver_data.detach().cpu().numpy()
            out["receiver_coords"] = self.receiver_coords_sem

        # Persist artifacts (snapshots / animation) if enabled
        self._save_wavefield_snapshots()
        self._save_visualization()
        return out
    
    def _sparse_matvec(self, K_torch, x):
        if K_torch.is_sparse:
            return torch.sparse.mm(K_torch, x.unsqueeze(1)).squeeze()
        else:
            return K_torch @ x

    def _time_stepping_loop_numpy(self):
        #print(f"Running SEM time stepping for {self.nt} steps (NumPy)...")
        src_M_factor = self.M_diag[self.src_node]

        for it in range(self.nt):
            t = it * self.dt

            source_value = self.src_wavelet[it]

            self.K_u = stiffness_matrix_vector_product(
                self.u_curr, self.velocity_model, self.global_connectivity,
                self.Gx_global, self.Gz_global, self.S_interp_global, self.w2D_global,
                self.npol, self.npoints
            )
            self.rhs.fill(0.0)
            np.negative(self.K_u, out=self.rhs)
            self.rhs[self.src_node] += source_value * src_M_factor

            np.multiply(2.0, self.u_curr, out=self.u_next)
            np.subtract(self.u_next, self.u_prev, out=self.u_next)
            np.multiply(self.Minv_dt2, self.rhs, out=self.u_prev)
            np.add(self.u_next, self.u_prev, out=self.u_next)

            if self.save_seismograms:
                self.receiver_data[it] = self.R_sparse.dot(self.u_curr)

            self.u_prev, self.u_curr, self.u_next = apply_precomputed_pml_all_layers(
                self.u_prev, self.u_curr, self.u_next, self.pml_mask_sem, self.decay_sem
            )

            if self.save_wavefield and it % self.snapshot_interval == 0:
                snapshot = self.u_curr[self.roi_mask].astype(self.snapshot_dtype)
                self.wavefield_snapshots.append(snapshot)

            if self.visualize:
                self._update_visualization(it)

            self.u_prev, self.u_curr, self.u_next = self.u_curr, self.u_next, self.u_prev

            #if it % 100 == 0:
            #    print(f"Time step {it}/{self.nt}, t={t:.3f}s, max displacement: {np.max(np.abs(self.u_curr)):.2e}")

    def _time_stepping_loop_torch(self):
        #print(f"Running SEM time stepping for {self.nt} steps (PyTorch)...")
        src_M_factor = self.M_diag_torch[self.src_node]

        # Initialize visualization for torch path (if requested)
        if self.visualize:
            self._setup_visualization()

        nt = int(self.nt)

        for it in range(nt):
            t = it * self.dt
            source_value = self.src_wavelet_torch[it]

            # Ku
            try:
                self.K_u_torch = torch.matmul(self.K_csr_torch, self.u_curr_torch)
            except Exception:
                self.K_u_torch = torch.sparse.mm(
                    self.K_csr_torch, self.u_curr_torch.unsqueeze(1)
                ).squeeze(1)

            # RHS = -Ku + source
            self.rhs_torch.fill_(0.0)
            torch.neg(self.K_u_torch, out=self.rhs_torch)
            self.rhs_torch[self.src_node] += source_value * src_M_factor

            # u_next = 2u_curr - u_prev + dt^2 * Minv * rhs
            self.u_next_torch.fill_(0.0)
            torch.mul(2.0, self.u_curr_torch, out=self.u_next_torch)
            torch.sub(self.u_next_torch, self.u_prev_torch, out=self.u_next_torch)
            temp = self.Minv_dt2_torch * self.rhs_torch
            torch.add(self.u_next_torch, temp, out=self.u_next_torch)

            # Record seismograms if receivers are present
            if hasattr(self, "R_sparse_torch") and self.R_sparse_torch is not None and hasattr(self, "receiver_data"):
                try:
                    self.receiver_data[it] = torch.matmul(self.R_sparse_torch, self.u_curr_torch)
                except Exception:
                    self.receiver_data[it] = torch.sparse.mm(
                        self.R_sparse_torch, self.u_curr_torch.unsqueeze(1)
                    ).squeeze(1)

            # Apply PML and rotate states
            self.u_prev_torch, self.u_curr_torch, self.u_next_torch = apply_pml_torch_optimized(
                self.u_prev_torch, self.u_curr_torch, self.u_next_torch,
                self.pml_indices_sem, self.decay_sem_torch
            )
            self.u_prev_torch, self.u_curr_torch, self.u_next_torch = self.u_curr_torch, self.u_next_torch, self.u_prev_torch

            # Save wavefield snapshot in ROI (numpy) to keep file size reasonable
            if self.save_wavefield and it % self.snapshot_interval == 0:
                snap_t = self.u_curr_torch[self.roi_mask_torch].to(self.snapshot_dtype).detach().cpu().numpy()
                self.wavefield_snapshots.append(snap_t)

            # Update visualization / append frame
            if self.visualize:
                self._update_visualization(it)

            # if it % 1000 == 0:
            #     max_disp = torch.max(torch.abs(self.u_curr_torch)).item()
            #     print(f"Time step {it}/{self.nt}, t={t:.3f}s, max displacement: {max_disp:.2e}")

    def _setup_visualization(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 6))

        x_grid = np.linspace(self.xmin, self.xmax, self.nelem_x * 2)
        z_grid = np.linspace(self.zmin, self.zmax, self.nelem_z * 2)
        self.xx_grid, self.zz_grid = np.meshgrid(x_grid, z_grid, indexing='ij')

        self._setup_interpolation_matrix()

        u_grid = self._interpolate_to_grid(self.u_curr if hasattr(self, 'u_curr') else
                                           self.u_curr_torch.detach().numpy())
        self.im = self.ax1.imshow(u_grid.T,
                                  extent=[self.xmin, self.xmax, self.zmin, self.zmax],
                                  cmap='seismic', vmin=-8e-8, vmax=8e-8,
                                  origin='lower')
        plt.colorbar(self.im, ax=self.ax1, label='Displacement')
        self.ax1.plot(self.src_x, self.src_z, 'r*', markersize=12, label='Source')

        for rx, rz in self.receiver_coords_sem:
            self.ax1.plot(rx, rz, 'g^', markersize=6, markeredgecolor='black')
        self.ax1.plot(self.receiver_coords_sem[0][0], self.receiver_coords_sem[0][1],
                    'g^', markersize=6, markeredgecolor='black', label='Receiver')

        self.ax1.set_title('Wave Propagation')
        self.ax1.set_xlabel('X (m)')
        self.ax1.set_ylabel('Z (m)')
        self.ax1.legend()

        self.time_axis = np.arange(self.nt) * self.dt
        self.lines = []
        for i in range(len(self.receiver_coords_sem)):
            line, = self.ax2.plot([], [], lw=1, label=f'R{i+1}')
            self.lines.append(line)

        self.ax2.set_xlim(0, self.nt * self.dt)
        self.ax2.set_ylim(-8e-8, 8e-8)
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Amplitude')
        self.ax2.set_title('Receiver Waveforms')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='upper right', ncol=2)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    def _setup_interpolation_matrix(self):
        grid_points = np.column_stack([self.xx_grid.ravel(), self.zz_grid.ravel()])
        tree = cKDTree(self.global_coords)
        distances, indices = tree.query(grid_points, k=4)

        weights = []
        row_indices = []
        col_indices = []

        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if np.any(dist == 0):
                exact_idx = idx[np.where(dist == 0)[0][0]]
                weights.append(1.0)
                row_indices.append(i)
                col_indices.append(exact_idx)
            else:
                w = 1.0 / (dist + 1e-10)
                w /= np.sum(w)
                for j, weight in enumerate(w):
                    weights.append(weight)
                    row_indices.append(i)
                    col_indices.append(idx[j])

        self.interp_matrix = sp.csr_matrix(
            (weights, (row_indices, col_indices)),
            shape=(len(grid_points), len(self.global_coords))
        )

    def _interpolate_to_grid(self, u_data):
        u_grid = self.interp_matrix.dot(u_data).reshape(self.xx_grid.shape)
        return u_grid

    def _update_visualization(self, it):
        if it % 20 != 0:
            return

        t = it * self.dt
        if hasattr(self, 'u_curr'):
            u_data = self.u_curr
        else:
            u_data = self.u_curr_torch.detach().numpy()

        u_grid = self._interpolate_to_grid(u_data)

        self.im.set_array(u_grid.T)
        self.im.set_clim(-8*1e-8, 8*1e-8)
        self.ax1.set_title(f'Wave Propagation at t={t:.3f}s')

        if hasattr(self, 'receiver_data'):
            for i, line in enumerate(self.lines):
                if i < len(self.receiver_coords_sem):
                    if hasattr(self, 'receiver_data',) and hasattr(self.receiver_data, 'detach'):
                        data = self.receiver_data[:it+1, i].detach().numpy()
                    else:
                        data = self.receiver_data[:it+1, i]
                    line.set_data(self.time_axis[:it+1], data)

        self.ax2.set_ylim(-8*1e-8, 8*1e-8)
        plt.draw()
        plt.pause(0.01)

        self.fig.canvas.draw()
        try:
            frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        except AttributeError:
            frame = np.frombuffer(self.fig.canvas.tostring_argb(), dtype=np.uint8)
            frame = frame.reshape(-1, 4)[:, 1:4].reshape(-1, 3)
        frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self.frames.append(frame)

    def _sigmoid_derivative(self, x):
        s = 1.0 / (1.0 + np.exp(-x))
        return s * (1.0 - s)

    def bspline_basis(self, t, i, degree, knots):
        if degree == 0:
            return 1.0 if knots[i] <= t < knots[i+1] else 0.0
        term1 = 0.0
        denom1 = knots[i+degree] - knots[i]
        if abs(denom1) > 1e-10:
            term1 = (t - knots[i]) / denom1 * self.bspline_basis(t, i, degree-1, knots)
        term2 = 0.0
        denom2 = knots[i+degree+1] - knots[i+1]
        if abs(denom2) > 1e-10:
            term2 = (knots[i+degree+1] - t) / denom2 * self.bspline_basis(t, i+1, degree-1, knots)
        return term1 + term2

    def bspline_basis_derivative(self, t, i, degree, knots):
        if degree == 0:
            return 0.0
        term1 = 0.0
        denom1 = knots[i+degree] - knots[i]
        if abs(denom1) > 1e-10:
            term1 = degree / denom1 * self.bspline_basis(t, i, degree-1, knots)
        term2 = 0.0
        denom2 = knots[i+degree+1] - knots[i+1]
        if abs(denom2) > 1e-10:
            term2 = degree / denom2 * self.bspline_basis(t, i+1, degree-1, knots)
        return term1 - term2

    def _create_periodic_knots(self, n_ctrl, degree):
        knots = np.arange(-degree, n_ctrl + degree + 1)
        return knots

    def _precompute_basis_derivatives(self, t_samples, n_ctrl, degree, knots):
        basis_derivs = np.zeros((len(t_samples), n_ctrl + 1))
        for i, t in enumerate(t_samples):
            for j in range(n_ctrl + 1):
                basis_derivs[i, j] = self.bspline_basis_derivative(t, j, degree, knots)
        return basis_derivs

    def _compute_spline_and_derivatives(self, control_points, t_samples, basis_derivs, n_ctrl):
        n_samples = len(t_samples)
        spline_points = np.zeros((n_samples, 2))
        ds_dctrl = np.zeros((n_samples, n_ctrl, 2))
        for i, t in enumerate(t_samples):
            s_x, s_z = 0.0, 0.0
            for j in range(n_ctrl + 1):
                basis_val = self.bspline_basis(t, j, 3, self._create_periodic_knots(n_ctrl, 3))
                s_x += basis_val * control_points[j % n_ctrl, 0]
                s_z += basis_val * control_points[j % n_ctrl, 1]
            spline_points[i] = [s_x, s_z]
            for j in range(n_ctrl):
                ds_dctrl[i, j, 0] = basis_derivs[i, j]
                ds_dctrl[i, j, 1] = basis_derivs[i, j]
        return spline_points, ds_dctrl

    def _eval_dC_dctrl(self, ctrl_points, t_samples, degree=3):
        """usebasis function valueinstead of deviations """
        ctrl_points = np.asarray(ctrl_points, dtype=np.float64)
        if ctrl_points.shape != (6, 2):
            raise ValueError(f"_eval_dC_dctrl expects ctrl_points (6,2), got {ctrl_points.shape}")

        t_samples = np.asarray(t_samples, dtype=np.float64).reshape(-1)
        S = t_samples.shape[0]
        n_ctrl = 6

        # velocity.py knot construction
        k = 3
        ctrl_closed = np.vstack([ctrl_points, np.tile(ctrl_points[0], (k, 1))])
        n = len(ctrl_closed) - 1
        total_knots = n + k + 2
        knots = np.zeros(total_knots, dtype=np.float64)
        knots[:k+1] = 0.0
        knots[-k-1:] = 1.0
        inner = np.linspace(0, 1, n - k + 2)[1:-1]
        knots[k+1:-k-1] = inner

        dC = np.zeros((S, n_ctrl, 2), dtype=np.float64)

        # Key fix:usebasis function valueinstead ofderivative
        for i, t in enumerate(t_samples):
            for j in range(n_ctrl):
                Nj = self.bspline_basis(t, j, degree, knots)  # basis function value
                dC[i, j, 0] = Nj  # ∂C_x/∂ctrl_{j,x} = N_j
                dC[i, j, 1] = Nj  # ∂C_z/∂ctrl_{j,z} = N_j

        return dC

    def _clamped_basis_values_collapsed(self, ctrl_points, u, degree=3, knots=None, ctrl_closed=None):
        """
        Exactly consistent with velocity.py "triple copy of first point + open interval endpoint knots" basis function value (for 6 free control points).
        Returns: B6 (M,6)
        """
        P = np.asarray(ctrl_points, dtype=np.float64).reshape(-1, 2)  # (6,2)
        k = int(degree)
        if ctrl_closed is None or knots is None:
            ctrl_closed = np.vstack([P, np.tile(P[0], (k, 1))])  # (9,2)
            n = len(ctrl_closed) - 1
            total_knots = n + k + 2
            knots = np.zeros(total_knots, dtype=np.float64)
            knots[:k+1] = 0.0
            knots[-k-1:] = 1.0
            inner = np.linspace(0.0, 1.0, n - k + 2)[1:-1]
            knots[k+1:-k-1] = inner

        u = np.asarray(u, dtype=np.float64).reshape(-1)
        t_hat = knots[k] + (knots[-(k+1)] - knots[k]) * u

        M = u.shape[0]
        B9 = np.zeros((M, 9), dtype=np.float64)
        for j in range(9):
            coeff = np.zeros(9, dtype=np.float64); coeff[j] = 1.0
            sp = BSpline(knots, coeff, k, extrapolate=False)
            B9[:, j] = sp(t_hat)

        B6 = np.zeros((M, 6), dtype=np.float64)
        B6[:, 0] = B9[:, 0] + B9[:, 6] + B9[:, 7] + B9[:, 8]  # collapse only P0
        B6[:, 1] = B9[:, 1]
        B6[:, 2] = B9[:, 2]
        B6[:, 3] = B9[:, 3]
        B6[:, 4] = B9[:, 4]
        B6[:, 5] = B9[:, 5]
        return B6, knots, ctrl_closed
  
    def _analytical_gradient_projection_improved(
        self,
        dL_dv_np: np.ndarray,
        bspline_ctrl: np.ndarray,
        tau: float,
        signed_dist: np.ndarray,
        extras: dict,
        spline_samples: int = 200
    ):
        """
        "forward the same spline geometry" analytical projection:
        ∂d/∂C = n_out,  ∂C/∂ctrl_j = N_j(u_star)  =>  ∂d/∂ctrl_j = N_j(u_star) * n_out
         (dL/dv)*(dv/ddsoft), 9->6 collapse (only 6,7,8->0).
         **self.spline_extras/self.signed_dist**,use explicitly passed  `signed_dist` `extras`.
        """
       
        ctrl6 = np.asarray(bspline_ctrl, dtype=np.float64).reshape(6, 2)
        vmin  = float(self.config['velocity']['inside_velocity'])
        vmax  = float(self.config['velocity']['outside_velocity'])
        dL_dv = np.asarray(dL_dv_np, dtype=np.float64).reshape(-1)

        #print(f"vmin == {vmin}")
        #print(f"vmax == {vmax}")

        # Use the explicitly passed geometric quantities
        sd         = np.asarray(signed_dist, dtype=np.float64).reshape(-1)
        u_star     = np.asarray(extras['u_star'],    dtype=np.float64).reshape(-1)
        normal_out = np.asarray(extras['normal_out'], dtype=np.float64).reshape(-1, 2)

        tau_safe = max(float(tau), 1e-12)
        x   = np.clip(sd / tau_safe, -50.0, 50.0)
        sig = 1.0 / (1.0 + np.exp(x))                    # = sigmoid(-sd/tau)
        dv_ddsoft = (vmin - vmax) * sig * (1.0 - sig) * (-1.0 / tau_safe)

        # Keep only effective contributions
        mask = np.abs(dv_ddsoft) > 1e-8 * (vmax - vmin)
        if not np.any(mask):
            return {'grad_ctrl_x': np.zeros(6, dtype=np.float64),
                    'grad_ctrl_z': np.zeros(6, dtype=np.float64)}

        w_node = (dL_dv[mask] * dv_ddsoft[mask]).reshape(-1, 1)   # (M,1)
        n_out  = normal_out[mask]                                  # (M,2)
        u_hat  = u_star[mask]                                      # (M,)

        # Closed spline (9 control point), u_star "basis function value"
        _tprobe = np.array([0.5])
        _, _, knots, ctrl9 = self._eval_closed_bspline_and_tangent(ctrl6, _tprobe, degree=3)

        M  = u_hat.shape[0]
        B9 = np.zeros((M, 9), dtype=np.float64)
        for j in range(9):
            coeff = np.zeros(9, dtype=np.float64); coeff[j] = 1.0
            sp = BSpline(knots, coeff, 3, extrapolate=False)
            B9[:, j] = sp(u_hat)                                  # N_j(u_star)

        nx = n_out[:, 0:1]
        nz = n_out[:, 1:2]
        grad9_x = np.sum(w_node * (B9 * nx), axis=0)              # (9,)
        grad9_z = np.sum(w_node * (B9 * nz), axis=0)              # (9,)

        # 9->6 collapse (only 6,7,8 fold 0)
        grad_x = np.zeros(6, dtype=np.float64)
        grad_z = np.zeros(6, dtype=np.float64)
        grad_x[0] = grad9_x[0] + grad9_x[6] + grad9_x[7] + grad9_x[8]
        grad_z[0] = grad9_z[0] + grad9_z[6] + grad9_z[7] + grad9_z[8]
        grad_x[1:6] = grad9_x[1:6]
        grad_z[1:6] = grad9_z[1:6]
    
        return {'grad_ctrl_x': grad_x, 'grad_ctrl_z': grad_z}

    def _eval_closed_bspline_and_tangent(self, ctrl_points, t_vals, degree=3):
        """
        - first point k  (only P0 repeated 3 times)
        - endpoint  (open interval endpoint knots):knots[:k+1]=0, knots[-k-1:]=1, inner knots
        - Sampling interval is [knots[k], knots[-(k+1)]].
        Returns:C, T, knots, ctrl_closed
        """
        P = np.asarray(ctrl_points, dtype=np.float64).reshape(-1, 2)  # (6,2)
        k = int(degree)
        if P.shape[0] != 6:
            raise ValueError(f"_eval_closed_bspline_and_tangent expects 6 ctrl points, got {P.shape}")

        # first point repeated k times
        ctrl_closed = np.vstack([P, np.tile(P[0], (k, 1))])  # (9,2)

        # Open-interval endpoint knot vector
        n = len(ctrl_closed) - 1                    # 8
        total_knots = n + k + 2                     # 13
        knots = np.zeros(total_knots, dtype=np.float64)
        knots[:k+1]   = 0.0
        knots[-k-1:]  = 1.0
        inner = np.linspace(0.0, 1.0, n - k + 2)[1:-1]   # uniform inner knots
        knots[k+1:-k-1] = inner

        # Map the input normalized t_vals in [0,1] to the actual parameter interval
        t_vals = np.asarray(t_vals, dtype=np.float64).reshape(-1)
        t_hat  = knots[k] + (knots[-(k+1)] - knots[k]) * t_vals

        # Construct spline and first derivative
        spl_x  = BSpline(knots, ctrl_closed[:,0], k, extrapolate=False)
        spl_z  = BSpline(knots, ctrl_closed[:,1], k, extrapolate=False)
        dspl_x = spl_x.derivative()
        dspl_z = spl_z.derivative()

        Cx  = spl_x(t_hat)
        Cz  = spl_z(t_hat)
        dCx = dspl_x(t_hat)
        dCz = dspl_z(t_hat)

        C = np.stack([Cx, Cz], axis=-1)
        T = np.stack([dCx, dCz], axis=-1)
        T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-15)
        return C, T, knots, ctrl_closed

    def _finite_difference_gradient_projection(self, perturbations, dJ_dv,
                                               spline_samples, tau, fd_eps):
        vel_cfg = self.config['velocity']
        #vmin = float(vel_cfg.get('inside_velocity', getattr(self, 'vmin', 2000.0)))
        #vmax = float(vel_cfg.get('outside_velocity', getattr(self, 'vmax', 3000.0)))
        vmin  = float(self.config['velocity']['inside_velocity'])
        vmax  = float(self.config['velocity']['outside_velocity'])

        #print(f"vmin == {vmin}")
        #print(f"vmax == {vmax}")

        ctrl6_base = np.asarray(perturbations, dtype=np.float64).reshape(6, 2)  # perts "absolute control point"
        dJ_dctrl = np.zeros(12, dtype=np.float64)

        for j in range(12):
            dp = np.zeros((6, 2), dtype=np.float64)
            if j < 6:
                dp[j, 0] = fd_eps
            else:
                dp[j-6, 1] = fd_eps

            ctrl6_plus  = ctrl6_base + dp
            ctrl6_minus = ctrl6_base - dp

            v_plus, _, _ = build_velocity_on_sem_nodes(
                nodes_xy=self.global_coords,
                ctrl6_xy=ctrl6_plus,
                v_inside=vmin, v_outside=vmax,
                tau=tau, samples=spline_samples, newton_steps=7
            )
            v_minus, _, _ = build_velocity_on_sem_nodes(
                nodes_xy=self.global_coords,
                ctrl6_xy=ctrl6_minus,
                v_inside=vmin, v_outside=vmax,
                tau=tau, samples=spline_samples, newton_steps=7
            )
            dv_dzj = (v_plus - v_minus) / (2.0 * fd_eps)
            dJ_dctrl[j] = float(np.sum(np.asarray(dJ_dv) * dv_dzj))

        return dJ_dctrl

    def _ensure_torch64_compatible(self, x, reference_tensor):
        """ensurex reference_tensor type, device precision """
        if isinstance(x, torch.Tensor):
            x = x.to(dtype=torch.float64, device=reference_tensor.device)
        else:
            x = torch.from_numpy(np.asarray(x, dtype=np.float64)).to(
                dtype=torch.float64, device=reference_tensor.device
            )
        return x

    def run_forward_and_adjoint(self, params_dict):
        """
        Forward + Adjoint for Gaussian log-likelihood and its gradient wrt 12 B-spline parameters.
        Returns:
        {'loglik': float, 'grad_wrt_ctrl': (12,), 'clean_data': (nt,nrec), 'dt': float, 'nt': int}
        -  (velocity, signed_dist, extras)
        - projection explicitly passed signed_dist/extras (_analytical_gradient_projection_explicit)
        """
        cfg     = self.config
        verbose = bool(params_dict.get('verbose', True))

        # 0) 
        perts = np.asarray(params_dict['bspline_ctrl'], dtype=np.float64)
        if perts.shape == (12,):
            perts = perts.reshape(6, 2)      # absolute control point
        y_obs  = np.asarray(params_dict['y_obs'], dtype=np.float64)
        obs_dt = float(params_dict['obs_dt'])
        obs_nt = int(params_dict['obs_nt'])
        sigma  = float(params_dict.get('noise_std', 0.0))
        sigma2 = sigma * sigma + 1e-24

        # 1) 
        vel_cfg = self.config['velocity']
        # velocities already validated and stored on self in _initialize_components
        #vmin = float(vel_cfg.get('inside_velocity', getattr(self, 'vmin', 2000.0)))
        #vmax = float(vel_cfg.get('outside_velocity', getattr(self, 'vmax', 3000.0)))
        vmin  = float(self.config['velocity']['inside_velocity'])
        vmax  = float(self.config['velocity']['outside_velocity'])

        #print(f"vmin == {vmin}")
        #print(f"vmax == {vmax}")

        n_curve_samples  = int(self.config['velocity']['spline_samples'])
        #n_curve_samples = int(vel_cfg.get('spline_samples', 1200))
        #print(f"n_curve_samples = {n_curve_samples}")

        tau  = float(self.config['velocity']['tau'])
        #tau = float(vel_cfg.get('tau', 10.0))
        #print(f"tau = {tau}")
       
        #vel_cfg = self.config['velocity']
        #vmin = float(vel_cfg.get('inside_velocity', getattr(self, 'vmin', 2000.0)))
        #vmax = float(vel_cfg.get('outside_velocity', getattr(self, 'vmax', 3000.0)))
        #n_curve_samples = int(vel_cfg.get('spline_samples', 1200))
        #tau = float(vel_cfg.get('tau', 10.0))

        ctrl6 = perts.reshape(6, 2)
        # geometry result for this sample forward, adjoint projection
        velocity_model, signed_dist, spline_extras = build_velocity_on_sem_nodes(
            nodes_xy=self.global_coords,
            ctrl6_xy=ctrl6,
            v_inside=vmin,
            v_outside=vmax,
            tau=tau,
            samples=n_curve_samples,
            newton_steps=7
        )
        #  
        self.velocity_model   = velocity_model
        self.signed_dist      = signed_dist
        self.spline_extras    = spline_extras
        self.velocity_model_torch = torch.from_numpy(self.velocity_model).to(torch.float64)

        # 2) CFL
        #self.dt, self.nt = adjust_time_step_for_stability(
        #    self.velocity_model, getattr(self, 'dt', 8.0e-5), self.total_time,
        #    self.npol, self.dx_elem, self.dz_elem, self.vmax
        #)
        self.dt, self.nt = adjust_time_step_for_stability(
            self.velocity_model, self.dt, self.total_time,
            self.npol, self.dx_elem, self.dz_elem, self.vmax
        )
        self.dt2 = self.dt * self.dt
        self.dt2_torch = torch.tensor(self.dt2, dtype=torch.float64)

        #pml_thickness = self.config['method'].get('pml_thickness', 300.0)
        pml_thickness = self.config['method']['pml_thickness']
        #print("==== PML Parameters (run_forward_and_adjoint) ====")
        #print(f"pml_thickness = {pml_thickness}")

        self.pml_indices_sem, self.pml_mask_sem, self.decay_sem_torch = setup_pml(
            self.xmin, self.xmax, self.zmin, self.zmax, pml_thickness, self.global_coords, self.dt, c_max=self.vmax 
        )
        self.decay_sem = self.decay_sem_torch.numpy()

        self.time_array = np.arange(self.nt, dtype=np.float64) * self.dt
        self.src_wavelet = create_source_wavelet(
            self.nt, self.dt, f0=self.src_f0, t0=0.1, gate_time=0.3, amp=self.src_amp
        )
        self.src_wavelet_torch = torch.from_numpy(self.src_wavelet).double()

        if verbose:
            #print("\n=== Starting SEM Forward Simulation (PyTorch) ===\n")
            _stable, _dtmax = check_cfl_condition(
                self.velocity_model, self.dt, self.npol,
                self.dx_elem, self.dz_elem, self.vmax
            )
            #print("CFL condition satisfied for SEM")
            #print("Assembling global stiffness matrix (Torch, COO -> CSR) ...")

        # 3) Mass matrix & stiffness matrix
        M_lumped = assemble_mass_matrix_lumped_torch(
            torch.from_numpy(self.global_connectivity), self.dx_elem, self.dz_elem, self.npol, self.npoints
        )
        self.M_diag_torch   = M_lumped.clone().to(torch.float64)
        self.Minv_diag_torch= (1.0 / (M_lumped + 1e-30)).to(torch.float64)

        K_coo = assemble_stiffness_matrix_torch(
            torch.from_numpy(self.velocity_model).to(dtype=torch.float64),
            self.global_connectivity,
            self.Gx_global, self.Gz_global,
            self.S_interp_global, self.w2D_global,
            self.npol, self.npoints
        )
        self.K_csr_torch = K_coo.coalesce().to_sparse_csr()

        # 4) forward (adjoint)
        u_prev = torch.zeros(self.npoints, dtype=torch.float64)
        u_curr = torch.zeros(self.npoints, dtype=torch.float64)
        u_next = torch.zeros(self.npoints, dtype=torch.float64)
        self.u_hist = torch.zeros((self.nt, self.npoints), dtype=torch.float64)

        R = self.R_torch
        if R is not None and (not R.is_sparse):
            R = R.to_sparse_csr()
        self.R_torch  = R
        self.RT_torch = None if R is None else R.transpose(0, 1)

        src_node    = int(self.src_node)
        src_M_fac   = self.M_diag_torch[src_node]
        for it in range(self.nt):
            # Ku
            try:
                Ku = torch.matmul(self.K_csr_torch, u_curr)
            except Exception:
                Ku = torch.sparse.mm(self.K_csr_torch, u_curr.unsqueeze(1)).squeeze(1)
            rhs = -Ku
            rhs[src_node] = rhs[src_node] + self.src_wavelet_torch[it] * src_M_fac
            u_next = 2.0 * u_curr - u_prev + self.dt2_torch * (self.Minv_diag_torch * rhs)
            # PML
            u_prev, u_curr, u_next = apply_pml_torch_optimized(
                u_prev, u_curr, u_next, self.pml_indices_sem, self.decay_sem_torch
            )
            u_prev, u_curr, u_next = u_curr, u_next, u_prev
            self.u_hist[it, :] = u_curr

            #if (it % 1000 == 0) and verbose:
                #umax = float(torch.max(torch.abs(u_curr)))
                #print(f"  [FWD] step {it}/{self.nt}, |u|_inf={umax:.2e}")

        # Synthesize receiver records
        if self.R_torch is not None:
            d_syn = (self.R_torch @ self.u_hist.T).T.detach().cpu().numpy()
        else:
            d_syn = np.zeros((self.nt, 0), dtype=np.float64)

        # Data alignment and likelihood
        if (abs(self.dt - obs_dt) < 1e-12) and (self.nt == obs_nt):
            residual = d_syn - y_obs
        else:
            residual = self._align_observation_data(d_syn, self.dt, self.nt) - \
                       self._align_observation_data(y_obs, obs_dt, obs_nt)
        res_norm2 = float(np.sum(residual * residual))
        loglik    = -0.5 * (res_norm2 / sigma2)
        #if verbose:
        #    print(f"NLL (data term): {0.5 * (res_norm2 / sigma2):.6f}, residual norm: {np.sqrt(res_norm2):.6e}")

        # 5) adjoint, node dJ/dv
        conn = torch.from_numpy(self.global_connectivity)
        Gx_t = torch.from_numpy(self.Gx_global).to(torch.float64)
        Gz_t = torch.from_numpy(self.Gz_global).to(torch.float64)
        S_t  = torch.from_numpy(self.S_interp_global).to(torch.float64)
        w2D  = torch.from_numpy(self.w2D_global).to(torch.float64)

        if self.RT_torch is not None:
            residual_t = torch.from_numpy(residual.T).to(torch.float64)
            residual_scaled = (1.0 / sigma2) * residual_t
        else:
            residual_scaled = None

        lam_prev = torch.zeros(self.npoints, dtype=torch.float64)
        lam_curr = torch.zeros(self.npoints, dtype=torch.float64)
        lam_next = torch.zeros(self.npoints, dtype=torch.float64)
        dJ_dv    = torch.zeros(self.npoints, dtype=torch.float64)

        for it in range(self.nt - 1, -1, -1):
            if self.RT_torch is not None:
                Rt_rt = self.RT_torch @ residual_scaled[:, it]
            else:
                Rt_rt = torch.zeros(self.npoints, dtype=torch.float64)

            try:
                Ku_adj = torch.matmul(self.K_csr_torch, lam_curr)
            except Exception:
                Ku_adj = torch.sparse.mm(self.K_csr_torch, lam_curr.unsqueeze(1)).squeeze(1)
            rhs_adj    = Rt_rt - Ku_adj
            lam_prev_n = 2.0 * lam_curr - lam_next + self.dt2_torch * (self.Minv_diag_torch * rhs_adj)

            gu = self.u_hist[it, :]
            gl = lam_curr
            dJ_dv = dJ_dv + accumulate_sensitivity_dJdc_torch(
                gu, gl, conn, Gx_t, Gz_t, S_t, w2D,
                self.velocity_model_torch, self.npol, self.npoints
            )

            lam_next, lam_curr = lam_curr, lam_prev_n
            #if (it % 1000 == 0) and verbose:
                #lmax = float(torch.max(torch.abs(lam_curr)))
                #print(f"  [ADJ] step {it}/{self.nt-1}, |λ|_inf={lmax:.2e}")

        # 6) projection 12 control point
        dL_dv_np = (-dJ_dv).detach().cpu().numpy()  # L = -J
        proj = self._analytical_gradient_projection_improved(
            dL_dv_np=dL_dv_np,
            bspline_ctrl=perts,
            tau=tau,
            signed_dist=signed_dist,
            extras=spline_extras,
            spline_samples=n_curve_samples
        )
        grad_ctrl = np.concatenate([proj['grad_ctrl_x'], proj['grad_ctrl_z']], axis=0)

        # ===================================================
        # ADJ vs FD checks (projection + full log-likelihood)
        # ===================================================
        #VERIFY_PROJECTION = True
        VERIFY_PROJECTION = self.config['method'].get('VERIFY_PROJECTION', False)
        if VERIFY_PROJECTION:
            # 1) Projection FD (node-space -> ctrl)
            #    _finite_difference_gradient_projection(),
            #    adjoint dL/dv (L = -J)
            fd_eps = 1e-6
            proj_fd = -self._finite_difference_gradient_projection(
                perturbations=perts,                  # (6,2) control pointoffset
                dJ_dv=dL_dv_np,                       # dL/dv (= -dJ/dv)
                spline_samples=n_curve_samples,
                tau=tau,
                fd_eps=fd_eps
            )

            # 2) dimension:projection  ADJ vs projectionFD
            proj_an = np.concatenate([proj['grad_ctrl_x'], proj['grad_ctrl_z']], axis=0)  # length 12
            print("\nPer-dimension gradient comparison:")
            print("Dim |     Adjoint     |    Numerical    |    Ratio    |  Relative error")
            print("-----------------------------------------------------------------")
            for i in range(12):
                adj_val = proj_an[i]
                num_val = proj_fd[i]
                if np.abs(num_val) > 0:
                    ratio = adj_val / num_val
                    rel_err = abs(adj_val - num_val) / (abs(num_val) + 1e-30)
                else:
                    ratio = float('inf')
                    rel_err = float('inf')
                print(f"{i:2d}   | {adj_val:14.6e} | {num_val:14.6e} | {ratio:11.3f} | {rel_err:11.3e}")

            sign_agreement = np.sum(np.sign(proj_an) == np.sign(proj_fd))
            print(f"\nSign consistency: {sign_agreement}/{len(proj_an)} ({sign_agreement/len(proj_an)*100:.1f}%)")

            # 3) fullFD:" L"finite difference 
            print('\nPerforming full finite-difference check on log-likelihood (this is slower).')
            fd_eps_full = 1e-6
            loglik_fd = np.zeros(12, dtype=np.float64)

            for k in range(12):
                dp = np.zeros((6, 2), dtype=np.float64)
                if k < 6:
                    dp[k, 0] = fd_eps_full
                else:
                    dp[k - 6, 1] = fd_eps_full

                # forward with +eps
                pp = (perts + dp).reshape(12,)
                res_p = self.run_forward_only_for_loglik(pp, sigma2, obs_dt, obs_nt, y_obs)
                # forward with -eps
                pm = (perts - dp).reshape(12,)
                res_m = self.run_forward_only_for_loglik(pm, sigma2, obs_dt, obs_nt, y_obs)

                # res_* "L", dL/d(ctrl_k)
                loglik_fd[k] = (res_p - res_m) / (2.0 * fd_eps_full)

            # Compare projected ADJ (dL/dctrl from projection) vs true FD(dL/dctrl)
            print("\nFull FD vs ADJ (projected) comparison:")
            for k in range(12):
                if np.abs(loglik_fd[k]) > 0:
                    ratio = proj_an[k] / loglik_fd[k]
                else:
                    ratio = float('inf')
                print(f"{k:2d} | ADJ_proj={proj_an[k]: .6e} | FD_full={loglik_fd[k]: .6e} | ratio={ratio:.3f}")

        # grad_ctrl is currently [x0..x5, z0..z5]
        gx = proj['grad_ctrl_x'].reshape(6)
        gz = proj['grad_ctrl_z'].reshape(6)

        grad_ctrl_interleaved = np.empty(12, dtype=np.float64)
        grad_ctrl_interleaved[0::2] = gx  # x0,x1,...,x5 -> positions 0,2,4,6,8,10
        grad_ctrl_interleaved[1::2] = gz  # z0,z1,...,z5 -> positions 1,3,5,7,9,11

        return {
            'loglik': float(loglik),
            'grad_wrt_ctrl': grad_ctrl_interleaved,
            'clean_data': d_syn,
            'dt': float(self.dt),
            'nt': int(self.nt)
        }

    def _forward_pred_with_velocity(self, velocity_torch):
        """
        Using the given nodal velocity (torch.float64, (npoints,))forward,
        and return the predicted receiver data d_pred (np.ndarray, (nt, nrec)).
        only for dot-test.
        """
        torch.set_default_dtype(torch.float64)
        device = torch.device("cpu")

        nt   = int(self.nt)
        npts = int(self.npoints)
        dt2  = torch.tensor(self.dt * self.dt, dtype=torch.float64, device=device)

        Minv_diag = self.Minv_diag_torch
        R  = self.R_torch
        if R is not None and (not R.is_sparse):
            R = R.to_sparse_csr()
        src_sig   = self.src_wavelet_torch
        src_node  = int(self.src_node)
        src_M_factor = self.M_diag_torch[src_node]

        K_coo = assemble_stiffness_matrix_torch(
            velocity_torch,
            self.global_connectivity,
            self.Gx_global, self.Gz_global,
            self.S_interp_global, self.w2D_global,
            self.npol, self.npoints
        )
        K = K_coo.coalesce().to_sparse_csr()

        u_prev = torch.zeros(npts, dtype=torch.float64, device=device)
        u_curr = torch.zeros(npts, dtype=torch.float64, device=device)
        u_next = torch.zeros(npts, dtype=torch.float64, device=device)

        nrec = int(self.num_receivers) if hasattr(self, 'num_receivers') else 0
        d_pred = torch.zeros((nt, nrec), dtype=torch.float64, device=device) if nrec > 0 else None

        for it in range(nt):
            try:
                Ku = torch.matmul(K, u_curr)
            except Exception:
                Ku = torch.sparse.mm(K, u_curr.unsqueeze(1)).squeeze(1)

            rhs = -Ku
            rhs[src_node] = rhs[src_node] + src_sig[it] * src_M_factor

            u_next.copy_(u_curr).mul_(2.0).sub_(u_prev)
            u_next.add_(Minv_diag * rhs * dt2)

            if nrec > 0:
                try:
                    d_now = torch.matmul(R, u_curr)
                except Exception:
                    d_now = torch.sparse.mm(R, u_curr.unsqueeze(1)).squeeze(1)
                d_pred[it].copy_(d_now)

            if hasattr(self, 'pml_indices_sem') and hasattr(self, 'decay_sem_torch'):
                u_prev, u_curr, u_next = apply_pml_torch_optimized(
                    u_prev, u_curr, u_next, self.pml_indices_sem, self.decay_sem_torch
                )

            u_prev, u_curr, u_next = u_curr, u_next, u_prev

        if d_pred is None:
            return None
        return d_pred.detach().cpu().numpy()

    def run_forward_only_for_loglik(self, ctrl12, sigma2, obs_dt, obs_nt, y_obs):
        """
        "full finite-difference"check forward:
        L = -0.5/σ^2 * || d_pred - y_obs ||^2
        **will not** self.velocity_model / self.signed_dist / self.spline_extras.
        dt/nt/M/R/PML (run_forward_and_adjoint).
        """
        # 1) 
        perts = np.asarray(ctrl12, dtype=np.float64).reshape(6, 2)

        #vel_cfg = self.config['velocity']
        vmin  = float(self.config['velocity']['inside_velocity'])
        vmax  = float(self.config['velocity']['outside_velocity'])
        #vmin = float(vel_cfg.get('inside_velocity', getattr(self, 'vmin', 2000.0)))
        #vmax = float(vel_cfg.get('outside_velocity', getattr(self, 'vmax', 3000.0)))
        #n_curve_samples = int(vel_cfg.get('spline_samples', 1200))
        #n_curve_samples = int(vel_cfg.get('spline_samples', 1200))
        n_curve_samples = int(self.config['velocity']['spline_samples'])
        #tau = float(vel_cfg.get('tau', 10.0))
        tau = float(self.config['velocity']['tau'])


        #vel_cfg = self.config['velocity']
        #vmin = float(vel_cfg.get('inside_velocity', getattr(self, 'vmin', 2000.0)))
        #vmax = float(vel_cfg.get('outside_velocity', getattr(self, 'vmax', 3000.0)))
        #n_curve_samples = int(vel_cfg.get('spline_samples', 1200))
        #tau = float(vel_cfg.get('tau', 10.0))

        v_local, sd_local, ex_local = build_velocity_on_sem_nodes(
            nodes_xy=self.global_coords,
            ctrl6_xy=perts,
            v_inside=vmin, v_outside=vmax,
            tau=tau, samples=n_curve_samples, newton_steps=7
        )
        v_torch = torch.from_numpy(v_local).to(torch.float64)

        # 2) 
        d_pred = self._forward_pred_with_velocity(v_torch)  # (nt, nrec) or None
        if d_pred is None:
            return -0.0

        # 3) 
        if (abs(self.dt - obs_dt) < 1e-12) and (self.nt == obs_nt):
            r = d_pred - y_obs
        else:
            d_pred_aligned = self._align_observation_data(d_pred, self.dt, self.nt)
            y_obs_aligned  = self._align_observation_data(y_obs,  obs_dt,  obs_nt)
            r = d_pred_aligned - y_obs_aligned

        res2   = float(np.sum(r * r))
        loglik = -0.5 * (res2 / float(sigma2))
        
        return loglik

    def dot_test_nodes_fd(self, dJ_dc_torch, y_obs, obs_dt, obs_nt, sigma,
                          nvec=1, eps=1e-4, seed=0, use_roi=True, verbose=True):
        """
        "node c(x)"finite difference:
          LHS = <dJ/dc, δc>
          RHS = [J(c + eps*δc) - J(c - eps*δc)] / (2*eps)
        J  Gaussian NLL , run_forward_and_adjoint.
        """
        torch.set_default_dtype(torch.float64)
        device = torch.device("cpu")

        y_obs = np.asarray(y_obs, dtype=np.float64)
        if (abs(obs_dt - self.dt) > 1e-15) or (int(obs_nt) != int(self.nt)):
            y_obs_aligned = self._align_observation_data(y_obs, obs_dt, obs_nt)
        else:
            y_obs_aligned = y_obs
        sigma2 = float(sigma) * float(sigma) + 1e-24

        npts = int(self.npoints)
        rng = np.random.RandomState(seed)

        c0 = torch.from_numpy(np.asarray(self.velocity_model, dtype=np.float64))

        if use_roi and hasattr(self, 'roi_mask') and self.roi_mask is not None:
            roi = np.asarray(self.roi_mask, dtype=bool)
        else:
            roi = np.ones(npts, dtype=bool)

        results = []
        for k in range(int(nvec)):
            delta = np.zeros(npts, dtype=np.float64)
            idx = np.where(roi)[0]
            v = rng.randn(idx.size)
            v /= (np.linalg.norm(v) + 1e-24)
            delta[idx] = v
            delta_t = torch.from_numpy(delta)

            lhs = float(torch.dot(dJ_dc_torch, delta_t))

            c_plus  = (c0 + eps * delta_t)
            c_minus = (c0 - eps * delta_t)

            d_pred_plus  = self._forward_pred_with_velocity(c_plus)
            d_pred_minus = self._forward_pred_with_velocity(c_minus)

            r_plus  = d_pred_plus  - y_obs_aligned
            r_minus = d_pred_minus - y_obs_aligned
            Jp = 0.5 / sigma2 * float(np.sum(r_plus  * r_plus))
            Jm = 0.5 / sigma2 * float(np.sum(r_minus * r_minus))
            rhs = (Jp - Jm) / (2.0 * eps)

            rat = lhs / (rhs + 1e-30)
            results.append((lhs, rhs, rat))

            #if verbose:
            #    print(f"[Node FD dot-test #{k}]  LHS={lhs:.6e}, RHS={rhs:.6e}, ratio={rat:.6e}")

        return results

    def _save_visualization(self):
        if not self.frames:
            return
        try:
            os.makedirs(self.output_dir, exist_ok=True)  # ensure folder at save time
            output_path = os.path.join(self.output_dir, 'wave_propagation.gif')
            imageio.mimsave(output_path, self.frames, duration=0.1)
            print(f"Saved visualization to {output_path}")
        except Exception as e:
            print(f"Failed to save visualization: {e}")

    def _save_wavefield_snapshots(self):
        """Persist stacked wavefield snapshots to NPZ when save_wavefield=True."""
        try:
            if not getattr(self, 'save_wavefield', False):
                return
            if not hasattr(self, 'wavefield_snapshots') or not self.wavefield_snapshots:
                return
            os.makedirs(self.output_dir, exist_ok=True)  # ensure folder at save time
            stack = np.stack([np.asarray(s, dtype=np.float32) for s in self.wavefield_snapshots], axis=0)
            out_path = os.path.join(self.output_dir, 'wavefield_snapshots.npz')
            np.savez(out_path,
                    snapshots=stack,
                    dt=self.dt, nt=self.nt,
                    xmin=self.xmin, xmax=self.xmax,
                    zmin=self.zmin, zmax=self.zmax)
            print(f"Saved wavefield snapshots to {out_path}")
        except Exception as e:
            print(f"Failed to save wavefield snapshots: {e}")

