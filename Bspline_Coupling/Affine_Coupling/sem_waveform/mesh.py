import numpy as np
from .utils import gll_points

def create_global_mesh(xmin, xmax, zmin, zmax, nelem_x, nelem_z, npol):
    """Create global mesh coordinates and connectivity"""
    ngll = npol + 1
    gll_nodes = gll_points(npol)
    
    elem_corners_x = np.linspace(xmin, xmax, nelem_x + 1)
    elem_corners_z = np.linspace(zmin, zmax, nelem_z + 1)
    
    global_coords = []
    global_connectivity = []
    node_id = 0
    node_map = {}
    
    for ez in range(nelem_z):
        for ex in range(nelem_x):
            x0 = elem_corners_x[ex]
            x1 = elem_corners_x[ex+1]
            z0 = elem_corners_z[ez]
            z1 = elem_corners_z[ez+1]
            
            elem_nodes = []
            for j in range(ngll):
                for i in range(ngll):
                    xi = gll_nodes[i]
                    eta = gll_nodes[j]
                    x = 0.5 * ((1 - xi) * x0 + (1 + xi) * x1)
                    z = 0.5 * ((1 - eta) * z0 + (1 + eta) * z1)
                    
                    coord_key = (round(x, 6), round(z, 6))
                    if coord_key not in node_map:
                        node_map[coord_key] = node_id
                        global_coords.append((x, z))
                        node_id += 1
                    
                    elem_nodes.append(node_map[coord_key])
            
            global_connectivity.append(elem_nodes)
    
    global_coords = np.array(global_coords)
    return global_coords, global_connectivity, node_map

def find_closest_node(x, z, global_coords, tol=1e-6):
    """Find the node closest to the given coordinates with tolerance check"""
    dist = np.sqrt((global_coords[:, 0] - x)**2 + (global_coords[:, 1] - z)**2)
    min_idx = np.argmin(dist)
    min_dist = dist[min_idx]
    
    if min_dist > tol:
        print(f"Warning: Closest node is {min_dist:.6f} m away from ({x}, {z})")
    
    return min_idx