import numpy as np

def ricker_wavelet(t, f0=20.0, t0=0.1):
    """Ricker wavelet source function"""
    return (1.0 - 2.0 * (np.pi * f0 * (t - t0))**2) * np.exp(-(np.pi * f0 * (t - t0))**2)

def create_source_wavelet(nt, dt, f0=20.0, t0=0.1, gate_time=0.3, amp=1.0):
    """
    Create source wavelet with DC removal and time gating
    """
    time_array = np.arange(nt) * dt
    src_wavelet = ricker_wavelet(time_array, f0, t0) * amp
    
    # Remove DC component
    src_wavelet -= np.trapezoid(src_wavelet, time_array) / (time_array[-1] - time_array[0])
    
    # Apply time gate
    gate_mask = time_array < gate_time
    src_wavelet[~gate_mask] = 0.0
    
    return src_wavelet.astype(np.float64)