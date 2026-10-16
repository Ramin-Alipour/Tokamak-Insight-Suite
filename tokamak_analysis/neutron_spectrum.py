"""Neutron energy-spectrum moments and peak extraction."""
import numpy as np
def analyze_neutron_spectrum(energy_keV, counts, *, background=None):
    e=np.asarray(energy_keV,dtype=float); c=np.asarray(counts,dtype=float)
    if e.ndim!=1 or c.shape!=e.shape or len(e)<2 or not np.all(np.isfinite(e)) or not np.all(np.isfinite(c)) or np.any(np.diff(e)<=0) or np.any(c<0): raise ValueError("Invalid energy spectrum")
    net=c.copy()
    if background is not None:
        b=np.asarray(background,dtype=float)
        if b.shape!=c.shape or not np.all(np.isfinite(b)) or np.any(b<0): raise ValueError("Invalid background")
        net=np.maximum(c-b,0)
    total=np.sum(net); mean=float(np.sum(e*net)/total) if total else float("nan")
    return {"energy_keV":e,"net_counts":net,"peak_energy_keV":float(e[np.argmax(net)]),"mean_energy_keV":mean,"total_counts":float(total),"caveat":"Detector unfolding and neutron-response calibration are required for a physical incident neutron spectrum."}

def neutron_energy_from_time_of_flight_keV(flight_path_m, flight_time_s, *, neutron_mass_kg=1.67492749804e-27):
    """Non-relativistic neutron TOF relation E=0.5*m*(L/t)^2, returned in keV."""
    length,time,mass=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (flight_path_m,flight_time_s,neutron_mass_kg)])
    if any(not np.all(np.isfinite(v)) for v in (length,time,mass)) or np.any(length<=0) or np.any(time<=0) or np.any(mass<=0):
        raise ValueError("Flight path, time and neutron mass must be finite and positive")
    energy_keV=0.5*mass*(length/time)**2/(1.602176634e-16)
    return {"neutron_energy_keV":energy_keV,"neutron_speed_m_s":length/time,
            "equation":"E_keV=0.5*m_n*(L/t)^2/(1.602176634e-16 J/keV)",
            "assumption":"Non-relativistic time-of-flight estimate; timing offsets and detector response must be calibrated."}
