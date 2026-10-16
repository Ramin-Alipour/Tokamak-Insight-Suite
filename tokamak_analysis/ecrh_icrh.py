"""ECRH/ICRH resonance, deposition and absorbed-power bookkeeping."""
import numpy as np
E=1.602176634e-19; ME=9.1093837139e-31; U=1.66053906660e-27
def ecrh_resonance_field(frequency_Hz, harmonic=1):
    f=np.asarray(frequency_Hz,dtype=float); h=int(harmonic)
    if not np.all(np.isfinite(f)) or np.any(f<=0) or h<1: raise ValueError("Positive frequency and harmonic required")
    return {"resonance_field_T":2*np.pi*ME*f/(E*h),"harmonic":h,"caveat":"Relativistic and Doppler shifts are not included."}
def icrh_resonance_field(frequency_Hz, ion_mass_amu, charge_state=1, harmonic=1):
    f=np.asarray(frequency_Hz,dtype=float); mass=float(ion_mass_amu); charge=int(charge_state); h=int(harmonic)
    if not np.all(np.isfinite(f)) or np.any(f<=0) or not np.isfinite(mass) or mass<=0 or charge<1 or h<1:
        raise ValueError("Frequency, ion mass, charge state, and harmonic must be positive and finite")
    field=2*np.pi*(mass*U)*f/(E*charge*h)
    return {"resonance_field_T":field,"ion_mass_amu":mass,"charge_state":charge,"harmonic":h,"caveat":"Uses the nonrelativistic ion-cyclotron resonance condition; species and harmonic must match the experiment."}
def rf_power_balance(input_power_W, reflected_power_W, transmission_loss_W=0.0):
    pin,pr,loss=map(float,(input_power_W,reflected_power_W,transmission_loss_W))
    if not all(np.isfinite(v) for v in (pin,pr,loss)) or pin<0 or pr<0 or loss<0 or pr+loss>pin: raise ValueError("Invalid RF power balance")
    return {"net_delivered_power_W":pin-pr-loss,"reflection_fraction":pr/pin if pin else float("nan"),"loss_fraction":loss/pin if pin else float("nan"),"caveat":"Delivered power is not necessarily absorbed power; plasma coupling and deposition are separate."}


def relativistic_ecrh_resonance_field(frequency_Hz, gamma_factor=1.0, harmonic=1):
    """Relativistic electron cyclotron resonance: omega=h*e*B/(gamma*m_e)."""
    f,gamma=np.broadcast_arrays(np.asarray(frequency_Hz,dtype=float),np.asarray(gamma_factor,dtype=float)); h=int(harmonic)
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(gamma)) or np.any(f<=0) or np.any(gamma<1) or h<1:
        raise ValueError("Frequency positive, gamma >=1, and harmonic positive required")
    return {"resonance_field_T":2*np.pi*ME*f*gamma/(E*h),"equation":"B_res=2*pi*m_e*f*gamma/(e*h)"}
