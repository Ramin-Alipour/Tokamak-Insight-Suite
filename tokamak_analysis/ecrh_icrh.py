"""ECRH/ICRH resonance, deposition and absorbed-power bookkeeping."""
import numpy as np
E=1.602176634e-19; ME=9.1093837139e-31
def ecrh_resonance_field(frequency_Hz, harmonic=1):
    f=np.asarray(frequency_Hz,dtype=float); h=int(harmonic)
    if not np.all(np.isfinite(f)) or np.any(f<=0) or h<1: raise ValueError("Positive frequency and harmonic required")
    return {"resonance_field_T":2*np.pi*ME*f/(E*h),"harmonic":h,"caveat":"Relativistic and Doppler shifts are not included."}
def rf_power_balance(input_power_W, reflected_power_W, transmission_loss_W=0.0):
    pin,pr,loss=map(float,(input_power_W,reflected_power_W,transmission_loss_W))
    if not all(np.isfinite(v) for v in (pin,pr,loss)) or pin<0 or pr<0 or loss<0 or pr+loss>pin: raise ValueError("Invalid RF power balance")
    return {"net_delivered_power_W":pin-pr-loss,"reflection_fraction":pr/pin if pin else float("nan"),"loss_fraction":loss/pin if pin else float("nan"),"caveat":"Delivered power is not necessarily absorbed power; plasma coupling and deposition are separate."}