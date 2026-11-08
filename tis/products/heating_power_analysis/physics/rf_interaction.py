"""RF-plasma coupling bookkeeping from incident/reflected/transmitted powers."""
import numpy as np
def analyze_rf_interaction(incident_W, reflected_W, transmitted_W=0.0, *, uncertainty_W=None):
    pi,pr,pt=map(float,(incident_W,reflected_W,transmitted_W))
    if not all(np.isfinite(v) for v in (pi,pr,pt)) or pi<0 or pr<0 or pt<0 or pr+pt>pi: raise ValueError("Invalid RF power terms")
    absorbed=pi-pr-pt; out={"incident_power_W":pi,"reflected_power_W":pr,"transmitted_power_W":pt,"unaccounted_power_W":absorbed,"coupling_fraction":absorbed/pi if pi else float("nan")}
    if uncertainty_W is not None:
        u=np.asarray(uncertainty_W,dtype=float)
        if u.shape!=(3,) or not np.all(np.isfinite(u)) or np.any(u<0): raise ValueError("uncertainty_W must be [incident, reflected, transmitted] nonnegative values")
        out["unaccounted_power_uncertainty_W"]=float(np.sqrt(np.sum(u**2)))
    out["caveat"]="Unaccounted power can include line losses and measurement bias; it is not automatically plasma-absorbed power."
    return out