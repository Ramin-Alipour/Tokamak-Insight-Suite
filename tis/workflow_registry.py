"""Lazy lookup for callable diagnostic workflows exposed by the workbench."""
from __future__ import annotations

from importlib import import_module
from inspect import signature
from typing import Any

_MODULES = (
    "core", "plasma", "signals", "events", "thomson", "ece", "cxrs", "molecular",
    "thomson_distribution", "magnetic", "polarimetry", "ballpen", "gundestrup",
    "ion_saturation", "potential_array", "probe_fluctuations", "optical_spectroscopy",
    "vuv", "charge_exchange", "instrument_response", "interferometry", "reflectometry",
    "raytrace", "collective_scattering", "fast_bolometry", "sxr", "hard_xray",
    "neutron_yield", "neutron_spectrometry", "neutron_tomography", "gamma_spectroscopy",
    "npa", "fida", "fild", "runaway", "pressure_gauge", "rga", "gas_flow",
    "neutral_recycling", "pumping_speed", "particle_transport", "energy_balance",
    "nbi", "ecrh", "icrh", "lower_hybrid", "heating_power", "thermocouple",
    "surface_change", "dust_monitor", "strike_point", "thermal_response", "noise_floor",
    "channel_sync", "ir_thermography", "visible_imaging", "equilibrium",
)


def resolve_workflow(name: str):
    """Return a named callable from a known TIS module without eager-importing all modules."""
    for module_name in _MODULES:
        try:
            module = import_module(f"tis.{module_name}")
        except Exception:
            continue
        value = getattr(module, name, None)
        if callable(value):
            return value
    raise LookupError(f"No callable workflow named {name!r} was found in the registered TIS modules.")


def workflow_signature(name: str) -> str:
    """Return a concise callable signature and docstring for UI help."""
    fn = resolve_workflow(name)
    doc = (fn.__doc__ or "").strip().splitlines()
    return f"{signature(fn)}\n\n{doc[0] if doc else 'No summary available.'}"
