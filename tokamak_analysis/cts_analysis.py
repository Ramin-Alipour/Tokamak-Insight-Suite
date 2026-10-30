"""Dedicated CTS entry point; microwave CTS inversion is not yet implemented."""


def analyze_cts_spectrum(*args, **kwargs):
    """Fail closed rather than routing CTS data through unrelated calculations."""
    raise NotImplementedError(
        "Microwave CTS inversion is not implemented: a frequency-domain collective "
        "dielectric forward model, beam geometry, instrument response and uncertainty-aware "
        "ion-parameter inversion are required."
    )
