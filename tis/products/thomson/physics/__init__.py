"""Thomson product-owned solver namespace.

Shared-domain kernels must be imported explicitly from ``tis.physics.shared``.
Keeping this package path local prevents accidental resolution of missing
product modules from a sibling shared directory and makes import ownership
visible to Python, static analysis, and the dependency inventory.
"""

