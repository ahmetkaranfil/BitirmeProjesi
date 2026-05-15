"""Shared pytest configuration.

Registers and loads a ``ddd`` Hypothesis profile that runs each
property test for ``max_examples=100`` iterations and disables
deadlines so slower property generators do not flake the suite. This
matches the property-based testing configuration documented in the
feature design (every property test must run at least 100 iterations).
"""

from hypothesis import HealthCheck, settings

# A single shared profile keeps every property test consistent with the
# design contract: exactly 100 examples, no deadline so heavier
# generators (e.g. long Prediction sequences) do not flake.
settings.register_profile(
    "ddd",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("ddd")
