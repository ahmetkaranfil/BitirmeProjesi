"""Hypothesis strategies shared across property-based tests.

These strategies match the table documented in the feature design under
"Hypothesis Stratejileri" and feed the property tests for the alert
logic core (P1-P7) and the configuration validator (Property 8, 9).

This module is a placeholder created by task 1.1; concrete generators
are filled in by the property-test tasks (2.3, 2.4, 4.3-4.9, 6.3, 7.3,
10.4) as the corresponding production modules come online.
"""

# NOTE: Strategy implementations import the production dataclasses
# (Prediction, AppConfig, etc.). Those types are stubs at this stage,
# so the bodies below are intentionally left as ``NotImplementedError``
# placeholders so tests that import them prematurely fail loudly
# instead of silently passing.

from hypothesis import strategies as st


def eye_label() -> st.SearchStrategy:
    """Generate ``"Open"``, ``"Closed"`` or ``None`` for an eye state.

    Used to build ``Prediction`` sequences for properties P1, P2, P5,
    P6, P7.
    """
    raise NotImplementedError("Filled in by property-test tasks")


def mouth_label() -> st.SearchStrategy:
    """Generate ``"yawn"``, ``"no_yawn"`` or ``None`` for a mouth state.

    Used to build ``Prediction`` sequences for properties P3, P4, P5,
    P6, P7.
    """
    raise NotImplementedError("Filled in by property-test tasks")


def prediction_sequences() -> st.SearchStrategy:
    """Generate lists of ``Prediction`` with monotonically increasing
    ``t_capture_s`` timestamps.

    Used by every alert-logic property (P1-P7).
    """
    raise NotImplementedError("Filled in by property-test tasks")


def closed_eye_durations() -> st.SearchStrategy:
    """Generate floats in ``[0.5, 10.0]`` for ``closed_eye_duration``.

    Used by P1, P5, and Property 8 (validator boundaries).
    """
    raise NotImplementedError("Filled in by property-test tasks")


def yawn_thresholds() -> st.SearchStrategy:
    """Generate ``(yawn_count, yawn_time_window)`` pairs.

    ``yawn_count`` is drawn from ``{1, ..., 20}`` and
    ``yawn_time_window`` from ``[10.0, 600.0]``. Used by P3 and P5.
    """
    raise NotImplementedError("Filled in by property-test tasks")


def frame_skip_values() -> st.SearchStrategy:
    """Generate frame-skip values in ``{1, ..., 10}`` (Property P7)."""
    raise NotImplementedError("Filled in by property-test tasks")


def valid_config_values(param: str) -> st.SearchStrategy:
    """Generate an in-range value for the named ``AppConfig`` parameter.

    Used by Property 8 (validator accepts every in-range value) and
    Property 9 (defaulting behaviour).
    """
    raise NotImplementedError("Filled in by property-test tasks")


def invalid_config_values(param: str) -> st.SearchStrategy:
    """Generate an out-of-range / wrong-type value for the named
    ``AppConfig`` parameter.

    Used by Property 8 to assert that ``validate`` raises
    ``ConfigError`` for every invalid value.
    """
    raise NotImplementedError("Filled in by property-test tasks")
