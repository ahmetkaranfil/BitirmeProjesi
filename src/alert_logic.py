"""Uyari_Mantigi module (saf cekirdek).

Defines the immutable ``Prediction``, ``AlertConfig``, ``AlertState``,
and ``AlertEvent`` dataclasses and the pure ``update`` function that
runs the eye / mouth state machines (Requirements 5.1-5.7, 6.1-6.7;
properties P1-P7).

This file implements tasks 4.1, 4.2 and 4.5:

* Frozen dataclasses for predictions, alert configuration, alert state
  and alert events.
* ``Literal`` aliases for the closed value-sets used by the system
  (eye label, mouth label, alert kind).
* ``INITIAL_STATE`` constant, the canonical empty starting state for
  the alert engine.
* Per-prediction invariant enforcement inside
  ``Prediction.__post_init__`` (``eye_conf`` and ``mouth_conf`` must
  lie in ``[0.0, 1.0]``).
* :func:`ensure_monotonic_capture` helper that enforces the
  cross-call monotonicity of ``t_capture_s``.
* :func:`update`, a pure function that drives both the eye-closure
  drowsy alert (Requirements 5.x) and the rolling-window fatigue
  alert (Requirements 6.x) state machines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Literal, Optional, Tuple


# ---------------------------------------------------------------------------
# Literal aliases for the closed value-sets used throughout the system.
# ---------------------------------------------------------------------------

EyeLabel = Literal["Open", "Closed"]
MouthLabel = Literal["yawn", "no_yawn"]
AlertKind = Literal["DROWSY", "FATIGUE"]


# ---------------------------------------------------------------------------
# Data models.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    """A single per-frame classification result handed to the alert engine.

    Attributes
    ----------
    eye:
        ``"Open"``, ``"Closed"`` or ``None``. ``None`` indicates that
        the eye confidence fell below ``confidence_threshold`` and the
        frame must be ignored by the eye state machine
        (Requirement 5.7, property P6).
    mouth:
        ``"yawn"``, ``"no_yawn"`` or ``None``. ``None`` indicates that
        the mouth confidence fell below ``confidence_threshold`` and
        the frame must be ignored by the mouth state machine
        (Requirement 6.7, property P6).
    eye_conf, mouth_conf:
        Confidence of the chosen label, in the closed interval
        ``[0.0, 1.0]``.
    raw_scores:
        Raw per-class scores for all four classes
        (``"Closed"``, ``"Open"``, ``"no_yawn"``, ``"yawn"``).
        Kept for logging / diagnostics; the alert engine does not
        depend on it.
    t_capture_s:
        Monotonic capture timestamp in seconds. Must be non-decreasing
        across consecutive calls to ``update`` (enforced there with
        :func:`ensure_monotonic_capture`).
    """

    eye: Optional[EyeLabel]
    mouth: Optional[MouthLabel]
    eye_conf: float
    mouth_conf: float
    raw_scores: Dict[str, float]
    t_capture_s: float

    def __post_init__(self) -> None:
        # Confidence values must lie in [0.0, 1.0]. NaN comparisons
        # always evaluate to False, so the chained comparison below
        # also rejects NaN.
        if not (0.0 <= self.eye_conf <= 1.0):
            raise ValueError(
                f"Prediction.eye_conf must be in [0.0, 1.0]; got {self.eye_conf!r}"
            )
        if not (0.0 <= self.mouth_conf <= 1.0):
            raise ValueError(
                f"Prediction.mouth_conf must be in [0.0, 1.0]; got {self.mouth_conf!r}"
            )
        # ``t_capture_s`` must be a finite real number; cross-call
        # monotonicity is enforced in ``update`` via
        # :func:`ensure_monotonic_capture`.
        if not math.isfinite(self.t_capture_s):
            raise ValueError(
                f"Prediction.t_capture_s must be finite; got {self.t_capture_s!r}"
            )


@dataclass(frozen=True)
class AlertConfig:
    """Pure-core subset of :class:`AppConfig` consumed by ``update``.

    Only the four parameters that influence the alert state machines
    are exposed here so the pure core stays decoupled from camera /
    logging / model parameters.

    Attributes
    ----------
    closed_eye_duration_s:
        Drowsy threshold in seconds (Requirement 5.5; range
        ``[0.5, 10.0]``).
    yawn_count:
        Fatigue threshold in number of yawn events (Requirement 6.5;
        range ``[1, 20]``).
    yawn_time_window_s:
        Rolling window in seconds in which yawn events are counted
        (Requirement 6.5; range ``[10.0, 600.0]``).
    confidence_threshold:
        Minimum eye / mouth confidence for a frame to influence the
        state machines (range ``[0.0, 1.0]``). Validation of this
        threshold itself is performed by :func:`config.validate`; the
        alert engine relies on the producer of :class:`Prediction` to
        have already collapsed sub-threshold labels to ``None``.
    """

    closed_eye_duration_s: float
    yawn_count: int
    yawn_time_window_s: float
    confidence_threshold: float


@dataclass(frozen=True)
class AlertState:
    """Immutable snapshot of the two state machines combined.

    Eye branch (Requirements 5.1-5.7):

    * ``eye_closed_start_s`` - monotonic timestamp at which the
      current ``Closed`` block began, or ``None`` when the eyes are
      currently open (or no observation has been seen yet).
    * ``eyes_currently_closed`` - convenience flag mirroring the
      latest non-``None`` eye observation.
    * ``drowsy_alert_active`` - latched while the driver remains
      drowsy; cleared by the first ``Open`` observation
      (Requirement 5.6).
    * ``last_eye_was_closed`` - retained so the mouth branch and
      external observers can detect ``Open -> Closed`` transitions.

    Mouth branch (Requirements 6.1-6.7):

    * ``yawn_event_times_s`` - timestamps of yawn events that fall
      inside the rolling ``yawn_time_window_s``, ordered oldest first.
    * ``in_yawn_block`` - latched while the mouth observation stays at
      ``yawn``; cleared on ``no_yawn`` so the next ``yawn`` counts as
      a new event (Requirement 6.6, property P4).
    * ``fatigue_alert_active`` - latched while the windowed event
      count is at or above ``yawn_count``; cleared as soon as the
      count drops below the threshold (Requirement 6.4).
    """

    eye_closed_start_s: Optional[float]
    eyes_currently_closed: bool
    drowsy_alert_active: bool
    last_eye_was_closed: bool
    yawn_event_times_s: Tuple[float, ...]
    in_yawn_block: bool
    fatigue_alert_active: bool


@dataclass(frozen=True)
class AlertEvent:
    """An alert produced by ``update``.

    ``kind`` is one of ``"DROWSY"`` (Requirement 5.3) or
    ``"FATIGUE"`` (Requirement 6.3). ``t_event_s`` always equals the
    ``t_capture_s`` of the triggering :class:`Prediction`, so it is
    monotonic by construction.
    """

    kind: AlertKind
    message: str
    t_event_s: float


# ---------------------------------------------------------------------------
# Initial state.
# ---------------------------------------------------------------------------

INITIAL_STATE: AlertState = AlertState(
    eye_closed_start_s=None,
    eyes_currently_closed=False,
    drowsy_alert_active=False,
    last_eye_was_closed=False,
    yawn_event_times_s=(),
    in_yawn_block=False,
    fatigue_alert_active=False,
)


# ---------------------------------------------------------------------------
# Cross-prediction invariants helper (used by ``update`` in 4.2 / 4.5).
# ---------------------------------------------------------------------------


def ensure_monotonic_capture(
    state: AlertState, pred: Prediction
) -> None:
    """Raise ``ValueError`` if ``pred.t_capture_s`` rewinds the clock.

    The alert engine's time reasoning (Requirements 5.1, 5.2, 6.2)
    requires a non-decreasing capture clock. The latest observed
    timestamp is the maximum of ``state.eye_closed_start_s`` and the
    last yawn-event timestamp; if neither is set the call is a no-op.

    This helper is provided here so it can be unit-tested in isolation
    and reused by ``update`` in tasks 4.2 / 4.5 without bloating that
    function.
    """

    candidates = []
    if state.eye_closed_start_s is not None:
        candidates.append(state.eye_closed_start_s)
    if state.yawn_event_times_s:
        candidates.append(state.yawn_event_times_s[-1])
    if not candidates:
        return
    last_seen = max(candidates)
    if pred.t_capture_s < last_seen:
        raise ValueError(
            "Prediction.t_capture_s must be non-decreasing across calls; "
            f"got {pred.t_capture_s!r} after {last_seen!r}"
        )


__all__ = [
    "EyeLabel",
    "MouthLabel",
    "AlertKind",
    "Prediction",
    "AlertConfig",
    "AlertState",
    "AlertEvent",
    "INITIAL_STATE",
    "ensure_monotonic_capture",
    "update",
]


# ---------------------------------------------------------------------------
# Pure update function (eye branch only; mouth branch is task 4.5).
# ---------------------------------------------------------------------------


_DROWSY_MESSAGE = "UYARI: Sürücü uyukluyor olabilir!"
_FATIGUE_MESSAGE = "UYARI: Sürücü yorgun olabilir!"


def update(
    state: AlertState,
    pred: Prediction,
    cfg: AlertConfig,
) -> Tuple[AlertState, List[AlertEvent]]:
    """Pure transition function for the alert state machines.

    This implementation covers both the eye branch (task 4.2,
    Requirements 5.1-5.4, 5.6, 5.7) and the mouth branch (task 4.5,
    Requirements 6.1-6.4, 6.6, 6.7).

    The function is total and side-effect free: it returns a new
    :class:`AlertState` and the (possibly empty) list of
    :class:`AlertEvent` instances triggered by ``pred``. Cross-call
    timestamp monotonicity is enforced first via
    :func:`ensure_monotonic_capture`.

    Eye branch behaviour:

    * ``pred.eye is None`` (low confidence) leaves the eye-related
      state and the event list completely unchanged (Requirement 5.7,
      property P6).
    * ``pred.eye == "Closed"`` starts a new closed block on the first
      ``Closed`` after an ``Open`` (or on the very first observation)
      by setting ``eye_closed_start_s = pred.t_capture_s``
      (Requirement 5.1). On every subsequent ``Closed`` the elapsed
      duration ``pred.t_capture_s - eye_closed_start_s`` is recomputed
      from the stored start time (Requirement 5.2). When that
      duration meets ``cfg.closed_eye_duration_s`` for the first time
      *while no drowsy alert is active*, exactly one ``DROWSY`` event
      is emitted and ``drowsy_alert_active`` is latched
      (Requirements 5.3, 5.6).
    * ``pred.eye == "Open"`` clears ``eye_closed_start_s``,
      ``eyes_currently_closed``, ``last_eye_was_closed`` and the
      latched ``drowsy_alert_active`` flag, so the next ``Closed``
      block starts fresh and a future threshold crossing fires a new
      event (Requirement 5.4).

    Mouth branch behaviour:

    * Stale events are pruned every call: timestamps ``t`` for which
      ``pred.t_capture_s - t > cfg.yawn_time_window_s`` are dropped
      from ``yawn_event_times_s`` (Requirement 6.2). The prune runs
      even when ``pred.mouth is None`` so events still expire as time
      advances.
    * ``pred.mouth is None`` (low confidence) leaves
      ``in_yawn_block`` untouched; only the time-window prune above
      runs (Requirement 6.7, property P6).
    * ``pred.mouth == "yawn"`` records ``pred.t_capture_s`` as a new
      yawn event only on a ``no_yawn -> yawn`` transition (i.e. when
      ``in_yawn_block`` is False); subsequent ``yawn`` frames inside
      the same block are debounced and do not create extra events
      (Requirements 6.1, 6.6, property P4). ``in_yawn_block`` is
      latched True for the duration of the block.
    * ``pred.mouth == "no_yawn"`` clears ``in_yawn_block`` so the next
      ``yawn`` will count as a fresh event (Requirement 6.6).
    * After updating, when the windowed event count reaches
      ``cfg.yawn_count`` *while no fatigue alert is active*, exactly
      one ``FATIGUE`` event is emitted and ``fatigue_alert_active``
      is latched (Requirement 6.3). When the count later drops below
      the threshold the flag is cleared so a future crossing can fire
      another event (Requirement 6.4).
    """

    ensure_monotonic_capture(state, pred)

    new_state = state
    events: List[AlertEvent] = []

    # --- Eye branch ---------------------------------------------------
    if pred.eye is None:
        # Low-confidence eye frames are skipped entirely; no field
        # related to the eye state machine changes (Req 5.7, P6).
        pass
    elif pred.eye == "Closed":
        if new_state.eye_closed_start_s is None:
            # First Closed after an Open (or the first valid eye
            # observation): anchor the closed-block start at the
            # current capture timestamp (Req 5.1, P2).
            new_state = replace(
                new_state,
                eye_closed_start_s=pred.t_capture_s,
                eyes_currently_closed=True,
                last_eye_was_closed=True,
            )
        else:
            # Continuing a closed block: keep the existing start time
            # so the accumulated duration grows with each frame
            # (Req 5.2).
            new_state = replace(
                new_state,
                eyes_currently_closed=True,
                last_eye_was_closed=True,
            )

        closed_duration = pred.t_capture_s - new_state.eye_closed_start_s
        if (
            closed_duration >= cfg.closed_eye_duration_s
            and not new_state.drowsy_alert_active
        ):
            events.append(
                AlertEvent(
                    kind="DROWSY",
                    message=_DROWSY_MESSAGE,
                    t_event_s=pred.t_capture_s,
                )
            )
            new_state = replace(new_state, drowsy_alert_active=True)
    else:  # pred.eye == "Open"
        new_state = replace(
            new_state,
            eye_closed_start_s=None,
            eyes_currently_closed=False,
            last_eye_was_closed=False,
            drowsy_alert_active=False,
        )

    # --- Mouth branch -------------------------------------------------
    # Step 1: handle the no_yawn -> yawn transition and the
    # debouncing flag. ``pred.mouth is None`` (low confidence) leaves
    # ``in_yawn_block`` untouched so the next valid frame still sees
    # the same edge state (Req 6.7, P6).
    yawn_times = new_state.yawn_event_times_s
    if pred.mouth == "yawn":
        if not new_state.in_yawn_block:
            # no_yawn -> yawn transition: record a new event
            # (Req 6.1) and latch the debounce flag (Req 6.6, P4).
            yawn_times = yawn_times + (pred.t_capture_s,)
            new_state = replace(
                new_state,
                yawn_event_times_s=yawn_times,
                in_yawn_block=True,
            )
        # Continuing a yawn block: no new event is recorded
        # (Req 6.6, P4).
    elif pred.mouth == "no_yawn":
        # Clear the debounce flag so the next yawn counts again
        # (Req 6.6).
        if new_state.in_yawn_block:
            new_state = replace(new_state, in_yawn_block=False)

    # Step 2: prune events older than the rolling window. This runs
    # every call (including ``pred.mouth is None``) so events
    # naturally expire as time advances (Req 6.2).
    window = cfg.yawn_time_window_s
    pruned = tuple(
        t for t in new_state.yawn_event_times_s
        if pred.t_capture_s - t <= window
    )
    if pruned != new_state.yawn_event_times_s:
        new_state = replace(new_state, yawn_event_times_s=pruned)

    # Step 3: emit / clear the FATIGUE alert based on the windowed
    # count.
    count = len(new_state.yawn_event_times_s)
    if count >= cfg.yawn_count and not new_state.fatigue_alert_active:
        events.append(
            AlertEvent(
                kind="FATIGUE",
                message=_FATIGUE_MESSAGE,
                t_event_s=pred.t_capture_s,
            )
        )
        new_state = replace(new_state, fatigue_alert_active=True)
    elif count < cfg.yawn_count and new_state.fatigue_alert_active:
        # Count dropped below the threshold; clear the latch so a
        # future crossing fires a new event (Req 6.4).
        new_state = replace(new_state, fatigue_alert_active=False)

    return new_state, events
