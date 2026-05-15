"""Kamera_Yakalayicisi and live-detection main loop.

Owns ``CameraCapture``, the overlay helper, and the wiring that joins
config, logger, predictor, alert engine, and sound alerter for live
detection (Requirements 4.1-4.7, 10.6, 10.7).

This file currently implements tasks 10.1, 10.2 and 10.3:

* :class:`CameraConfig` - a frozen dataclass holding the camera
  source identifier, the open timeout, and the target capture format
  (``640x480`` / ``>= 15 FPS`` per Requirements 4.1 and 10.2). The
  remaining fields (``frame_skip``, ``consecutive_read_fail_limit``,
  ``no_frame_timeout_s``) drive the :meth:`CameraCapture.frames`
  iterator below.
* :class:`CameraCapture` - opens a ``cv2.VideoCapture`` with a polling
  timeout, applies the target capture properties on success, exposes
  the :meth:`frames` generator that decimates frames for prediction
  per ``frame_skip`` and aborts on read failures (Requirements 4.6,
  4.7, 10.6, 10.7), and releases the device on close. Supports the
  context-manager protocol so callers can write
  ``with CameraCapture(cfg) as cap: ...``.
* :func:`draw_overlay` - draws ``Goz_Durumu``, ``Agiz_Durumu`` and
  active alert text on the frame with auto-sized font and contrasting
  background rectangles (Requirement 4.4). The font is scaled so each
  glyph is at least 3% of the frame height and the combined overlay
  rectangles cover at most 25% of the frame area; oversized layouts
  are shrunk until they fit.

The main loop (task 12.1) is intentionally NOT implemented here.

OpenCV (``cv2``) is imported lazily inside :meth:`CameraCapture.open`
and :meth:`CameraCapture.frames` so this module can be imported by
the alert-logic / config test suites on environments where OpenCV is
not installed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Iterator, Optional, Type, Union

if TYPE_CHECKING:  # pragma: no cover - hints only
    import numpy as np  # noqa: F401  (used in string annotations)


# Validation bounds for the ``source`` field, taken straight from
# Requirement 4.2: integer indices ``0..9`` or string sources whose
# length is in ``(0, 260]``.
_MAX_CAMERA_INDEX: int = 9
_MAX_SOURCE_STR_LEN: int = 260

# Polling cadence used while waiting for ``cv2.VideoCapture.isOpened``
# to flip to ``True``. 50 ms gives a sub-second worst-case wakeup
# without spinning the CPU.
_OPEN_POLL_INTERVAL_S: float = 0.05

# Inclusive bounds for the effective ``frame_skip`` value used by
# :meth:`CameraCapture.frames`. Values outside this range are clamped
# to ``_FRAME_SKIP_FALLBACK`` and a WARNING is logged
# (Requirement 10.7).
_FRAME_SKIP_MIN: int = 1
_FRAME_SKIP_MAX: int = 10
_FRAME_SKIP_FALLBACK: int = 1

# ``cv2.waitKey`` poll cadence used by the frames iterator. 1 ms is
# the documented minimum that still pumps the OpenCV event loop and
# keeps the ``q`` keypress latency well under the 1 s budget set by
# Requirement 4.7.
_WAITKEY_DELAY_MS: int = 1


# ---------------------------------------------------------------------------
# Overlay rendering constants (Requirement 4.4).
# ---------------------------------------------------------------------------

# Fraction of the frame height each glyph must reach. Requirement 4.4
# mandates >= 3 %; we aim for 4 % so a sensible margin survives the
# integer rounding done by ``cv2.getTextSize``.
_OVERLAY_GLYPH_FRACTION: float = 0.04
# Floor for the glyph height in pixels. ``cv2.FONT_HERSHEY_SIMPLEX``
# becomes unreadable below ~12 px regardless of frame size.
_OVERLAY_GLYPH_MIN_PX: int = 12
# ``cv2.FONT_HERSHEY_SIMPLEX`` produces glyphs roughly 22 px tall at
# ``fontScale=1.0``; this calibrates ``font_scale`` from the desired
# glyph height.
_OVERLAY_FONT_BASELINE_PX: int = 22
# Padding (in pixels) added to every side of the background rectangle
# behind a line of text. Keeps text from touching the rectangle edge
# while still being compact.
_OVERLAY_RECT_PADDING_PX: int = 4
# Vertical gap between consecutive overlay lines.
_OVERLAY_LINE_GAP_PX: int = 6
# Distance from the top-left corner of the frame to the first line.
_OVERLAY_MARGIN_PX: int = 6
# Maximum fraction of the frame area that the combined background
# rectangles may cover (Requirement 4.4).
_OVERLAY_MAX_AREA_FRACTION: float = 0.25
# Multiplicative factor applied to ``font_scale`` while shrinking the
# overlay so it fits inside the area budget. < 1.0 so the loop makes
# progress; not so small that we converge in many iterations.
_OVERLAY_SHRINK_FACTOR: float = 0.9
# Hard floor for ``font_scale`` while shrinking. Below this value the
# text is unreadable; we stop shrinking and accept the layout even if
# it nominally exceeds the 25 % budget (extremely small frames with
# many alert lines).
_OVERLAY_MIN_FONT_SCALE: float = 0.3
# Maximum number of shrink iterations. With factor 0.9 this lets us
# drop ``font_scale`` by ~88 % which is well past the readability
# floor; we exit early via ``_OVERLAY_MIN_FONT_SCALE`` in practice.
_OVERLAY_MAX_SHRINK_ITERS: int = 20

# BGR colours for the overlay. Black background with white text gives
# a strong contrast against arbitrary scene content (Requirement 4.4).
_OVERLAY_BG_COLOR: tuple[int, int, int] = (0, 0, 0)
_OVERLAY_FG_COLOR: tuple[int, int, int] = (255, 255, 255)


# ---------------------------------------------------------------------------
# Configuration dataclass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraConfig:
    """Immutable configuration for :class:`CameraCapture`.

    Attributes
    ----------
    source:
        Camera source. Either an integer device index in ``[0, 9]``
        (Requirement 4.2) or a non-empty string up to 260 characters
        (a file path or RTSP/USB descriptor, also Requirement 4.2).
        Defaults to ``0`` so live detection picks the system default
        camera (Requirement 4.1).
    open_timeout_s:
        Maximum number of seconds :meth:`CameraCapture.open` is
        allowed to wait for the device before raising. Defaults to
        ``5.0`` per Requirement 4.5.
    target_width, target_height:
        Capture resolution requested via
        ``cv2.CAP_PROP_FRAME_WIDTH`` / ``CAP_PROP_FRAME_HEIGHT``.
        Defaults to ``640x480`` per Requirements 4.1 and 10.2.
    target_fps:
        Capture frame rate requested via ``cv2.CAP_PROP_FPS``.
        Defaults to ``15.0`` to satisfy the ``>= 15 FPS`` floor in
        Requirements 4.1 and 10.2. Drivers may report or deliver a
        different rate; that is handled by the runtime FPS tracker
        (task 12.x), not here.
    frame_skip:
        Hint for task 10.2: only every ``frame_skip``-th frame is
        forwarded to the predictor. Validation and behaviour live in
        task 10.2; the field is kept here so the dataclass shape does
        not churn.
    consecutive_read_fail_limit:
        Maximum number of consecutive failed reads tolerated before
        the iterator (task 10.2) terminates the stream
        (Requirement 4.6).
    no_frame_timeout_s:
        Maximum number of seconds the iterator (task 10.2) may go
        without receiving a fresh frame before terminating
        (Requirement 4.6).
    """

    source: Union[int, str] = 0
    open_timeout_s: float = 5.0
    target_width: int = 640
    target_height: int = 480
    target_fps: float = 15.0
    frame_skip: int = 1
    consecutive_read_fail_limit: int = 30
    no_frame_timeout_s: float = 3.0


# ---------------------------------------------------------------------------
# Validation helpers.
# ---------------------------------------------------------------------------


def _validate_source(source: Any) -> None:
    """Validate the ``CameraConfig.source`` value (Requirement 4.2).

    Raises
    ------
    ValueError
        If ``source`` is neither an ``int`` in ``[0, 9]`` nor a
        non-empty string of at most 260 characters. ``bool`` values
        are rejected explicitly because Python treats them as
        integers, which would silently accept ``True`` as device
        index ``1``.
    """

    # ``bool`` is a subclass of ``int``; reject it explicitly so a
    # stray ``True`` / ``False`` cannot pose as a device index.
    if isinstance(source, bool):
        raise ValueError(
            f"CameraConfig.source must be an int in [0, {_MAX_CAMERA_INDEX}] "
            f"or a non-empty string up to {_MAX_SOURCE_STR_LEN} characters; "
            f"got bool {source!r}"
        )
    if isinstance(source, int):
        if not (0 <= source <= _MAX_CAMERA_INDEX):
            raise ValueError(
                f"CameraConfig.source integer must be in "
                f"[0, {_MAX_CAMERA_INDEX}]; got {source!r}"
            )
        return
    if isinstance(source, str):
        if len(source) == 0 or len(source) > _MAX_SOURCE_STR_LEN:
            raise ValueError(
                f"CameraConfig.source string length must be in "
                f"(0, {_MAX_SOURCE_STR_LEN}]; got length {len(source)}"
            )
        return
    raise ValueError(
        f"CameraConfig.source must be an int in [0, {_MAX_CAMERA_INDEX}] "
        f"or a non-empty string up to {_MAX_SOURCE_STR_LEN} characters; "
        f"got {type(source).__name__} {source!r}"
    )


def _validate_camera_config(cfg: CameraConfig) -> None:
    """Validate every field of ``cfg`` that :meth:`open` relies on."""

    _validate_source(cfg.source)
    if not (cfg.open_timeout_s > 0):
        raise ValueError(
            f"CameraConfig.open_timeout_s must be > 0; "
            f"got {cfg.open_timeout_s!r}"
        )
    if cfg.target_width <= 0:
        raise ValueError(
            f"CameraConfig.target_width must be > 0; "
            f"got {cfg.target_width!r}"
        )
    if cfg.target_height <= 0:
        raise ValueError(
            f"CameraConfig.target_height must be > 0; "
            f"got {cfg.target_height!r}"
        )
    if not (cfg.target_fps > 0):
        raise ValueError(
            f"CameraConfig.target_fps must be > 0; "
            f"got {cfg.target_fps!r}"
        )


# ---------------------------------------------------------------------------
# Camera capture wrapper.
# ---------------------------------------------------------------------------


class CameraCapture:
    """Thin wrapper around ``cv2.VideoCapture`` with timeout-bounded open.

    The class is kept deliberately small in this task; it exposes
    ``open`` / ``close`` and the context-manager protocol. The frame
    iteration logic lives in :meth:`frames` (task 10.2).

    Parameters
    ----------
    cfg:
        Validated :class:`CameraConfig` instance. The constructor
        re-validates every field so callers cannot mutate the
        dataclass via ``object.__setattr__`` and bypass the checks
        (the ``frozen=True`` dataclass hardens the common path).
    """

    def __init__(self, cfg: CameraConfig) -> None:
        _validate_camera_config(cfg)
        self._cfg: CameraConfig = cfg
        # ``Any`` rather than ``cv2.VideoCapture`` because cv2 is
        # imported lazily inside ``open`` to keep this module
        # importable in environments without OpenCV.
        self._capture: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public read-only accessors.
    # ------------------------------------------------------------------

    @property
    def cfg(self) -> CameraConfig:
        """Return the validated configuration backing this capture."""

        return self._cfg

    def is_open(self) -> bool:
        """Return ``True`` iff a backing ``VideoCapture`` is open."""

        cap = self._capture
        if cap is None:
            return False
        try:
            return bool(cap.isOpened())
        except Exception:
            # Defensive: a misbehaving backend should not hide the
            # fact that we hold a handle. Treat as "not open" so the
            # caller's recovery path runs.
            return False

    # ------------------------------------------------------------------
    # Lifecycle.
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the camera device, polling ``isOpened`` until ready.

        Raises
        ------
        RuntimeError
            If the device is not reported open within
            ``cfg.open_timeout_s`` seconds. The message names the
            source and the timeout (Requirement 4.5).
        """

        if self.is_open():
            # Idempotent: second ``open`` call is a no-op so callers
            # can defensively re-arm the capture without leaking
            # handles.
            return

        # Deferred import: keeps this module importable in test
        # environments that do not have OpenCV installed (the
        # alert-logic / config suites import the package eagerly).
        import cv2  # type: ignore[import-not-found]

        cap = cv2.VideoCapture(self._cfg.source)

        deadline = time.monotonic() + float(self._cfg.open_timeout_s)
        # Poll until the device flips to ``isOpened`` or the deadline
        # is hit. ``isOpened`` is cheap; a 50 ms sleep keeps the loop
        # responsive without spinning the CPU.
        while True:
            try:
                opened = bool(cap.isOpened())
            except Exception:
                opened = False
            if opened:
                break
            if time.monotonic() >= deadline:
                # Release whatever partial handle the backend created
                # before raising so the OS resource is not leaked.
                try:
                    cap.release()
                except Exception:
                    pass
                raise RuntimeError(
                    f"Camera source {self._cfg.source!r} could not be opened "
                    f"within {float(self._cfg.open_timeout_s):.2f}s"
                )
            time.sleep(_OPEN_POLL_INTERVAL_S)

        # Successful open: request the documented capture format
        # (Requirements 4.1, 10.2). ``VideoCapture.set`` returns a
        # bool indicating whether the backend accepted the property;
        # we do not fail the open on a rejection because some drivers
        # ignore unsupported properties silently. Performance issues
        # caused by an unsupported format are surfaced later by
        # ``FpsTracker`` (task 12.x).
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._cfg.target_width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._cfg.target_height))
            cap.set(cv2.CAP_PROP_FPS, float(self._cfg.target_fps))
        except Exception:
            # Defensive: if the backend explodes on ``set`` after a
            # successful open, release the device before propagating
            # so we do not leak the handle.
            try:
                cap.release()
            except Exception:
                pass
            raise

        self._capture = cap

    def close(self) -> None:
        """Release the backing ``VideoCapture`` if one is held.

        Safe to call multiple times. Exceptions raised by the OpenCV
        backend during release are swallowed so the caller's cleanup
        path always runs to completion.
        """

        cap = self._capture
        # Detach first so a double-close (or a release that throws)
        # cannot leave us in an inconsistent state.
        self._capture = None
        if cap is None:
            return
        try:
            cap.release()
        except Exception:
            # Deliberately swallowed: ``close`` is the cleanup path
            # and must not mask the original exception that caused
            # the caller to abandon the capture.
            pass

    # ------------------------------------------------------------------
    # Frame iteration (Requirements 4.6, 4.7, 10.6, 10.7).
    # ------------------------------------------------------------------

    def frames(
        self, logger: Optional[logging.Logger] = None
    ) -> Iterator[tuple[int, "np.ndarray", bool]]:
        """Yield ``(frame_index, frame_bgr, should_predict)`` tuples.

        ``should_predict`` is ``True`` only every ``frame_skip``-th
        yielded frame, so callers can keep the display path running
        on every frame while the predictor is throttled
        (Requirement 10.6). The iterator terminates cleanly when:

        * the user presses ``q`` (Requirement 4.7), surfaced via
          ``cv2.waitKey`` so the latency stays well under one second;
        * ``cfg.consecutive_read_fail_limit`` consecutive
          ``VideoCapture.read`` failures occur (Requirement 4.6); or
        * no successful frame is delivered for
          ``cfg.no_frame_timeout_s`` seconds (Requirement 4.6).

        On any termination path the camera resource is released
        before the generator returns (Requirement 4.6).

        Parameters
        ----------
        logger:
            Optional logger used for the ``frame_skip`` clamp warning
            (Requirement 10.7) and termination reasons. Falls back to
            the module logger when ``None``.

        Raises
        ------
        RuntimeError
            If the capture is not open. Callers must run inside the
            context manager (or call :meth:`open` first).
        """

        if not self.is_open():
            raise RuntimeError("CameraCapture is not open")

        log = logger if logger is not None else logging.getLogger(__name__)

        # Resolve the effective frame_skip up-front so we only log the
        # clamp warning once per iteration, no matter how many frames
        # are produced (Requirement 10.7).
        configured_skip = self._cfg.frame_skip
        if (
            not isinstance(configured_skip, int)
            or isinstance(configured_skip, bool)
            or configured_skip < _FRAME_SKIP_MIN
            or configured_skip > _FRAME_SKIP_MAX
        ):
            log.warning(
                "frame_skip=%r is outside [%d, %d]; falling back to %d",
                configured_skip,
                _FRAME_SKIP_MIN,
                _FRAME_SKIP_MAX,
                _FRAME_SKIP_FALLBACK,
            )
            effective_skip = _FRAME_SKIP_FALLBACK
        else:
            effective_skip = configured_skip

        # Deferred import: keeps the module importable in environments
        # without OpenCV (the alert-logic / config test suites import
        # this package eagerly).
        import cv2  # type: ignore[import-not-found]

        cap = self._capture
        # ``is_open`` already verified ``cap`` is non-None and live;
        # narrow the type for the loop below.
        assert cap is not None

        fail_limit = int(self._cfg.consecutive_read_fail_limit)
        no_frame_timeout = float(self._cfg.no_frame_timeout_s)

        consecutive_failures = 0
        frame_index = 0
        last_frame_time = time.monotonic()

        # ``cv2.waitKey`` requires a HighGUI window; on headless
        # builds it raises ``cv2.error``. Latch the failure so we
        # don't re-attempt (and re-log) on every frame.
        waitkey_unavailable = False

        try:
            while True:
                try:
                    ret, frame = cap.read()
                except Exception:
                    # Treat a backend exception the same as a clean
                    # ``ret == False`` so the consecutive-failure
                    # counter still terminates the loop deterministically.
                    ret, frame = False, None

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= fail_limit:
                        log.warning(
                            "Camera read failed %d consecutive times; "
                            "terminating capture",
                            consecutive_failures,
                        )
                        return
                    if (
                        time.monotonic() - last_frame_time
                        >= no_frame_timeout
                    ):
                        log.warning(
                            "No new frame received for >= %.2fs; "
                            "terminating capture",
                            no_frame_timeout,
                        )
                        return
                    # Brief sleep so a stuck backend does not spin the
                    # CPU while we wait for either the failure budget
                    # or the no-frame timeout to trip.
                    time.sleep(_OPEN_POLL_INTERVAL_S)
                    continue

                # Successful read: reset failure tracking before
                # yielding so caller exceptions cannot leave us in a
                # half-updated state.
                consecutive_failures = 0
                last_frame_time = time.monotonic()

                should_predict = (frame_index % effective_skip) == 0
                yield frame_index, frame, should_predict

                # Pump the HighGUI event loop so ``imshow`` windows
                # repaint and the ``q`` keypress is observed within
                # the 1 s budget mandated by Requirement 4.7. On
                # headless environments ``waitKey`` raises; we catch
                # once and skip subsequent calls.
                if not waitkey_unavailable:
                    try:
                        key = cv2.waitKey(_WAITKEY_DELAY_MS) & 0xFF
                    except Exception:
                        waitkey_unavailable = True
                    else:
                        if key == ord("q"):
                            log.info(
                                "Quit key pressed; terminating capture"
                            )
                            return

                frame_index += 1
        finally:
            # Whatever caused the loop to exit (return, exception,
            # caller closing the generator), release the device so we
            # do not leak the OS handle (Requirement 4.6).
            self.close()

    # ------------------------------------------------------------------
    # Context-manager protocol.
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Overlay helper (Goruntu_Bindirici, Requirement 4.4).
# ---------------------------------------------------------------------------


def _build_overlay_lines(
    eye_label: Optional[str],
    mouth_label: Optional[str],
    active_alerts: tuple[str, ...],
) -> list[str]:
    """Return the text lines drawn by :func:`draw_overlay`, in order.

    Eye/mouth labels are rendered with a ``"-"`` placeholder when
    ``None`` (no confident prediction yet) so the overlay never
    contains the word "None"; alert lines are appended verbatim.
    """

    eye_text = eye_label if eye_label is not None else "-"
    mouth_text = mouth_label if mouth_label is not None else "-"
    lines: list[str] = [
        f"Goz: {eye_text}",
        f"Agiz: {mouth_text}",
    ]
    lines.extend(active_alerts)
    return lines


def _measure_overlay(
    cv2_module: Any,
    lines: list[str],
    font_scale: float,
    thickness: int,
    frame_w: Optional[int] = None,
    frame_h: Optional[int] = None,
) -> tuple[list[tuple[int, int]], int]:
    """Measure each line and return ``(sizes, total_area)``.

    ``sizes[i]`` is the ``(rect_width, rect_height)`` of the
    background rectangle that will be drawn behind ``lines[i]``,
    including ``_OVERLAY_RECT_PADDING_PX`` of padding on every side.
    ``total_area`` is the sum of those rectangle areas (used to
    enforce the 25 % budget mandated by Requirement 4.4).

    When ``frame_w`` / ``frame_h`` are supplied the area is computed
    from rectangles clamped to the frame, mirroring what
    :func:`draw_overlay` actually paints; otherwise unclamped sizes
    are used (handy for tests that just want raw text metrics).
    """

    font = cv2_module.FONT_HERSHEY_SIMPLEX
    sizes: list[tuple[int, int]] = []
    total_area = 0
    pad = _OVERLAY_RECT_PADDING_PX
    for line in lines:
        (text_w, text_h), baseline = cv2_module.getTextSize(
            line, font, font_scale, thickness
        )
        # Include the descender (``baseline``) so glyphs with tails
        # such as 'g' / 'y' fit fully inside the background rectangle.
        rect_w = text_w + 2 * pad
        rect_h = text_h + baseline + 2 * pad
        sizes.append((rect_w, rect_h))
        # Use clamped dimensions for the budget so the area number
        # tracks what ``draw_overlay`` actually paints on the frame.
        eff_w = rect_w if frame_w is None else min(rect_w, frame_w)
        eff_h = rect_h if frame_h is None else min(rect_h, frame_h)
        total_area += max(0, eff_w) * max(0, eff_h)
    return sizes, total_area


def draw_overlay(
    frame: "np.ndarray",
    eye_label: Optional[str],
    mouth_label: Optional[str],
    active_alerts: tuple[str, ...] = (),
) -> "np.ndarray":
    """Draw ``Goz_Durumu``, ``Agiz_Durumu`` and active alert text on ``frame``.

    The text is drawn in place on ``frame`` (OpenCV mutates the
    underlying buffer). The same array is returned so callers can
    chain ``frame = draw_overlay(frame, ...)`` without worrying about
    whether a copy was made.

    The font scale is auto-sized so each glyph reaches at least
    ``_OVERLAY_GLYPH_FRACTION`` (4 %) of the frame height, comfortably
    above the 3 % minimum mandated by Requirement 4.4. If the
    resulting background rectangles would together cover more than
    25 % of the frame area (typical when the frame is tiny and there
    are many alert lines), the font is shrunk by
    ``_OVERLAY_SHRINK_FACTOR`` until the layout fits or the readable
    floor at ``_OVERLAY_MIN_FONT_SCALE`` is reached.

    Parameters
    ----------
    frame:
        BGR image to annotate. Must be a NumPy array with ``shape``
        ``(H, W, 3)`` (OpenCV's default frame layout).
    eye_label:
        Eye state label (e.g. ``"Open"`` / ``"Closed"``) or ``None``
        to render the placeholder ``"-"`` (the predictor returns
        ``None`` for low-confidence frames per design).
    mouth_label:
        Mouth state label (e.g. ``"yawn"`` / ``"no_yawn"``) or
        ``None``; rendered like ``eye_label``.
    active_alerts:
        Tuple of alert messages currently active (zero or more), each
        rendered on its own line below the eye/mouth labels.

    Returns
    -------
    np.ndarray
        ``frame`` itself, with the overlay drawn on top.
    """

    # Deferred import keeps this module importable in environments
    # without OpenCV (the alert-logic / config test suites import the
    # package eagerly via ``src.webcam_detect``).
    import cv2  # type: ignore[import-not-found]

    # Defensive shape check: a malformed array would otherwise produce
    # opaque ``cv2`` errors deep inside the drawing calls.
    if not hasattr(frame, "shape") or len(frame.shape) < 2:
        raise ValueError(
            "draw_overlay requires a 2-D image array; "
            f"got shape={getattr(frame, 'shape', None)!r}"
        )

    h = int(frame.shape[0])
    w = int(frame.shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(
            f"draw_overlay requires positive frame dimensions; "
            f"got h={h}, w={w}"
        )
    frame_area = h * w

    lines = _build_overlay_lines(eye_label, mouth_label, active_alerts)

    # Calibrate font_scale so the glyph height is ~4 % of the frame
    # height (well above the 3 % floor) but never below the 12 px
    # readability floor.
    glyph_height = max(int(_OVERLAY_GLYPH_FRACTION * h), _OVERLAY_GLYPH_MIN_PX)
    font_scale = glyph_height / float(_OVERLAY_FONT_BASELINE_PX)
    thickness = max(1, int(font_scale * 1.5))

    # Shrink the font until the rectangles fit inside the 25 % area
    # budget. The first call to ``_measure_overlay`` populates the
    # sizes used for drawing if the layout already fits.
    sizes, total_area = _measure_overlay(
        cv2, lines, font_scale, thickness, frame_w=w, frame_h=h
    )
    iterations = 0
    while (
        total_area > _OVERLAY_MAX_AREA_FRACTION * frame_area
        and font_scale > _OVERLAY_MIN_FONT_SCALE
        and iterations < _OVERLAY_MAX_SHRINK_ITERS
    ):
        font_scale *= _OVERLAY_SHRINK_FACTOR
        # Keep the stroke thickness in step with the new scale,
        # otherwise the strokes look disproportionately heavy on
        # small text.
        thickness = max(1, int(font_scale * 1.5))
        sizes, total_area = _measure_overlay(
            cv2, lines, font_scale, thickness, frame_w=w, frame_h=h
        )
        iterations += 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = _OVERLAY_RECT_PADDING_PX

    # Stack lines from the top-left corner. ``y_top`` tracks the top
    # edge of the next background rectangle; the text baseline sits
    # ``pad + text_h`` below that edge so descenders are not clipped.
    y_top = _OVERLAY_MARGIN_PX
    x_left = _OVERLAY_MARGIN_PX
    for line, (rect_w, rect_h) in zip(lines, sizes):
        # Clamp the rectangle to the frame so we do not draw off the
        # right/bottom edges on small frames.
        x0 = x_left
        y0 = y_top
        x1 = min(w, x0 + rect_w)
        y1 = min(h, y0 + rect_h)
        if x1 <= x0 or y1 <= y0:
            # Frame is too small for even one rectangle; bail out
            # gracefully instead of issuing zero-size draws.
            break

        cv2.rectangle(
            frame,
            (x0, y0),
            (x1, y1),
            _OVERLAY_BG_COLOR,
            thickness=-1,  # filled
        )
        # ``cv2.putText`` anchors at the text baseline; place it so
        # the glyph sits inside the padded rectangle. ``rect_h``
        # already includes the baseline + padding, so subtracting
        # ``pad`` lands the baseline just above the bottom padding.
        text_origin = (x0 + pad, y1 - pad)
        cv2.putText(
            frame,
            line,
            text_origin,
            font,
            font_scale,
            _OVERLAY_FG_COLOR,
            thickness,
            lineType=cv2.LINE_AA,
        )

        y_top = y1 + _OVERLAY_LINE_GAP_PX
        if y_top >= h:
            # No room for the remaining lines; stop early instead of
            # piling rectangles outside the frame.
            break

    return frame


__all__ = [
    "CameraConfig",
    "CameraCapture",
    "draw_overlay",
    "main",
]


# ---------------------------------------------------------------------------
# Live-detection main loop (task 12.1).
# ---------------------------------------------------------------------------

# Latched-alert message text, mirrored from ``src.alert_logic`` so the
# overlay can render the active alert lines while the alert is latched
# (no fresh ``AlertEvent`` is emitted between the rising edge and the
# clearing edge, so we cannot rely on incoming events alone).
_DROWSY_OVERLAY_MESSAGE = "UYARI: Surucu uyukluyor olabilir!"
_FATIGUE_OVERLAY_MESSAGE = "UYARI: Surucu yorgun olabilir!"

# Default sound files used for ``DROWSY`` / ``FATIGUE`` alerts. The
# files do not have to exist at import time; ``SoundAlerter`` only
# touches the filesystem when ``enable_sound=True`` and an alert
# fires, and any backend failure is logged then swallowed
# (Requirement 7.6).
_DEFAULT_DROWSY_SOUND_PATH = Path("assets/drowsy.wav")
_DEFAULT_FATIGUE_SOUND_PATH = Path("assets/fatigue.wav")

# Window title used by the live preview ``cv2.imshow`` calls. Kept as
# a module constant so the cleanup path in ``main`` can name the
# window when it tears it down.
_LIVE_WINDOW_TITLE = "DDD Live"


def _build_active_alerts(state: Any) -> tuple[str, ...]:
    """Return the overlay alert lines for the latched flags in ``state``.

    The ordering is deterministic (drowsy before fatigue) so the
    overlay layout stays stable across frames; this also matches the
    visual priority described in Requirement 4.4 where the more
    immediate drowsy alert appears first.
    """

    lines: list[str] = []
    if getattr(state, "drowsy_alert_active", False):
        lines.append(_DROWSY_OVERLAY_MESSAGE)
    if getattr(state, "fatigue_alert_active", False):
        lines.append(_FATIGUE_OVERLAY_MESSAGE)
    return tuple(lines)


def _flush_logger_handlers(logger: Optional[logging.Logger]) -> None:
    """Best-effort flush of every handler attached to ``logger``.

    Called from the ``finally`` block in :func:`main` so the rotating
    file handler commits pending records before the process exits.
    Failures are swallowed: logging must never raise into the
    shutdown path (Requirement 9.6).
    """

    if logger is None:
        return
    for handler in list(logger.handlers):
        try:
            handler.flush()
        except Exception:
            # Deliberately swallowed: ``flush`` is part of the cleanup
            # path; raising here would mask the real reason the loop
            # exited.
            pass


def _safe_imshow(title: str, frame: "np.ndarray") -> bool:
    """Show ``frame`` in ``title`` and report whether the call succeeded.

    Returns ``False`` when OpenCV's HighGUI is unavailable (typical
    for headless servers / CI workers) so the caller can latch the
    failure and stop attempting to render. Any exception is caught
    and treated as "headless": the live detection loop must keep
    running even if no display is attached.
    """

    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return False
    try:
        cv2.imshow(title, frame)
    except Exception:
        return False
    return True


def _safe_destroy_windows() -> None:
    """Tear down OpenCV HighGUI windows, ignoring headless errors."""

    try:
        import cv2  # type: ignore[import-not-found]
    except Exception:
        return
    try:
        cv2.destroyAllWindows()
        # ``destroyAllWindows`` returns immediately on most backends;
        # call ``waitKey(1)`` once so the window manager processes the
        # destroy event before the process exits.
        cv2.waitKey(1)
    except Exception:
        pass


def main(argv: Optional[list[str]] = None) -> int:
    """Run the end-to-end live drowsiness-detection loop.

    Wires :func:`src.config.load_config` ->
    :func:`src.logger.build_logger` -> :class:`src.predictor.Predictor`
    -> :class:`CameraCapture` -> :func:`src.alert_logic.update` ->
    :class:`src.sound_alert.SoundAlerter` -> :func:`draw_overlay`,
    and drives :class:`src.logger.FpsTracker` for the periodic FPS
    info line (Requirement 9.3).

    Parameters
    ----------
    argv:
        Optional argv list (excluding the program name) used by the
        unit tests. When ``None`` the standard ``sys.argv[1:]`` slice
        is parsed.

    Returns
    -------
    int
        Process exit code. ``0`` on a clean shutdown (user pressed
        ``q`` or sent ``Ctrl+C``), ``2`` on a configuration / camera
        failure (Requirements 4.5, 8.5, 8.6).
    """

    # Imports for the wired pipeline. Done at call time (not at
    # module import) so the module can still be imported on
    # environments without OpenCV / Ultralytics for the alert-logic
    # / config test suites.
    import argparse
    import sys
    import time as _time

    from src.alert_logic import (
        INITIAL_STATE,
        AlertConfig,
        update,
    )
    from src.config import ConfigError, load_config
    from src.logger import FpsTracker, build_logger
    from src.predictor import Predictor
    from src.sound_alert import SoundAlerter

    parser = argparse.ArgumentParser(
        prog="python -m src.webcam_detect",
        description=(
            "Surucu uyku ve yorgunluk tespiti canli akisi. "
            "config.yaml dosyasindaki parametreleri kullanir."
        ),
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help=(
            "Yapilandirma YAML dosyasinin yolu (varsayilan: "
            "config.yaml)."
        ),
    )
    args = parser.parse_args(argv)

    logger: Optional[logging.Logger] = None
    capture: Optional[CameraCapture] = None
    try:
        # --- Load and validate the YAML configuration -----------------
        # ``load_config`` performs every range / type / filesystem
        # check itself and raises ``ConfigError`` with parameter +
        # reason context (Requirements 8.5, 8.6).
        cfg = load_config(args.config)

        # --- Build the shared rotating-file + stdout logger ----------
        logger = build_logger(cfg)
        logger.info(
            "Canli tespit baslatiliyor (config=%s, camera=%r, "
            "eye=%s, mouth=%s)",
            args.config,
            cfg.camera_index,
            cfg.eye_model_path,
            cfg.mouth_model_path,
        )

        # --- Predictor (loads eye + mouth models) -------------------
        predictor = Predictor(
            eye_model_path=cfg.eye_model_path,
            mouth_model_path=cfg.mouth_model_path,
            inference_resolution=cfg.inference_resolution,
            confidence_threshold=cfg.confidence_threshold,
        )

        # --- Camera capture ------------------------------------------
        capture = CameraCapture(
            CameraConfig(
                source=cfg.camera_index,
                frame_skip=cfg.frame_skip,
            )
        )
        capture.open()

        # --- Pure-core alert configuration ---------------------------
        alert_cfg = AlertConfig(
            closed_eye_duration_s=cfg.closed_eye_duration,
            yawn_count=cfg.yawn_count,
            yawn_time_window_s=cfg.yawn_time_window,
            confidence_threshold=cfg.confidence_threshold,
        )

        # --- Sound alerter (no-op when ``enable_sound=False``) -------
        sound = SoundAlerter(
            sound_files={
                "DROWSY": _DEFAULT_DROWSY_SOUND_PATH,
                "FATIGUE": _DEFAULT_FATIGUE_SOUND_PATH,
            },
            enable_sound=cfg.enable_sound,
            logger=logger,
        )

        fps_tracker = FpsTracker(logger=logger)

        # --- Per-frame state carried across iterations ---------------
        # ``eye_label`` / ``mouth_label`` persist between predicted
        # frames so the overlay does not flicker on the frames where
        # ``frame_skip`` skipped the predictor (Requirement 10.6).
        eye_label: Optional[str] = None
        mouth_label: Optional[str] = None
        eye_conf: float = 0.0
        mouth_conf: float = 0.0
        active_alerts: tuple[str, ...] = ()
        alert_state = INITIAL_STATE

        # ``cv2.imshow`` is unavailable on headless builds; latch the
        # first failure so we do not flood the log with the same
        # error every frame.
        display_available = True

        for frame_index, frame, should_predict in capture.frames(logger):
            if should_predict:
                # ``predict_frame`` defaults ``t_capture_s`` to
                # ``time.monotonic()`` so the alert engine sees a
                # non-decreasing capture clock, as required by
                # Property 7 / Requirements 5.1, 5.2.
                pred = predictor.predict_frame(frame)

                alert_state, events = update(alert_state, pred, alert_cfg)
                for event in events:
                    # Requirement 9.1: ISO-8601 timestamp + level +
                    # message arrive on stdout via the configured
                    # formatter. Requirement 5.3 / 6.3: the
                    # human-readable Turkish alert text comes from the
                    # pure core, so we just relay it.
                    logger.info("%s: %s", event.kind, event.message)
                    sound.play(event.kind)

                eye_label = pred.eye
                mouth_label = pred.mouth
                eye_conf = pred.eye_conf
                mouth_conf = pred.mouth_conf
                active_alerts = _build_active_alerts(alert_state)

                # Drive the rolling-window FPS tracker once per
                # predicted frame so the 1-second average reflects
                # actual inference cadence rather than camera FPS
                # (Requirement 9.3).
                now = _time.monotonic()
                fps_tracker.tick(now)
                fps_tracker.maybe_log(now, cfg.fps_log_interval)

                if cfg.verbose:
                    # Requirement 9.5: per-frame number + labels +
                    # confidences (two decimals). The 1 s rolling
                    # FPS is included so verbose runs surface
                    # performance regressions immediately.
                    logger.debug(
                        "frame=%d eye=%s mouth=%s "
                        "eye_conf=%.2f mouth_conf=%.2f fps=%.2f",
                        frame_index,
                        eye_label,
                        mouth_label,
                        eye_conf,
                        mouth_conf,
                        fps_tracker.average_fps(1.0),
                    )

            # Always render the overlay on every yielded frame so the
            # display path stays at full FPS even when the predictor
            # is throttled by ``frame_skip`` (Requirement 10.6).
            frame = draw_overlay(frame, eye_label, mouth_label, active_alerts)

            if display_available:
                display_available = _safe_imshow(_LIVE_WINDOW_TITLE, frame)

        # ``cap.frames`` has already released the camera (it pumps the
        # ``q`` keypress + handles the read-failure / no-frame
        # timeouts described in Requirement 4.6 / 4.7); we only need
        # to tear down the HighGUI window here.
        logger.info("Canli tespit normal sekilde sonlandi")
        return 0

    except ConfigError as exc:
        # ``ConfigError`` carries parameter + value + reason; ``str``
        # already produces the uniform "Yapilandirma parametresi 'X'
        # gecersiz: ..." message required by Requirement 8.5.
        sys.stderr.write(f"Hata: {exc}\n")
        return 2

    except RuntimeError as exc:
        # ``CameraCapture.open`` raises ``RuntimeError`` when the
        # device cannot be opened within the timeout (Requirement 4.5).
        # Any other ``RuntimeError`` raised by the predictor is also
        # treated as a startup failure for the live loop.
        if logger is not None:
            logger.error("Canli tespit baslatilamadi: %s", exc)
        sys.stderr.write(f"Hata: {exc}\n")
        return 2

    except KeyboardInterrupt:
        # Ctrl+C: same clean-exit semantics as the ``q`` keypress
        # (Requirement 4.7). Keep the message short so it does not
        # interleave with the OpenCV cleanup output.
        if logger is not None:
            logger.info("Kullanici durdurdu")
        return 0

    finally:
        # Best-effort cleanup: release the camera (the frames iterator
        # already does this on every termination path; this is a
        # belt-and-braces ``close`` for the rare case where ``open``
        # succeeded but the loop never ran), tear down the HighGUI
        # window, and flush the logger so the rotating file handler
        # commits any buffered records before the process exits
        # (Requirements 4.6, 4.7, 9.2).
        if capture is not None:
            try:
                capture.close()
            except Exception:
                pass
        _safe_destroy_windows()
        _flush_logger_handlers(logger)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import sys as _sys

    _sys.exit(main())
