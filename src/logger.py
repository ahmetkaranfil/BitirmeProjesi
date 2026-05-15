"""Loglayici module.

Builds the shared ``logging.Logger`` (stream + rotating file handler)
and exposes the ``FpsTracker`` rolling-window FPS helper described in
the design (Requirements 9.1-9.7, 10.9, 10.10).

The ``build_logger`` factory wires a stdout :class:`logging.StreamHandler`
and a resilient :class:`logging.handlers.RotatingFileHandler` together
on a single named logger. The rotating handler is wrapped so that disk
write failures (permissions, full disk, removed directory) downgrade to
a one-line warning on ``stderr`` and are retried every 30 seconds
without ever propagating an exception into the live detection loop.

``FpsTracker`` keeps a rolling-window deque of frame timestamps so the
main loop can periodically log the recent average FPS (Requirement 9.3)
and emit a single ``WARNING`` with parameter-tuning advice once the
30-second average drops below 5 FPS for 30 seconds in a row
(Requirements 10.9, 10.10).
"""

from __future__ import annotations

import logging
import sys
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Protocol


# Fixed logger name so every component shares the same configured
# ``logging.Logger`` instance. Using ``logging.getLogger(LOGGER_NAME)``
# elsewhere in the codebase returns the logger built by this factory.
LOGGER_NAME = "ddd"

# Exact format strings required by Requirements 9.1 and 9.2: ISO 8601
# timestamp, log level in brackets, and the message.
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Retry window for the rotating file handler. After a write failure the
# handler drops records for this many seconds before attempting another
# write (Requirement 9.6).
RETRY_INTERVAL_S = 30.0


class _LoggerCfg(Protocol):
    """Duck-typed view of :class:`src.config.AppConfig`.

    Only the fields actually consumed by :func:`build_logger` are
    declared so this module does not have to import the full config
    type (which would create an import cycle once tasks 2.x land).
    """

    log_file: Path
    log_max_bytes: int


class _ResilientRotatingFileHandler(RotatingFileHandler):
    """Rotating file handler that degrades gracefully on write errors.

    Requirement 9.6 says that a failure to write to the log file must
    not stop live detection: instead the system prints a warning on
    ``stderr`` and retries the file write every 30 seconds. The base
    :class:`RotatingFileHandler` already calls :meth:`handleError` on
    failure (so it never raises into the caller), but it also keeps
    trying the broken file on every record. We override
    :meth:`handleError` to record the failure timestamp, then short
    circuit :meth:`emit` while the cooldown window is still active.
    """

    def __init__(
        self,
        *args,
        retry_interval_s: float = RETRY_INTERVAL_S,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._retry_interval_s = float(retry_interval_s)
        # Monotonic timestamp of the most recent write failure, or
        # ``None`` when the handler is healthy.
        self._last_failure_s: float | None = None

    def emit(self, record: logging.LogRecord) -> None:
        now = time.monotonic()
        if (
            self._last_failure_s is not None
            and now - self._last_failure_s < self._retry_interval_s
        ):
            # Still inside the cooldown window: drop the record so the
            # caller is not blocked re-raising into the same broken
            # disk on every frame.
            return

        # Snapshot the failure marker before delegating; the parent
        # class catches exceptions internally and routes them through
        # ``handleError`` (which we override below to update the
        # marker).
        prev_failure = self._last_failure_s
        super().emit(record)
        if self._last_failure_s == prev_failure:
            # No new failure was recorded by ``handleError``; clear
            # any stale cooldown so subsequent records flow normally.
            self._last_failure_s = None

    def handleError(self, record: logging.LogRecord) -> None:
        # Capture the active exception (if any) so the stderr warning
        # is informative, then mark the cooldown window so the next
        # ``emit`` call short-circuits for the next 30 seconds.
        exc_type, exc_value, _ = sys.exc_info()
        self._last_failure_s = time.monotonic()
        try:
            type_name = exc_type.__name__ if exc_type is not None else "Error"
            detail = "" if exc_value is None else f": {exc_value}"
            sys.stderr.write(
                f"[WARNING] log dosyasina yazilamadi "
                f"({type_name}{detail}); "
                f"{int(self._retry_interval_s)} sn sonra yeniden denenecek\n"
            )
            sys.stderr.flush()
        except Exception:
            # ``stderr`` itself can be closed (e.g. detached console);
            # swallow so the detection loop keeps running per Req 9.6.
            pass


def build_logger(cfg: _LoggerCfg) -> logging.Logger:
    """Configure and return the shared application logger.

    Wires a stdout :class:`logging.StreamHandler` plus a resilient
    :class:`logging.handlers.RotatingFileHandler` writing to
    ``cfg.log_file`` with ``maxBytes=cfg.log_max_bytes`` and
    ``backupCount=5``. Both handlers use the
    ``'%(asctime)s [%(levelname)s] %(message)s'`` format with the
    ``asctime`` field fixed to ``'%Y-%m-%d %H:%M:%S'``.

    File-handler write failures fall back to a one-line warning on
    ``stderr`` and are retried every 30 seconds without ever raising
    into the caller (Requirement 9.6). Rotation past
    ``cfg.log_max_bytes`` produces ``app.log.1`` ... ``app.log.5``
    archive files (Requirement 9.7).

    The logger is named ``"ddd"`` and propagation to the root logger is
    disabled so messages are not double-printed when callers already
    use ``logging.getLogger("ddd")`` directly.

    Calling ``build_logger`` more than once is safe: previously
    attached handlers are detached and closed before the new pair is
    installed, which keeps tests and re-initialisation idempotent.
    """

    log_file = Path(cfg.log_file)

    # Best-effort directory creation. If creation fails (read-only fs,
    # missing permissions) we do not abort logger construction; the
    # resilient file handler will catch the open failure on the first
    # emit and route it through the regular retry path.
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Detach any handlers from a previous build so handlers do not pile
    # up across calls (matters in tests and on config-reload paths).
    for old in list(logger.handlers):
        logger.removeHandler(old)
        try:
            old.close()
        except Exception:
            pass

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # ``delay=True`` defers the actual file open until the first record
    # is emitted, which lets the resilient ``handleError`` path catch
    # initial open failures (missing directory, no write permission)
    # alongside ongoing write failures.
    file_handler = _ResilientRotatingFileHandler(
        filename=str(log_file),
        maxBytes=int(cfg.log_max_bytes),
        backupCount=5,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Bounds for ``cfg.fps_log_interval`` (Requirements 9.3, 9.4). Values
# outside this closed interval are clamped to ``FPS_LOG_INTERVAL_DEFAULT``
# and a one-time WARNING is emitted on the underlying logger.
FPS_LOG_INTERVAL_MIN = 1.0
FPS_LOG_INTERVAL_MAX = 60.0
FPS_LOG_INTERVAL_DEFAULT = 1.0

# Performance-warning thresholds for the Jetson Nano deployment target
# (Requirements 10.9, 10.10). When the rolling 30-second average FPS
# stays below ``LOW_FPS_THRESHOLD`` for at least ``LOW_FPS_WINDOW_S``
# seconds, ``maybe_log`` emits a single WARNING with parameter advice.
LOW_FPS_THRESHOLD = 5.0
LOW_FPS_WINDOW_S = 30.0

# Maximum age of timestamps the tracker keeps in its deque. We keep at
# least the longest window we ever query (the 30 s low-FPS window) so
# ``average_fps`` is correct for both 1 s and 30 s queries.
_FPS_TRACKER_HISTORY_S = LOW_FPS_WINDOW_S


class FpsTracker:
    """Rolling-window frame-rate tracker.

    The main loop calls :meth:`tick` once per processed frame and then
    :meth:`maybe_log` to emit periodic FPS reports plus the Jetson
    performance advisory described in Requirements 10.9 and 10.10.

    All time arguments are in seconds and are expected to come from a
    monotonic clock (``time.monotonic``). The tracker itself never
    reads a clock so the unit/property tests can drive it with a fake
    clock.

    Parameters
    ----------
    logger:
        Optional :class:`logging.Logger` used for verbose per-interval
        FPS lines and the low-FPS warning. ``None`` means the tracker
        falls back to ``logging.getLogger(LOGGER_NAME)`` so callers
        that already configured ``build_logger`` get the shared
        handler set up automatically.
    low_fps_threshold:
        FPS value at or below which the tracker considers the system
        underperforming. Defaults to ``5.0`` to match Requirement 10.9.
    low_fps_window_s:
        Sustained duration (seconds) the average FPS must stay below
        ``low_fps_threshold`` before a single warning is emitted.
        Defaults to ``30.0`` (Requirement 10.10).
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        low_fps_threshold: float = LOW_FPS_THRESHOLD,
        low_fps_window_s: float = LOW_FPS_WINDOW_S,
    ) -> None:
        self._logger = logger if logger is not None else logging.getLogger(LOGGER_NAME)
        self._low_fps_threshold = float(low_fps_threshold)
        self._low_fps_window_s = float(low_fps_window_s)

        # We size the deque so the longest window we ever query
        # (``low_fps_window_s`` for the Jetson check, or the 1 s
        # report, whichever is bigger) stays fully represented.
        self._history_s = max(_FPS_TRACKER_HISTORY_S, self._low_fps_window_s)
        self._ticks: deque[float] = deque()

        # Monotonic timestamp of the most recent verbose FPS line. Set
        # to ``None`` so the very first ``maybe_log`` call establishes
        # the cadence reference instead of immediately emitting a line.
        self._last_log_s: float | None = None

        # Monotonic timestamp at which the average FPS first dropped
        # below the threshold during the current sub-threshold streak.
        # Reset to ``None`` whenever the average recovers above the
        # threshold.
        self._low_fps_started_s: float | None = None

        # ``True`` once the parameter-advice WARNING has been emitted
        # for the current sub-threshold streak; cleared when the
        # average recovers, so a later relapse can produce a fresh
        # warning (Req 10.10 only forbids spam *within* one streak).
        self._low_fps_warning_emitted = False

        # Tracks whether we have already warned about an out-of-range
        # ``interval_s`` value so we do not spam the log at every
        # frame.
        self._interval_warning_emitted = False

    def tick(self, t_now_s: float) -> None:
        """Record that a frame was processed at ``t_now_s`` seconds.

        Old timestamps that fall outside the longest window we track
        are dropped here so the deque stays bounded in size.
        """

        t = float(t_now_s)
        self._ticks.append(t)
        cutoff = t - self._history_s
        # Drop expired timestamps from the front of the deque. Using a
        # ``while`` loop keeps the operation amortised O(1) per tick.
        while self._ticks and self._ticks[0] < cutoff:
            self._ticks.popleft()

    def average_fps(self, window_s: float = 1.0) -> float:
        """Return the average FPS over the last ``window_s`` seconds.

        The result is the count of ticks whose timestamps fall within
        ``[t_latest - window_s, t_latest]`` divided by ``window_s``. A
        non-positive ``window_s`` or an empty tick history both yield
        ``0.0`` so callers can use the value safely in arithmetic
        without special-casing.
        """

        window = float(window_s)
        if window <= 0.0 or not self._ticks:
            return 0.0

        latest = self._ticks[-1]
        cutoff = latest - window
        # ``self._ticks`` is monotonic non-decreasing, so a linear scan
        # from the right end stops as soon as we cross the cutoff.
        count = 0
        for t in reversed(self._ticks):
            if t < cutoff:
                break
            count += 1
        return count / window

    def maybe_log(self, t_now_s: float, interval_s: float) -> None:
        """Emit periodic FPS info + the Jetson low-FPS warning.

        ``interval_s`` is clamped to ``[1.0, 60.0]`` (Requirement 9.4):
        out-of-range values fall back to ``1.0`` and a one-time
        WARNING is logged. The tracker emits a verbose INFO line with
        the 1-second average FPS at most once per ``interval_s`` and,
        independently, a single WARNING with parameter-tuning advice
        whenever the 30-second average has stayed below
        ``low_fps_threshold`` for at least ``low_fps_window_s`` seconds
        (Requirements 10.9, 10.10).
        """

        t = float(t_now_s)
        effective_interval = self._effective_interval_s(interval_s)

        # First call: anchor the cadence and skip the verbose line so
        # we do not emit before any frame has been ticked through.
        if self._last_log_s is None:
            self._last_log_s = t
        elif t - self._last_log_s >= effective_interval:
            avg_1s = self.average_fps(window_s=1.0)
            self._logger.info("FPS: %.2f", avg_1s)
            self._last_log_s = t

        self._update_low_fps_state(t)

    # ------------------------------------------------------------------
    # Helpers (kept private; not part of the public API).
    # ------------------------------------------------------------------

    def _effective_interval_s(self, interval_s: float) -> float:
        """Return a sanitised ``interval_s`` and warn once on misuse."""

        try:
            value = float(interval_s)
        except (TypeError, ValueError):
            value = float("nan")

        in_range = (
            value == value  # NaN guard: NaN != NaN
            and FPS_LOG_INTERVAL_MIN <= value <= FPS_LOG_INTERVAL_MAX
        )
        if in_range:
            return value

        if not self._interval_warning_emitted:
            self._logger.warning(
                "fps_log_interval=%s gecersiz (izin verilen aralik "
                "%.1f-%.1f sn); varsayilan %.1f sn kullanilacak",
                interval_s,
                FPS_LOG_INTERVAL_MIN,
                FPS_LOG_INTERVAL_MAX,
                FPS_LOG_INTERVAL_DEFAULT,
            )
            self._interval_warning_emitted = True
        return FPS_LOG_INTERVAL_DEFAULT

    def _update_low_fps_state(self, t_now_s: float) -> None:
        """Track the sustained low-FPS streak and emit a single warning.

        The streak starts the first time the 30-second average dips at
        or below ``low_fps_threshold`` and ends as soon as the average
        recovers above the threshold. While the streak is active, a
        single WARNING with parameter-tuning advice is emitted once
        the streak has lasted at least ``low_fps_window_s`` seconds.
        """

        avg_window = self.average_fps(window_s=self._low_fps_window_s)
        below_threshold = avg_window < self._low_fps_threshold

        if not below_threshold:
            # FPS recovered: reset the streak so a subsequent dip can
            # produce a fresh warning later.
            self._low_fps_started_s = None
            self._low_fps_warning_emitted = False
            return

        if self._low_fps_started_s is None:
            self._low_fps_started_s = t_now_s
            return

        sustained_s = t_now_s - self._low_fps_started_s
        if sustained_s >= self._low_fps_window_s and not self._low_fps_warning_emitted:
            self._logger.warning(
                "Performans dusuk: son %.0f sn ortalama FPS=%.2f "
                "(hedef >=%.1f). inference_resolution / frame_skip "
                "parametrelerini ayarlamayi deneyin.",
                self._low_fps_window_s,
                avg_window,
                self._low_fps_threshold,
            )
            self._low_fps_warning_emitted = True
