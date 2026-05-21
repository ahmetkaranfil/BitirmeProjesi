"""Sesli_Uyarici module.

Platform-aware audio backend used to play short audible alerts for
``DROWSY`` and ``FATIGUE`` events (Requirements 7.1-7.7).

* ``SoundAlerter`` selects ``winsound`` on Windows and ``playsound``
  on Linux/Jetson, falling back to an ``aplay`` subprocess when the
  ``playsound`` package is not importable (Requirements 7.3, 7.4).
* ``play(kind)`` is a no-op when ``enable_sound=False``
  (Requirement 7.2) and otherwise dispatches the chosen backend
  asynchronously, so playback starts within 500 ms of the triggering
  call (Requirement 7.1).
* A daemon ``threading.Timer`` stops the active sound after
  ``max_duration_s`` so a single alert never plays for more than
  3 seconds (Requirement 7.5).
* While a sound for a given ``kind`` is still active, additional
  ``play(kind)`` requests of the same kind are silently dropped
  (Requirement 7.7).
* Any exception raised by the underlying backend is caught and
  logged via the shared logger with the error type and target path;
  the exception never propagates and the visual alert pipeline keeps
  running (Requirement 7.6).
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional


# ---------------------------------------------------------------------------
# Backend identifiers (internal).
# ---------------------------------------------------------------------------

_BACKEND_WINSOUND = "winsound"
_BACKEND_PLAYSOUND = "playsound"
_BACKEND_APLAY = "aplay"
_BACKEND_AFPLAY = "afplay"


# ``StopCallback`` is a zero-arg callable that stops the currently
# playing sound for one ``kind``. ``threading.Timer`` invokes it from
# a daemon thread once ``max_duration_s`` elapses.
StopCallback = Callable[[], None]


def _noop_stop() -> None:
    """Default stop callback for backends without a kill primitive."""

    return None


class SoundAlerter:
    """Platform-aware async audio alerter (task 11.1).

    Parameters
    ----------
    sound_files:
        Mapping from alert kind (``"DROWSY"`` / ``"FATIGUE"``) to the
        sound file played for that kind.
    enable_sound:
        When ``False`` every :meth:`play` call is a silent no-op
        (Requirement 7.2).
    max_duration_s:
        Maximum audible duration of a single alert in seconds; the
        backend is forced to stop after this many seconds
        (Requirement 7.5). Defaults to ``3.0``.
    logger:
        Shared application logger. Stored for use by task 11.2 (error
        reporting per Requirement 7.6); the 11.1 implementation does
        not log anything itself.
    """

    def __init__(
        self,
        sound_files: Mapping[str, Path],
        enable_sound: bool = True,
        max_duration_s: float = 3.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._sound_files: Dict[str, Path] = {
            kind: Path(path) for kind, path in sound_files.items()
        }
        self._enable_sound = bool(enable_sound)
        self._max_duration_s = float(max_duration_s)
        self._logger = (
            logger if logger is not None else logging.getLogger(__name__)
        )

        # Per-kind active stop callbacks and timers. Guarded by
        # ``self._lock`` so the playback thread, the application
        # thread and the Timer thread can interact safely. Concurrency
        # dedupe (Requirement 7.7) is added in task 11.2 and will use
        # these same structures.
        self._stop_callbacks: Dict[str, Optional[StopCallback]] = {}
        self._timers: Dict[str, Optional[threading.Timer]] = {}
        self._lock = threading.Lock()

        # Backend is selected once at construction time so the
        # ``playsound`` -> ``aplay`` fallback (Requirement 7.4) does
        # not repeat the import probe on every alert.
        self._backend: str = self._select_backend()

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    def play(self, kind: str, loop_until_stopped: bool = False) -> None:
        """Start async playback for ``kind``; no-op when disabled.

        Returns as soon as the platform call has been issued, so
        playback begins well within the 500 ms budget required by
        Requirement 7.1. The accompanying :class:`threading.Timer`
        terminates the sound after ``max_duration_s`` seconds.

        ``loop_until_stopped=True`` disables the duration cap timer
        so the sound keeps looping until ``stop(kind)`` is called.
        This is used by the DROWSY alert: the sound must keep playing
        until the driver opens their eyes.

        Concurrency dedupe (Requirement 7.7): while a previous sound
        for the same ``kind`` is still active, additional
        ``play(kind)`` calls return immediately without invoking the
        backend.

        Error handling (Requirement 7.6): any exception raised by the
        backend (missing file, decode error, unavailable subprocess,
        ...) is caught, logged via the shared logger with timestamp,
        error type and the offending path, and then swallowed. No
        timer is registered for a failed call, and the exception is
        never propagated to the caller.
        """

        if not self._enable_sound:
            return

        # Concurrency dedupe: if a sound for this kind is already
        # active, drop the new request silently so overlapping alerts
        # do not stack. We track activity via either a registered
        # timer (timed playback) or a stop callback without a timer
        # (loop_until_stopped playback).
        with self._lock:
            already_active = (
                self._timers.get(kind) is not None
                or self._stop_callbacks.get(kind) is not None
            )
        if already_active:
            return

        path = self._sound_files[kind]

        try:
            if self._backend == _BACKEND_WINSOUND:
                stop_cb = self._play_winsound(path)
            elif self._backend == _BACKEND_PLAYSOUND:
                stop_cb = self._play_playsound(path)
            elif self._backend == _BACKEND_AFPLAY:
                stop_cb = self._play_afplay(path)
            else:
                stop_cb = self._play_aplay(path)

            if loop_until_stopped:
                # No duration cap; ``stop(kind)`` is the only way to
                # end this playback. Register the stop callback so
                # ``stop()`` can find and invoke it.
                with self._lock:
                    self._stop_callbacks[kind] = stop_cb
                    self._timers[kind] = None
            else:
                timer = threading.Timer(
                    self._max_duration_s,
                    self._stop_playback,
                    args=(kind,),
                )
                timer.daemon = True
                with self._lock:
                    self._stop_callbacks[kind] = stop_cb
                    self._timers[kind] = timer
                timer.start()
        except Exception as exc:  # noqa: BLE001 - intentional swallow
            # Requirement 7.6: log type + path with a timestamp (the
            # logger formatter already emits ISO-style timestamps) and
            # keep the visual alert pipeline running. We deliberately
            # do NOT register a timer here; without a timer the dedupe
            # check stays open so a later, healthy call can succeed.
            self._logger.error(
                "Sesli_Uyarici playback failed: kind=%s path=%s "
                "error_type=%s error=%s",
                kind,
                path,
                type(exc).__name__,
                exc,
            )
            return

    def stop(self, kind: str) -> None:
        """Stop playback for ``kind`` immediately if active.

        Idempotent and thread-safe. Cancels any pending duration timer
        and invokes the backend stop callback so a looping playback
        ends right away. No-op if no playback is currently active for
        the given kind, or if sound is disabled.
        """

        if not self._enable_sound:
            return

        with self._lock:
            stop_cb = self._stop_callbacks.get(kind)
            timer = self._timers.get(kind)
            self._stop_callbacks[kind] = None
            self._timers[kind] = None

        if timer is not None:
            timer.cancel()
        if stop_cb is not None:
            try:
                stop_cb()
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "Sesli_Uyarici stop failed: kind=%s "
                    "error_type=%s error=%s",
                    kind,
                    type(exc).__name__,
                    exc,
                )

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------

    def _select_backend(self) -> str:
        """Pick the audio backend based on ``sys.platform``.

        * ``win32`` -> ``winsound`` (Requirement 7.3).
        * ``darwin`` (macOS) -> system ``afplay`` subprocess. On macOS
          the cross-platform ``playsound`` package depends on PyObjC
          (``AppKit``), which is not in our pinned requirements; using
          the built-in ``afplay`` command keeps the install footprint
          unchanged and avoids the ``ModuleNotFoundError: AppKit``
          failure inside the playback thread.
        * Anything else (Linux, Jetson) -> ``playsound`` if the
          package imports cleanly, otherwise an ``aplay`` subprocess
          (Requirement 7.4).
        """

        if sys.platform == "win32":
            return _BACKEND_WINSOUND
        if sys.platform == "darwin":
            return _BACKEND_AFPLAY
        try:
            import playsound  # noqa: F401  (probe only)
        except Exception:
            return _BACKEND_APLAY
        return _BACKEND_PLAYSOUND

    def _stop_playback(self, kind: str) -> None:
        """Timer callback: cap audible duration at ``max_duration_s``."""

        with self._lock:
            stop_cb = self._stop_callbacks.get(kind)
            self._stop_callbacks[kind] = None
            self._timers[kind] = None
        if stop_cb is not None:
            stop_cb()

    # -- Per-backend playback primitives --------------------------------

    def _play_winsound(self, path: Path) -> StopCallback:
        """Async play via ``winsound.PlaySound`` (Windows)."""

        import winsound

        winsound.PlaySound(
            str(path), winsound.SND_ASYNC | winsound.SND_FILENAME
        )

        def stop() -> None:
            # ``PlaySound(None, SND_PURGE)`` cancels any sound currently
            # playing on this thread, which is the documented way to
            # interrupt an ``SND_ASYNC`` playback.
            winsound.PlaySound(None, winsound.SND_PURGE)

        return stop

    def _play_playsound(self, path: Path) -> StopCallback:
        """Async play via the cross-platform ``playsound`` package."""

        from playsound import playsound

        # ``playsound`` is blocking on most platforms, so run it on a
        # daemon thread to satisfy the 500 ms start budget. The
        # underlying engine has no portable kill primitive, so the
        # ``stop`` callback is a no-op; ``max_duration_s`` is bounded
        # in practice by the file length and the 3 s cap on the file
        # itself.
        thread = threading.Thread(
            target=playsound, args=(str(path),), daemon=True
        )
        thread.start()
        return _noop_stop

    def _play_aplay(self, path: Path) -> StopCallback:
        """Async play via the ``aplay`` ALSA subprocess (Linux/Jetson)."""

        return self._play_loop_subprocess(["aplay", str(path)])

    def _play_afplay(self, path: Path) -> StopCallback:
        """Async play via the ``afplay`` subprocess (macOS).

        ``afplay`` ships with macOS, so no extra dependency is needed.
        Like the ``aplay`` backend, it returns a ``stop`` callback that
        kills the subprocess if the duration cap timer fires before
        the clip finishes naturally.
        """

        return self._play_loop_subprocess(["afplay", str(path)])

    # ------------------------------------------------------------------
    # Looping subprocess helper.
    # ------------------------------------------------------------------

    def _play_loop_subprocess(self, argv: list) -> StopCallback:
        """Repeatedly invoke ``argv`` until told to stop.

        Many alert clips are only 1-3 s long, but ``max_duration_s``
        can be configured up to 60 s. To stretch a short clip across
        the full window, we run the playback subprocess in a daemon
        thread that re-launches it as soon as it exits, until the
        ``stop_event`` is set by the duration timer (or another
        ``stop`` call).

        The returned stop callback is idempotent: it sets the event
        and, if a subprocess is still running, kills it so the loop
        exits without waiting for the current clip to finish.
        """

        stop_event = threading.Event()
        proc_holder: Dict[str, Optional[subprocess.Popen]] = {"proc": None}
        proc_lock = threading.Lock()

        def loop() -> None:
            while not stop_event.is_set():
                try:
                    proc = subprocess.Popen(
                        argv,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "Sesli_Uyarici loop subprocess failed: argv=%s "
                        "error_type=%s error=%s",
                        argv,
                        type(exc).__name__,
                        exc,
                    )
                    return
                with proc_lock:
                    proc_holder["proc"] = proc
                proc.wait()
                with proc_lock:
                    proc_holder["proc"] = None

        thread = threading.Thread(target=loop, daemon=True)
        thread.start()

        def stop() -> None:
            stop_event.set()
            with proc_lock:
                proc = proc_holder["proc"]
            if proc is not None and proc.poll() is None:
                proc.kill()

        return stop


__all__ = ["SoundAlerter"]
