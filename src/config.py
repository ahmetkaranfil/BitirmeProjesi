"""Yapilandirma module.

Holds the immutable :class:`AppConfig` dataclass, the :data:`DEFAULTS`
dict, the :class:`ConfigError` exception and the pure :func:`validate`
function described in the design (Requirements 5.5, 6.5, 8.1, 8.4,
8.5, 9.4, 10.7).

The validator is intentionally split from any I/O. The YAML loader
lives in :func:`load_config` (task 2.2); :func:`validate` is a pure
function that takes a ``dict`` and returns an :class:`AppConfig`,
which makes it directly addressable by Property 8 and Property 9.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# Python 3.8 ships ``Literal`` in ``typing``; we keep that import so the
# dataclass annotation matches the design verbatim and is checkable on
# the supported interpreter range (3.8-3.11, see requirements.txt).
try:  # pragma: no cover - exercised implicitly on every supported runtime
    from typing import Literal
except ImportError:  # pragma: no cover - 3.7 fallback, kept for safety
    from typing_extensions import Literal  # type: ignore[assignment]


__all__ = [
    "AppConfig",
    "ConfigError",
    "DEFAULTS",
    "load_config",
    "validate",
]


_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AppConfig:
    """Immutable runtime configuration.

    The field order, names and types mirror the table documented in
    ``design.md`` under "Yapilandirma" so that property tests for
    Property 8 / Property 9 can introspect the dataclass via
    :func:`dataclasses.fields`.
    """

    # Alert thresholds (Requirements 5.5, 6.5)
    closed_eye_duration: float
    yawn_count: int
    yawn_time_window: float
    confidence_threshold: float

    # Camera and performance (Requirements 4.x, 10.6, 10.7, 10.8, 9.3)
    camera_index: int
    frame_skip: int
    inference_resolution: Optional[int]
    fps_log_interval: float

    # Model and deployment (Requirements 3.5, 8.5, 10.4)
    eye_model_path: Path
    mouth_model_path: Path
    export_format: Literal["pt", "onnx", "tensorrt"]

    # Sound and logging (Requirements 7.x, 9.x)
    enable_sound: bool
    alert_sound_duration: float
    log_file: Path
    log_max_bytes: int
    verbose: bool


# ---------------------------------------------------------------------------
# Defaults (Requirement 8.4)
# ---------------------------------------------------------------------------


# Documented defaults applied when a parameter is missing from the raw
# input. Keys map 1:1 to ``AppConfig`` fields; values match the
# defaults table in the design.
DEFAULTS: Dict[str, Any] = {
    "closed_eye_duration": 2.0,
    "yawn_count": 3,
    "yawn_time_window": 60.0,
    "confidence_threshold": 0.5,
    "camera_index": 0,
    "frame_skip": 1,
    "inference_resolution": None,
    "fps_log_interval": 1.0,
    "eye_model_path": "models/eye_best.pt",
    "mouth_model_path": "models/mouth_best.pt",
    "export_format": "pt",
    "enable_sound": False,
    "alert_sound_duration": 3.0,
    "log_file": "logs/app.log",
    "log_max_bytes": 10 * 1024 * 1024,
    "verbose": False,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigError(Exception):
    """Raised when a configuration value is invalid.

    Carries the offending parameter name, the rejected value and the
    reason text so the CLI layer can format a uniform error message
    (Requirement 8.5).
    """

    def __init__(self, parameter: str, value: Any, reason: str) -> None:
        self.parameter = parameter
        self.value = value
        self.reason = reason
        super().__init__(
            "Yapilandirma parametresi '{name}' gecersiz: {reason} "
            "(alinan deger={value!r})".format(
                name=parameter, reason=reason, value=value
            )
        )


# ---------------------------------------------------------------------------
# Validation helpers (private)
# ---------------------------------------------------------------------------


# ``bool`` is a subclass of ``int`` in Python, so an isinstance check
# would accept ``True`` / ``False`` where a numeric field is expected.
# This guard rejects booleans for numeric fields (Requirement 8.5).
def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _check_float_range(name: str, value: Any, lo: float, hi: float) -> float:
    if _is_bool(value) or not isinstance(value, (int, float)):
        raise ConfigError(
            name, value, "kayan noktali sayi bekleniyor"
        )
    f = float(value)
    if not (lo <= f <= hi):
        raise ConfigError(
            name,
            value,
            "{lo} ile {hi} arasinda olmali (uc degerler dahil)".format(
                lo=lo, hi=hi
            ),
        )
    return f


def _check_int_range(name: str, value: Any, lo: int, hi: int) -> int:
    if _is_bool(value) or not isinstance(value, int):
        raise ConfigError(name, value, "tam sayi bekleniyor")
    if not (lo <= value <= hi):
        raise ConfigError(
            name,
            value,
            "{lo} ile {hi} arasinda olmali (uc degerler dahil)".format(
                lo=lo, hi=hi
            ),
        )
    return value


def _check_bool(name: str, value: Any) -> bool:
    if not _is_bool(value):
        raise ConfigError(name, value, "true/false bekleniyor")
    return value


def _check_inference_resolution(name: str, value: Any) -> Optional[int]:
    if value is None:
        return None
    if _is_bool(value) or not isinstance(value, int):
        raise ConfigError(
            name, value, "null veya tam sayi bekleniyor"
        )
    if not (160 <= value <= 1280):
        raise ConfigError(
            name, value, "160 ile 1280 arasinda olmali"
        )
    if value % 32 != 0:
        raise ConfigError(name, value, "32'nin kati olmali")
    return value


def _check_export_format(name: str, value: Any) -> str:
    allowed = ("pt", "onnx", "tensorrt")
    if not isinstance(value, str) or value not in allowed:
        raise ConfigError(
            name,
            value,
            "izin verilen degerler: {0}".format(", ".join(allowed)),
        )
    return value


def _check_path_string(name: str, value: Any, max_len: int) -> Path:
    if not isinstance(value, (str, Path)):
        raise ConfigError(name, value, "dosya yolu dizgisi bekleniyor")
    text = str(value)
    if len(text) == 0:
        raise ConfigError(name, value, "bos olmamali")
    if len(text) > max_len:
        raise ConfigError(
            name,
            value,
            "en fazla {0} karakter olabilir".format(max_len),
        )
    return Path(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate(raw: Dict[str, Any]) -> AppConfig:
    """Validate a raw configuration mapping and return an
    :class:`AppConfig`.

    Behaviour (matches Property 8 and Property 9 in the design):

    * Missing keys are filled from :data:`DEFAULTS` and a
      ``WARNING``-level record is emitted per missing key
      (Requirement 8.4).
    * Each value is range / type checked against the contract listed
      in Property 8.
    * ``model_path`` is resolved and verified to exist on disk
      (Requirements 3.5 / 8.5).
    * On the first invalid value a :class:`ConfigError` is raised; no
      partial :class:`AppConfig` is returned.

    The function is pure with respect to the configuration mapping:
    only logging and a single read-only ``Path.exists`` filesystem
    check are performed.
    """

    if not isinstance(raw, dict):
        raise ConfigError(
            "<root>", raw, "yapilandirma kok nesnesi bir sozluk olmali"
        )

    # --- Step 1: detect unknown keys (cheap typo guard) -----------------
    unknown = set(raw.keys()) - set(DEFAULTS.keys())
    if unknown:
        first = sorted(unknown)[0]
        raise ConfigError(
            first,
            raw[first],
            "bilinmeyen yapilandirma anahtari",
        )

    # --- Step 2: fill in defaults with a warning per missing key --------
    merged: Dict[str, Any] = {}
    for key, default in DEFAULTS.items():
        if key in raw:
            merged[key] = raw[key]
        else:
            merged[key] = default
            _LOGGER.warning(
                "Yapilandirma parametresi '%s' eksik, varsayilan kullaniliyor: %r",
                key,
                default,
            )

    # --- Step 3: per-field range / type validation ----------------------
    closed_eye_duration = _check_float_range(
        "closed_eye_duration", merged["closed_eye_duration"], 0.5, 10.0
    )
    yawn_count = _check_int_range(
        "yawn_count", merged["yawn_count"], 1, 20
    )
    yawn_time_window = _check_float_range(
        "yawn_time_window", merged["yawn_time_window"], 10.0, 600.0
    )
    confidence_threshold = _check_float_range(
        "confidence_threshold", merged["confidence_threshold"], 0.0, 1.0
    )

    camera_index = _check_int_range(
        "camera_index", merged["camera_index"], 0, 10
    )
    frame_skip = _check_int_range(
        "frame_skip", merged["frame_skip"], 0, 30
    )
    inference_resolution = _check_inference_resolution(
        "inference_resolution", merged["inference_resolution"]
    )
    fps_log_interval = _check_float_range(
        "fps_log_interval", merged["fps_log_interval"], 1.0, 60.0
    )

    eye_model_path = _check_path_string(
        "eye_model_path", merged["eye_model_path"], 512
    )
    mouth_model_path = _check_path_string(
        "mouth_model_path", merged["mouth_model_path"], 512
    )
    export_format = _check_export_format(
        "export_format", merged["export_format"]
    )

    enable_sound = _check_bool("enable_sound", merged["enable_sound"])
    alert_sound_duration = _check_float_range(
        "alert_sound_duration", merged["alert_sound_duration"], 0.5, 60.0
    )
    log_file = _check_path_string(
        "log_file", merged["log_file"], 512
    )
    log_max_bytes = _check_int_range(
        "log_max_bytes",
        merged["log_max_bytes"],
        1 * 1024 * 1024,
        100 * 1024 * 1024,
    )
    verbose = _check_bool("verbose", merged["verbose"])

    # --- Step 4: filesystem precondition for model paths ---------------
    if not eye_model_path.exists():
        raise ConfigError(
            "eye_model_path",
            str(eye_model_path),
            "goz model agirlik dosyasi mevcut degil",
        )
    if not mouth_model_path.exists():
        raise ConfigError(
            "mouth_model_path",
            str(mouth_model_path),
            "agiz model agirlik dosyasi mevcut degil",
        )

    return AppConfig(
        closed_eye_duration=closed_eye_duration,
        yawn_count=yawn_count,
        yawn_time_window=yawn_time_window,
        confidence_threshold=confidence_threshold,
        camera_index=camera_index,
        frame_skip=frame_skip,
        inference_resolution=inference_resolution,
        fps_log_interval=fps_log_interval,
        eye_model_path=eye_model_path,
        mouth_model_path=mouth_model_path,
        export_format=export_format,  # type: ignore[arg-type]
        enable_sound=enable_sound,
        alert_sound_duration=alert_sound_duration,
        log_file=log_file,
        log_max_bytes=log_max_bytes,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# YAML loader (Requirements 8.2, 8.3, 8.6)
# ---------------------------------------------------------------------------


def load_config(path: Union[str, Path] = Path("config.yaml")) -> AppConfig:
    """Read a YAML configuration file and return a validated :class:`AppConfig`.

    The function performs three discrete failure-mode checks before
    delegating to :func:`validate` so the user gets a uniform
    :class:`ConfigError` regardless of whether the source is missing
    (Requirement 8.6), unparsable (Requirement 8.6), or structurally
    not a mapping. The validated parameters are then ready to be logged
    on startup (Requirement 8.2) and reloaded on a subsequent run
    (Requirement 8.3).

    Parameters
    ----------
    path:
        Filesystem path to the YAML file. Accepts ``str`` or
        :class:`~pathlib.Path`; defaults to ``config.yaml`` at the
        process working directory.

    Raises
    ------
    ConfigError
        * ``parameter="config_path"`` when ``path`` does not exist.
        * ``parameter="config_yaml"`` when YAML parsing fails.
        * ``parameter="<root>"`` when the parsed document is not a
          mapping (e.g. a top-level list or scalar).
        * Re-raises any :class:`ConfigError` from :func:`validate`
          (invalid value / type / missing ``model_path`` file).
    """

    config_path = Path(path)

    if not config_path.exists():
        raise ConfigError(
            parameter="config_path",
            value=str(config_path),
            reason="yapilandirma dosyasi bulunamadi",
        )

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(
            parameter="config_yaml",
            value=str(config_path),
            reason="YAML ayristirma hatasi: {0}".format(exc),
        ) from exc

    # An empty YAML file parses to ``None``. Treat that the same way as
    # a non-mapping document so the user gets a clear failure rather
    # than a silent fallback to defaults (Req 8.6).
    if not isinstance(parsed, dict):
        raise ConfigError(
            parameter="<root>",
            value=parsed,
            reason="yapilandirma kok nesnesi bir sozluk olmali",
        )

    return validate(parsed)
