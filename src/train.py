"""Egitici module.

Validates training parameters for YOLOv8 classification training,
exposes a CLI-override merge helper, and provides the
``run_training`` entry point plus a ``python -m src.train`` argparse
interface.

Public surface:

* :data:`ALLOWED_MODEL_SIZES` and the documented numeric ranges
* :class:`TrainConfig` (frozen dataclass)
* :func:`validate_train_config` (pure validator)
* :func:`merge_cli_overrides` (CLI > file precedence helper)
* :func:`run_training` (Ultralytics wrapper, copies best.pt)
* :func:`main` (CLI entry point)

Implements tasks 8.1 and 8.2 of the driver-drowsiness-detection spec.

Validates Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

import yaml

# ---------------------------------------------------------------------------
# Allowed values / ranges (Requirement 2.2)
# ---------------------------------------------------------------------------

#: Set of YOLOv8 classification model sizes accepted by the trainer.
ALLOWED_MODEL_SIZES: Tuple[str, ...] = (
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
)

# Inclusive integer ranges and steps documented in Requirement 2.2.
EPOCHS_MIN: int = 1
EPOCHS_MAX: int = 500

IMGSZ_MIN: int = 320
IMGSZ_MAX: int = 1280
IMGSZ_STEP: int = 32  # ``imgsz`` must be a multiple of 32

BATCH_MIN: int = 1
BATCH_MAX: int = 128

# Documented defaults (Requirement 2.2).
DEFAULT_MODEL_SIZE: str = "yolov8n"
DEFAULT_EPOCHS: int = 50
DEFAULT_IMGSZ: int = 640
DEFAULT_BATCH: int = 16
DEFAULT_DATA: str = "dataset"
DEFAULT_OUTPUT_DIR: str = "models"
DEFAULT_OUTPUT_FILENAME: str = "best.pt"


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    """Immutable, validated training configuration.

    Construct instances through :func:`validate_train_config` so range
    invariants from Requirement 2.2 are guaranteed; constructing the
    dataclass directly is technically allowed but bypasses validation.

    Attributes
    ----------
    model_size:
        One of :data:`ALLOWED_MODEL_SIZES`.
    epochs:
        Number of training epochs in ``[EPOCHS_MIN, EPOCHS_MAX]``.
    imgsz:
        Square training resolution in ``[IMGSZ_MIN, IMGSZ_MAX]`` and a
        multiple of :data:`IMGSZ_STEP`.
    batch:
        Mini-batch size in ``[BATCH_MIN, BATCH_MAX]``.
    data:
        Path to the YOLOv8 classification dataset root (the directory
        produced by :mod:`src.data_prep` containing ``train/``,
        ``test/`` and ``data.yaml``). Stored as ``str`` for direct use
        by the Ultralytics API.
    output_dir:
        Directory under which task 8.2 will copy the best checkpoint to
        ``best.pt`` (Requirement 2.4 / 2.5).
    """

    model_size: str = DEFAULT_MODEL_SIZE
    epochs: int = DEFAULT_EPOCHS
    imgsz: int = DEFAULT_IMGSZ
    batch: int = DEFAULT_BATCH
    data: str = DEFAULT_DATA
    output_dir: str = DEFAULT_OUTPUT_DIR
    output_filename: str = DEFAULT_OUTPUT_FILENAME


# ---------------------------------------------------------------------------
# Deferred ConfigError import (task 2.1 may not be merged yet)
# ---------------------------------------------------------------------------


def _raise_config_error(parameter: str, value: Any, reason: str) -> None:
    """Raise ``src.config.ConfigError`` with parameter context.

    The import is deferred to call time so :mod:`src.train` can still
    be imported when :mod:`src.config` is a placeholder (task 2.1
    implements ``ConfigError``). When the symbol is unavailable we
    re-raise a clear :class:`RuntimeError` so callers still see the
    diagnostic instead of a cryptic ``ImportError``.
    """
    try:
        from src.config import ConfigError  # type: ignore[attr-defined]
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Cannot raise ConfigError because src.config.ConfigError is "
            "not implemented yet (task 2.1). "
            f"parameter={parameter!r}, value={value!r}, reason={reason}"
        ) from exc

    # ``src.config.ConfigError`` signature is
    # ``__init__(parameter, value, reason)``; pass positionally so this
    # call survives keyword renames in :mod:`src.config`.
    raise ConfigError(parameter, value, reason)


# ---------------------------------------------------------------------------
# CLI override merging (Requirement 2.2)
# ---------------------------------------------------------------------------


def merge_cli_overrides(
    file_cfg: Mapping[str, Any],
    cli_overrides: Mapping[str, Any],
) -> dict:
    """Merge CLI overrides on top of file-config values.

    CLI overrides win over file-config values for any key whose CLI
    value is **not** ``None``. ``argparse`` populates unset arguments
    with ``None`` by default, so ``None`` is interpreted as "user did
    not supply this flag" and the file value (or, later, the
    dataclass default) is preserved.

    Parameters
    ----------
    file_cfg:
        Mapping read from ``config.yaml`` or any other file source.
    cli_overrides:
        Mapping derived from parsed CLI arguments. ``None`` values are
        ignored.

    Returns
    -------
    dict
        A new ``dict`` (neither input is mutated) suitable for passing
        to :func:`validate_train_config`.
    """
    if not isinstance(file_cfg, ABCMapping):
        raise TypeError(
            "file_cfg must be a Mapping; got "
            f"{type(file_cfg).__name__}"
        )
    if not isinstance(cli_overrides, ABCMapping):
        raise TypeError(
            "cli_overrides must be a Mapping; got "
            f"{type(cli_overrides).__name__}"
        )

    merged: dict = dict(file_cfg)
    for key, value in cli_overrides.items():
        if value is None:
            continue
        merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Validation (Requirements 2.2, 2.3, 2.7)
# ---------------------------------------------------------------------------


def _validate_int_in_range(
    name: str, value: Any, low: int, high: int
) -> int:
    """Return ``value`` as an ``int`` if it lies in ``[low, high]``.

    Rejects ``bool`` even though ``bool`` is a subclass of ``int`` in
    Python; treating ``True``/``False`` as a valid epoch count would
    silently corrupt training runs.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        _raise_config_error(
            parameter=name,
            value=value,
            reason=(
                f"{name} must be an integer in [{low}, {high}]; "
                f"received type: {type(value).__name__}"
            ),
        )
    if value < low or value > high:
        _raise_config_error(
            parameter=name,
            value=value,
            reason=f"{name} must be in [{low}, {high}]",
        )
    return int(value)


def _validate_path_string(name: str, value: Any, default: str) -> str:
    """Coerce ``value`` to a non-empty path string.

    ``Path`` instances are accepted and converted via ``str``.
    """
    if value is None:
        return default
    if isinstance(value, Path):
        text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        _raise_config_error(
            parameter=name,
            value=value,
            reason=(
                f"{name} must be a string path; "
                f"received type: {type(value).__name__}"
            ),
        )
        return default  # unreachable; satisfies type-checker
    if not text:
        _raise_config_error(
            parameter=name,
            value=value,
            reason=f"{name} must be a non-empty path string",
        )
    return text


def validate_train_config(raw: Mapping[str, Any]) -> TrainConfig:
    """Validate ``raw`` and return a frozen :class:`TrainConfig`.

    Missing keys fall back to the documented defaults from Requirement
    2.2. Out-of-range or wrong-type values raise
    :class:`src.config.ConfigError` carrying the parameter name, the
    received value and the expected range / value-set, *without*
    starting any training (Requirement 2.3, 2.7).

    Unknown keys are ignored so callers can pass a YAML config dict
    that contains additional unrelated fields (e.g. logging settings).

    Parameters
    ----------
    raw:
        Mapping of parameter names to values. Typically the dictionary
        produced by :func:`merge_cli_overrides`.
    """
    if not isinstance(raw, ABCMapping):
        _raise_config_error(
            parameter="<train_config>",
            value=raw,
            reason=(
                "train config must be a mapping; received type: "
                f"{type(raw).__name__}"
            ),
        )

    # ---- model_size --------------------------------------------------
    model_size = raw.get("model_size", DEFAULT_MODEL_SIZE)
    if not isinstance(model_size, str) or model_size not in ALLOWED_MODEL_SIZES:
        _raise_config_error(
            parameter="model_size",
            value=model_size,
            reason=(
                "model_size must be one of "
                f"{list(ALLOWED_MODEL_SIZES)}"
            ),
        )

    # ---- epochs ------------------------------------------------------
    epochs = _validate_int_in_range(
        "epochs",
        raw.get("epochs", DEFAULT_EPOCHS),
        EPOCHS_MIN,
        EPOCHS_MAX,
    )

    # ---- imgsz (range + multiple-of-32) ------------------------------
    imgsz = _validate_int_in_range(
        "imgsz",
        raw.get("imgsz", DEFAULT_IMGSZ),
        IMGSZ_MIN,
        IMGSZ_MAX,
    )
    if imgsz % IMGSZ_STEP != 0:
        _raise_config_error(
            parameter="imgsz",
            value=imgsz,
            reason=(
                f"imgsz must be a multiple of {IMGSZ_STEP} within "
                f"[{IMGSZ_MIN}, {IMGSZ_MAX}]"
            ),
        )

    # ---- batch -------------------------------------------------------
    batch = _validate_int_in_range(
        "batch",
        raw.get("batch", DEFAULT_BATCH),
        BATCH_MIN,
        BATCH_MAX,
    )

    # ---- data --------------------------------------------------------
    data = _validate_path_string("data", raw.get("data"), DEFAULT_DATA)

    # ---- output_dir --------------------------------------------------
    output_dir = _validate_path_string(
        "output_dir", raw.get("output_dir"), DEFAULT_OUTPUT_DIR
    )

    # ---- output_filename ---------------------------------------------
    output_filename = _validate_path_string(
        "output_filename", raw.get("output_filename"), DEFAULT_OUTPUT_FILENAME
    )

    return TrainConfig(
        model_size=model_size,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        data=data,
        output_dir=output_dir,
        output_filename=output_filename,
    )


# ---------------------------------------------------------------------------
# Training entry point (Requirements 2.1, 2.4, 2.5, 2.6, 2.7, 2.8)
# ---------------------------------------------------------------------------


def run_training(cfg: TrainConfig) -> Path:
    """Run YOLOv8 classification training and copy the best checkpoint.

    Validates that ``cfg.data`` and ``cfg.data/data.yaml`` exist
    *before* touching ``cfg.output_dir`` so a missing dataset never
    perturbs existing weights (Requirement 2.7). On success the best
    checkpoint produced by the Ultralytics trainer is copied to
    ``<cfg.output_dir>/best.pt`` (Requirements 2.4, 2.5) and the path
    to that copy is returned.

    Parameters
    ----------
    cfg:
        Validated training configuration. Must originate from
        :func:`validate_train_config` so range invariants are
        guaranteed.

    Returns
    -------
    Path
        Absolute / relative path of the copied best checkpoint
        (``<cfg.output_dir>/best.pt``).

    Raises
    ------
    RuntimeError
        When the dataset root or its ``data.yaml`` file is missing.
    ImportError
        When the optional ``ultralytics`` dependency is not installed.
    """
    data_root = Path(cfg.data)
    data_yaml = data_root / "data.yaml"
    if not data_root.exists() or not data_yaml.exists():
        raise RuntimeError(
            "Veri seti bulunamadi: "
            f"data_root={data_root!s}, data.yaml={data_yaml!s}"
        )

    # Lazy import so module import does not fail when ultralytics is
    # absent (e.g. during unit-testing of validate_train_config).
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment dep
        raise ImportError(
            "ultralytics paketi kurulu degil. "
            "requirements.txt'i kurarak tekrar deneyin "
            "(pip install -r requirements.txt)."
        ) from exc

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(f"{cfg.model_size}-cls.pt")
    result = model.train(
        data=cfg.data,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        task="classify",
        verbose=True,
    )

    # Locate the best checkpoint produced by the Ultralytics trainer.
    best = getattr(getattr(model, "trainer", None), "best", None)
    if best is None:
        save_dir = getattr(result, "save_dir", None)
        if save_dir is not None:
            best = Path(save_dir) / "weights" / "best.pt"
    if best is None:
        raise RuntimeError(
            "Egitim tamamlandi ancak best.pt konumu bulunamadi."
        )

    dest = output_dir / cfg.output_filename
    shutil.copy2(best, dest)

    # Best-effort final-metrics print (Requirement 2.6).
    try:
        metrics = getattr(result, "results_dict", {}) or {}
        accuracy = metrics.get("metrics/accuracy_top1", "N/A")
        print(f"Egitim tamamlandi. accuracy: {accuracy}")
    except Exception:  # pragma: no cover - defensive
        pass

    return dest


# ---------------------------------------------------------------------------
# CLI entry point (Requirement 2.2: CLI overrides file config)
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for ``python -m src.train``.

    All defaults are ``None`` so :func:`merge_cli_overrides` correctly
    skips flags the user did not supply, allowing values from
    ``--config`` (or :class:`TrainConfig` defaults) to take effect.
    """
    parser = argparse.ArgumentParser(
        prog="src.train",
        description=(
            "YOLOv8 sinflandirma egitimi. CLI argumanlari --config "
            "dosyasindaki degerleri ezer."
        ),
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Veri seti kok dizini (data.yaml iceren YOLOv8-cls kokunu).",
    )
    parser.add_argument(
        "--model-size",
        default=None,
        choices=ALLOWED_MODEL_SIZES,
        help=f"Model boyutu (varsayilan: {DEFAULT_MODEL_SIZE}).",
    )
    parser.add_argument(
        "--epochs",
        default=None,
        type=int,
        help=f"Epoch sayisi [{EPOCHS_MIN}, {EPOCHS_MAX}].",
    )
    parser.add_argument(
        "--imgsz",
        default=None,
        type=int,
        help=(
            f"Goruntu boyutu [{IMGSZ_MIN}, {IMGSZ_MAX}], "
            f"{IMGSZ_STEP} kati."
        ),
    )
    parser.add_argument(
        "--batch",
        default=None,
        type=int,
        help=f"Batch boyutu [{BATCH_MIN}, {BATCH_MAX}].",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Opsiyonel YAML yapilandirma dosyasi yolu.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="best.pt'nin kopyalanacagi cikti dizini.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Cikti agirlik dosyasi adi (varsayilan: best.pt).",
    )
    return parser


def main(argv: list[str] | None = None) -> Path:
    """CLI entry point for ``python -m src.train``.

    Reads optional YAML config, merges CLI overrides on top, validates
    the resulting parameters and delegates to :func:`run_training`.
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    file_cfg: dict = {}
    if args.config is not None:
        config_path = Path(args.config)
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if loaded is None:
            file_cfg = {}
        elif isinstance(loaded, ABCMapping):
            file_cfg = dict(loaded)
        else:
            raise RuntimeError(
                f"Yapilandirma dosyasi bir mapping olmali: {config_path!s}"
            )

    cli_keys = {"data", "model_size", "epochs", "imgsz", "batch", "output_dir", "output_name"}
    cli = {k: v for k, v in vars(args).items() if k in cli_keys}
    # argparse gives us ``output_name``; rename to match TrainConfig field.
    if "output_name" in cli:
        cli["output_filename"] = cli.pop("output_name")

    merged = merge_cli_overrides(file_cfg, cli)
    cfg = validate_train_config(merged)
    return run_training(cfg)


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    try:
        main()
    except Exception as exc:
        print(f"HATA: {exc}", file=sys.stderr)
        raise


__all__ = [
    "ALLOWED_MODEL_SIZES",
    "EPOCHS_MIN",
    "EPOCHS_MAX",
    "IMGSZ_MIN",
    "IMGSZ_MAX",
    "IMGSZ_STEP",
    "BATCH_MIN",
    "BATCH_MAX",
    "DEFAULT_MODEL_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_IMGSZ",
    "DEFAULT_BATCH",
    "DEFAULT_DATA",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_OUTPUT_FILENAME",
    "TrainConfig",
    "validate_train_config",
    "merge_cli_overrides",
    "run_training",
    "main",
]
