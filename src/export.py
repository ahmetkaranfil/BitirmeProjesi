"""Model export helper for Jetson Nano deployment.

Validates Requirements 10.1-10.5: convert .pt weights to ONNX or TensorRT
and gracefully fall back to .pt on any failure.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Union

ALLOWED_FORMATS = ("pt", "onnx", "tensorrt")
_LOGGER = logging.getLogger("ddd")

def export_model(
    path: Union[str, Path],
    fmt: str,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Export a YOLOv8 .pt model to ONNX or TensorRT.

    Returns the exported artifact path. On any failure, logs ERROR and
    returns the original .pt path so inference can still run (Req 10.5).
    """
    log = logger if logger is not None else _LOGGER

    if fmt not in ALLOWED_FORMATS:
        raise ValueError(
            f"Desteklenmeyen export formati: {fmt!r}; "
            f"izin verilen: {ALLOWED_FORMATS}"
        )

    src_path = Path(path)

    if fmt == "pt":
        return src_path  # No-op for native format

    if not src_path.exists():
        log.error(
            "Model dosyasi bulunamadi: %s; export atlandi", src_path
        )
        return src_path

    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError as exc:
        log.error(
            "ultralytics paketi kurulu degil: %s; .pt dosyasi kullanilacak",
            exc,
        )
        return src_path

    try:
        model = YOLO(str(src_path))
        # Ultralytics: ONNX uses format="onnx", TensorRT uses format="engine"
        ultralytics_fmt = "engine" if fmt == "tensorrt" else "onnx"
        exported = model.export(format=ultralytics_fmt)
        # Ultralytics returns either a string path or a list of paths
        if isinstance(exported, (list, tuple)):
            exported = exported[0]
        result_path = Path(exported)
        log.info("Model basariyla export edildi (%s): %s", fmt, result_path)
        return result_path
    except Exception as exc:
        log.error(
            "Model export (%s) basarisiz: %s; .pt dosyasi kullanilacak",
            fmt,
            exc,
        )
        return src_path

__all__ = ["export_model", "ALLOWED_FORMATS"]
