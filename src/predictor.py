"""Tahminci - iki ayri YOLOv8 sinflandirma modeli sarmalayicisi.

* Goz modeli (``eye_model_path``) kirpilmis goz goruntulerinde
  ``Closed`` / ``Open`` ayrimini yapar.
* Agiz modeli (``mouth_model_path``) tam yuz goruntusunde
  ``yawn`` / ``no_yawn`` ayrimini yapar.

Tam yuz girisinden gozleri kirpmak icin OpenCV haarcascade
``haarcascade_eye.xml`` kullanilir. Goz tespit edilemezse ``eye``
``None`` doner ve kare belirsiz olarak isaretlenir.

Validates Requirements 3.1, 3.2, 3.4, 3.5, 4.3, 8.5, 10.8.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from src.alert_logic import Prediction
    from src.config import AppConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raise_config_error(parameter: str, value: Any, reason: str) -> None:
    """Raise ``src.config.ConfigError`` with parameter context."""
    from src.config import ConfigError  # local import; avoids cycles

    raise ConfigError(parameter, value, reason)


def _load_yolo(model_path: Path):
    """Load an Ultralytics YOLO model from ``model_path``.

    Lazy-imports ``ultralytics`` so this module stays importable when
    the dependency is missing (test environments).
    """
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "ultralytics paketi kurulu degil. "
            "`pip install -r requirements.txt` ile kurun."
        ) from exc
    return YOLO(str(model_path))


def _scores_from_result(result: Any) -> Tuple[List[float], dict]:
    """Extract per-class probabilities from an Ultralytics ``Results``.

    Returns ``(scores, index_to_name)``.
    """
    probs = getattr(result, "probs", None)
    if probs is None or getattr(probs, "data", None) is None:
        raise RuntimeError(
            "Ultralytics sonucu sinif olasiliklari icermiyor; "
            "modelin sinflandirma modunda egitildiginden emin olun."
        )

    tensor = probs.data
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "tolist"):
        scores = tensor.tolist()
    else:
        scores = list(tensor)

    names_map = getattr(result, "names", None) or {}
    if isinstance(names_map, dict):
        index_to_name = {int(k): str(v) for k, v in names_map.items()}
    else:
        index_to_name = {i: str(n) for i, n in enumerate(names_map)}

    if len(scores) != len(index_to_name):
        raise RuntimeError(
            "Sinif sayisi ile olasilik vektor uzunlugu uyusmuyor: "
            f"len(scores)={len(scores)}, len(names)={len(index_to_name)}"
        )

    return scores, index_to_name


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class Predictor:
    """Iki YOLOv8-cls modelini ve goz tespitini sarmalayan adaptor.

    Parametreler
    ------------
    eye_model_path:
        Closed/Open icin egitilmis ``.pt`` agirlik dosyasi.
    mouth_model_path:
        yawn/no_yawn icin egitilmis ``.pt`` agirlik dosyasi.
    inference_resolution:
        Agiz modelinin cikarim cozunurlugu (None -> orijinal boyut).
        Goz modeli her zaman kucuk kirpik aldigi icin yeniden boyut
        uygulanmaz.
    confidence_threshold:
        Bu degerin altinda kalan tahminler ``None`` olarak raporlanir.
    config:
        :class:`AppConfig` ornegi; verilmisse yukaridaki alanlar
        ondan okunur (acik argumanlar varsa onlar oncelikli).
    """

    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
    EYE_CLASSES: Tuple[str, ...] = ("Closed", "Open")
    MOUTH_CLASSES: Tuple[str, ...] = ("no_yawn", "yawn")
    SUPPORTED_IMAGE_EXTENSIONS: Tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    )

    def __init__(
        self,
        eye_model_path: Optional[Union[str, Path]] = None,
        mouth_model_path: Optional[Union[str, Path]] = None,
        inference_resolution: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        *,
        config: Optional["AppConfig"] = None,
    ) -> None:
        # --- Resolve parameters (explicit > config > defaults) -----------
        if eye_model_path is None and config is not None:
            eye_model_path = getattr(config, "eye_model_path", None)
        if mouth_model_path is None and config is not None:
            mouth_model_path = getattr(config, "mouth_model_path", None)

        if eye_model_path is None:
            _raise_config_error(
                "eye_model_path",
                None,
                "Goz modeli yolu belirtilmedi",
            )
        if mouth_model_path is None:
            _raise_config_error(
                "mouth_model_path",
                None,
                "Agiz modeli yolu belirtilmedi",
            )

        eye_resolved = Path(eye_model_path)  # type: ignore[arg-type]
        mouth_resolved = Path(mouth_model_path)  # type: ignore[arg-type]

        if not eye_resolved.exists():
            _raise_config_error(
                "eye_model_path",
                str(eye_resolved),
                f"Goz modeli bulunamadi: {eye_resolved}",
            )
        if not mouth_resolved.exists():
            _raise_config_error(
                "mouth_model_path",
                str(mouth_resolved),
                f"Agiz modeli bulunamadi: {mouth_resolved}",
            )

        if inference_resolution is None and config is not None:
            inference_resolution = getattr(config, "inference_resolution", None)
        if confidence_threshold is None and config is not None:
            confidence_threshold = getattr(config, "confidence_threshold", None)
        if confidence_threshold is None:
            confidence_threshold = self.DEFAULT_CONFIDENCE_THRESHOLD

        # --- Load both models -----------------------------------------
        self.eye_model_path: Path = eye_resolved
        self.mouth_model_path: Path = mouth_resolved
        self.inference_resolution: Optional[int] = inference_resolution
        self.confidence_threshold: float = float(confidence_threshold)

        self._eye_model = _load_yolo(eye_resolved)
        self._mouth_model = _load_yolo(mouth_resolved)
        self._eye_cascade = None  # lazy-loaded on first frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_image(self, path: Union[str, Path]) -> "Prediction":
        """Tek bir goruntu dosyasini siniflandirir."""
        import cv2  # type: ignore[import-not-found]

        if path is None:
            raise ValueError("Goruntu yolu bos olamaz.")

        image_path = Path(path)
        if not image_path.exists():
            raise ValueError(f"Goruntu dosyasi bulunamadi: {image_path}")
        if not image_path.is_file():
            raise ValueError(f"Goruntu yolu bir dosya degil: {image_path}")

        suffix = image_path.suffix.lower()
        if suffix not in self.SUPPORTED_IMAGE_EXTENSIONS:
            allowed = ", ".join(self.SUPPORTED_IMAGE_EXTENSIONS)
            raise ValueError(
                f"Desteklenmeyen goruntu uzantisi {suffix!r}; "
                f"izin verilenler: {allowed}"
            )

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(
                f"Goruntu dosyasi okunamadi (bozuk veya desteklenmeyen format): "
                f"{image_path}"
            )
        return self.predict_frame(frame)

    def predict_frame(
        self,
        frame: "np.ndarray",
        t_capture_s: Optional[float] = None,
    ) -> "Prediction":
        """BGR kare uzerinde iki modeli ve haarcascade goz kirpmayi calistirir."""
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]

        from src.alert_logic import Prediction

        if frame is None:
            raise ValueError("Tahmin icin verilen kare bos olamaz.")
        if not isinstance(frame, np.ndarray):
            raise ValueError(
                "Tahmin icin verilen kare bir numpy.ndarray olmali; "
                f"alinan tip={type(frame).__name__}"
            )
        if frame.size == 0:
            raise ValueError("Tahmin icin verilen kare bos.")

        if t_capture_s is None:
            t_capture_s = time.monotonic()

        # --- Mouth: full-frame inference ------------------------------
        mouth_frame = frame
        if self.inference_resolution is not None:
            target = int(self.inference_resolution)
            mouth_frame = cv2.resize(
                frame, (target, target), interpolation=cv2.INTER_LINEAR
            )

        mouth_results = self._mouth_model(mouth_frame, verbose=False)
        if not mouth_results:
            raise RuntimeError("Agiz modeli bos sonuc dondurdu.")
        mouth_scores, mouth_names = _scores_from_result(mouth_results[0])
        mouth_raw = self._build_raw_scores(
            mouth_scores, mouth_names, self.MOUTH_CLASSES
        )

        # --- Eye: crop with haarcascade, then classify largest crop ----
        eye_crop = self._extract_eye_crop(frame)
        if eye_crop is None:
            eye_label: Optional[str] = None
            eye_conf: float = 0.0
            eye_raw = {"Closed": 0.0, "Open": 0.0}
        else:
            eye_results = self._eye_model(eye_crop, verbose=False)
            if not eye_results:
                raise RuntimeError("Goz modeli bos sonuc dondurdu.")
            eye_scores, eye_names = _scores_from_result(eye_results[0])
            eye_raw = self._build_raw_scores(
                eye_scores, eye_names, self.EYE_CLASSES
            )
            if eye_raw["Closed"] >= eye_raw["Open"]:
                eye_label, eye_conf = "Closed", eye_raw["Closed"]
            else:
                eye_label, eye_conf = "Open", eye_raw["Open"]

        # --- Mouth argmax + confidence threshold ---------------------
        if mouth_raw["yawn"] >= mouth_raw["no_yawn"]:
            mouth_label: Optional[str] = "yawn"
            mouth_conf: float = mouth_raw["yawn"]
        else:
            mouth_label = "no_yawn"
            mouth_conf = mouth_raw["no_yawn"]

        if eye_label is not None and eye_conf < self.confidence_threshold:
            eye_label = None
        if mouth_conf < self.confidence_threshold:
            mouth_label = None

        # Combined raw_scores for diagnostics / format_prediction.
        raw_scores = {**eye_raw, **mouth_raw}

        return Prediction(
            eye=eye_label,  # type: ignore[arg-type]
            mouth=mouth_label,  # type: ignore[arg-type]
            eye_conf=eye_conf,
            mouth_conf=mouth_conf,
            raw_scores=raw_scores,
            t_capture_s=float(t_capture_s),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_raw_scores(
        scores: List[float],
        index_to_name: dict,
        required: Tuple[str, ...],
    ) -> dict:
        """Sinif adi -> [0,1] araliginda skor sozlugu uretir."""
        raw: dict = {}
        for idx, score in enumerate(scores):
            name = index_to_name.get(idx)
            if name is None:
                raise RuntimeError(
                    f"Sinif indeksi {idx} icin isim bulunamadi."
                )
            raw[name] = max(0.0, min(1.0, float(score)))
        for cls in required:
            if cls not in raw:
                raise RuntimeError(
                    f"Beklenen sinif '{cls}' modelin sinif kumesinde yok."
                )
        return raw

    def _extract_eye_crop(self, frame: "np.ndarray") -> Optional["np.ndarray"]:
        """OpenCV haarcascade ile goz tespit eder ve en buyuk kirpigi doner.

        Hicbir goz bulunamazsa ``None`` doner. Birden fazla goz varsa
        en buyuk dikdortgenli olani secer (daha guvenilir kirpik).
        """
        import cv2  # type: ignore[import-not-found]

        if self._eye_cascade is None:
            cascade_path = (
                Path(cv2.data.haarcascades) / "haarcascade_eye.xml"
            )
            if not cascade_path.exists():
                # OpenCV kurulumu cascade dosyasini icermiyor; goz
                # tespiti devre disi.
                return None
            self._eye_cascade = cv2.CascadeClassifier(str(cascade_path))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = self._eye_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(eyes) == 0:
            return None

        # En buyuk dikdortgen (en guvenilir kirpik).
        x, y, w, h = max(eyes, key=lambda r: r[2] * r[3])
        # Hafif bir kenar ekle (bazi haarcascade kirpimlari kasi keser).
        margin = max(2, int(0.1 * max(w, h)))
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(frame.shape[1], x + w + margin)
        y1 = min(frame.shape[0], y + h + margin)
        return frame[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Format helper for CLI output
# ---------------------------------------------------------------------------


def format_prediction(p: "Prediction") -> str:
    """Bir :class:`Prediction`'i iki bagimsiz satir olarak yazdirir.

    Cikti sablonu::

        Goz:  <Open|Closed|Belirsiz> (<score>)
        Agiz: <yawn|no_yawn|Belirsiz> (<score>)

        Tum siniflar:
          Closed:  0.NN
          Open:    0.NN
          no_yawn: 0.NN
          yawn:    0.NN
    """
    raw = dict(p.raw_scores)
    if not raw:
        raise ValueError("Prediction.raw_scores bos; format edilemiyor.")

    closed_s = raw.get("Closed", 0.0)
    open_s = raw.get("Open", 0.0)
    if closed_s >= open_s:
        eye_label, eye_conf = "Closed", closed_s
    else:
        eye_label, eye_conf = "Open", open_s

    yawn_s = raw.get("yawn", 0.0)
    no_yawn_s = raw.get("no_yawn", 0.0)
    if yawn_s >= no_yawn_s:
        mouth_label, mouth_conf = "yawn", yawn_s
    else:
        mouth_label, mouth_conf = "no_yawn", no_yawn_s

    eye_text = (
        f"{eye_label} ({eye_conf:.2f})"
        if p.eye is not None
        else f"Belirsiz ({eye_conf:.2f})"
    )
    mouth_text = (
        f"{mouth_label} ({mouth_conf:.2f})"
        if p.mouth is not None
        else f"Belirsiz ({mouth_conf:.2f})"
    )

    canonical_order = ("Closed", "Open", "no_yawn", "yawn")
    label_width = max(len(name) for name in canonical_order) + 1

    lines = [
        f"Goz:  {eye_text}",
        f"Agiz: {mouth_text}",
        "",
        "Tum siniflar:",
    ]
    for name in canonical_order:
        score = raw.get(name)
        if score is None:
            continue
        label = ("  " + name + ":").ljust(label_width + 2)
        lines.append(f"{label} {float(score):.2f}")

    return "\n".join(lines)


__all__ = ["Predictor", "format_prediction"]
