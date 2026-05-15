"""Static-image prediction CLI.

Argparse entry point that loads the trained classifier, runs inference
on a single image, prints the top-class label plus the four per-class
scores to stdout, and either displays the annotated frame in an OpenCV
window or saves it to disk.

Usage::

    python -m src.static_predict --image path/to/photo.jpg
    python -m src.static_predict --image path/to/photo.jpg --save
    python -m src.static_predict --image photo.jpg --save --output out.png
    python -m src.static_predict --image photo.jpg --model-path models/best.pt

When neither ``--show`` nor ``--save`` is provided the CLI defaults to
displaying the result in a window titled ``"Tahmin"``. ``--model-path``
bypasses :func:`src.config.load_config` so the CLI can be exercised
before ``config.yaml`` is fully populated.

Validates Requirements 3.1, 3.2, 3.3, 3.4, 3.5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser used by :func:`main`.

    Kept separate so unit tests (task 9.2) can exercise the parser
    without invoking the full predictor pipeline.
    """

    parser = argparse.ArgumentParser(
        prog="python -m src.static_predict",
        description=(
            "Statik bir goruntu uzerinde surucu uyku/yorgunluk modelini "
            "calistirir; ust sinifi ve dort sinif icin guven skorlarini "
            "yazdirir, annotated goruntuyu ekrana basar veya diske yazar."
        ),
    )
    parser.add_argument(
        "--image",
        required=True,
        type=str,
        help="Tahmin yapilacak goruntu dosyasinin yolu.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Annotated goruntuyu OpenCV penceresinde goster.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Annotated goruntuyu diske kaydet (varsayilan ad: <stem>_pred<ext>).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Kayit yolu (sadece --save ile birlikte kullanilir).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Yapilandirma dosyasi yolu (varsayilan: config.yaml).",
    )
    parser.add_argument(
        "--eye-model-path",
        type=str,
        default=None,
        help=(
            "Goz modeli yolu. Verilirse config.yaml okunmaz; "
            "--mouth-model-path da verilmeli."
        ),
    )
    parser.add_argument(
        "--mouth-model-path",
        type=str,
        default=None,
        help=(
            "Agiz modeli yolu. Verilirse config.yaml okunmaz; "
            "--eye-model-path da verilmeli."
        ),
    )
    return parser


def _resolve_output_path(image_path: Path, output: Optional[str]) -> Path:
    """Resolve the destination path for ``--save``.

    Without an explicit ``--output`` we derive a sibling file by
    inserting ``_pred`` between the stem and the extension, so repeated
    runs do not silently overwrite the original.
    """

    if output:
        return Path(output)
    return image_path.with_name(
        image_path.stem + "_pred" + image_path.suffix
    )


def _annotate_top_label(frame, text: str):
    """Draw ``text`` near the top-left corner with a black backing box.

    Sized via :func:`cv2.getTextSize` plus a 6 px padding so the
    overlay stays compact across resolutions. Mutates ``frame`` in
    place and returns it so callers can chain calls fluently.
    """

    import cv2  # type: ignore[import-not-found]

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    pad = 6

    (text_w, text_h), baseline = cv2.getTextSize(
        text, font_face, font_scale, thickness
    )

    # Top-left anchor with 6 px padding on every side of the rectangle.
    rect_x1 = pad
    rect_y1 = pad
    rect_x2 = pad + text_w + 2 * pad
    rect_y2 = pad + text_h + baseline + 2 * pad

    cv2.rectangle(
        frame,
        (rect_x1, rect_y1),
        (rect_x2, rect_y2),
        color=(0, 0, 0),
        thickness=cv2.FILLED,
    )

    # Text baseline sits inside the rectangle with the same 6 px pad.
    text_x = rect_x1 + pad
    text_y = rect_y1 + pad + text_h
    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        font_face,
        font_scale,
        color=(255, 255, 255),
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    return frame


def _annotate_two_lines(frame, line1: str, line2: str):
    """Draw two stacked text lines (eye + mouth) at the top-left corner.

    Each line gets its own black backing rectangle so the overlay stays
    readable on busy backgrounds. Sized via :func:`cv2.getTextSize` plus
    a 6 px padding to mirror :func:`_annotate_top_label`'s look.
    """

    import cv2  # type: ignore[import-not-found]

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    pad = 6
    gap = 4  # vertical gap between the two lines

    y_top = pad
    for text in (line1, line2):
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font_face, font_scale, thickness
        )
        rect_x1 = pad
        rect_y1 = y_top
        rect_x2 = pad + text_w + 2 * pad
        rect_y2 = y_top + text_h + baseline + 2 * pad

        cv2.rectangle(
            frame,
            (rect_x1, rect_y1),
            (rect_x2, rect_y2),
            color=(0, 0, 0),
            thickness=cv2.FILLED,
        )
        cv2.putText(
            frame,
            text,
            (rect_x1 + pad, rect_y1 + pad + text_h),
            font_face,
            font_scale,
            color=(255, 255, 255),
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
        y_top = rect_y2 + gap

    return frame


def _print_error(message: str) -> None:
    """Write ``Hata: <message>`` to stderr."""

    print(f"Hata: {message}", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Returns the process exit code so unit tests can drive ``main``
    directly without spawning a subprocess.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    image_path = Path(args.image)

    # --- Step 2: resolve weight path / inference parameters --------------
    # ``--model-path`` short-circuits config loading so smoke tests work
    # before ``config.yaml`` is populated; otherwise pull everything
    # from the validated AppConfig.
    if args.eye_model_path or args.mouth_model_path:
        if not (args.eye_model_path and args.mouth_model_path):
            _print_error(
                "--eye-model-path ve --mouth-model-path birlikte verilmelidir."
            )
            return 2
        eye_model_path = args.eye_model_path
        mouth_model_path = args.mouth_model_path
        inference_resolution: Optional[int] = None
        confidence_threshold: float = 0.5
    else:
        from src.config import ConfigError, load_config

        try:
            cfg = load_config(args.config)
        except ConfigError as exc:
            _print_error(str(exc))
            return 2

        eye_model_path = str(cfg.eye_model_path)
        mouth_model_path = str(cfg.mouth_model_path)
        inference_resolution = cfg.inference_resolution
        confidence_threshold = float(cfg.confidence_threshold)

    # --- Step 3: build the predictor (may raise ConfigError) -------------
    from src.config import ConfigError  # local import keeps argparse cheap
    from src.predictor import Predictor, format_prediction

    try:
        predictor = Predictor(
            eye_model_path=eye_model_path,
            mouth_model_path=mouth_model_path,
            inference_resolution=inference_resolution,
            confidence_threshold=confidence_threshold,
        )
    except ConfigError as exc:
        _print_error(str(exc))
        return 2

    # --- Step 4: run inference -------------------------------------------
    try:
        pred = predictor.predict_image(image_path)
    except (ValueError, FileNotFoundError) as exc:
        _print_error(str(exc))
        return 2

    # --- Step 5: print top-class + four scores (Req 3.2, 3.3) ------------
    print(format_prediction(pred))

    # --- Step 6: re-decode the source image for annotation --------------
    # Lazy import keeps the CLI importable on machines without OpenCV
    # (e.g. argparse-only unit tests).
    import cv2  # type: ignore[import-not-found]

    img = cv2.imread(str(image_path))
    if img is None:
        _print_error(
            f"Goruntu dosyasi okunamadi (bozuk veya desteklenmeyen format): "
            f"{image_path}"
        )
        return 2

    if not pred.raw_scores:
        _print_error("Tahmin sonucu sinif skoru icermiyor.")
        return 2

    # Compute eye / mouth labels independently from raw_scores so the
    # overlay matches what ``format_prediction`` printed (göz ve ağız
    # bağımsız raporlanır).
    rs = pred.raw_scores
    closed_s = rs.get("Closed", 0.0)
    open_s = rs.get("Open", 0.0)
    eye_label, eye_conf = ("Closed", closed_s) if closed_s >= open_s else ("Open", open_s)

    yawn_s = rs.get("yawn", 0.0)
    no_yawn_s = rs.get("no_yawn", 0.0)
    mouth_label, mouth_conf = (
        ("yawn", yawn_s) if yawn_s >= no_yawn_s else ("no_yawn", no_yawn_s)
    )

    # Render two stacked lines: eye then mouth.
    line1 = f"Goz: {eye_label} ({eye_conf:.2f})"
    line2 = f"Agiz: {mouth_label} ({mouth_conf:.2f})"
    _annotate_two_lines(img, line1, line2)

    # --- Step 7: --save (defaults to ``<stem>_pred<ext>``) ---------------
    if args.save:
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = image_path.with_name(
                image_path.stem + "_pred" + image_path.suffix
            )
        cv2.imwrite(str(out_path), img)
        print(f"Kaydedildi: {out_path}")

    # --- Step 8: --show, plus default-when-neither (Req 3.3) -------------
    # Default behaviour when the user passes neither flag is to display
    # the annotated frame so the CLI is useful out of the box.
    if args.show or (not args.save and not args.show):
        try:
            cv2.imshow("Tahmin", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except (cv2.error, Exception):
            # Headless environments (CI, Jetson without X) raise
            # ``cv2.error`` from ``imshow``. Degrade gracefully so the
            # CLI still exits 0 with a useful message.
            print("Goruntu acilamadi (headless ortam)")

    return 0


if __name__ == "__main__":  # pragma: no cover - manual CLI entry point
    sys.exit(main())
