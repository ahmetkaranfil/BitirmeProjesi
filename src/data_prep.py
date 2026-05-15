"""Veri_Hazirlayici - iki bagimsiz veri seti hazirlar.

Kaynak ``eye_yawn_dataset/`` icindeki dort sinifi iki YOLOv8
siniflandirma veri setine ayirir:

* ``<output>/eye/`` - ``Closed`` / ``Open`` (kirpilmis goz goruntuleri)
* ``<output>/mouth/`` - ``no_yawn`` / ``yawn`` (tam yuz goruntuleri)

Her alt veri setinin altinda ``train/`` ve ``test/`` dizinleri ile
sinif klasorleri ve YOLO siniflandirma ``data.yaml`` dosyasi
olusturulur. CLI giriaaa noktasi ``python -m src.data_prep``.

Validates Requirements 1.1-1.7.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple, Union

import yaml


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

EYE_CLASSES: Tuple[str, ...] = ("Closed", "Open")
"""Goz alt veri setinin sinif sirasi (indeks 0..1)."""

MOUTH_CLASSES: Tuple[str, ...] = ("no_yawn", "yawn")
"""Agiz alt veri setinin sinif sirasi (indeks 0..1)."""

ALLOWED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")
"""Buyuk/kucuk harf duyarsiz kabul edilen goruntu uzantilari."""

_SPLITS: Tuple[str, ...] = ("train", "test")
_ALL_CLASSES: Tuple[str, ...] = EYE_CLASSES + MOUTH_CLASSES

PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# Exceptions and data model
# ---------------------------------------------------------------------------


class DatasetError(Exception):
    """Veri seti dogrulama veya hazirlama hatasi."""


@dataclass
class SubsetReport:
    """Tek bir alt veri seti (eye veya mouth) raporu."""

    name: str
    classes: Tuple[str, ...]
    counts_train: Dict[str, int]
    counts_test: Dict[str, int]
    total_train: int
    total_test: int


@dataclass
class DatasetReport:
    """Iki alt veri setinin birlesik raporu (Requirement 1.7)."""

    root: Path
    eye: SubsetReport
    mouth: SubsetReport

    @property
    def grand_total(self) -> int:
        return (
            self.eye.total_train
            + self.eye.total_test
            + self.mouth.total_train
            + self.mouth.total_test
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_valid_images(folder: Path) -> Iterator[Path]:
    """``folder`` icindeki kabul edilen uzantili dosyalari sirali doner."""
    if not folder.is_dir():
        return
    for entry in sorted(folder.iterdir(), key=lambda p: p.name):
        if entry.is_file() and entry.suffix.lower() in ALLOWED_EXTENSIONS:
            yield entry


def _validate_layout(source_root: Path) -> None:
    """``source_root`` altinda gerekli dort sinifin train/test klasorlerini dogrular."""
    if not source_root.exists():
        raise DatasetError(f"Veri seti kok dizini bulunamadi: {source_root}")
    if not source_root.is_dir():
        raise DatasetError(f"Veri seti kokuu bir dizin degil: {source_root}")
    for split in _SPLITS:
        split_dir = source_root / split
        if not split_dir.is_dir():
            raise DatasetError(f"Beklenen alt dizin bulunamadi: {split_dir}")
        for cls in _ALL_CLASSES:
            class_dir = split_dir / cls
            if not class_dir.is_dir():
                raise DatasetError(f"Beklenen sinif klasoru bulunamadi: {class_dir}")


def _count_split(
    source_root: Path, split: str, classes: Tuple[str, ...]
) -> Dict[str, int]:
    """Belirli bir split'te sinif basina kabul edilen goruntu sayisini doner.

    Bos sinif klasoru :class:`DatasetError` yukseltir.
    """
    counts: Dict[str, int] = {}
    for cls in classes:
        class_dir = source_root / split / cls
        n = sum(1 for _ in _iter_valid_images(class_dir))
        if n == 0:
            raise DatasetError(
                f"Sinif klasoru gecerli goruntu icermiyor: "
                f"sinif='{cls}', yol={class_dir}"
            )
        counts[cls] = n
    return counts


def _copy_split(
    source_root: Path,
    output_root: Path,
    split: str,
    classes: Tuple[str, ...],
) -> None:
    """``source_root/<split>/<cls>`` -> ``output_root/<split>/<cls>``."""
    for cls in classes:
        src_dir = source_root / split / cls
        dst_dir = output_root / split / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img_path in _iter_valid_images(src_dir):
            shutil.copy2(img_path, dst_dir / img_path.name)


def _write_data_yaml(output_root: Path, classes: Tuple[str, ...]) -> None:
    """YOLOv8 siniflandirma ``data.yaml`` yazar (deterministik indeksler)."""
    names: Dict[int, str] = {idx: cls for idx, cls in enumerate(classes)}
    payload = {
        "path": str(output_root.resolve()),
        "train": "train",
        "val": "test",
        "names": names,
    }
    data_yaml = output_root / "data.yaml"
    with data_yaml.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)


def _build_subset(
    source_root: Path,
    output_root: Path,
    name: str,
    classes: Tuple[str, ...],
) -> SubsetReport:
    """Tek bir alt veri seti olustur (validate, count, copy, yaml)."""
    counts_train = _count_split(source_root, "train", classes)
    counts_test = _count_split(source_root, "test", classes)

    output_root.mkdir(parents=True, exist_ok=True)
    _copy_split(source_root, output_root, "train", classes)
    _copy_split(source_root, output_root, "test", classes)
    _write_data_yaml(output_root, classes)

    return SubsetReport(
        name=name,
        classes=classes,
        counts_train=counts_train,
        counts_test=counts_test,
        total_train=sum(counts_train.values()),
        total_test=sum(counts_test.values()),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_dataset(
    source_root: PathLike, output_root: PathLike
) -> DatasetReport:
    """``source_root`` icindeki eye_yawn_dataset'i iki alt veri setine ayirir.

    Cikti yapisi::

        <output_root>/
          eye/
            train/{Closed,Open}/...
            test/{Closed,Open}/...
            data.yaml
          mouth/
            train/{no_yawn,yawn}/...
            test/{no_yawn,yawn}/...
            data.yaml

    Once tum sinif klasorlerini dogrular ve sayar, ardindan ciktiya
    yazar; herhangi bir dogrulama hatasi olursa ``output_root``
    altina hicbir dosya yazilmaz (Requirement 1.5, 1.6).
    """
    src = Path(os.fspath(source_root))
    out = Path(os.fspath(output_root))

    # ---- Phase 1: validate every class folder up front ----
    _validate_layout(src)

    # Pre-count both subsets to fail fast before any I/O on `out`.
    eye_train = _count_split(src, "train", EYE_CLASSES)
    eye_test = _count_split(src, "test", EYE_CLASSES)
    mouth_train = _count_split(src, "train", MOUTH_CLASSES)
    mouth_test = _count_split(src, "test", MOUTH_CLASSES)

    # ---- Phase 2: materialise both subsets ----
    out.mkdir(parents=True, exist_ok=True)

    eye_root = out / "eye"
    eye_root.mkdir(parents=True, exist_ok=True)
    _copy_split(src, eye_root, "train", EYE_CLASSES)
    _copy_split(src, eye_root, "test", EYE_CLASSES)
    _write_data_yaml(eye_root, EYE_CLASSES)
    eye_report = SubsetReport(
        name="eye",
        classes=EYE_CLASSES,
        counts_train=eye_train,
        counts_test=eye_test,
        total_train=sum(eye_train.values()),
        total_test=sum(eye_test.values()),
    )

    mouth_root = out / "mouth"
    mouth_root.mkdir(parents=True, exist_ok=True)
    _copy_split(src, mouth_root, "train", MOUTH_CLASSES)
    _copy_split(src, mouth_root, "test", MOUTH_CLASSES)
    _write_data_yaml(mouth_root, MOUTH_CLASSES)
    mouth_report = SubsetReport(
        name="mouth",
        classes=MOUTH_CLASSES,
        counts_train=mouth_train,
        counts_test=mouth_test,
        total_train=sum(mouth_train.values()),
        total_test=sum(mouth_test.values()),
    )

    return DatasetReport(root=out.resolve(), eye=eye_report, mouth=mouth_report)


__all__: Iterable[str] = (
    "EYE_CLASSES",
    "MOUTH_CLASSES",
    "ALLOWED_EXTENSIONS",
    "DatasetError",
    "SubsetReport",
    "DatasetReport",
    "prepare_dataset",
)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def _print_subset_report(subset: SubsetReport, label: str) -> None:
    """Tek bir alt veri setinin sayilarini ve toplamlarini yazdirir."""
    print(f"=== {label} ({subset.name}/) ===")
    for cls in subset.classes:
        print(f"  train/{cls}: {subset.counts_train[cls]}")
    for cls in subset.classes:
        print(f"  test/{cls}:  {subset.counts_test[cls]}")
    print(f"  Toplam train: {subset.total_train}")
    print(f"  Toplam test:  {subset.total_test}")


def main() -> None:
    """``python -m src.data_prep`` giris noktasi (Requirement 1.7)."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 siniflandirma icin iki ayri veri seti hazirlar"
    )
    parser.add_argument("--source", required=True, help="Kaynak veri seti kokuu")
    parser.add_argument("--output", required=True, help="Cikti veri seti kokuu")
    args = parser.parse_args()

    try:
        report = prepare_dataset(args.source, args.output)
    except DatasetError as error:
        print(f"Hata: {error}", file=sys.stderr)
        sys.exit(2)

    _print_subset_report(report.eye, "Goz veri seti")
    print()
    _print_subset_report(report.mouth, "Agiz veri seti")
    print()
    print(f"Genel toplam: {report.grand_total}")


if __name__ == "__main__":
    main()
