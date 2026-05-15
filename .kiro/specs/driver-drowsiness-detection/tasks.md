# Implementation Plan: Driver Drowsiness Detection

## Overview

Convert the feature design into a series of prompts for a code-generation LLM
that will implement each step with incremental progress. Make sure that each
prompt builds on the previous prompts, and ends with wiring things together.
There should be no hanging or orphaned code that isn't integrated into a
previous step. Focus ONLY on tasks that involve writing, modifying, or
testing code.

The implementation language is **Python** (3.8 - 3.11). Tasks are ordered to
build the pure core (Uyari_Mantigi, Yapilandirma) first so property-based
tests (P1-P9) can be exercised early, then layer adapters (Tahminci,
Kamera_Yakalayicisi) and finally the kabuk components (Sesli_Uyarici, main
loop, Jetson optimizations).

## Tasks

- [x] 1. Set up project skeleton and dependencies
  - [x] 1.1 Create directory and module skeleton with pinned dependencies
    - Create `dataset/`, `models/`, `logs/`, `assets/`, `src/`, `tests/`
      folders
    - Create empty module files: `src/__init__.py`, `src/config.py`,
      `src/logger.py`, `src/alert_logic.py`, `src/data_prep.py`,
      `src/predictor.py`, `src/train.py`, `src/static_predict.py`,
      `src/webcam_detect.py`, `src/sound_alert.py`, `src/export.py`,
      `src/utils.py`
    - Create `tests/__init__.py` and `tests/conftest.py` with a shared
      Hypothesis profile (`max_examples=100, deadline=None`)
    - Write `requirements.txt` with pinned versions for `ultralytics`,
      `opencv-python`, `pyyaml`, `numpy`, `playsound`, `pytest`,
      `hypothesis`, `pytest-cov`, `pyfakefs`; add a header comment that
      states supported Python is 3.8-3.11
    - Create `tests/strategies.py` with placeholders for the Hypothesis
      strategies listed in the design
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

  - [x] 1.2 Create README.md with required Turkish sections
    - Sections: Kurulum, Eğitim, Statik Tahmin, Canlı Tespit, Jetson Nano
      Taşıma (en az 3 adım)
    - Include runnable command-line examples for each section
    - _Requirements: 11.5, 11.6_

- [x] 2. Implement configuration module (saf çekirdek)
  - [x] 2.1 Implement AppConfig dataclass and validate function in `src/config.py`
    - Define frozen `AppConfig` dataclass with all fields listed in the
      design (alert thresholds, camera, performance, model, sound,
      logging)
    - Define `DEFAULTS` dict with documented default values
    - Define `ConfigError` exception that carries parameter name, value,
      and reason
    - Implement pure `validate(raw: dict) -> AppConfig` that:
      enforces ranges from Property 8, fills missing keys from `DEFAULTS`
      and emits a WARNING-level record per missing key, raises
      `ConfigError` for out-of-range / wrong-type values, and verifies
      that `model_path` exists on disk
    - _Requirements: 5.5, 6.5, 8.1, 8.4, 8.5, 9.4, 10.7_

  - [x] 2.2 Implement YAML loader and default `config.yaml`
    - Add `load_config(path: Path = Path("config.yaml")) -> AppConfig` to
      `src/config.py` that reads YAML and calls `validate`; raise
      `ConfigError` on parse errors / missing source
    - Create top-level `config.yaml` with the schema documented in the
      design (all defaults populated)
    - _Requirements: 8.2, 8.3, 8.6_

  - [ ]* 2.3 Write property test for configuration validator boundaries
    - **Property 8: Yapılandırma doğrulayıcı sınırları korur**
    - **Validates: Requirements 5.5, 6.5, 8.1, 8.5, 9.4, 10.7**

  - [ ]* 2.4 Write property test for missing-parameter defaulting
    - **Property 9: Eksik parametreler varsayılana düşer ve uyarı log'u
      üretir**
    - **Validates: Requirements 8.4**

- [x] 3. Implement logger and FPS tracker
  - [x] 3.1 Implement `build_logger` with rotating file handler
    - In `src/logger.py`, build a `logging.Logger` with a
      `StreamHandler` (stdout) and a `RotatingFileHandler`
      (`cfg.log_file`, `maxBytes=cfg.log_max_bytes`, `backupCount=5`)
    - Format: `'%(asctime)s [%(levelname)s] %(message)s'` with
      `asctime` fixed to `'%Y-%m-%d %H:%M:%S'`
    - Wrap the file handler so write failures fall back to a stderr
      warning without raising and retry every 30 seconds
    - _Requirements: 9.1, 9.2, 9.6, 9.7_

  - [x] 3.2 Implement `FpsTracker` with rolling-window FPS
    - Add `FpsTracker.tick(t_now_s)`, `average_fps(window_s=1.0)`,
      `maybe_log(t_now_s, interval_s)` to `src/logger.py`
    - 30-second rolling window for the Jetson performance check; emit
      `WARNING` with parameter advice when avg FPS < 5 for >=30 s
    - Use `cfg.fps_log_interval` (clamped to 1.0..60.0; emit warning
      when out of range and fall back to 1.0)
    - _Requirements: 9.3, 9.4, 10.9, 10.10_

  - [ ]* 3.3 Write unit and property tests for logger and FpsTracker
    - Unit: rotation when size exceeds limit, ISO 8601 format, retry on
      write failure, `verbose` per-frame line
    - Property: monotonic FPS averaging and `fps_log_interval` clamping
    - _Requirements: 9.1, 9.2, 9.5, 9.6, 9.7_

- [x] 4. Implement Uyari_Mantigi pure core
  - [x] 4.1 Define alert data models and initial state
    - In `src/alert_logic.py`, define frozen dataclasses `Prediction`,
      `AlertConfig`, `AlertState`, `AlertEvent`; define `INITIAL_STATE`
      and `Literal` types for `kind`
    - Add invariants enforcement: `eye_conf` and `mouth_conf` must lie
      in `[0.0, 1.0]`; `t_capture_s` must be non-decreasing across calls
    - _Requirements: 5.1, 6.1_

  - [x] 4.2 Implement eye state machine inside pure `update`
    - Implement the `update(state, pred, cfg) -> (AlertState, [AlertEvent])`
      function for the eye branch only: set `eye_closed_start_s` on the
      first `Closed` after `Open`, accumulate `Kapali_Goz_Suresi`, fire
      one `DROWSY` event when threshold reached, suppress repeats while
      eyes stay closed, clear on `Open`, ignore `eye=None`
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.6, 5.7_

  - [ ]* 4.3 Write property test for drowsy alert count
    - **Property 1: Uyku uyarısı yalnızca eşik aşıldığında üretilir**
    - **Validates: Requirements 5.3, 5.6**

  - [ ]* 4.4 Write property test for open-eye reset
    - **Property 2: Açılan göz sayacı sıfırlar**
    - **Validates: Requirements 5.1, 5.2, 5.4**

  - [x] 4.5 Implement mouth state machine inside pure `update`
    - Extend `update` for the mouth branch: append `t_capture_s` to
      `yawn_event_times_s` on `no_yawn -> yawn` transitions, debounce
      while `in_yawn_block`, drop events older than `yawn_time_window`,
      fire one `FATIGUE` event when count >= `yawn_count`, clear when
      count drops back below, ignore `mouth=None`
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.6, 6.7_

  - [ ]* 4.6 Write property test for yawn-window metamorphic behaviour
    - **Property 3: Esneme zaman penceresi metamorfik özelliği**
    - **Validates: Requirements 6.2, 6.4**

  - [ ]* 4.7 Write property test for single-yawn-block counting
    - **Property 4: Tek esneme tek olay sayılır**
    - **Validates: Requirements 6.1, 6.6**

  - [ ]* 4.8 Write property test for threshold monotonicity
    - **Property 5: Yapılandırma eşikleri davranışı monoton belirler**
    - **Validates: Requirements 5.5, 6.5**

  - [ ]* 4.9 Write property test for low-confidence frames being ignored
    - **Property 6: Düşük güvenli kareler durumu değiştirmez**
    - **Validates: Requirements 5.7, 6.7**

- [x] 5. Checkpoint - core property tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Veri_Hazirlayici (data preparation)
  - [x] 6.1 Implement `prepare_dataset` and `DatasetReport`
    - In `src/data_prep.py`, define `EXPECTED_CLASSES`,
      `ALLOWED_EXTENSIONS`, `DatasetError`, and `DatasetReport`
    - Implement `prepare_dataset(source_root, output_root)` that
      validates the four class folders under `train/` and `test/`,
      counts valid images by extension, writes the YOLOv8
      classification `data.yaml` (deterministic class indices), and
      returns a populated `DatasetReport`
    - Raise `DatasetError` (exit code 2) for missing or empty class
      folders without producing any output artifacts
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x] 6.2 Add CLI entry point and per-class reporting
    - Add `python -m src.data_prep` argparse interface; on success
      print each class line, train/test totals, and grand total
    - _Requirements: 1.7_

  - [ ]* 6.3 Write unit and property tests for data preparation
    - Use `pyfakefs` for filesystem isolation
    - Unit: missing class, empty class, mixed extensions, deterministic
      indices in `data.yaml`
    - Property: extension-filter invariant (count of accepted images is
      independent of order, equals number of `.jpg|.jpeg|.png` files)
    - _Requirements: 1.1, 1.2, 1.5, 1.6, 1.7_

- [x] 7. Implement Tahminci (prediction adapter)
  - [x] 7.1 Implement `Predictor` class with model loading
    - In `src/predictor.py`, implement `Predictor.__init__` that loads
      `models/best.pt` (or path from `AppConfig.model_path`), validates
      that the file exists, stores `inference_resolution` and
      `confidence_threshold`
    - Raise `ConfigError` with the missing file path when the weight
      file is absent
    - _Requirements: 3.5, 8.5, 10.8_

  - [x] 7.2 Implement `predict_image` and `predict_frame`
    - Resize input to `inference_resolution` when set, run the
      Ultralytics classification model, return a `Prediction` with
      `argmax({Closed, Open})` and `argmax({yawn, no_yawn})`
    - Replace eye / mouth labels with `None` whenever the corresponding
      confidence falls below `confidence_threshold`
    - Validate input file path / extension and raise a clear error for
      missing or unsupported files
    - Print top-class label and four per-class scores formatted to two
      decimals when called from CLI
    - _Requirements: 3.1, 3.2, 3.4, 4.3, 10.8_

  - [ ]* 7.3 Write unit and property tests for Predictor
    - Mock the Ultralytics call to return controlled score arrays
    - Unit: missing weight file, unsupported extension, corrupted image
    - Property: top class equals argmax of returned scores; all per-class
      scores within `[0.0, 1.0]`; low-confidence collapses to `None`
    - _Requirements: 3.1, 3.2, 3.4, 3.5_

- [x] 8. Implement Egitici (training pipeline)
  - [x] 8.1 Implement `TrainConfig` and `validate_train_config`
    - In `src/train.py`, implement the dataclass, range checks
      (`epochs 1..500`, `imgsz 320..1280` step 32, `batch 1..128`,
      `model_size` set), and CLI override merge logic
    - Raise `ConfigError` with parameter name / received value /
      expected range for invalid inputs without starting training
    - _Requirements: 2.2, 2.3, 2.7_

  - [x] 8.2 Implement `run_training` entry point
    - Wrap `ultralytics.YOLO(...).train(...)` for classification mode,
      ensure `models/` exists, copy the best checkpoint to
      `models/best.pt` after training completes
    - Print per-epoch line (`current/total`, loss, val metric, elapsed)
      and a single final-metrics line (`accuracy` for classification)
    - Bail out without modifying `models/` when the data root or
      `data.yaml` is missing
    - _Requirements: 2.1, 2.4, 2.5, 2.6, 2.8_

  - [ ]* 8.3 Write unit tests for trainer with mocked Ultralytics
    - Verify CLI overrides win over config; per-epoch log format;
      `models/best.pt` written only on success; data-missing path leaves
      existing weights untouched
    - _Requirements: 2.2, 2.3, 2.4, 2.5, 2.7, 2.8_

- [x] 9. Implement static-image prediction CLI
  - [x] 9.1 Implement `static_predict.py` entry point
    - Argparse: `--image`, `--show/--save`, `--output`
    - Use `Predictor.predict_image`, render top-class label + score on
      the image (`cv2.putText` + background rect), default to display
      when neither show nor save is provided
    - Exit non-zero with explicit message for invalid path / unsupported
      extension / missing weight file
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

  - [ ]* 9.2 Write unit tests for static prediction CLI
    - Cover: success path with `--show`, success with `--save`, missing
      file, unsupported extension, missing weights
    - _Requirements: 3.3, 3.4, 3.5_

- [x] 10. Implement Kamera_Yakalayicisi and overlay
  - [x] 10.1 Implement `CameraCapture.open/close` with timeout handling
    - In `src/webcam_detect.py`, implement `CameraConfig` and
      `CameraCapture` using `cv2.VideoCapture`; bail out with
      non-zero exit code if the source cannot open within
      `open_timeout_s` (5 s default), accept index `0..9` or string
      sources up to 260 chars
    - On open success, set capture properties to target >= 15 FPS at
      640x480
    - _Requirements: 4.1, 4.2, 4.5_

  - [x] 10.2 Implement `frames` iterator with `frame_skip` and failure detection
    - Yield only every n-th frame for prediction (`frame_skip` 1..10,
      clamp out-of-range to 1 with a WARNING), keep the display loop on
      every frame, surface `q` keypress within 1 s to terminate cleanly
    - Trigger termination after 30 consecutive read failures or 3 s
      without a new frame; release resources before exit
    - _Requirements: 4.6, 4.7, 10.6, 10.7_

  - [x] 10.3 Implement `Goruntu_Bindirici` overlay helper
    - Draw `Goz_Durumu`, `Agiz_Durumu`, and active alert text on the
      frame using `cv2.putText` plus a contrasting background
      rectangle; auto-size font so glyphs are >= 3% of frame height and
      total overlay area is <= 25% of the frame
    - _Requirements: 4.4_

  - [ ]* 10.4 Write property test for frame-skip alert timing
    - **Property 7: Frame-skip uyarı zamanlamasını yalnızca tek kare
      çözünürlüğünde geciktirir**
    - **Validates: Requirements 10.6**

- [x] 11. Implement Sesli_Uyarici
  - [x] 11.1 Implement platform-aware audio backends
    - In `src/sound_alert.py`, implement `SoundAlerter` that selects
      `winsound.PlaySound(..., SND_ASYNC)` on Windows and `playsound`
      (with `aplay` subprocess fallback) on Linux/Jetson
    - Honor `enable_sound=False` by making `play` a no-op; cap audible
      duration to 3 s
    - Start playback within 500 ms of the triggering call when enabled
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [x] 11.2 Implement concurrency dedupe and error handling
    - Suppress new `play(kind)` calls of the same kind while a previous
      sound for that kind is still active
    - Catch any audio exception, log it via the shared logger with
      timestamp + error type, and never propagate
    - _Requirements: 7.6, 7.7_

  - [ ]* 11.3 Write unit and property tests for SoundAlerter
    - Unit: file-not-found, decode error, disabled flag, both backends
    - Property: among any sequence of `play(kind)` calls, the number
      reaching the backend within one playback window equals 1 per kind
      (concurrency dedupe invariant)
    - _Requirements: 7.1, 7.2, 7.6, 7.7_

- [x] 12. Wire end-to-end webcam detection and Jetson optimisations
  - [x] 12.1 Wire main loop in `webcam_detect.py`
    - Compose `load_config` -> `build_logger` -> `Predictor` ->
      `CameraCapture` -> `AlertEngine` -> `SoundAlerter` ->
      `Goruntu_Bindirici`
    - Per accepted frame: predict, call `update`, push events to
      logger and `SoundAlerter`, draw overlay, log per-frame verbose
      output when configured, drive `FpsTracker.maybe_log`
    - Add `python -m src.webcam_detect` argparse entry point with
      `--config` override and clean shutdown handling
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 5.3, 6.3, 7.1, 9.1, 9.2, 9.3, 9.5_

  - [x] 12.2 Implement Jetson model export and performance advisory
    - Add `src/export.py` with `export_model(path, fmt)` that calls the
      Ultralytics export API for `onnx`, builds a TensorRT engine for
      `tensorrt`, falls back to the original `.pt` on failure with an
      ERROR log entry
    - Wire `cfg.export_format` into `Predictor.__init__` so the
      converted artifact is used at inference time
    - Hook `FpsTracker` into the main loop so the < 5 FPS / 30 s
      condition emits the parameter-advice WARNING described in design
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.9, 10.10_

  - [ ]* 12.3 Write integration tests for the wired pipeline
    - Use a mocked `VideoCapture` that yields a deterministic sequence
      of frames and a mocked `Predictor` that returns scripted
      `Prediction` objects
    - Assert: drowsy alert fires when synthetic eyes stay closed past
      threshold; fatigue alert fires when synthetic yawns exceed
      `yawn_count` within window; `q` key exits within 1 s; failed
      ONNX export falls back to `.pt`
    - _Requirements: 4.3, 4.7, 5.3, 6.3, 10.5_

- [x] 13. Final checkpoint - full suite passes
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP;
  every property test is marked optional but each one maps 1:1 to a
  design property and is the cheapest way to catch regressions in the
  pure core.
- Each task references the specific acceptance-criteria sub-numbers it
  covers, so traceability flows requirements -> design -> task -> code.
- Property tests live in `tests/test_property_<n>_*.py` and import
  shared Hypothesis strategies from `tests/strategies.py`.
- Checkpoints at tasks 5 and 13 give natural break points to run the
  full suite before moving on.
- The implementation order builds the pure core (config + alert logic)
  first so most properties are exercised before any camera or model
  dependency is introduced.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "3.1", "4.1", "6.1", "7.1", "8.1", "10.1", "11.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4", "3.2", "4.2", "6.2", "7.2", "8.2", "10.2", "11.2"] },
    { "id": 3, "tasks": ["3.3", "4.3", "4.4", "4.5", "6.3", "7.3", "8.3", "9.1", "10.3", "11.3"] },
    { "id": 4, "tasks": ["4.6", "4.7", "4.8", "4.9", "9.2", "10.4", "12.1"] },
    { "id": 5, "tasks": ["12.2"] },
    { "id": 6, "tasks": ["12.3"] }
  ]
}
```
